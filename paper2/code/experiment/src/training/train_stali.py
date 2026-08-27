"""STALI training entrypoint — real PyLate + PEFT wiring.

Trains a LoRA adapter on top of lightonai/GTE-ModernColBERT-v1 using DocHop-QA
triples. STALI's section-type signal is injected via a natural-language prefix
in the document text ("[SECTION: {type}] ..."), so the LoRA adapters learn the
section-type-aware re-weighting end-to-end without custom loss/collator.

Usage:
    python -m src.training.train_stali --config configs/s1_stali.yaml
    python -m src.training.train_stali --config configs/s2_plain_lora.yaml --no-section-prefix
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from pylate.losses import Contrastive
from pylate.models import ColBERT
from pylate.utils import ColBERTCollator
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from src.data.hard_negatives import filter_by_jaccard

log = logging.getLogger("train_stali")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SECTION_PREFIX_TEMPLATE = "[SECTION: {section_type}] {text}"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _maybe_prefix(text: str, section_type: str | None, use_prefix: bool) -> str:
    if use_prefix and section_type:
        return SECTION_PREFIX_TEMPLATE.format(section_type=section_type, text=text)
    return text


def _triples_to_hf_dataset(
    triples_path: Path,
    use_section_prefix: bool,
    n_hard_neg: int,
    n_random_neg: int,
    filter_hard_neg_only: bool = False,
    hard_neg_threshold: float | None = None,
) -> Dataset:
    """Convert JSONL triples to a sentence-transformers-compatible HF Dataset.

    Each row emits: query, positive, negative_1, ..., negative_N.
    Documents are prefixed with "[SECTION: {type}]" if use_section_prefix=True.
    Token truncation is handled by the configured ColBERT query/document lengths.
    Cohort filtering and Jaccard thresholds come from the selected YAML config.
    """
    total_neg = n_hard_neg + n_random_neg
    rows = []
    kept = skipped = filtered_no_hard = 0
    with triples_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            positives = rec.get("positives") or []
            hard_negs = filter_by_jaccard(
                rec.get("hard_negatives") or [], hard_neg_threshold
            )
            if filter_hard_neg_only and not hard_negs:
                filtered_no_hard += 1
                continue
            hard_negs = hard_negs[:n_hard_neg]
            random_negs = (rec.get("random_negatives") or [])[:n_random_neg]
            if not positives or (len(hard_negs) + len(random_negs)) < 1:
                skipped += 1
                continue

            # Pad negatives to exactly total_neg by cycling
            all_negs = hard_negs + random_negs
            while len(all_negs) < total_neg and all_negs:
                all_negs.append(all_negs[len(all_negs) % len(hard_negs + random_negs)])
            all_negs = all_negs[:total_neg]

            # Use the first positive per query (ColBERT Contrastive expects 1 pos + N neg)
            pos = positives[0]
            row = {
                "query": rec.get("query", ""),
                "positive": _maybe_prefix(
                    pos.get("text", ""),
                    pos.get("section_type"),
                    use_section_prefix,
                ),
            }
            for i, neg in enumerate(all_negs):
                row[f"negative_{i+1}"] = _maybe_prefix(
                    neg.get("text", ""),
                    neg.get("section_type"),
                    use_section_prefix,
                )
            rows.append(row)
            kept += 1

    log.info(
        "triples loaded: kept=%d skipped=%d filtered_no_hard_neg=%d",
        kept,
        skipped,
        filtered_no_hard,
    )
    return Dataset.from_list(rows)


def _attach_lora(model: ColBERT, lora_cfg_dict: dict) -> ColBERT:
    """Wrap the inner HF ModernBERT with PEFT LoRA adapters."""
    inner = model[0].auto_model if hasattr(model[0], "auto_model") else model.model
    import torch.nn as nn
    # Auto-discover attention linear leaves (resilient to naming drift)
    leaf_names = sorted({
        n.split(".")[-1]
        for n, m in inner.named_modules()
        if isinstance(m, nn.Linear) and any(k in n for k in ("attn", "attention"))
    })
    preferred = [n for n in ("Wqkv", "Wo") if n in leaf_names]
    if len(preferred) < 2:
        preferred = leaf_names[:4]  # fallback: any 4 attn linears
    log.info("LoRA target_modules=%s", preferred)

    cfg = LoraConfig(
        r=lora_cfg_dict.get("r", 16),
        lora_alpha=lora_cfg_dict.get("alpha", 32),
        lora_dropout=lora_cfg_dict.get("dropout", 0.1),
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=preferred,
    )
    wrapped = get_peft_model(inner, cfg)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    log.info("LoRA attached: trainable=%s total=%s pct=%.3f",
             f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Replace the inner model with the PEFT-wrapped one
    if hasattr(model[0], "auto_model"):
        model[0].auto_model = wrapped
    else:
        model.model = wrapped
    return model


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--no-section-prefix", is_flag=True,
              help="Strip section-type prefix for S2 plain ablation")
@click.option("--triples", type=click.Path(exists=True, path_type=Path), default=None,
              help="Override training.data.triples_path from the YAML config")
@click.option("--seed", type=int, default=None, help="Override the YAML training seed")
def main(config: Path, no_section_prefix: bool, triples: Path | None, seed: int | None) -> int:
    cfg = _load_config(config)
    data_cfg = cfg["training"]["data"]
    triples = triples or Path(data_cfg["triples_path"])
    if not triples.exists():
        raise click.ClickException(f"training triples not found: {triples}")
    system_id = cfg.get("system_id", "s_stali")
    use_prefix = not no_section_prefix
    log.info("system_id=%s use_section_prefix=%s triples=%s",
             system_id, use_prefix, triples)

    # ---------- 1. load ColBERT ----------
    model_cfg = cfg["model"]
    base_model = model_cfg["base"]
    log.info("loading base model %s", base_model)
    model = ColBERT(
        model_name_or_path=base_model,
        device="cuda",
        query_length=model_cfg.get("max_query_length"),
        document_length=model_cfg.get("max_doc_length"),
    )

    # ---------- 2. attach LoRA ----------
    model = _attach_lora(model, cfg["lora"])

    # ---------- 3. build HF dataset ----------
    n_hard = data_cfg.get("hard_neg_top_k", 1)
    n_rand = data_cfg.get("random_neg_k", 2)
    ds = _triples_to_hf_dataset(
        triples_path=triples,
        use_section_prefix=use_prefix,
        n_hard_neg=n_hard,
        n_random_neg=n_rand,
        filter_hard_neg_only=data_cfg.get("filter_hard_neg_only", False),
        hard_neg_threshold=data_cfg.get("hard_neg_threshold"),
    )
    log.info("dataset size=%d columns=%s", len(ds), ds.column_names)

    # ---------- 4. loss + collator ----------
    loss_fn = Contrastive(model=model)
    collator = ColBERTCollator(tokenize_fn=model.tokenize)

    # ---------- 5. training args ----------
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    run_seed = cfg["training"].get("seed", 42) if seed is None else seed
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["training"].get("num_epochs", 3),
        per_device_train_batch_size=cfg["training"].get("per_device_batch_size", 16),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 2),
        learning_rate=float(cfg["training"].get("learning_rate", 3.0e-5)),
        warmup_steps=cfg["training"].get("warmup_steps", 200),
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        seed=run_seed,
        remove_unused_columns=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        loss=loss_fn,
        data_collator=collator,
    )

    log.info("starting training: %d epochs, batch=%d, grad_accum=%d",
             args.num_train_epochs, args.per_device_train_batch_size,
             args.gradient_accumulation_steps)
    trainer.train()

    # ---------- 6. save LoRA adapter + metadata ----------
    adapter_dir = out_dir / "adapter"
    inner = model[0].auto_model if hasattr(model[0], "auto_model") else model.model
    inner.save_pretrained(str(adapter_dir))
    log.info("saved LoRA adapter to %s", adapter_dir)

    meta = {
        "system_id": system_id,
        "base_model": base_model,
        "use_section_prefix": use_prefix,
        "n_train_samples": len(ds),
        "epochs": args.num_train_epochs,
        "batch_size": args.per_device_train_batch_size,
        "seed": run_seed,
        "lora": cfg["lora"],
        "triples_path": str(triples),
        "filter_hard_neg_only": data_cfg.get("filter_hard_neg_only", False),
        "hard_neg_threshold": data_cfg.get("hard_neg_threshold"),
        "query_length": model.query_length,
        "document_length": model.document_length,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("[TRAIN DONE] %s", system_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(standalone_mode=False) or 0)
