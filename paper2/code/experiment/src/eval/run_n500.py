"""n=500 DocHop-QA eval harness — section_r@5 computation.

Scores one system (baseline OR trained STALI adapter) on the fixed 500-query eval
set, computing:
    - section_recall_at_k for k in {5, 10}
    - document_recall_at_k for k in {5, 10}
    - per-query correctness breakdown (for paired McNemar later)

Gold labels are loaded from data/eval/gold_n500.json (Paper 1 §4.1 n=500 mixed split).
Candidate corpus is assembled from the union of gold sections across all 500 queries
plus a configurable number of distractor sections from the DocHop-QA corpus.

This is the only path by which numbers enter Paper 2's Table 1.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import torch
import yaml

log = logging.getLogger("run_n500")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _section_id(pmc_id: str, section: str, subsection: str | None) -> str:
    """Composite section ID matching the released gold-data builder."""
    return f"{pmc_id}||{section}/{subsection or ''}".rstrip("/")


def _load_dochop(path: Path) -> list[dict]:
    with path.open() as fh:
        return json.load(fh)


def _build_corpus(
    dochop_records: list[dict],
    gold_query_ids: set[str],
) -> tuple[list[dict], dict[str, int]]:
    """Build a section corpus from DocHop-QA records.

    Strategy: include every section that any of the 500 eval queries cite as gold
    (required for recall to be computable) + every other unique section from the
    same papers (so retrieval ranks gold against within-paper distractors). This
    matches Paper 1's corpus construction where the candidate set = union of
    sections from the queries' gold papers.
    """
    eval_pmcs: set[str] = set()
    for rec in dochop_records:
        qid = str(rec.get("id", ""))
        if qid in gold_query_ids:
            for ctx in rec.get("context_list") or []:
                if ctx.get("pmc_id"):
                    eval_pmcs.add(ctx["pmc_id"])

    sections: list[dict] = []
    seen: set[str] = set()
    for rec in dochop_records:
        for ctx in rec.get("context_list") or []:
            pmc = ctx.get("pmc_id") or ""
            if pmc not in eval_pmcs:
                continue
            sid = _section_id(pmc, ctx.get("section") or "", ctx.get("subsection"))
            if sid in seen:
                continue
            seen.add(sid)
            text = ctx.get("content") or ""
            if not text:
                continue
            sections.append({
                "sid": sid,
                "pmc_id": pmc,
                "section": ctx.get("section") or "",
                "subsection": ctx.get("subsection"),
                "text": text,
            })
    sid_to_idx = {s["sid"]: i for i, s in enumerate(sections)}
    return sections, sid_to_idx


def _maybe_prefix_section(section: dict, use_prefix: bool) -> str:
    if not use_prefix:
        return section["text"]
    # Re-run the section classifier at query time to avoid needing a cached index
    try:
        from src.data.section_classifier import classify_composite_key
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.data.section_classifier import classify_composite_key
    stype = classify_composite_key(section["pmc_id"], section["section"], section.get("subsection"))
    return f"[SECTION: {stype}] {section['text']}"


def _maxsim_score(q_embed: torch.Tensor, d_embed: torch.Tensor) -> float:
    """Standard ColBERT MaxSim: sum over query tokens of max dot-product with doc tokens."""
    # q_embed: (Tq, H), d_embed: (Td, H), assume L2-normalised
    sim = torch.einsum("qh,th->qt", q_embed.float(), d_embed.float())
    return sim.max(dim=1).values.sum().item()


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None,
              help="Optional system YAML supplying the model and token limits")
@click.option("--model", default=None, help="HF model id or local path for base model")
@click.option("--adapter", type=click.Path(path_type=Path), default=None,
              help="Optional PEFT LoRA adapter dir to load on top of --model")
@click.option("--dochop-json", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--gold-path", type=click.Path(exists=True, path_type=Path),
              default="data/eval/gold_n500.json")
@click.option("--use-section-prefix/--no-section-prefix", default=None,
              help="Override training.data.use_section_prefix from --config")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--system-id", default=None)
@click.option("--top-k", type=int, default=None,
              help="Override eval.top_k from --config (must be at least 10)")
@click.option("--max-query-length", type=click.IntRange(min=1), default=None,
              help="Override model.max_query_length from --config")
@click.option("--max-doc-length", type=click.IntRange(min=1), default=None,
              help="Override model.max_doc_length from --config")
def main(
    config: Path | None,
    model: str | None,
    adapter: Path | None,
    dochop_json: Path,
    gold_path: Path,
    use_section_prefix: bool | None,
    output: Path,
    system_id: str | None,
    top_k: int | None,
    max_query_length: int | None,
    max_doc_length: int | None,
) -> int:
    from pylate.models import ColBERT

    cfg = (yaml.safe_load(config.read_text()) or {}) if config else {}
    if not isinstance(cfg, dict):
        raise click.ClickException("--config must contain a YAML mapping")
    model_cfg = cfg.get("model") or {}
    if not isinstance(model_cfg, dict):
        raise click.ClickException("config model must be a YAML mapping")
    model = model or model_cfg.get("base")
    if not model:
        raise click.ClickException("provide --model or a --config containing model.base")
    system_id = system_id or cfg.get("system_id", "unknown")
    data_cfg = (cfg.get("training") or {}).get("data") or {}
    eval_cfg = cfg.get("eval") or {}
    if use_section_prefix is None:
        use_section_prefix = bool(data_cfg.get("use_section_prefix", False))
    if top_k is None:
        top_k = eval_cfg.get("top_k", 10)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 10:
        raise click.ClickException("--top-k/config eval.top_k must be an integer >= 10")
    max_query_length = max_query_length or model_cfg.get("max_query_length")
    max_doc_length = max_doc_length or model_cfg.get("max_doc_length")
    for name, value in (("max_query_length", max_query_length), ("max_doc_length", max_doc_length)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise click.ClickException(f"model.{name} must be a positive integer")

    log.info("system=%s model=%s adapter=%s prefix=%s top_k=%s query_length=%s document_length=%s",
             system_id, model, adapter, use_section_prefix, top_k,
             max_query_length, max_doc_length)

    log.info("loading gold labels")
    gold = json.loads(Path(gold_path).read_text())
    log.info("  n_eval_queries=%d", len(gold))

    # Build eval query → gold section set
    eval_qid_to_gold: dict[str, set[str]] = {}
    eval_qid_to_gold_docs: dict[str, set[str]] = {}
    for qid, g in gold.items():
        eval_qid_to_gold[qid] = set(g.get("relevant_section_ids") or [])
        eval_qid_to_gold_docs[qid] = set(g.get("relevant_doc_ids") or [])

    log.info("loading DocHop-QA records")
    dochop_records = _load_dochop(dochop_json)
    log.info("  total records: %d", len(dochop_records))

    gold_query_ids = set(gold.keys())
    sections, sid_to_idx = _build_corpus(dochop_records, gold_query_ids)
    log.info("section corpus: %d unique sections from gold papers", len(sections))

    # Warn: how many gold section ids are actually findable in the corpus?
    findable = 0
    for qid, gold_sids in eval_qid_to_gold.items():
        findable += sum(1 for sid in gold_sids if sid in sid_to_idx)
    total_gold = sum(len(v) for v in eval_qid_to_gold.values())
    log.info("  findable gold sections: %d / %d (%.1f%%)",
             findable, total_gold, 100 * findable / max(total_gold, 1))

    log.info("loading ColBERT base=%s", model)
    m = ColBERT(
        model_name_or_path=model,
        device="cuda",
        query_length=max_query_length,
        document_length=max_doc_length,
    )

    if adapter is not None:
        log.info("loading PEFT adapter from %s", adapter)
        from peft import PeftModel
        inner = m[0].auto_model if hasattr(m[0], "auto_model") else m.model
        wrapped = PeftModel.from_pretrained(inner, str(adapter))
        if hasattr(m[0], "auto_model"):
            m[0].auto_model = wrapped
        else:
            m.model = wrapped
        log.info("adapter attached")

    # ----- encode corpus -----
    log.info("encoding corpus (batched)")
    section_texts = [_maybe_prefix_section(s, use_section_prefix) for s in sections]
    batch_size = 16
    doc_embeds: list[torch.Tensor] = []
    for i in range(0, len(section_texts), batch_size):
        batch = section_texts[i : i + batch_size]
        emb = m.encode(batch, is_query=False, convert_to_tensor=True, show_progress_bar=False)
        # PyLate returns list when doc lengths differ
        if isinstance(emb, list):
            doc_embeds.extend([e.detach() for e in emb])
        else:
            doc_embeds.extend([emb[j].detach() for j in range(emb.shape[0])])
        if (i // batch_size) % 20 == 0:
            log.info("  corpus encoded %d/%d", min(i + batch_size, len(section_texts)), len(section_texts))

    # ----- eval loop -----
    log.info("scoring %d eval queries against %d sections", len(gold), len(sections))
    per_query = []
    for qi, (qid, g) in enumerate(gold.items()):
        q_text = g.get("query", "")
        q_emb = m.encode([q_text], is_query=True, convert_to_tensor=True, show_progress_bar=False)
        q0 = q_emb[0] if isinstance(q_emb, list) else q_emb[0]

        # Score all sections
        scores = torch.tensor([
            _maxsim_score(q0, d) for d in doc_embeds
        ])
        ranked_idx = torch.argsort(scores, descending=True).tolist()
        top_idx = ranked_idx[:top_k]
        top_sids = [sections[i]["sid"] for i in top_idx]
        top_pmcs = list(dict.fromkeys(sections[i]["pmc_id"] for i in ranked_idx))[:top_k]

        gold_sids = eval_qid_to_gold.get(qid, set())
        gold_docs = eval_qid_to_gold_docs.get(qid, set())

        sec_hit_5 = any(sid in gold_sids for sid in top_sids[:5])
        sec_hit_10 = any(sid in gold_sids for sid in top_sids[:10])
        doc_hit_5 = any(p in gold_docs for p in top_pmcs[:5])
        doc_hit_10 = any(p in gold_docs for p in top_pmcs[:10])

        # Section recall@5: fraction of gold sections in top-5
        sec_r5 = (
            sum(1 for sid in gold_sids if sid in top_sids[:5]) / max(len(gold_sids), 1)
        )
        doc_r5 = (
            sum(1 for p in gold_docs if p in top_pmcs[:5]) / max(len(gold_docs), 1)
        )

        per_query.append({
            "qid": qid,
            "query": q_text[:200],
            "n_gold_sections": len(gold_sids),
            "n_gold_docs": len(gold_docs),
            "top5_sids": top_sids[:5],
            "top5_pmcs": top_pmcs[:5],
            "section_recall_at_5": sec_r5,
            "doc_recall_at_5": doc_r5,
            "section_hit_at_5": sec_hit_5,
            "section_hit_at_10": sec_hit_10,
            "doc_hit_at_5": doc_hit_5,
            "doc_hit_at_10": doc_hit_10,
        })
        if (qi + 1) % 50 == 0:
            log.info("  query %d/%d done", qi + 1, len(gold))

    n = len(per_query)
    summary = {
        "system_id": system_id,
        "model": model,
        "adapter": str(adapter) if adapter else None,
        "use_section_prefix": use_section_prefix,
        "top_k": top_k,
        "query_length": m.query_length,
        "document_length": m.document_length,
        "n_queries": n,
        "section_recall_at_5_mean": sum(r["section_recall_at_5"] for r in per_query) / n,
        "doc_recall_at_5_mean": sum(r["doc_recall_at_5"] for r in per_query) / n,
        "section_hit_at_5_rate": sum(1 for r in per_query if r["section_hit_at_5"]) / n,
        "section_hit_at_10_rate": sum(1 for r in per_query if r["section_hit_at_10"]) / n,
        "doc_hit_at_5_rate": sum(1 for r in per_query if r["doc_hit_at_5"]) / n,
        "doc_hit_at_10_rate": sum(1 for r in per_query if r["doc_hit_at_10"]) / n,
    }
    log.info("[EVAL SUMMARY] %s", json.dumps(summary, indent=2))

    out = {"summary": summary, "per_query": per_query}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    log.info("wrote results to %s", output)
    log.info("[EVAL DONE] %s", system_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(standalone_mode=False) or 0)
