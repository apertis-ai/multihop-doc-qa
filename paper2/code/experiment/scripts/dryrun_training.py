"""Dry-run smoke test of the STALI training pipeline.

Runs: load ColBERT → attach LoRA → build HF dataset → collate batch → forward pass
through Contrastive loss. No actual training — just verifies every integration
point works end-to-end before committing GPU hours to a real run.
"""
import sys
from pathlib import Path

# Allow running from experiment/ root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from pylate.models import ColBERT
from pylate.losses import Contrastive
from pylate.utils import ColBERTCollator

from src.training.train_stali import _triples_to_hf_dataset, _attach_lora

print("[DRYRUN] loading ColBERT", flush=True)
m = ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1", device="cuda")
print("[DRYRUN] attaching LoRA", flush=True)
m = _attach_lora(m, {"r": 16, "alpha": 32, "dropout": 0.1})

print("[DRYRUN] building dataset", flush=True)
ds = _triples_to_hf_dataset(
    Path("data/train/dochop_qa_triples_t02.jsonl"),
    use_section_prefix=True,
    n_hard_neg=3,
    n_random_neg=3,
)
print(f"[DRYRUN] dataset: n={len(ds)} cols={ds.column_names}", flush=True)
print(f"[DRYRUN] sample query: {ds[0]['query'][:80]}", flush=True)
print(f"[DRYRUN] sample positive: {ds[0]['positive'][:120]}", flush=True)
print(f"[DRYRUN] sample negative_1: {ds[0]['negative_1'][:120]}", flush=True)

print("[DRYRUN] testing collator", flush=True)
coll = ColBERTCollator(tokenize_fn=m.tokenize)
batch = coll([ds[0], ds[1]])
print(f"[DRYRUN] collator batch keys: {sorted(batch.keys())}", flush=True)
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)}", flush=True)

print("[DRYRUN] testing Contrastive loss forward", flush=True)
loss_fn = Contrastive(model=m)

sentence_features = []
for prefix in ["query", "positive", "negative_1", "negative_2", "negative_3",
               "negative_4", "negative_5", "negative_6"]:
    if f"{prefix}_input_ids" not in batch:
        continue
    sentence_features.append({
        "input_ids": batch[f"{prefix}_input_ids"].cuda(),
        "attention_mask": batch[f"{prefix}_attention_mask"].cuda(),
    })
print(f"[DRYRUN] sentence_features: {len(sentence_features)} groups", flush=True)

loss = loss_fn(sentence_features=sentence_features)
print(f"[DRYRUN] loss={loss.item():.4f}", flush=True)
print("[DRYRUN] backward pass", flush=True)
loss.backward()
print("[DRYRUN DONE] all pipeline components wired", flush=True)
