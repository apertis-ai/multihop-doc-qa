"""Build E6 MultiHop-RAG training triples.

Mirrors the A2 recipe (random-only negatives, k=3):
  - Positives: gold paragraphs resolved via evidence_list[i].fact substring match.
  - Random negatives: paragraphs sampled uniformly from the corpus, excluding any
    paragraph that shares a doc_id with any gold paragraph for this query.
  - Held-out queries (n=500, seed=42) are excluded from training (no leak).
  - Queries with zero gold paragraphs (the 301 null_query distractors) are skipped.

Output JSONL schema matches train_stali.py expectations:
  {
    "query_id": str,
    "query": str,
    "positives":      [{"text": str, "doc_id": str, "para_id": str}],
    "hard_negatives": [],   # E6 follows A2 random-only; left empty
    "random_negatives": [{"text": str, "doc_id": str, "para_id": str}],
  }

Usage:
    cd code/harness
    python build_triples_multihoprag.py \
      --corpus /path/to/multihoprag/corpus.json \
      --queries /path/to/multihoprag/MultiHopRAG.json \
      --out ../experiment/data/train/multihoprag_triples.jsonl \
      --random-neg-k 3 --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path


def _load_loader():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "mh_loader", here / "datasets" / "multihoprag.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mh_loader"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-neg-k", type=int, default=3)
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mh = _load_loader()
    chunks, t2d = mh.load_corpus(Path(args.corpus))
    queries = mh.load_queries(Path(args.queries), chunks, t2d)
    train_q, heldout_q = mh.split_train_heldout(
        queries, n_heldout=args.n_heldout, seed=args.seed
    )

    chunks_by_id = {c.id: c for c in chunks}
    all_para_ids = [c.id for c in chunks]
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped_no_gold = 0
    with out_path.open("w") as fh:
        for q in train_q:
            if not q.gold_paragraph_ids:
                skipped_no_gold += 1
                continue
            # Positives: all gold paragraphs (train_stali uses positives[0])
            positives = []
            for pid in sorted(q.gold_paragraph_ids):
                c = chunks_by_id[pid]
                positives.append(
                    {"text": c.text, "doc_id": c.doc_id, "para_id": c.id}
                )
            # Random negatives: sample paragraphs whose doc_id ∉ gold_doc_ids
            random_negatives = []
            tries = 0
            while len(random_negatives) < args.random_neg_k and tries < args.random_neg_k * 50:
                pid = all_para_ids[rng.randrange(len(all_para_ids))]
                c = chunks_by_id[pid]
                if c.doc_id in q.gold_doc_ids:
                    tries += 1
                    continue
                random_negatives.append(
                    {"text": c.text, "doc_id": c.doc_id, "para_id": c.id}
                )
            rec = {
                "query_id": q.qid,
                "query": q.question,
                "positives": positives,
                "hard_negatives": [],
                "random_negatives": random_negatives,
            }
            fh.write(json.dumps(rec) + "\n")
            kept += 1

    manifest = {
        "n_train_triples": kept,
        "n_skipped_no_gold": skipped_no_gold,
        "n_heldout_excluded": len(heldout_q),
        "total_queries": len(queries),
        "random_neg_k": args.random_neg_k,
        "hard_neg_k": 0,
        "seed": args.seed,
        "corpus_paragraphs": len(chunks),
        "corpus_docs": len(t2d),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Also dump heldout query ids for downstream eval
    heldout_path = out_path.parent / "multihoprag_heldout_qids.json"
    heldout_path.write_text(
        json.dumps([q.qid for q in heldout_q], indent=2)
    )

    print(json.dumps(manifest, indent=2))
    print(f"wrote {kept} triples -> {out_path}")
    print(f"wrote {len(heldout_q)} heldout qids -> {heldout_path}")


if __name__ == "__main__":
    main()
