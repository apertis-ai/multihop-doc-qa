"""Prepare HotpotQA distractor dataset for STALI A2 training.

HotpotQA distractor setting: supporting facts are Wikipedia paragraphs.
We treat each paragraph as a retrieval unit (section-level grain, consistent with MultiHop-RAG).

Samples n=500 from the HotpotQA dev set and generates:
- passages.jsonl: {passage_id, title, text} (one paragraph per line)
- queries.jsonl: {qid, question, answer} (one query per line)
- qrels.jsonl: {qid, passage_id, rel} (positive judgments)

Usage:
    python3 w3_prep_hotpotqa.py

Output paths (relative to the working directory):
    e_hotpotqa_prep/passages.jsonl
    e_hotpotqa_prep/queries.jsonl
    e_hotpotqa_prep/qrels.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import random

# Fixed seed for reproducibility
SEED = 42
N_SAMPLES = 500

OUT_DIR = Path("data/hotpotqa")

def main() -> None:
    """Load HotpotQA, sample, convert to section-grain triplet format."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets library not found; install: pip install datasets")

    random.seed(SEED)

    # Load HotpotQA distractor validation set (HF naming for "dev")
    print("Loading HotpotQA distractor validation set...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    print(f"  Total dev queries: {len(ds)}")

    # Sample n=500
    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)
    sampled_indices = all_indices[:N_SAMPLES]
    print(f"  Sampled: {N_SAMPLES} queries")

    # State: {passage_id → passage, qid → qrel_list}
    passages = {}  # passage_id → {passage_id, title, text}
    qrels = {}     # qid → list of passage_ids
    queries = {}   # qid → {qid, question, answer}

    passage_counter = 0

    for idx in sampled_indices:
        example = ds[idx]

        qid = f"hotpot_{idx}"
        question = example.get("question", "")
        answer = example.get("answer", "")

        # HF dict-of-lists format: supporting_facts = {"title": [...], "sent_id": [...]}
        sf = example.get("supporting_facts", {}) or {}
        support_titles = set(sf.get("title", []))

        # context = {"title": [...], "sentences": [[...], [...], ...]} in dict-of-lists format
        ctx = example.get("context", {}) or {}
        ctx_titles = ctx.get("title", [])
        ctx_sentences = ctx.get("sentences", [])
        ctx_by_title = dict(zip(ctx_titles, ctx_sentences))

        # Add ALL 10 paragraphs (gold + distractors) to the haystack;
        # qrels point only at gold (supporting-fact) paragraphs.
        qrel_list = []
        for title, sents in ctx_by_title.items():
            para_text = " ".join(sents).strip()
            if not para_text:
                continue

            pid = f"hotpot_p{passage_counter}"
            passages[pid] = {
                "passage_id": pid,
                "title": title,
                "text": para_text,
            }
            if title in support_titles:
                qrel_list.append(pid)
            passage_counter += 1

        if not qrel_list:
            print(f"  WARN: qid {qid} has no supporting passages")
            continue

        queries[qid] = {
            "qid": qid,
            "question": question,
            "answer": answer,
        }
        qrels[qid] = qrel_list

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    passages_path = OUT_DIR / "passages.jsonl"
    with open(passages_path, "w") as f:
        for pid in sorted(passages.keys()):
            f.write(json.dumps(passages[pid]) + "\n")

    queries_path = OUT_DIR / "queries.jsonl"
    with open(queries_path, "w") as f:
        for qid in sorted(queries.keys()):
            f.write(json.dumps(queries[qid]) + "\n")

    qrels_path = OUT_DIR / "qrels.jsonl"
    with open(qrels_path, "w") as f:
        for qid in sorted(qrels.keys()):
            for pid in qrels[qid]:
                f.write(json.dumps({"qid": qid, "passage_id": pid, "rel": 1}) + "\n")

    print(f"✓ passages written: {passages_path}  n={len(passages)}")
    print(f"✓ queries  written: {queries_path}  n={len(queries)}")
    print(f"✓ qrels    written: {qrels_path}  n={sum(len(v) for v in qrels.values())}")
    print(f"  example query: {list(queries.values())[0] if queries else 'NONE'}")

if __name__ == "__main__":
    main()
