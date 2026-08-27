"""E6 reader+eval driver — feeds top-5 retrieved paragraphs to LLM provider Llama-3.1-8B,
computes paragraph_recall@{5,10}, doc_recall@5, EM, F1, and writes per-query
predictions for downstream stat tests.

Run AFTER run_e6_retrieval.py for each retriever:
  python run_e6_reader_eval.py \
    --retrieval runs/e6_bge_m3/retrieval.json \
    --corpus    /path/to/multihoprag/corpus.json \
    --out       runs/e6_bge_m3/eval.json \
    --top-k-reader 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))


def _imp(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, H / relpath)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k-reader", type=int, default=5)
    args = ap.parse_args()

    mh = _imp("mh_loader", "datasets/multihoprag.py")
    metrics = _imp("metrics", "metrics.py")
    llm = _imp("llm_reader", "readers/llm_reader.py")

    chunks, _ = mh.load_corpus(args.corpus)
    chunks_by_id = {c.id: c for c in chunks}

    rdata = json.loads(args.retrieval.read_text())
    queries = rdata["queries"]
    print(f"[eval] system={rdata['system_id']} queries={len(queries)} top-k-reader={args.top_k_reader}", flush=True)

    reader = llm.LLMReader()

    rows = []
    for i, q in enumerate(queries):
        gold_paras = set(q["gold_paragraph_ids"])
        gold_docs = set(q["gold_doc_ids"])
        top_ids = [r["chunk_id"] for r in q["results"]]
        top_doc_ids = list(dict.fromkeys(
            chunks_by_id[cid].doc_id for cid in top_ids if cid in chunks_by_id
        ))

        para_r5 = metrics.recall_at_k(top_ids, gold_paras, 5)
        para_r10 = metrics.recall_at_k(top_ids, gold_paras, 10)
        para_hit5 = metrics.hit_at_k(top_ids, gold_paras, 5)
        doc_r5 = metrics.recall_at_k(top_doc_ids, gold_docs, 5)
        doc_hit5 = metrics.hit_at_k(top_doc_ids, gold_docs, 5)

        # Reader: top-K paragraphs as context
        top_para_texts = [chunks_by_id[cid].text for cid in top_ids[:args.top_k_reader] if cid in chunks_by_id]
        prompt = llm.build_user_prompt(q["question"], top_para_texts)
        try:
            res = reader.chat(llm.SYSTEM_QA, prompt)
            pred = res["content"].strip()
            usage = res.get("usage", {}).get("total_tokens", None)
        except Exception as e:
            pred = ""
            usage = None
            print(f"[reader-error] qid={q['qid']}: {e}", flush=True)

        em = metrics.exact_match(pred, q["answer"])
        f1v = metrics.f1(pred, q["answer"])
        err = metrics.classify_error(top_ids, gold_paras, args.top_k_reader, pred, q["answer"])

        rows.append({
            "qid": q["qid"],
            "question_type": q.get("question_type", ""),
            "has_retrieval_gold": bool(gold_paras or gold_docs),
            "para_r5": para_r5, "para_r10": para_r10, "para_hit5": para_hit5,
            "doc_r5": doc_r5, "doc_hit5": doc_hit5,
            "pred": pred, "gold": q["answer"],
            "em": em, "f1": f1v,
            "error_class": err,
            "tokens": usage,
        })
        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(queries)} done", flush=True)

    retrieval_metrics = ("para_r5", "para_r10", "para_hit5", "doc_r5", "doc_hit5")
    summary = {k: float(sum(r[k] for r in rows) / len(rows))
               for k in (*retrieval_metrics, "em", "f1")}
    answerable_rows = [r for r in rows if r["has_retrieval_gold"]]
    if not answerable_rows:
        raise RuntimeError("retrieval evaluation has no answerable rows")
    answerable_summary = {
        key: float(sum(row[key] for row in answerable_rows) / len(answerable_rows))
        for key in retrieval_metrics
    }
    err_counts = {}
    for r in rows:
        err_counts[r["error_class"]] = err_counts.get(r["error_class"], 0) + 1

    out = {
        "system_id": rdata["system_id"],
        "n_queries": len(rows),
        "top_k_reader": args.top_k_reader,
        "summary": summary,
        "summary_scope": "all queries; empty-gold retrieval metrics score 0",
        "retrieval_summary_answerable": answerable_summary,
        "n_answerable_retrieval": len(answerable_rows),
        "n_empty_gold_retrieval": len(rows) - len(answerable_rows),
        "n_null_queries": sum(r["question_type"] == "null_query" for r in rows),
        "error_counts": err_counts,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}", flush=True)
    print(f"[summary] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[errors] {err_counts}", flush=True)


if __name__ == "__main__":
    main()
