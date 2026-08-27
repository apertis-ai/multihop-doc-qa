"""Post-hoc retrieval extraction — loads a saved HippoRAG graph+embeddings and runs
retrieve() only (no QA Reading), saving per-query top-k results as JSON.

Reuses all costly artifacts (openie, entity embeddings, fact embeddings, saved graph).
Only runs per-query retrieval + PPR + fact reranking via LLM (LLM provider). Outputs
per-query top-k for downstream paired-bootstrap.

Usage:
    cd /path/to/HippoRAG
    python3 extract_retrieval_perquery.py \
        --dataset multihoprag_full \
        --save_dir /path/to/outputs_full_multihoprag \
        --topk 5 \
        --out /path/to/retrieval_c_perquery.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_HIPPO = _HERE if (_HERE / "src" / "hipporag").exists() else (_HERE / "HippoRAG")
if (_HIPPO / "src").exists():
    sys.path.insert(0, str(_HIPPO))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--llm_base_url", default="${LLM_BASE_URL}")
    ap.add_argument("--llm_name", default="claude-sonnet-4-6")
    ap.add_argument("--embedding_name", default="nvidia/NV-Embed-v2")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    reproduce_dir = _HIPPO / "reproduce" / "dataset"
    queries_path = reproduce_dir / f"{a.dataset}.json"
    corpus_path = reproduce_dir / f"{a.dataset}_corpus.json"
    samples = json.load(queries_path.open())
    corpus = json.load(corpus_path.open())

    config = BaseConfig(
        save_dir=a.save_dir,
        llm_base_url=a.llm_base_url,
        llm_name=a.llm_name,
        dataset=a.dataset,
        embedding_model_name=a.embedding_name,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=max(a.topk, 200),
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=a.topk,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=2,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode="online",
    )

    hipporag = HippoRAG(global_config=config)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    hipporag.index(docs)

    queries = [s["question"] for s in samples]
    qids = [s.get("_id", s.get("id", str(i))) for i, s in enumerate(samples)]

    results = hipporag.retrieve(queries=queries, num_to_retrieve=a.topk)

    idx_by_title = {doc["title"]: doc["idx"] for doc in corpus}
    out_rows = []
    for qid, q, r in zip(qids, queries, results):
        candidates = getattr(r, "docs", None) or getattr(r, "retrieved_docs", None) or r
        if isinstance(candidates, dict):
            candidates = candidates.get("docs") or candidates.get("retrieved_docs") or []
        docs_out = []
        for cand in (candidates or []):
            if isinstance(cand, str):
                title = cand.split("\n", 1)[0]
            elif isinstance(cand, dict):
                title = cand.get("title") or cand.get("idx") or ""
            elif isinstance(cand, (list, tuple)) and cand:
                title = str(cand[0])
            else:
                title = str(cand)
            docs_out.append({"title": title, "idx": idx_by_title.get(title)})
        out_rows.append({"qid": str(qid), "question": q, "retrieved": docs_out[:a.topk]})

    a.out.write_text(json.dumps({
        "dataset": a.dataset, "topk": a.topk, "n": len(out_rows),
        "queries": out_rows,
    }, ensure_ascii=False, indent=2))
    print(f"OK wrote {a.out} n={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
