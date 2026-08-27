"""E6 retrieval driver — MultiHop-RAG n=500 heldout.

Indexes the 4988-paragraph corpus with one of:
  --retriever stali           # GTE-ModernColBERT + E6 LoRA adapter (MaxSim)
  --retriever bge-m3          # BAAI/bge-m3 dense (cosine)
  --retriever jina-v3         # jinaai/jina-embeddings-v3 dense (cosine, retrieval task)
  --retriever qwen3-emb-4b    # Qwen/Qwen3-Embedding-4B dense (cosine, last-token)

Writes to runs/e6_<retriever>/retrieval.json:
  {
    "system_id": str,
    "queries": [
      {"qid": str, "question": str, "answer": str,
       "gold_paragraph_ids": [...], "gold_doc_ids": [...],
       "results": [{"chunk_id": str, "score": float, "rank": int}, ...top_k]
      }, ...
    ]
  }

Usage (example):
  python run_e6_retrieval.py --retriever bge-m3 \
    --corpus /path/to/multihoprag/corpus.json \
    --queries /path/to/multihoprag/MultiHopRAG.json \
    --heldout ../experiment/data/train/multihoprag_heldout_qids.json \
    --out runs/e6_bge_m3/retrieval.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))


def _imp(modname: str, relpath: str):
    spec = importlib.util.spec_from_file_location(modname, H / relpath)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


def _load_data(corpus_p: Path, queries_p: Path, heldout_p: Path):
    mh = _imp("mh_loader", "datasets/multihoprag.py")
    chunks, t2d = mh.load_corpus(corpus_p)
    qs = mh.load_queries(queries_p, chunks, t2d)
    heldout_ids = set(json.loads(heldout_p.read_text()))
    heldout = [q for q in qs if q.qid in heldout_ids]
    return chunks, heldout


def _retain_metric_depth(ranked_idx, chunks, paragraph_k, document_k=5):
    """Retain at least paragraph_k hits and document_k unique documents."""
    selected = []
    for query_idx, ranking in enumerate(ranked_idx):
        kept, seen_docs = [], set()
        for chunk_idx in ranking:
            kept.append(chunk_idx)
            seen_docs.add(chunks[chunk_idx].doc_id)
            if len(kept) >= paragraph_k and len(seen_docs) >= document_k:
                break
        if len(kept) < paragraph_k or len(seen_docs) < document_k:
            raise ValueError(
                f"query {query_idx} has only {len(kept)} paragraphs and "
                f"{len(seen_docs)} unique documents in the corpus"
            )
        selected.append(kept)
    return selected


def _rank_for_metrics(scores, chunks, paragraph_k):
    scores_cpu = scores.detach().float().cpu()
    ranked_idx = scores_cpu.argsort(dim=1, descending=True).tolist()
    top_idx = _retain_metric_depth(ranked_idx, chunks, paragraph_k)
    top_scores = [
        [float(scores_cpu[query_idx, chunk_idx].item()) for chunk_idx in indices]
        for query_idx, indices in enumerate(top_idx)
    ]
    return top_idx, top_scores


# ── Dense bi-encoder retrievers ─────────────────────────────────────────────

def _dense_retrieve(
    chunks, queries, top_k, model_name, query_prefix="", doc_prefix="",
    pooling="mean", batch_size=16, max_len=512, use_jina_task=False,
):
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    print(f"[dense] loading {model_name}", flush=True)
    if use_jina_task:
        st = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16})
    else:
        st = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        )
    st = st.cuda()
    st.max_seq_length = max_len

    docs = [doc_prefix + c.text for c in chunks]
    print(f"[dense] encoding {len(docs)} docs ...", flush=True)
    t0 = time.time()
    if use_jina_task:
        d_emb = st.encode(docs, batch_size=batch_size, show_progress_bar=True,
                          convert_to_tensor=True, normalize_embeddings=True,
                          task="retrieval.passage")
    else:
        d_emb = st.encode(docs, batch_size=batch_size, show_progress_bar=True,
                          convert_to_tensor=True, normalize_embeddings=True)
    print(f"[dense] doc encoding {time.time()-t0:.1f}s", flush=True)

    qtexts = [query_prefix + q.question for q in queries]
    if use_jina_task:
        q_emb = st.encode(qtexts, batch_size=batch_size, convert_to_tensor=True,
                          normalize_embeddings=True, task="retrieval.query")
    else:
        q_emb = st.encode(qtexts, batch_size=batch_size, convert_to_tensor=True,
                          normalize_embeddings=True)

    print(f"[dense] scoring {len(qtexts)}x{len(docs)} ...", flush=True)
    sims = q_emb @ d_emb.T  # (Q, D)
    top_idx, top_scores = _rank_for_metrics(sims, chunks, top_k)
    return _format_results(queries, chunks, top_idx, top_scores)


def _format_results(queries, chunks, top_idx, top_scores):
    out = []
    for qi, q in enumerate(queries):
        results = [
            {"chunk_id": chunks[ci].id, "score": float(top_scores[qi][k]), "rank": k + 1}
            for k, ci in enumerate(top_idx[qi])
        ]
        out.append({
            "qid": q.qid,
            "question": q.question,
            "answer": q.answer,
            "question_type": q.question_type,
            "gold_paragraph_ids": sorted(q.gold_paragraph_ids),
            "gold_doc_ids": sorted(q.gold_doc_ids),
            "results": results,
        })
    return out


# ── ColBERT (STALI) retriever ───────────────────────────────────────────────

def _stali_retrieve(
    chunks, queries, top_k, base_model, adapter_path,
    batch_size=8, max_len=512, query_len=64,
):
    import torch
    from peft import PeftModel
    from pylate.models import ColBERT

    print(f"[stali] loading {base_model}", flush=True)
    model = ColBERT(
        model_name_or_path=base_model,
        device="cuda",
        query_length=query_len,
        document_length=max_len,
    )
    inner = model[0].auto_model if hasattr(model[0], "auto_model") else model.model
    print(f"[stali] attaching adapter {adapter_path}", flush=True)
    wrapped = PeftModel.from_pretrained(inner, str(adapter_path))
    if hasattr(model[0], "auto_model"):
        model[0].auto_model = wrapped
    else:
        model.model = wrapped

    docs = [c.text for c in chunks]
    print(f"[stali] encoding {len(docs)} docs ...", flush=True)
    t0 = time.time()
    d_emb = model.encode(docs, batch_size=batch_size, is_query=False, show_progress_bar=True,
                         convert_to_tensor=True)
    print(f"[stali] doc encoding {time.time()-t0:.1f}s", flush=True)

    qtexts = [q.question for q in queries]
    q_emb = model.encode(qtexts, batch_size=batch_size, is_query=True, show_progress_bar=True,
                         convert_to_tensor=True)

    # MaxSim scoring with masked padding. pylate.encode returns a list of
    # variable-length (n_tokens, dim) tensors; pad both sides and use a doc
    # mask to ignore padding tokens in the max.
    print(f"[stali] MaxSim scoring {len(qtexts)}x{len(docs)} ...", flush=True)
    from torch.nn.utils.rnn import pad_sequence

    def _pad(emb_list):
        # emb_list: list of (n_i, dim) tensors. Returns (N, max_n, dim) and (N, max_n) bool mask (True = real token).
        lens = torch.tensor([e.size(0) for e in emb_list], device=emb_list[0].device)
        padded = pad_sequence(emb_list, batch_first=True)  # (N, max_n, dim)
        mask = torch.arange(padded.size(1), device=padded.device).unsqueeze(0) < lens.unsqueeze(1)
        return padded, mask

    qP, _ = _pad(q_emb)        # (Q, max_q, dim)
    dP, dMask = _pad(d_emb)    # (D, max_d, dim), (D, max_d)
    Q, D, dim = qP.size(0), dP.size(0), qP.size(-1)

    # Chunked scoring to avoid OOM: process docs in batches.
    chunk = 256
    sims = torch.empty((Q, D), device=qP.device)
    neg_inf = torch.finfo(qP.dtype).min
    for ds in range(0, D, chunk):
        de = min(D, ds + chunk)
        d_chunk = dP[ds:de]                 # (Dc, max_d, dim)
        m_chunk = dMask[ds:de]              # (Dc, max_d)
        # (Q, max_q, dim) x (Dc, dim, max_d) -> (Q, Dc, max_q, max_d)
        # Compute per-doc using batched einsum to manage memory:
        # qP: (Q, max_q, dim); d_chunk: (Dc, max_d, dim)
        # scores: (Q, Dc, max_q, max_d) = qP[:,None,:,:] @ d_chunk[None,:,:,:].transpose(-1,-2)
        scores = torch.einsum("qhd,xtd->qxht", qP, d_chunk)
        # Mask padded doc tokens to -inf so they don't win the max.
        # m_chunk: (Dc, max_d) -> (1, Dc, 1, max_d)
        scores = scores.masked_fill(~m_chunk[None, :, None, :], neg_inf)
        # MaxSim: max over doc tokens, sum over query tokens.
        sims[:, ds:de] = scores.max(dim=-1).values.sum(dim=-1)

    top_idx, top_scores = _rank_for_metrics(sims, chunks, top_k)
    return _format_results(queries, chunks, top_idx, top_scores)


# ── main ────────────────────────────────────────────────────────────────────

RETRIEVER_REGISTRY = {
    "bge-m3": dict(
        impl="dense",
        model="BAAI/bge-m3",
        max_len=8192,
        batch_size=8,
        pooling="cls",
    ),
    "jina-v3": dict(
        impl="dense",
        model="jinaai/jina-embeddings-v3",
        max_len=2048,
        batch_size=16,
        use_jina_task=True,
    ),
    "qwen3-emb-4b": dict(
        impl="dense",
        model="Qwen/Qwen3-Embedding-4B",
        max_len=2048,
        batch_size=4,
        query_prefix="Instruct: Given a multi-hop question, retrieve relevant paragraphs.\nQuery: ",
        env_dtype="bf16",
        env_padding="left",
    ),
    "stali": dict(
        impl="colbert",
        base="lightonai/GTE-ModernColBERT-v1",
        adapter="runs/e6_multihoprag/adapter",
        query_len=64,
        max_len=512,
        batch_size=8,
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", required=True, choices=list(RETRIEVER_REGISTRY.keys()))
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--heldout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    if args.top_k < 10:
        ap.error("--top-k must be at least 10 for the released paragraph metrics")

    cfg = RETRIEVER_REGISTRY[args.retriever]
    chunks, queries = _load_data(args.corpus, args.queries, args.heldout)
    print(f"[main] retriever={args.retriever} corpus={len(chunks)} queries={len(queries)}", flush=True)

    if cfg["impl"] == "dense":
        results = _dense_retrieve(
            chunks, queries, args.top_k,
            model_name=cfg["model"],
            query_prefix=cfg.get("query_prefix", ""),
            doc_prefix=cfg.get("doc_prefix", ""),
            pooling=cfg.get("pooling", "mean"),
            batch_size=cfg.get("batch_size", 16),
            max_len=cfg.get("max_len", 512),
            use_jina_task=cfg.get("use_jina_task", False),
        )
    elif cfg["impl"] == "colbert":
        results = _stali_retrieve(
            chunks, queries, args.top_k,
            base_model=cfg["base"],
            adapter_path=Path(cfg["adapter"]),
            batch_size=cfg.get("batch_size", 8),
            max_len=cfg.get("max_len", 512),
            query_len=cfg.get("query_len", 64),
        )
    else:
        raise ValueError(cfg["impl"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "system_id": f"e6_{args.retriever}",
        "retriever_config": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))},
        "n_corpus": len(chunks),
        "n_queries": len(queries),
        "paragraph_top_k": args.top_k,
        "minimum_unique_documents": 5,
        "queries": results,
    }, indent=2))
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
