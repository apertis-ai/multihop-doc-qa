"""Convert MultiHop-RAG corpus + queries to HippoRAG 2 reproduce/dataset format.

HippoRAG corpus schema: list of {idx, title, text}
HippoRAG query schema : list of {_id, question, answer, supporting_facts, context, ...}

We reuse the 500 QIDs already locked-in by STALI E6 retrieval.json so the comparison
is apples-to-apples. For each MHRAG query we derive supporting_facts = [[title, 0]]
from the title list in evidence_list (paragraph-level gold is the best MHRAG exposes).

Usage:
    python3 prep_multihoprag_for_hipporag.py --corpus corpus.json \
      --queries MultiHopRAG.json --stali-e6 retrieval.json --output-dir dataset/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--stali-e6", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    corpus_raw = json.loads(args.corpus.read_text())
    qa_raw = json.loads(args.queries.read_text())
    stali = json.loads(args.stali_e6.read_text())
    out_queries = args.output_dir / "multihoprag_full.json"
    out_corpus = args.output_dir / "multihoprag_full_corpus.json"
    qid_map_out = args.output_dir / "multihoprag_full_qidmap.json"

    # 500 QIDs locked by E6 (same ordering as STALI eval)
    stali_qids = [q["qid"] for q in stali["queries"]]
    qid_set    = set(stali_qids)

    # MHRAG queries are indexed 0..2555 in source order; STALI qids look like "mhrag_0042"
    # — reconstruct mapping from the question text.
    q_by_question = {q["query"]: (i, q) for i, q in enumerate(qa_raw)}
    kept_queries  = []
    qidmap = {}
    for sq in stali["queries"]:
        qtext = sq["question"]
        if qtext not in q_by_question:
            print(f"  WARN: qid {sq['qid']} not found in MHRAG source")
            continue
        src_idx, src_q = q_by_question[qtext]
        titles = sorted({ev["title"] for ev in src_q["evidence_list"] if ev.get("title")})
        # HippoRAG expects context=[[title, [sentences]]] & supporting_facts=[[title, 0]]
        ctx = []
        for t in titles:
            # Find corpus entry with this title (may be multiple; take first)
            doc = next((c for c in corpus_raw if c.get("title") == t), None)
            if doc is None:
                continue
            body = (doc.get("body") or "").strip()
            # Split on double-newline for sentence-ish granularity (good enough for HippoRAG)
            sents = [p.strip() for p in body.split("\n") if p.strip()] or [body]
            ctx.append([t, sents])
        supporting = [[t, 0] for t, _ in ctx]

        kept_queries.append({
            "_id": sq["qid"],
            "question": qtext,
            "answer": src_q.get("answer", ""),
            "question_type": src_q.get("question_type", ""),
            "supporting_facts": supporting,
            "context": ctx,
            "type": src_q.get("question_type", ""),
            "level": "hard",
        })
        qidmap[sq["qid"]] = src_idx

    # Corpus: HippoRAG wants unique idx, title, text
    hippo_corpus = []
    seen_titles = set()
    for i, c in enumerate(corpus_raw):
        t = c.get("title", "") or f"doc_{i}"
        if t in seen_titles:
            t = f"{t} [{i}]"
        seen_titles.add(t)
        hippo_corpus.append({
            "idx": i,
            "title": t,
            "text": (c.get("body") or "").strip(),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_queries.write_text(json.dumps(kept_queries, ensure_ascii=False))
    out_corpus.write_text(json.dumps(hippo_corpus, ensure_ascii=False))
    qid_map_out.write_text(json.dumps(qidmap, ensure_ascii=False))

    print(f"✓ queries written: {out_queries}  n={len(kept_queries)} (of {len(stali_qids)} E6 common)")
    print(f"✓ corpus  written: {out_corpus}  n={len(hippo_corpus)}")
    print(f"✓ qidmap  written: {qid_map_out}")
    print(f"  kept_queries examples: {[q['_id'] for q in kept_queries[:3]]}")


if __name__ == "__main__":
    main()
