"""Convert DocHop-QA corpus + 500 gold queries to HippoRAG 2 reproduce/dataset format.

DocHop-QA is PMC-based with per-query context_list of 5 candidate paragraphs.
We build a global corpus of unique (pmc_id, section, subsection) chunks from all queries
(11,379 queries -> ~23,158 unique paragraph chunks) and use the 500 gold_n500 QIDs as the
eval subset (matching STALI a2_seed1 baseline).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def chunk_title(pmc_id: str, section: str, subsection: str) -> str:
    """Canonical title used for HippoRAG supporting_facts and corpus."""
    return f"{pmc_id}||{section}/{subsection or ''}".rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dochop", required=True, type=Path)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiment/data/eval/gold_n500.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    dochop = json.loads(args.dochop.read_text())
    gold = json.loads(args.gold.read_text())
    out_queries = args.output_dir / "dochop_full.json"
    out_corpus = args.output_dir / "dochop_full_corpus.json"
    qid_map_out = args.output_dir / "dochop_full_qidmap.json"

    print(f"DocHop queries: {len(dochop)}")
    print(f"gold_n500 eval subset: {len(gold)}")

    # 1. Build unique corpus from ALL queries' contexts
    seen = {}
    for q in dochop:
        for ctx in (q.get("context_list") or []):
            key = (ctx.get("pmc_id",""), ctx.get("section",""), ctx.get("subsection",""))
            if key not in seen:
                seen[key] = ctx
    print(f"unique corpus chunks: {len(seen)}")

    corpus = []
    title_to_idx = {}
    for i, ((pmc, sec, sub), ctx) in enumerate(sorted(seen.items())):
        title = chunk_title(pmc, sec, sub)
        text = (ctx.get("content") or ctx.get("Raw content") or "").strip()
        # Strip HTML if using Raw content
        if "<" in text and ">" in text:
            import re
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        corpus.append({"idx": i, "title": title, "text": text})
        title_to_idx[title] = i

    # 2. Build queries from gold_n500 only
    dochop_by_id = {str(q["id"]): q for q in dochop}
    kept_queries = []
    qidmap = {}
    misses = 0
    for qid, meta in gold.items():
        src = dochop_by_id.get(qid)
        if src is None:
            misses += 1
            continue
        # supporting_facts derived from gold's relevant_section_ids (PMC||section/subsec format)
        supporting_titles = list(meta.get("relevant_section_ids") or [])
        # fallback: use all contexts listed for this query
        if not supporting_titles:
            supporting_titles = [chunk_title(c.get("pmc_id",""), c.get("section",""), c.get("subsection",""))
                                 for c in (src.get("context_list") or [])]
        unresolved = [title for title in supporting_titles if title not in title_to_idx]
        if unresolved:
            raise ValueError(f"query {qid} has unresolved supporting sections: {unresolved}")
        supporting = [[title, 0] for title in supporting_titles]

        # context: full candidate list (5 contexts per query)
        ctx = []
        for c in (src.get("context_list") or []):
            t = chunk_title(c.get("pmc_id",""), c.get("section",""), c.get("subsection",""))
            raw = (c.get("content") or c.get("Raw content") or "").strip()
            if "<" in raw and ">" in raw:
                import re
                raw = re.sub(r"<[^>]+>", " ", raw)
                raw = re.sub(r"\s+", " ", raw).strip()
            sents = [p.strip() for p in raw.split("\n") if p.strip()] or [raw]
            ctx.append([t, sents])

        kept_queries.append({
            "_id": qid,
            "question": meta.get("query", src.get("question", "")),
            "answer": "",
            "question_type": meta.get("type", ""),
            "supporting_facts": supporting,
            "context": ctx,
            "type": meta.get("type", src.get("hop_type", "")),
            "level": f"hop{meta.get('hop','?')}",
        })
        qidmap[qid] = src["id"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_queries.write_text(json.dumps(kept_queries, ensure_ascii=False))
    out_corpus.write_text(json.dumps(corpus, ensure_ascii=False))
    qid_map_out.write_text(json.dumps(qidmap, ensure_ascii=False))

    print(f"queries written: {out_queries}  n={len(kept_queries)}  (misses={misses})")
    print(f"corpus  written: {out_corpus}  n={len(corpus)}")
    print(f"qidmap  written: {qid_map_out}")
    n_with_gold = sum(1 for q in kept_queries if q["supporting_facts"])
    print(f"  queries with resolved supporting_facts: {n_with_gold} / {len(kept_queries)}")


if __name__ == "__main__":
    main()
