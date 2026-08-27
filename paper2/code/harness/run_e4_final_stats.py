"""Final E4 paired-bootstrap stats using extracted retrieval_*.json.

Runs BOTH:
  - E4.6 MultiHop-RAG: HippoRAG 2 vs STALI E6 (doc_r@5, doc_hit@5)
  - E4.5 DocHop-QA:    HippoRAG 2 vs STALI a2_seed1 (section_r@5, section_hit@5, doc_r@5, doc_hit@5)

Writes one Markdown and JSON report for each comparison.
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
from pathlib import Path


def paired_bootstrap(diffs, n_boot=10000, seed=42):
    rng = random.Random(seed)
    n = len(diffs)
    if not n:
        return 0.0, (0.0, 0.0), 1.0
    means = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    mean = sum(diffs) / n
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    p = 2 * min(sum(1 for m in means if m <= 0) / n_boot,
                sum(1 for m in means if m >= 0) / n_boot)
    return mean, (lo, hi), p


def doc_from_title(t):
    return t.split("||", 1)[0] if "||" in t else t


# ---------------- E4.6 MultiHop-RAG ----------------
def run_multihoprag(hippo_path, stali_path, corpus_path, out_md):
    import hashlib
    hippo = json.load(open(hippo_path))
    stali = json.load(open(stali_path))
    corpus = json.load(open(corpus_path))
    # Build title -> stali doc_id (sha1(url)[:12]) mapping
    title_to_docid = {}
    for c in corpus:
        t = c.get("title", "")
        url = c.get("url", "")
        if t and url:
            title_to_docid[t] = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    stali_by_qid = {q["qid"]: q for q in stali["queries"]}

    recall_diffs, hit_diffs = [], []
    rows = []
    k = 5
    for hq in hippo["queries"]:
        qid = str(hq["qid"])
        sq = stali_by_qid.get(qid)
        if not sq: continue
        gold_docs = set(str(d) for d in sq.get("gold_doc_ids", []))
        if not gold_docs: continue
        # HippoRAG top-k -> map article titles to sha1 doc_ids
        h_top = [title_to_docid.get(d["title"], d["title"]) for d in hq["retrieved"][:k]]
        # STALI top-k doc ids (derived from chunk_id "docid||p####")
        seen = []
        for r in (sq.get("results") or []):
            cid = r.get("chunk_id") or r.get("doc_id") or r.get("did") or ""
            did = str(cid).split("||", 1)[0] if cid else ""
            if did and did not in seen:
                seen.append(did)
            if len(seen) >= k: break
        s_top = seen
        h_rec = sum(1 for t in h_top if t in gold_docs) / len(gold_docs)
        h_hit = 1.0 if any(t in gold_docs for t in h_top) else 0.0
        s_rec = sum(1 for t in s_top if t in gold_docs) / len(gold_docs)
        s_hit = 1.0 if any(t in gold_docs for t in s_top) else 0.0
        recall_diffs.append(h_rec - s_rec)
        hit_diffs.append(h_hit - s_hit)
        rows.append({"qid": qid, "h_rec": h_rec, "h_hit": h_hit,
                     "s_rec": s_rec, "s_hit": s_hit})

    n = len(rows)
    rm, rci, rp = paired_bootstrap([r["h_rec"] - r["s_rec"] for r in rows])
    hm, hci, hp = paired_bootstrap([r["h_hit"] - r["s_hit"] for r in rows])
    bonf = 0.05 / 2

    hipp = {"rec": statistics.mean(r["h_rec"] for r in rows),
            "hit": statistics.mean(r["h_hit"] for r in rows)}
    stal = {"rec": statistics.mean(r["s_rec"] for r in rows),
            "hit": statistics.mean(r["s_hit"] for r in rows)}

    md = [
        "# E4.6 MultiHop-RAG: HippoRAG 2 vs STALI E6 (paired-bootstrap)",
        f"- n = {n} paired queries (k={k})",
        f"- Bonferroni alpha = {bonf:.4f} (2 tests)",
        "",
        "## Per-system means",
        f"| system | doc_r@{k} | doc_hit@{k} |",
        f"|---|---|---|",
        f"| HippoRAG 2 | {hipp['rec']:.4f} | {hipp['hit']:.4f} |",
        f"| STALI E6   | {stal['rec']:.4f} | {stal['hit']:.4f} |",
        "",
        "## Paired-bootstrap (HippoRAG 2 - STALI E6, 10k)",
        f"| metric | delta | 95% CI | p | Bonf.sig |",
        f"|---|---|---|---|---|",
        f"| doc_r@{k}   | {rm:+.4f} | [{rci[0]:+.4f}, {rci[1]:+.4f}] | {rp:.4f} | {'YES' if rp < bonf else 'no'} |",
        f"| doc_hit@{k} | {hm:+.4f} | [{hci[0]:+.4f}, {hci[1]:+.4f}] | {hp:.4f} | {'YES' if hp < bonf else 'no'} |",
        "",
        "## Note on HippoRAG EM/F1 (from main.py log)",
        "- HippoRAG 2 reported EM=0.5060, F1=0.5356 (QA Reading with gold answers)",
        "- STALI E6 reported EM=0.3440, F1=0.4163 (from e6_stats.json)",
        "- Delta: +0.1620 EM, +0.1193 F1 (aggregate only, not paired-bootstrapped since HippoRAG per-query QA results not persisted)",
    ]
    Path(out_md).write_text("\n".join(md))
    json.dump({"n": n, "hipporag": hipp, "stali": stal,
               "delta_rec": rm, "ci_rec": rci, "p_rec": rp,
               "delta_hit": hm, "ci_hit": hci, "p_hit": hp,
               "bonf_alpha": bonf, "per_query": rows},
              Path(out_md).with_suffix(".json").open("w"), indent=2)
    print(f"OK {out_md} (n={n})")


# ---------------- E4.5 DocHop-QA ----------------
def run_dochop(hippo_path, stali_path, gold_path, out_md):
    hippo = json.load(open(hippo_path))
    stali = json.load(open(stali_path))
    gold = json.load(open(gold_path))
    stali_by_qid = {r["qid"]: r for r in stali["per_query"]}

    rows = []
    k = 5
    for hq in hippo["queries"]:
        qid = str(hq["qid"])
        gmeta = gold.get(qid)
        sq = stali_by_qid.get(qid)
        if not (gmeta and sq): continue
        gsids = set(gmeta.get("relevant_section_ids") or [])
        gdocs = set(gmeta.get("relevant_doc_ids") or [])
        if not gsids: continue
        h_top = [d["title"] for d in hq["retrieved"][:k]]
        h_sec_hit = 1.0 if any(t in gsids for t in h_top) else 0.0
        h_sec_rec = sum(1 for t in h_top if t in gsids) / len(gsids)
        h_pmcs = {doc_from_title(t) for t in h_top}
        h_doc_hit = 1.0 if any(p in gdocs for p in h_pmcs) else 0.0
        h_doc_rec = sum(1 for p in h_pmcs if p in gdocs) / len(gdocs)
        s_sec_rec = float(sq.get("section_recall_at_5", 0.0))
        s_sec_hit = float(sq.get("section_hit_at_5", False))
        s_doc_rec = float(sq.get("doc_recall_at_5", 0.0))
        s_doc_hit = float(sq.get("doc_hit_at_5", False))
        rows.append({"qid": qid,
                     "h_sec_rec": h_sec_rec, "h_sec_hit": h_sec_hit,
                     "h_doc_rec": h_doc_rec, "h_doc_hit": h_doc_hit,
                     "s_sec_rec": s_sec_rec, "s_sec_hit": s_sec_hit,
                     "s_doc_rec": s_doc_rec, "s_doc_hit": s_doc_hit})
    n = len(rows)
    boots = {
        "sec_rec": paired_bootstrap([r["h_sec_rec"] - r["s_sec_rec"] for r in rows]),
        "sec_hit": paired_bootstrap([r["h_sec_hit"] - r["s_sec_hit"] for r in rows]),
        "doc_rec": paired_bootstrap([r["h_doc_rec"] - r["s_doc_rec"] for r in rows]),
        "doc_hit": paired_bootstrap([r["h_doc_hit"] - r["s_doc_hit"] for r in rows]),
    }
    bonf = 0.05 / 4
    hipp = {kk: statistics.mean(r[f"h_{kk}"] for r in rows) for kk in ["sec_rec", "sec_hit", "doc_rec", "doc_hit"]}
    stal = {kk: statistics.mean(r[f"s_{kk}"] for r in rows) for kk in ["sec_rec", "sec_hit", "doc_rec", "doc_hit"]}

    md = [
        "# E4.5 DocHop-QA: HippoRAG 2 vs STALI a2_seed1 (paired-bootstrap)",
        f"- n = {n} paired queries (k={k})",
        f"- Bonferroni alpha = {bonf:.4f} (4 tests)",
        "",
        "## Per-system means",
        f"| system | section_r@{k} | section_hit@{k} | doc_r@{k} | doc_hit@{k} |",
        f"|---|---|---|---|---|",
        f"| HippoRAG 2     | {hipp['sec_rec']:.4f} | {hipp['sec_hit']:.4f} | {hipp['doc_rec']:.4f} | {hipp['doc_hit']:.4f} |",
        f"| STALI a2_seed1 | {stal['sec_rec']:.4f} | {stal['sec_hit']:.4f} | {stal['doc_rec']:.4f} | {stal['doc_hit']:.4f} |",
        "",
        "## Paired-bootstrap (HippoRAG 2 - STALI a2_seed1, 10k)",
        "| metric | delta | 95% CI | p | Bonf.sig |",
        "|---|---|---|---|---|",
    ]
    metric_labels = {"sec_rec": f"section_r@{k}", "sec_hit": f"section_hit@{k}",
                     "doc_rec": f"doc_r@{k}", "doc_hit": f"doc_hit@{k}"}
    for kk, label in metric_labels.items():
        m, (lo, hi), p = boots[kk]
        md.append(f"| {label} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} | {'YES' if p < bonf else 'no'} |")
    Path(out_md).write_text("\n".join(md))
    json.dump({"n": n, "hipporag": hipp, "stali": stal, "boots": {k: {"mean": v[0], "ci": v[1], "p": v[2]} for k, v in boots.items()},
               "bonf_alpha": bonf, "per_query": rows},
              Path(out_md).with_suffix(".json").open("w"), indent=2)
    print(f"OK {out_md} (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/hipporag_results")
    ap.add_argument("--out-dir", default="results/hipporag_results")
    a = ap.parse_args()
    base = Path(a.base)
    out = Path(a.out_dir)

    run_multihoprag(
        hippo_path=base / "retrieval_perquery" / "retrieval_c.json",
        stali_path=base / "retrieval.json",
        corpus_path=base / "mhrag_corpus.json",
        out_md=out / "e46_comparison.md",
    )
    run_dochop(
        hippo_path=base / "retrieval_perquery" / "retrieval_b.json",
        stali_path=base / "eval_n500.json",
        gold_path=base / "gold_n500.json",
        out_md=out / "e45_comparison.md",
    )


if __name__ == "__main__":
    main()
