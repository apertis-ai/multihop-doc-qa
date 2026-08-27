"""E4.5 paired-bootstrap statistical comparison: HippoRAG 2 vs STALI a2_seed1 on DocHop-QA.

Per-query metrics: section_r@5, section_hit@5, doc_r@5, doc_hit@5.
Paired-bootstrap (10k) + Bonferroni over 4 tests (alpha = 0.05/4 = 0.0125).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def load_stali_a2(p):
    d = json.load(p.open())
    return {r["qid"]: r for r in d["per_query"]}


def load_gold(p):
    return json.load(p.open())   # dict qid -> {query, hop, type, relevant_doc_ids, relevant_section_ids}


def find_hipporag_retr(save_dir):
    for pat in ("retrieval_*.json", "results_*.json", "result_*.json", "retrieval.json"):
        for fp in sorted(save_dir.rglob(pat)):
            try:
                obj = json.load(fp.open())
            except Exception:
                continue
            if isinstance(obj, (list, dict)) and len(str(obj)) > 1000:
                return {"path": str(fp), "data": obj}
    raise RuntimeError(f"No retrieval result file found under {save_dir}")


def extract_hippo_ranked(hippo_obj, qidmap):
    out = {}
    data = hippo_obj["data"]
    iterable = data.items() if isinstance(data, dict) else \
               [(r.get("_id") or r.get("id") or r.get("qid"), r) for r in data]
    for key, row in iterable:
        qid = row.get("_id") or row.get("qid") or key
        if qid not in qidmap and str(qid) not in qidmap:
            continue
        docs = (row.get("retrieved_docs") or row.get("retrieved_passages")
                or row.get("docs") or row.get("retrieval") or [])
        titles = []
        for d in docs:
            if isinstance(d, str):
                titles.append(d.split("\n", 1)[0])
            elif isinstance(d, dict):
                titles.append(d.get("title") or "")
            elif isinstance(d, (list, tuple)) and d:
                titles.append(str(d[0]))
        out[str(qid)] = titles
    return out


def doc_from_title(t):
    return t.split("||", 1)[0] if "||" in t else t


def metrics_hipporag(ranked, gold_sids, gold_docs, k=5):
    topk = ranked[:k]
    g_sec = set(gold_sids)
    g_doc = set(gold_docs)
    sec_hit = 1.0 if any(t in g_sec for t in topk) else 0.0
    sec_rec = (sum(1 for t in topk if t in g_sec) / len(g_sec)) if g_sec else 0.0
    top_pmcs = {doc_from_title(t) for t in topk}
    doc_hit = 1.0 if any(p in g_doc for p in top_pmcs) else 0.0
    doc_rec = (sum(1 for p in top_pmcs if p in g_doc) / len(g_doc)) if g_doc else 0.0
    return sec_rec, sec_hit, doc_rec, doc_hit


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hipporag-out", type=Path, required=True)
    ap.add_argument("--stali-a2",     type=Path, required=True)
    ap.add_argument("--gold",         type=Path, required=True)
    ap.add_argument("--qidmap",       type=Path, required=True)
    ap.add_argument("--out",          type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    stali = load_stali_a2(a.stali_a2)
    gold  = load_gold(a.gold)
    qidmap = json.load(a.qidmap.open())
    hippo = find_hipporag_retr(a.hipporag_out)
    print(f"HippoRAG retrieval file: {hippo['path']}")
    hippo_ranked = extract_hippo_ranked(hippo, qidmap)

    diffs = {"sec_r": [], "sec_hit": [], "doc_r": [], "doc_hit": []}
    rows = []
    for qid, gmeta in gold.items():
        if qid not in stali or qid not in hippo_ranked:
            continue
        gsids = gmeta.get("relevant_section_ids") or []
        gdocs = gmeta.get("relevant_doc_ids") or []
        if not gsids:
            continue
        r_sec_hp, h_sec_hp, r_doc_hp, h_doc_hp = metrics_hipporag(
            hippo_ranked[qid], gsids, gdocs, a.k)
        s = stali[qid]
        r_sec_st = float(s.get("section_recall_at_5", 0.0))
        h_sec_st = float(s.get("section_hit_at_5", False))
        r_doc_st = float(s.get("doc_recall_at_5", 0.0))
        h_doc_st = float(s.get("doc_hit_at_5", False))
        diffs["sec_r"].append(r_sec_hp - r_sec_st)
        diffs["sec_hit"].append(h_sec_hp - h_sec_st)
        diffs["doc_r"].append(r_doc_hp - r_doc_st)
        diffs["doc_hit"].append(h_doc_hp - h_doc_st)
        rows.append({"qid": qid,
                     "hipp_sec_r": r_sec_hp, "hipp_sec_hit": h_sec_hp,
                     "hipp_doc_r": r_doc_hp, "hipp_doc_hit": h_doc_hp,
                     "stali_sec_r": r_sec_st, "stali_sec_hit": h_sec_st,
                     "stali_doc_r": r_doc_st, "stali_doc_hit": h_doc_st})
    n = len(rows)
    if not n:
        raise SystemExit("No paired qids — inspect qidmap / hipporag output format")

    bonf = 0.05 / 4
    metric_names = {"sec_r": "section_r@5", "sec_hit": "section_hit@5",
                    "doc_r": "doc_r@5", "doc_hit": "doc_hit@5"}
    boots = {k: paired_bootstrap(v) for k, v in diffs.items()}

    hipp_mean = {
        "sec_r":   statistics.mean(r["hipp_sec_r"]   for r in rows),
        "sec_hit": statistics.mean(r["hipp_sec_hit"] for r in rows),
        "doc_r":   statistics.mean(r["hipp_doc_r"]   for r in rows),
        "doc_hit": statistics.mean(r["hipp_doc_hit"] for r in rows),
    }
    stal_mean = {
        "sec_r":   statistics.mean(r["stali_sec_r"]   for r in rows),
        "sec_hit": statistics.mean(r["stali_sec_hit"] for r in rows),
        "doc_r":   statistics.mean(r["stali_doc_r"]   for r in rows),
        "doc_hit": statistics.mean(r["stali_doc_hit"] for r in rows),
    }

    md = [
        "# E4.5 Full DocHop-QA: HippoRAG 2 vs STALI a2_seed1",
        f"- n = {n} paired queries (k={a.k})",
        f"- Bonferroni alpha = {bonf:.4f} (4 tests)",
        "",
        "## Per-system means",
        f"| system | section_r@{a.k} | section_hit@{a.k} | doc_r@{a.k} | doc_hit@{a.k} |",
        f"|---|---|---|---|---|",
        f"| HippoRAG 2       | {hipp_mean['sec_r']:.4f} | {hipp_mean['sec_hit']:.4f} | {hipp_mean['doc_r']:.4f} | {hipp_mean['doc_hit']:.4f} |",
        f"| STALI a2_seed1   | {stal_mean['sec_r']:.4f} | {stal_mean['sec_hit']:.4f} | {stal_mean['doc_r']:.4f} | {stal_mean['doc_hit']:.4f} |",
        "",
        "## Paired-bootstrap (HippoRAG 2 - STALI a2_seed1, 10k)",
        "| metric | delta | 95% CI | p | Bonf.sig |",
        "|---|---|---|---|---|",
    ]
    for k, label in metric_names.items():
        m, (lo, hi), p = boots[k]
        md.append(f"| {label} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} | {'YES' if p < bonf else 'no'} |")
    md += ["", f"_HippoRAG retrieval file: `{hippo['path']}`_"]

    a.out.write_text("\n".join(md))
    json.dump({
        "n": n, "bonf_alpha": bonf,
        "hipporag": hipp_mean, "stali": stal_mean,
        "deltas": {k: boots[k][0] for k in boots},
        "ci":     {k: boots[k][1] for k in boots},
        "p":      {k: boots[k][2] for k in boots},
        "per_query": rows,
    }, a.out.with_suffix(".json").open("w"), indent=2)
    print(f"OK {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
