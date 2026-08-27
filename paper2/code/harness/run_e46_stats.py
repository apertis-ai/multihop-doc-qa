"""E4.6 paired-bootstrap statistical comparison: HippoRAG 2 vs STALI E6 baseline.

Aligns per-query retrieval on the 500 common QIDs, computes doc-level Recall@5 / Hit@5
for HippoRAG, pairs against STALI E6 results, and applies paired-bootstrap + Bonferroni.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def load_stali_e6(p):
    d = json.load(p.open())
    return {q["qid"]: q for q in d["queries"]}


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
                titles.append(d.get("title") or d.get("idx") or "")
            elif isinstance(d, (list, tuple)) and d:
                titles.append(str(d[0]))
        out[str(qid)] = titles
    return out


def metrics_hipporag(ranked, gold, k):
    topk = list(dict.fromkeys(ranked))[:k]
    gset = set(gold)
    hit = 1.0 if any(t in gset for t in topk) else 0.0
    recall = (sum(1 for t in topk if t in gset) / len(gset)) if gset else 0.0
    return recall, hit


def gold_titles_from_stali(q):
    return sorted({pid.split("#")[0] for pid in q.get("gold_doc_ids", q.get("gold_paragraph_ids", []))})


def metrics_stali(q, k=5):
    ranked = []
    for r in (q.get("results") or []):
        chunk_id = r.get("chunk_id") or r.get("doc_id") or r.get("did") or ""
        did = str(chunk_id).split("||", 1)[0]
        if did and did not in ranked:
            ranked.append(did)
        if len(ranked) >= k:
            break
    gold = set(gold_titles_from_stali(q))
    topk = ranked[:k]
    recall = (sum(1 for t in topk if t in gold) / len(gold)) if gold else 0.0
    hit = 1.0 if any(t in gold for t in topk) else 0.0
    return recall, hit


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
    ap.add_argument("--stali-e6",     type=Path, required=True)
    ap.add_argument("--e6-stats",     type=Path, required=True)
    ap.add_argument("--qidmap",       type=Path, required=True)
    ap.add_argument("--out",          type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    stali_by_qid = load_stali_e6(a.stali_e6)
    qidmap = json.load(a.qidmap.open())
    hippo = find_hipporag_retr(a.hipporag_out)
    print(f"HippoRAG retrieval file: {hippo['path']}")
    hippo_ranked = extract_hippo_ranked(hippo, qidmap)

    r_diffs, h_diffs, rows = [], [], []
    for qid, sq in stali_by_qid.items():
        if qid not in hippo_ranked:
            continue
        gold = gold_titles_from_stali(sq)
        if not gold:
            continue
        r_hp, h_hp = metrics_hipporag(hippo_ranked[qid], gold, a.k)
        r_st, h_st = metrics_stali(sq, k=a.k)
        r_diffs.append(r_hp - r_st)
        h_diffs.append(h_hp - h_st)
        rows.append({"qid": qid, "r_hipp": r_hp, "r_stali": r_st,
                     "hit_hipp": h_hp, "hit_stali": h_st})

    n = len(rows)
    if not n:
        raise SystemExit("No paired qids — inspect qidmap / hipporag output format")

    rm, rci, rp = paired_bootstrap(r_diffs)
    hm, hci, hp = paired_bootstrap(h_diffs)
    bonf = 0.05 / 2

    r_hp_mean = statistics.mean(r["r_hipp"] for r in rows)
    r_st_mean = statistics.mean(r["r_stali"] for r in rows)
    h_hp_mean = statistics.mean(r["hit_hipp"] for r in rows)
    h_st_mean = statistics.mean(r["hit_stali"] for r in rows)

    md = [
        "# E4.6 Full MultiHop-RAG: HippoRAG 2 vs STALI E6",
        f"- n = {n} paired queries (k={a.k})",
        f"- Bonferroni alpha = {bonf:.4f} (2 tests)",
        "",
        "## Per-system means",
        f"| system | doc_r@{a.k} | doc_hit@{a.k} |",
        f"|---|---|---|",
        f"| HippoRAG 2 | {r_hp_mean:.4f} | {h_hp_mean:.4f} |",
        f"| STALI E6   | {r_st_mean:.4f} | {h_st_mean:.4f} |",
        "",
        "## Paired-bootstrap (HippoRAG 2 - STALI E6, 10k)",
        f"| metric | delta | 95% CI | p | Bonf.sig |",
        f"|---|---|---|---|---|",
        f"| doc_r@{a.k}   | {rm:+.4f} | [{rci[0]:+.4f}, {rci[1]:+.4f}] | {rp:.4f} | {'YES' if rp < bonf else 'no'} |",
        f"| doc_hit@{a.k} | {hm:+.4f} | [{hci[0]:+.4f}, {hci[1]:+.4f}] | {hp:.4f} | {'YES' if hp < bonf else 'no'} |",
        "",
        f"_HippoRAG retrieval file: `{hippo['path']}`_",
    ]
    a.out.write_text("\n".join(md))
    json.dump({
        "n": n,
        "hipporag": {"doc_r_at_k": r_hp_mean, "doc_hit_at_k": h_hp_mean},
        "stali":    {"doc_r_at_k": r_st_mean, "doc_hit_at_k": h_st_mean},
        "delta":    {"doc_r_at_k": rm, "doc_hit_at_k": hm},
        "ci":       {"doc_r_at_k": rci, "doc_hit_at_k": hci},
        "p":        {"doc_r_at_k": rp, "doc_hit_at_k": hp},
        "bonf_alpha": bonf, "per_query": rows,
    }, a.out.with_suffix(".json").open("w"), indent=2)
    print(f"OK {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
