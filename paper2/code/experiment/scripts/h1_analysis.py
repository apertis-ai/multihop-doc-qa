"""H1 gate statistical analysis: paired McNemar + slice breakdown.

Computes:
- Paired McNemar S2 vs S3 (LoRA effect)
- Paired McNemar S1 vs S2 (prefix effect)
- Paired McNemar S1 vs S3 (total method effect)
- Slice analysis by query type (text vs table)
- Slice analysis by hop count (if available)
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import binomtest, chi2 as chi2dist


def mcnemar(a_hits: list[bool], b_hits: list[bool]) -> dict:
    a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    b_only = sum(1 for a, b in zip(a_hits, b_hits) if not a and b)
    both = sum(1 for a, b in zip(a_hits, b_hits) if a and b)
    neither = sum(1 for a, b in zip(a_hits, b_hits) if not a and not b)
    n_disc = a_only + b_only
    if n_disc == 0:
        return {
            "a_only": a_only, "b_only": b_only,
            "both": both, "neither": neither,
            "chi2": 0.0, "p_chi2": 1.0, "p_exact": 1.0,
        }
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n_disc
    p_chi2 = float(chi2dist.sf(chi2, 1))
    p_exact = float(binomtest(min(a_only, b_only), n_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "a_only": a_only, "b_only": b_only,
        "both": both, "neither": neither,
        "chi2": chi2, "p_chi2": p_chi2, "p_exact": p_exact,
    }


def main() -> int:
    runs = Path("runs")
    s1 = json.loads((runs / "s1_stali" / "eval_n500.json").read_text())["per_query"]
    s2 = json.loads((runs / "s2_plain_lora" / "eval_n500.json").read_text())["per_query"]
    s3 = json.loads((runs / "s3_zero_shot_gte" / "eval_n500.json").read_text())["per_query"]

    hit1 = [r["section_hit_at_5"] for r in s1]
    hit2 = [r["section_hit_at_5"] for r in s2]
    hit3 = [r["section_hit_at_5"] for r in s3]

    print("=" * 70)
    print("S2 plain LoRA vs S3 zero-shot (primary LoRA effect)")
    print("=" * 70)
    r = mcnemar(hit2, hit3)
    print(f"  S2_only = {r['a_only']}, S3_only = {r['b_only']}")
    print(f"  both = {r['both']}, neither = {r['neither']}")
    print(f"  McNemar chi2_cc = {r['chi2']:.3f}  p_chi2 = {r['p_chi2']:.4f}  p_exact = {r['p_exact']:.4f}")

    print()
    print("=" * 70)
    print("S1 STALI vs S2 plain LoRA (prefix effect)")
    print("=" * 70)
    r = mcnemar(hit1, hit2)
    print(f"  S1_only = {r['a_only']}, S2_only = {r['b_only']}")
    print(f"  both = {r['both']}, neither = {r['neither']}")
    print(f"  McNemar chi2_cc = {r['chi2']:.3f}  p_chi2 = {r['p_chi2']:.4f}  p_exact = {r['p_exact']:.4f}")

    print()
    print("=" * 70)
    print("S1 STALI vs S3 zero-shot (total method effect)")
    print("=" * 70)
    r = mcnemar(hit1, hit3)
    print(f"  S1_only = {r['a_only']}, S3_only = {r['b_only']}")
    print(f"  both = {r['both']}, neither = {r['neither']}")
    print(f"  McNemar chi2_cc = {r['chi2']:.3f}  p_chi2 = {r['p_chi2']:.4f}  p_exact = {r['p_exact']:.4f}")

    # Slice by query type
    gold = json.loads(Path("data/eval/gold_n500.json").read_text())

    def sliced_mean(per_q: list[dict], qids: set[str], field: str = "section_recall_at_5") -> float:
        vals = [r[field] for r in per_q if r["qid"] in qids]
        return sum(vals) / len(vals) if vals else 0.0

    def sliced_hitrate(per_q: list[dict], qids: set[str]) -> float:
        vals = [1 if r["section_hit_at_5"] else 0 for r in per_q if r["qid"] in qids]
        return sum(vals) / len(vals) if vals else 0.0

    print()
    print("=" * 70)
    print("Slice analysis: section_recall_at_5 by query type")
    print("=" * 70)
    for qtype in ("text", "table"):
        qids = {qid for qid, g in gold.items() if g.get("type") == qtype}
        n = len(qids)
        s3v = sliced_mean(s3, qids)
        s2v = sliced_mean(s2, qids)
        s1v = sliced_mean(s1, qids)
        print(f"  {qtype} (n={n}): S3={s3v:.4f}  S2={s2v:.4f}  S1={s1v:.4f}  "
              f"S2-S3={s2v-s3v:+.4f}  S1-S2={s1v-s2v:+.4f}")

    print()
    print("Slice analysis: section_hit@5 rate by query type")
    for qtype in ("text", "table"):
        qids = {qid for qid, g in gold.items() if g.get("type") == qtype}
        n = len(qids)
        s3v = sliced_hitrate(s3, qids)
        s2v = sliced_hitrate(s2, qids)
        s1v = sliced_hitrate(s1, qids)
        print(f"  {qtype} (n={n}): S3={s3v:.4f}  S2={s2v:.4f}  S1={s1v:.4f}  "
              f"S2-S3={s2v-s3v:+.4f}  S1-S2={s1v-s2v:+.4f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
