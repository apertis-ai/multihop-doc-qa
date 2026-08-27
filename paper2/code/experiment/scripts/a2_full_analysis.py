"""Paired McNemar + slice analysis including A2 random-only."""
from __future__ import annotations
import json
from pathlib import Path
from scipy.stats import binomtest, chi2 as chi2dist


def mcnemar(a_hits, b_hits, label_a="A", label_b="B"):
    a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    b_only = sum(1 for a, b in zip(a_hits, b_hits) if not a and b)
    both = sum(1 for a, b in zip(a_hits, b_hits) if a and b)
    neither = sum(1 for a, b in zip(a_hits, b_hits) if not a and not b)
    n_disc = a_only + b_only
    if n_disc == 0:
        print(f"  {label_a}_only={a_only} {label_b}_only={b_only} both={both} neither={neither}")
        print(f"  no discordant pairs; p = 1.0")
        return
    chi2_val = (abs(a_only - b_only) - 1) ** 2 / n_disc
    p_chi2 = float(chi2dist.sf(chi2_val, 1))
    p_exact = float(binomtest(min(a_only, b_only), n_disc, 0.5, alternative="two-sided").pvalue)
    print(f"  {label_a}_only={a_only} {label_b}_only={b_only} both={both} neither={neither}")
    print(f"  chi2_cc={chi2_val:.3f}  p_chi2={p_chi2:.4f}  p_exact={p_exact:.4f}")


def main():
    runs = Path("runs")
    systems = {}
    for sid in ("s1_stali", "s2_plain_lora", "s3_zero_shot_gte", "s6_reason_modern",
                "s7_colbertv2", "a2_random_only"):
        p = runs / sid / "eval_n500.json"
        if p.exists():
            systems[sid] = json.loads(p.read_text())["per_query"]

    print("=" * 72)
    print("System summary (section_recall_at_5_mean, section_hit_at_5_rate)")
    print("=" * 72)
    for sid, per_q in systems.items():
        mean_r5 = sum(r["section_recall_at_5"] for r in per_q) / len(per_q)
        hit5 = sum(1 for r in per_q if r["section_hit_at_5"]) / len(per_q)
        print(f"  {sid:30s}  mean_r@5={mean_r5:.4f}  hit@5={hit5:.4f}")

    hit = {sid: [r["section_hit_at_5"] for r in pq] for sid, pq in systems.items()}

    print()
    print("=" * 72)
    print("Paired McNemar (section_hit_at_5)")
    print("=" * 72)

    print("\nA2 random-only vs S3 zero-shot (pure LoRA effect, no mechanism-targeted negs)")
    mcnemar(hit["a2_random_only"], hit["s3_zero_shot_gte"], "A2", "S3")

    print("\nA2 random-only vs S2 hard-neg (does §5.1 mining help?)")
    mcnemar(hit["a2_random_only"], hit["s2_plain_lora"], "A2", "S2")

    print("\nS2 hard-neg vs S3 zero-shot (hard-neg LoRA effect)")
    mcnemar(hit["s2_plain_lora"], hit["s3_zero_shot_gte"], "S2", "S3")

    print("\nS1 STALI vs A2 random-only (prefix + hard-neg vs nothing)")
    mcnemar(hit["s1_stali"], hit["a2_random_only"], "S1", "A2")

    # Slice analysis
    gold = json.loads(Path("data/eval/gold_n500.json").read_text())

    def slice_mean(per_q, qids, field="section_recall_at_5"):
        vals = [r[field] for r in per_q if r["qid"] in qids]
        return sum(vals) / len(vals) if vals else 0.0

    print()
    print("=" * 72)
    print("Slice by query type (section_r@5)")
    print("=" * 72)
    print(f"  {'slice':<15}{'n':>5}  {'S3':>8}  {'A2':>8}  {'S2':>8}  {'A2-S3':>8}  {'S2-A2':>8}")
    for qtype in ("text", "table"):
        qids = {qid for qid, g in gold.items() if g.get("type") == qtype}
        n = len(qids)
        s3v = slice_mean(systems["s3_zero_shot_gte"], qids)
        a2v = slice_mean(systems["a2_random_only"], qids)
        s2v = slice_mean(systems["s2_plain_lora"], qids)
        print(f"  {qtype:<15}{n:>5}  {s3v:>8.4f}  {a2v:>8.4f}  {s2v:>8.4f}  {a2v-s3v:>+8.4f}  {s2v-a2v:>+8.4f}")


if __name__ == "__main__":
    main()
