"""E6 stats driver — paired bootstrap + Bonferroni across four retrievers,
using per-query scores from each runs/e6_*/eval.json.

Comparisons reported (primary endpoint = paragraph_recall@5):
  - stali vs each baseline (3 pairs × 3 metrics = 9 tests → Bonferroni α=0.05/9)
  - Also report EM/F1 summary per system for reference.

Writes the requested JSON and Markdown summary paths.
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


SYSTEMS = ["stali", "bge_m3", "jina_v3", "qwen3_emb_4b"]
METRICS = ["para_r5", "para_r10", "para_hit5", "doc_r5", "em", "f1"]
PRIMARY_METRIC = "para_r5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, type=Path,
                    help="directory containing the e6_* run folders")
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    args = ap.parse_args()

    stats_mod = _imp("stats", "stats.py")

    systems_data: dict[str, dict] = {}
    for sys_key in SYSTEMS:
        path = args.runs_dir / f"e6_{sys_key}" / "eval.json"
        if not path.exists():
            print(f"[stats] missing {path} — skipping system {sys_key}", flush=True)
            continue
        systems_data[sys_key] = json.loads(path.read_text())

    if "stali" not in systems_data:
        raise RuntimeError("stali eval missing; cannot run primary comparison")

    def per_query(system, metric):
        rows = systems_data[system]["rows"]
        return {r["qid"]: float(r[metric]) for r in rows}

    # Align on qids present in ALL systems
    qid_sets = [set(per_query(s, PRIMARY_METRIC).keys()) for s in systems_data]
    common_qids = sorted(set.intersection(*qid_sets), key=lambda x: int(x))

    summary: dict[str, dict[str, float]] = {}
    for s, d in systems_data.items():
        summary[s] = dict(d["summary"])
        summary[s]["n_rows"] = d["n_queries"]
    answerable_summary = {
        s: d["retrieval_summary_answerable"]
        for s, d in systems_data.items()
        if "retrieval_summary_answerable" in d
    }

    # Bonferroni: STALI vs each of 3 baselines × 3 primary metrics (para_r5, em, f1)
    cmp_metrics = ["para_r5", "em", "f1"]
    baselines = [s for s in systems_data if s != "stali"]
    n_tests = len(baselines) * len(cmp_metrics)
    alpha_corrected = 0.05 / max(n_tests, 1)

    comparisons = []
    for metric in cmp_metrics:
        stali_scores = [per_query("stali", metric)[q] for q in common_qids]
        for base in baselines:
            base_scores = [per_query(base, metric)[q] for q in common_qids]
            res = stats_mod.paired_bootstrap(stali_scores, base_scores,
                                             n_boot=10_000, alpha=0.05)
            comparisons.append({
                "metric": metric,
                "a": "stali", "b": base,
                "mean_a": res.mean_a, "mean_b": res.mean_b,
                "diff": res.diff,
                "ci_low": res.ci_low, "ci_high": res.ci_high,
                "p_value": res.p_value, "cohens_d": res.cohens_d,
                "classification": res.classification,
                "bonferroni_alpha": alpha_corrected,
                "bonferroni_significant": (
                    res.p_value is not None and res.p_value < alpha_corrected
                ),
            })

    out = {
        "n_common_qids": len(common_qids),
        "systems_present": list(systems_data.keys()),
        "summary": summary,
        "summary_scope": "all queries; empty-gold retrieval metrics score 0",
        "retrieval_summary_answerable": answerable_summary,
        "retrieval_counts": {
            s: {
                "n_answerable": d.get("n_answerable_retrieval"),
                "n_empty_gold": d.get("n_empty_gold_retrieval"),
                "n_null_queries": d.get("n_null_queries"),
            }
            for s, d in systems_data.items()
        },
        "comparisons": comparisons,
        "bonferroni_n_tests": n_tests,
        "bonferroni_alpha": alpha_corrected,
    }
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out_json}", flush=True)

    # Markdown table
    lines = [
        "# E6 MultiHop-RAG — Stats Summary",
        "",
        f"n_common_qids = {len(common_qids)}; bonferroni α = {alpha_corrected:.4f} ({n_tests} tests)",
        "",
        "## Per-system all-query summary",
        "",
        "Empty-gold retrieval rows score 0; answerable-only retrieval metrics are reported below when available.",
        "",
        "| system | para_r@5 | para_r@10 | para_hit@5 | doc_r@5 | EM | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in SYSTEMS:
        if s not in summary:
            continue
        r = summary[s]
        lines.append(
            f"| {s} | {r['para_r5']:.4f} | {r['para_r10']:.4f} | "
            f"{r['para_hit5']:.4f} | {r['doc_r5']:.4f} | "
            f"{r['em']:.4f} | {r['f1']:.4f} |"
        )
    if answerable_summary:
        lines += [
            "",
            "## Answerable-only retrieval summary",
            "",
            "| system | para_r@5 | para_r@10 | para_hit@5 | doc_r@5 | doc_hit@5 |",
            "|---|---|---|---|---|---|",
        ]
        for s in SYSTEMS:
            if s not in answerable_summary:
                continue
            r = answerable_summary[s]
            lines.append(
                f"| {s} | {r['para_r5']:.4f} | {r['para_r10']:.4f} | "
                f"{r['para_hit5']:.4f} | {r['doc_r5']:.4f} | {r['doc_hit5']:.4f} |"
            )
    lines += [
        "",
        "## STALI vs baselines (paired bootstrap 10k, Bonferroni)",
        "",
        "| metric | vs | Δ (stali − base) | 95% CI | p | d | class | Bonf. sig |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        sig = "**YES**" if c["bonferroni_significant"] else "no"
        lines.append(
            f"| {c['metric']} | {c['b']} | {c['diff']:+.4f} | "
            f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
            f"{c['p_value']:.4f} | {c['cohens_d']:+.2f} | "
            f"{c['classification']} | {sig} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
