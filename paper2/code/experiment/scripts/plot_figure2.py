"""Figure 2 — Slice analysis: text vs table queries across S3 / S2 / A2.

Publication-quality: 300 DPI, ACL/EMNLP single-column width (3.4 × 3.0 in)
or double-column (5.5 × 3.2 in, configurable).

Usage (from repo root):
    python paper2/code/experiment/scripts/plot_figure2.py \
        --runs-dir /path/to/runs \
        --gold paper2/code/experiment/data/eval/gold_n500.json \
        --out paper2/figures/figure2_slice.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})
import matplotlib.pyplot as plt
import numpy as np

C_TEXT  = "#0072B2"   # text queries (Wong blue)
C_TABLE = "#E69F00"   # table queries (Wong amber)


def load_per_query(path: Path) -> list[dict]:
    return json.loads(path.read_text())["per_query"]


def slice_recall(per_query: list[dict], qids: set[str],
                 field: str = "section_recall_at_5") -> float:
    vals = [r[field] for r in per_query if r["qid"] in qids]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("runs"))
    ap.add_argument("--gold", type=Path,
                    default=Path("code/experiment/data/eval/gold_n500.json"))
    ap.add_argument("--metric", default="section_recall_at_5")
    ap.add_argument("--wide", action="store_true",
                    help="Double-column width (5.5 in) instead of single-column (3.4 in)")
    ap.add_argument("--out", type=Path,
                    default=Path("figures/figure2_slice.png"))
    args = ap.parse_args()

    R = args.runs_dir

    # ── Load gold qid → type mapping ─────────────────────────────────────────
    gold = json.loads(args.gold.read_text())
    text_qids  = {qid for qid, g in gold.items() if g.get("type") == "text"}
    table_qids = {qid for qid, g in gold.items() if g.get("type") == "table"}
    n_text  = len(text_qids)
    n_table = len(table_qids)

    # ── Systems to compare ────────────────────────────────────────────────────
    system_defs = [
        ("S3\nGTE-MC\nzero-shot",  R / "s3_zero_shot_gte" / "eval_n500.json"),
        ("S2\nLoRA +\nhard-neg",   R / "s2_plain_lora"    / "eval_n500.json"),
        ("A2\nLoRA +\nrandom-neg", R / "a2_random_only"   / "eval_n500.json"),
    ]

    systems = []
    for label, path in system_defs:
        if not path.exists():
            print(f"[warn] missing {path}, skipping")
            continue
        pq = load_per_query(path)
        systems.append({
            "label": label,
            "text":  slice_recall(pq, text_qids,  args.metric),
            "table": slice_recall(pq, table_qids, args.metric),
        })

    if not systems:
        print("[error] no systems found")
        return 1

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_w = 5.5 if args.wide else 3.4
    fig, ax = plt.subplots(figsize=(fig_w, 3.2), constrained_layout=True)

    n_groups = len(systems)
    x = np.arange(n_groups)
    bar_w = 0.34

    text_vals  = [s["text"]  for s in systems]
    table_vals = [s["table"] for s in systems]
    xlabels    = [s["label"] for s in systems]

    bars_text  = ax.bar(x - bar_w / 2, text_vals,  bar_w, color=C_TEXT,
                        label=f"Text queries  (n = {n_text})",
                        edgecolor="white", linewidth=0.5)
    bars_table = ax.bar(x + bar_w / 2, table_vals, bar_w, color=C_TABLE,
                        label=f"Table queries (n = {n_table})",
                        edgecolor="white", linewidth=0.5)

    # ── Value labels above each bar ───────────────────────────────────────────
    for bar, v in zip(list(bars_text) + list(bars_table),
                      text_vals + table_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#222222")

    # ── Delta annotations between systems (table slice only) ─────────────────
    for i in range(1, len(systems)):
        x0 = x[i - 1] + bar_w / 2
        x1 = x[i]     + bar_w / 2
        y0 = table_vals[i - 1]
        y1 = table_vals[i]
        delta = y1 - y0
        y_mid = (y0 + y1) / 2 + 0.008
        ax.annotate(
            "", xy=(x1, y1 + 0.018), xytext=(x0, y0 + 0.018),
            arrowprops=dict(arrowstyle="-", color="#888888",
                            lw=0.9, linestyle="dashed"),
        )
        ax.text((x0 + x1) / 2, y_mid + 0.026,
                f"{delta:+.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#555555")

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, ha="center", multialignment="center")
    ax.set_ylabel("Section recall@5")
    ax.set_ylim(0, max(max(text_vals), max(table_vals)) + 0.13)
    ax.set_title(
        "Figure 2 — Slice analysis: text vs. table queries\n"
        "LoRA benefit is 2.8× larger on table queries; "
        "hard-neg adds no further gain",
        pad=6,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.legend(loc="lower right", ncol=1, frameon=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}  ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
