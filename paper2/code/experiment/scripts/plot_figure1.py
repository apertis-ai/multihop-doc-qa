"""Figure 1 — Section-recall@5 bar chart across retrieval systems.

Publication-quality: 300 DPI, ACL/EMNLP two-column full-width (7 × 5.2 in).

Usage (from repo root):
    python paper2/code/experiment/scripts/plot_figure1.py \
        --runs-dir /path/to/runs \
        --out paper2/figures/figure1_teaser.png
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
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Wong (2011) colour-blind-safe palette
C_GREY   = "#bbbbbb"   # Paper 1 reference
C_AMBER  = "#E69F00"   # S7 ColBERTv2 (different family)
C_SKY    = "#56B4E9"   # S3 GTE-MC zero-shot
C_PURPLE = "#CC79A7"   # S6 Reason-MC zero-shot
C_BLUE   = "#0072B2"   # S2 LoRA + hard-neg (ours)
C_GREEN  = "#009E73"   # S1 STALI (ours)
C_BEST   = "#D55E00"   # A2 LoRA + random-neg (ours, BEST)


def read_metric(path: Path | None,
                metric: str = "section_recall_at_5_mean") -> float | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("summary", {}).get(metric)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("runs"))
    ap.add_argument("--metric", default="section_recall_at_5_mean")
    ap.add_argument("--ceiling-doc", type=float, default=0.664)
    ap.add_argument("--out", type=Path,
                    default=Path("figures/figure1_teaser.png"))
    args = ap.parse_args()
    R = args.runs_dir

    # ── System list: (display_label, value_or_path, colour, is_baseline_hatch) ──
    # Paper 1 reference values (hardcoded; not re-run)
    bars: list[tuple[str, float, str, bool]] = [
        ("P1: BM25",                               0.386,  C_GREY,   False),
        ("P1: OpenAI text-emb-3-small",            0.514,  C_GREY,   False),
        ("P1b: jina-v3 (dense baseline)",          0.514,  C_GREY,   True),   # hatched = primary baseline
    ]

    experimental = [
        ("S7: ColBERTv2-512 (zero-shot)",          R / "s7_colbertv2"     / "eval_n500.json", C_AMBER,  False),
        ("S6: Reason-ModernColBERT (zero-shot)",   R / "s6_reason_modern" / "eval_n500.json", C_PURPLE, False),
        ("S3: GTE-ModernColBERT (zero-shot)",      R / "s3_zero_shot_gte" / "eval_n500.json", C_SKY,    False),
        ("S2: LoRA + hard-neg (ours)",             R / "s2_plain_lora"    / "eval_n500.json", C_BLUE,   False),
        ("S1: STALI — LoRA + type prefix (ours)",  R / "s1_stali"         / "eval_n500.json", C_GREEN,  False),
        ("A2: LoRA + random-neg (ours)  ★ BEST",  R / "a2_random_only"   / "eval_n500.json", C_BEST,   False),
    ]
    for label, path, colour, hatch in experimental:
        v = read_metric(path, args.metric)
        if v is None:
            print(f"[warn] skipping {label}: {path}")
            continue
        bars.append((label, v, colour, hatch))

    labels  = [b[0] for b in bars]
    values  = [b[1] for b in bars]
    colours = [b[2] for b in bars]
    hatches = ["//" if b[3] else None for b in bars]

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.33, right=0.96, top=0.88, bottom=0.28)

    rects = ax.barh(
        labels, values,
        color=colours,
        hatch=hatches,
        edgecolor=["#555555" if h else "#dddddd" for h in hatches],
        linewidth=0.7,
        height=0.68,
    )
    # Bold border on BEST bar
    rects[-1].set_edgecolor("#111111")
    rects[-1].set_linewidth(1.4)

    # Reference vertical lines
    ax.axvline(x=0.514, color="#555555", linestyle=":",
               linewidth=1.2, alpha=0.85, label="P1b dense baseline (0.514)")
    ax.axvline(x=args.ceiling_doc, color="#222222", linestyle="--",
               linewidth=1.1, alpha=0.65, label=f"Doc r@5 ceiling ({args.ceiling_doc:.3f})")

    # Value labels at bar ends
    for rect, v in zip(rects, values):
        ax.text(
            v + 0.004,
            rect.get_y() + rect.get_height() / 2,
            f"{v:.4f}",
            va="center", ha="left",
            fontsize=8, color="#111111",
        )

    ax.set_xlim(0, args.ceiling_doc + 0.10)
    ax.set_xlabel("Section recall@5  (DocHop-QA, n = 500)", labelpad=5)
    ax.set_title(
        "Figure 1 — Section-recall@5 across retrieval systems\n"
        "A2 (LoRA + random-neg) closes 55.5 % of the baseline → ceiling gap",
        pad=8,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linestyle="--", zorder=0)

    # ── Legend 1: reference lines — lower right (bars are short there) ───────
    ax.legend(
        loc="lower right",
        fontsize=7.5,
        title="Reference lines",
        title_fontsize=7.5,
        frameon=True,
    )

    # ── Legend 2: system categories — BELOW the axes ─────────────────────────
    cat_patches = [
        mpatches.Patch(color=C_GREY,   label="Paper 1 reference (P1/P1b)"),
        mpatches.Patch(color=C_AMBER,  label="ColBERTv2 (S7, existing)"),
        mpatches.Patch(color=C_PURPLE, label="Reason-ModernColBERT (S6)"),
        mpatches.Patch(color=C_SKY,    label="GTE-ModernColBERT zero-shot (S3)"),
        mpatches.Patch(color=C_BLUE,   label="LoRA + hard-neg (S2, ours)"),
        mpatches.Patch(color=C_GREEN,  label="STALI / LoRA + type prefix (S1, ours)"),
        mpatches.Patch(color=C_BEST,   label="LoRA + random-neg (A2, ours) ★"),
    ]
    fig.legend(
        handles=cat_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        fontsize=7.5,
        frameon=True,
        title="System category",
        title_fontsize=7.5,
        columnspacing=1.2,
        handlelength=1.4,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}  ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
