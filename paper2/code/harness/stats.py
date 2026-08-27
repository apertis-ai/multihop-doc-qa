"""Statistical protocol for Paper 2 paired comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


@dataclass
class ComparisonResult:
    mean_a: float
    mean_b: float
    diff: float  # mean_a - mean_b
    ci_low: float
    ci_high: float
    p_value: float | None
    cohens_d: float
    n: int
    classification: str  # "wins" / "matches" / "inferior" / "n.s."


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> ComparisonResult:
    """Paired bootstrap CI on mean difference (a - b)."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap requires equal-length arrays, got {len(a)} vs {len(b)}")
    n = len(a)
    rng = np.random.default_rng(rng_seed)
    diffs = a - b
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    # two-sided p-value via proportion of resamples crossing 0
    p = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())
    d = cohens_d(a, b)
    return ComparisonResult(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        diff=float(diffs.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=float(p),
        cohens_d=d,
        n=n,
        classification=classify_comparison(float(diffs.mean()), float(lo), float(hi), float(p), d),
    )


def mcnemar_binary(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> float:
    """Two-sided McNemar exact test on paired binary outcomes. Returns p-value."""
    ca = np.asarray(correct_a, dtype=bool)
    cb = np.asarray(correct_b, dtype=bool)
    b = int(((~ca) & cb).sum())  # a wrong, b right
    c = int((ca & (~cb)).sum())  # a right, b wrong
    n = b + c
    if n == 0:
        return 1.0
    # exact binomial 2-sided
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for paired samples using pooled SD of the difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Return list of `is_significant_after_correction`."""
    k = len(p_values)
    if k == 0:
        return []
    adj = alpha / k
    return [p < adj for p in p_values]


def classify_comparison(
    diff_pp: float,
    ci_low: float,
    ci_high: float,
    p_value: float,
    d: float,
) -> Literal["wins", "matches", "inferior", "n.s."]:
    """Classify a comparison using fixed significance and effect thresholds.

    - wins: p<0.05 AND d≥0.2 AND diff>0
    - inferior: p<0.05 AND d≥0.2 AND diff<0
    - matches: CI contained within ±2pp of zero with significance (non-inferiority)
    - n.s.: CI straddles 0 OR effect too small
    """
    if p_value is None:
        return "n.s."
    significant = p_value < 0.05
    effect_meaningful = abs(d) >= 0.2
    if significant and effect_meaningful and diff_pp > 0:
        return "wins"
    if significant and effect_meaningful and diff_pp < 0:
        return "inferior"
    # non-inferiority: CI fully within ±2pp (0.02 for proportions)
    if abs(ci_low) < 0.02 and abs(ci_high) < 0.02:
        return "matches"
    return "n.s."
