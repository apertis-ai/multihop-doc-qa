"""Retrieval + QA metrics for unified harness.

All metric computations in the released Paper 2 harness route through this module.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Fraction of gold chunks appearing in top-k.

    For multi-gold questions this is |top_k ∩ gold| / |gold|, clamped to [0, 1].
    For single-gold it reduces to hit@k.
    """
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & gold_ids) / len(gold_ids)


def hit_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """1 if any gold in top-k, else 0."""
    if not gold_ids:
        return 0.0
    return 1.0 if (set(retrieved_ids[:k]) & gold_ids) else 0.0


def mrr_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Reciprocal rank of first gold chunk within top-k, else 0."""
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


# ---------- QA metrics (EM / F1) follow SQuAD/HotpotQA convention ----------

_ARTICLES = {"a", "an", "the"}


def _normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation/articles/extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gold: str | list[str]) -> float:
    golds = [gold] if isinstance(gold, str) else gold
    return max(
        float(_normalize_answer(pred) == _normalize_answer(g))
        for g in golds
    )


def f1(pred: str, gold: str | list[str]) -> float:
    golds = [gold] if isinstance(gold, str) else gold
    return max(_f1_single(pred, g) for g in golds)


def _f1_single(pred: str, gold: str) -> float:
    p_toks = _normalize_answer(pred).split()
    g_toks = _normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    prec = n_same / len(p_toks)
    rec = n_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)


# ---------- Error decomposition (per Zarrinkia2026GraphRAGBottleneck) ----------


def classify_error(
    retrieved_ids: list[str],
    gold_ids: set[str],
    k: int,
    pred: str,
    gold_answer: str | list[str],
    em_threshold: float = 1.0,
) -> str:
    """Return one of: 'correct', 'retrieval_failure', 'reasoning_failure'.

    retrieval_failure: gold not in top-k.
    reasoning_failure: gold in top-k but EM below threshold.
    correct: EM ≥ threshold.
    """
    em = exact_match(pred, gold_answer)
    if em >= em_threshold:
        return "correct"
    if not (set(retrieved_ids[:k]) & gold_ids):
        return "retrieval_failure"
    return "reasoning_failure"
