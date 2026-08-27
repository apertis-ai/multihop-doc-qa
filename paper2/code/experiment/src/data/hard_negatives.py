"""Hard-negative miner grounded in Paper 1 §5.1 lexical-overlap analysis.

Given a query q and its gold sections (pmc_id, section) in a correctly-shortlisted
paper P, the §5.1 failure-mode population is:

    { s ∈ P : s ∉ gold_sections(q, P) AND jaccard(tokens(q), tokens(s)) ≥ τ }

These sections lie on the exact axis dense retrievers get wrong — high lexical
overlap with the question but not the evidential content. Training with these
hard negatives teaches the retriever to discriminate lexical bait from true
evidence.

The computation reimplements Paper 1's token-Jaccard utility with identical
tokenisation (lowercased alphanumeric, English stopwords removed).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

# English stopwords matching Paper 1's lexical-overlap analysis
_STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all am an and any are aren as at be because been
    before being below between both but by can could did do does doing down during each few
    for from further had has have having he her here hers herself him himself his how i if
    in into is it its itself just me more most my myself no nor not now of off on once only
    or other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up very
    was we were what when where which while who whom why will with you your yours yourself
    yourselves""".split()
)

_TOKEN_RE = re.compile(r"\b[a-z0-9]+\b")


def tokenise(text: str) -> set[str]:
    """Paper 1 tokenisation: lowercase, alphanumeric, stopwords removed."""
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS and len(tok) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(union, 1)


def filter_by_jaccard(negatives: Iterable[dict], threshold: float | None) -> list[dict]:
    """Keep mined negatives at or above `threshold`; `None` keeps all."""
    if threshold is None:
        return list(negatives)
    return [neg for neg in negatives if float(neg.get("jaccard", 0.0)) >= threshold]


@dataclass
class Section:
    pmc_id: str
    section_title: str
    subsection_title: str | None
    text: str

    @property
    def composite_key(self) -> tuple[str, str, str | None]:
        return (self.pmc_id, self.section_title, self.subsection_title)


@dataclass
class MinedNegative:
    query_id: str
    section: Section
    jaccard_score: float
    reason: str  # "high_overlap_non_gold" / "random_wrong_paper" / etc.


class HardNegativeMiner:
    """§5.1-grounded hard negative miner.

    Usage:
        miner = HardNegativeMiner(threshold=0.05, top_k=10)
        hard_negs = miner.mine_for_query(query_text, gold_sections, candidate_sections_in_paper)
    """

    def __init__(self, threshold: float = 0.05, top_k: int = 10):
        self.threshold = threshold
        self.top_k = top_k

    def mine_for_query(
        self,
        query_id: str,
        query_text: str,
        gold_section_keys: set[tuple[str, str, str | None]],
        candidate_sections: Iterable[Section],
    ) -> list[MinedNegative]:
        """Return up to `top_k` hard negatives sorted by descending Jaccard.

        Candidates are filtered to only include sections from the correctly-shortlisted
        paper(s) that are NOT in the gold set — i.e., same-paper lexical-lookalikes.
        """
        q_tokens = tokenise(query_text)
        scored: list[tuple[float, Section]] = []
        for sec in candidate_sections:
            if sec.composite_key in gold_section_keys:
                continue
            s_tokens = tokenise(sec.text)
            j = jaccard(q_tokens, s_tokens)
            if j >= self.threshold:
                scored.append((j, sec))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].pmc_id,
                item[1].section_title,
                item[1].subsection_title or "",
            )
        )
        return [
            MinedNegative(
                query_id=query_id,
                section=sec,
                jaccard_score=j,
                reason="high_overlap_non_gold",
            )
            for j, sec in scored[: self.top_k]
        ]

    def mine_random_negatives(
        self,
        query_id: str,
        wrong_paper_sections: list[Section],
        n: int = 5,
        seed: int = 0,
    ) -> list[MinedNegative]:
        """Uniform random sampling from sections in wrong papers.

        Used for the A2 random-neg ablation baseline.
        """
        import random

        query_seed = int.from_bytes(hashlib.sha256(query_id.encode()).digest()[:8], "big")
        rng = random.Random(seed + query_seed)
        sample = rng.sample(wrong_paper_sections, min(n, len(wrong_paper_sections)))
        return [
            MinedNegative(
                query_id=query_id,
                section=sec,
                jaccard_score=0.0,
                reason="random_wrong_paper",
            )
            for sec in sample
        ]


__all__ = [
    "Section",
    "MinedNegative",
    "HardNegativeMiner",
    "tokenise",
    "jaccard",
    "filter_by_jaccard",
]
