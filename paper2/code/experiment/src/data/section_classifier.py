"""Rule-based section-type classifier.

Maps (pmc_id, section, subsection) composite keys to one of 7 section types:
    abstract, introduction, methods, results, discussion, conclusion, tables_figures.

Validation target: ≥ 95% precision on a 100-sample hand-labelled DocHop-QA set.

The classifier is intentionally simple: it operates on the section title string
(from the DocHop-QA composite key) and uses a cascade of regex rules. Order matters
— more specific patterns checked before generic ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.model.stali import SECTION_TYPES, SECTION_TYPE_TO_IDX


@dataclass
class SectionTypeRule:
    type_name: str
    pattern: re.Pattern[str]
    priority: int  # lower = checked first


# Patterns are intentionally pmc-centric; DocHop-QA is PubMed Central text.
_RULES: list[SectionTypeRule] = [
    # --- priority 0: high-confidence keyword triggers ---
    SectionTypeRule("tables_figures", re.compile(r"^\s*(table|figure|fig\.?)\s*\d", re.I), 0),
    SectionTypeRule("tables_figures", re.compile(r"(supp(lementary)?\s*)?(tab(le)?|fig(ure)?)\s+\w+", re.I), 0),

    # --- priority 1: canonical section names ---
    SectionTypeRule("abstract", re.compile(r"^\s*abstract\b", re.I), 1),
    SectionTypeRule("introduction", re.compile(r"^\s*(introduction|background|motivation)\b", re.I), 1),
    SectionTypeRule("methods", re.compile(r"^\s*(methods?|materials?\s+and\s+methods?|experimental\s+(setup|procedure|design)|methodology|study\s+design)\b", re.I), 1),
    SectionTypeRule("results", re.compile(r"^\s*(results?|findings?|observations?|experimental\s+results)\b", re.I), 1),
    SectionTypeRule("discussion", re.compile(r"^\s*(discussion|analysis|interpretation)\b", re.I), 1),
    SectionTypeRule("conclusion", re.compile(r"^\s*(conclusions?|summary|concluding\s+remarks|final\s+remarks)\b", re.I), 1),

    # --- priority 2: biomedical-specific variants ---
    SectionTypeRule("methods", re.compile(r"(patients?|subjects?|participants?|data\s+collection|statistical\s+analys[ie]s|randomi[sz]ation|inclusion\s+criteria)", re.I), 2),
    SectionTypeRule("results", re.compile(r"(outcome|efficacy|safety|baseline\s+characteristics|adverse\s+events|primary\s+endpoint)", re.I), 2),
    SectionTypeRule("introduction", re.compile(r"(related\s+work|prior\s+art|literature\s+review)", re.I), 2),
    SectionTypeRule("discussion", re.compile(r"(limitations?|implications?|future\s+work|future\s+directions?)", re.I), 2),

    # --- priority 3: weak signal fallbacks ---
    SectionTypeRule("methods", re.compile(r"(algorithm|model|architecture|implementation|preprocess)", re.I), 3),
    SectionTypeRule("results", re.compile(r"(evaluation|benchmark|experiment|ablation)", re.I), 3),
]


def classify_section_title(title: str | None) -> str:
    """Return one of SECTION_TYPES given a section title string.

    Default fallback is "introduction" (the most common section type in PMC abstracts).
    """
    if not title:
        return "introduction"
    title_norm = title.strip()
    if not title_norm:
        return "introduction"

    for rule in sorted(_RULES, key=lambda r: r.priority):
        if rule.pattern.search(title_norm):
            return rule.type_name

    return "introduction"


def classify_composite_key(pmc_id: str, section: str | None, subsection: str | None = None) -> str:
    """Classify from the DocHop-QA `(pmc_id, section, subsection)` composite key.

    Subsection is checked first because it is usually more specific than the section.
    """
    # Try subsection first (more specific), then section
    for candidate in (subsection, section):
        if candidate:
            result = classify_section_title(candidate)
            if result != "introduction":  # "introduction" is the generic fallback; keep searching
                return result
    # No specific signal found — return intro as fallback or let downstream check
    return classify_section_title(section or subsection or "")


def classify_section_idx(pmc_id: str, section: str | None, subsection: str | None = None) -> int:
    """Return the int index for the classified section type."""
    return SECTION_TYPE_TO_IDX[classify_composite_key(pmc_id, section, subsection)]


def evaluate_on_validation_set(val_samples: list[tuple[str, str, str, str]]) -> dict[str, float]:
    """Evaluate precision on a hand-labelled validation set.

    Args:
        val_samples: list of (pmc_id, section, subsection, gold_type) tuples

    Returns:
        dict with overall precision and per-type breakdown
    """
    correct = 0
    per_type: dict[str, list[int]] = {t: [0, 0] for t in SECTION_TYPES}  # [correct, total]
    for pmc_id, section, subsection, gold in val_samples:
        pred = classify_composite_key(pmc_id, section, subsection)
        per_type[gold][1] += 1
        if pred == gold:
            correct += 1
            per_type[gold][0] += 1

    n = len(val_samples)
    return {
        "precision_overall": correct / max(n, 1),
        "n_samples": n,
        "per_type_recall": {
            t: per_type[t][0] / max(per_type[t][1], 1)
            for t in SECTION_TYPES
        },
        "passes_H1b": (correct / max(n, 1)) >= 0.95,
    }


__all__ = [
    "classify_section_title",
    "classify_composite_key",
    "classify_section_idx",
    "evaluate_on_validation_set",
]
