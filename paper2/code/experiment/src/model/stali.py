"""Canonical section-type labels shared by data preparation and evaluation."""

SECTION_TYPES: tuple[str, ...] = (
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "tables_figures",
)
N_SECTION_TYPES = len(SECTION_TYPES)
SECTION_TYPE_TO_IDX = {name: index for index, name in enumerate(SECTION_TYPES)}

__all__ = ["SECTION_TYPES", "SECTION_TYPE_TO_IDX", "N_SECTION_TYPES"]
