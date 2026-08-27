"""Unified retriever Protocol for Paper 2 revision experiments (E4/E5/E6).

Every system (A2 adapter, HippoRAG 2, LAD-RAG, dense bi-encoder baselines)
implements this Protocol so the reader + metrics pipeline is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable


ChunkUnit = Literal["section", "paragraph"]


@dataclass(frozen=True)
class Chunk:
    """Atomic retrieval unit.

    id: composite identifier stable across systems within a dataset.
        DocHop section format: f"{pmc_id}||{section}/{subsection}"
        MultiHop-RAG paragraph format: f"{doc_id}||p{para_idx:04d}"
    text: surface text fed to encoder/reader.
    doc_id: parent document id (for doc_r@k).
    unit: "section" or "paragraph". Cross-dataset comparisons MUST normalize to "paragraph".
    meta: dataset-specific metadata (section title, page num, ...).
    """

    id: str
    text: str
    doc_id: str
    unit: ChunkUnit
    meta: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    rank: int  # 1-indexed


@dataclass
class IndexCosts:
    """Reported per-system, used for Pareto-frontier table."""

    indexing_usd: float = 0.0
    index_build_seconds: float = 0.0
    per_query_usd: float = 0.0
    per_query_latency_ms: float = 0.0
    notes: str = ""


@runtime_checkable
class Retriever(Protocol):
    """Minimal interface every retriever in Paper 2 E4/E5/E6 must satisfy."""

    name: str

    def index(self, corpus: list[Chunk]) -> None:
        """Build (or load) index over the given corpus. Idempotent per system."""
        ...

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """Return top-k chunk ids sorted by score desc, rank 1-indexed."""
        ...

    def costs(self) -> IndexCosts:
        """Report aggregated cost/latency for the Pareto table."""
        ...
