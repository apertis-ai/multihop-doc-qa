"""Normalize DocHop-QA sections and 512-token cross-dataset paragraphs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from retriever_protocol import Chunk, ChunkUnit

_MAX_PARA_TOKENS = 512


def estimate_tokens(text: str) -> int:
    """Rough token count (words * 1.3). Only for chunking decisions, NOT for
    context budget accounting — that uses the actual tokenizer in reader.py."""
    return int(len(text.split()) * 1.3)


def split_to_paragraphs(text: str, max_tokens: int = _MAX_PARA_TOKENS) -> list[str]:
    """Split a long section into paragraph-sized chunks.

    Strategy:
      1. Split on blank lines.
      2. If a paragraph exceeds `max_tokens`, sentence-split and greedy-pack.
      3. Whitespace normalize.
    """
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    out: list[str] = []
    for p in paras:
        if estimate_tokens(p) <= max_tokens:
            out.append(" ".join(p.split()))
            continue
        # sentence split + greedy pack
        sents = re.split(r"(?<=[.!?])\s+", p)
        buf: list[str] = []
        buf_tokens = 0
        for s in sents:
            t = estimate_tokens(s)
            if buf_tokens + t > max_tokens and buf:
                out.append(" ".join(buf))
                buf, buf_tokens = [s], t
            else:
                buf.append(s)
                buf_tokens += t
        if buf:
            out.append(" ".join(buf))
    return out


def section_to_paragraphs(section_chunk: Chunk) -> list[Chunk]:
    """Convert a section-unit Chunk into paragraph-unit Chunks.

    Paragraph ID format: f"{section_id}||p{idx:04d}"
    """
    if section_chunk.unit != "section":
        raise ValueError(f"expected section unit, got {section_chunk.unit}")
    paras = split_to_paragraphs(section_chunk.text)
    return [
        Chunk(
            id=f"{section_chunk.id}||p{i:04d}",
            text=p,
            doc_id=section_chunk.doc_id,
            unit="paragraph",
            meta={**section_chunk.meta, "section_id": section_chunk.id, "para_idx": i},
        )
        for i, p in enumerate(paras)
    ]


def paragraph_to_section_rollup(
    retrieved_para_ids: list[str],
) -> list[str]:
    """Roll up paragraph hits to their parent section, preserving order.

    Used when a paragraph-unit retriever is scored against section-unit gold.
    """
    seen: set[str] = set()
    out: list[str] = []
    for pid in retrieved_para_ids:
        if "||p" not in pid:
            out.append(pid)
            seen.add(pid)
            continue
        sid = pid.rsplit("||p", 1)[0]
        if sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out
