"""MultiHop-RAG dataset loader for unified harness (E6).

MultiHop-RAG is used for in-dataset LoRA training (E6) and paragraph-unit
evaluation, avoiding a section-versus-paragraph granularity mismatch.

Corpus: 609 news articles from yixuantt/MultiHopRAG (corpus.json).
Queries: 2556 multi-hop queries (MultiHopRAG.json) with evidence_list
specifying gold (article_title, fact) pairs.

Unit: paragraph. We split each article body on blank lines, strip headings,
then merge consecutive short fragments to ~256-token windows (the MultiHop-RAG
paper convention). Gold paragraph membership is determined by substring match
of evidence_list[i].fact within the paragraph text (case-insensitive,
whitespace-normalized).

Doc ID: stable hash of article URL (sha1[:12]).
Paragraph ID: f"{doc_id}||p{para_idx:04d}".
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from retriever_protocol import Chunk  # noqa: E402

_TARGET_TOKENS = 256
_TOKENS_PER_WORD = 1.3  # rough for English news


@dataclass
class MultiHopQuery:
    qid: str
    question: str
    answer: str
    question_type: str
    gold_paragraph_ids: set[str]
    gold_doc_ids: set[str]
    gold_facts: list[str]  # original evidence facts, for fallback/debug


def _doc_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _para_id(doc_id: str, para_idx: int) -> str:
    return f"{doc_id}||p{para_idx:04d}"


_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s).strip().lower()


def _split_paragraphs(body: str) -> list[str]:
    """Split body into ~256-token paragraphs.

    Strategy: split on blank lines, drop empty/very-short fragments,
    then greedily merge until each paragraph is >= 200 words (~256 tokens).
    """
    raw = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not raw:
        return []
    target_words = int(_TARGET_TOKENS / _TOKENS_PER_WORD)  # ~196
    out: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for frag in raw:
        wc = len(frag.split())
        if buf and buf_words + wc > target_words * 1.6:
            out.append("\n\n".join(buf))
            buf, buf_words = [frag], wc
        else:
            buf.append(frag)
            buf_words += wc
            if buf_words >= target_words:
                out.append("\n\n".join(buf))
                buf, buf_words = [], 0
    if buf:
        out.append("\n\n".join(buf))
    return out


def load_corpus(corpus_path: Path) -> tuple[list[Chunk], dict[str, str]]:
    """Build paragraph-unit corpus from MultiHopRAG corpus.json.

    Returns (chunks, title_to_doc_id). title_to_doc_id maps article title →
    doc_id, used by load_queries to resolve evidence_list[i].title → gold doc.
    """
    with corpus_path.open() as fh:
        records = json.load(fh)
    chunks: list[Chunk] = []
    title_to_doc_id: dict[str, str] = {}
    for rec in records:
        url = rec.get("url") or ""
        title = rec.get("title") or ""
        body = rec.get("body") or ""
        if not url or not body.strip():
            continue
        doc_id = _doc_id(url)
        if title:
            title_to_doc_id[_norm(title)] = doc_id
        for idx, para_text in enumerate(_split_paragraphs(body)):
            chunks.append(
                Chunk(
                    id=_para_id(doc_id, idx),
                    text=para_text,
                    doc_id=doc_id,
                    unit="paragraph",
                    meta={
                        "title": title,
                        "url": url,
                        "source": rec.get("source"),
                        "category": rec.get("category"),
                        "published_at": rec.get("published_at"),
                    },
                )
            )
    return chunks, title_to_doc_id


def load_queries(
    queries_path: Path,
    corpus_chunks: list[Chunk],
    title_to_doc_id: dict[str, str],
) -> list[MultiHopQuery]:
    """Load queries with gold paragraph + doc annotations resolved against corpus.

    Gold paragraph match: case-insensitive whitespace-normalized substring of
    evidence_list[i].fact within the paragraph text. Gold doc: resolved by
    article title (evidence_list[i].title) via title_to_doc_id.
    """
    with queries_path.open() as fh:
        records = json.load(fh)

    by_doc: dict[str, list[Chunk]] = {}
    for c in corpus_chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    out: list[MultiHopQuery] = []
    for i, rec in enumerate(records):
        gold_facts: list[str] = []
        gold_para_ids: set[str] = set()
        gold_doc_ids: set[str] = set()
        for ev in rec.get("evidence_list") or []:
            fact = (ev.get("fact") or "").strip()
            title = (ev.get("title") or "").strip()
            if not fact:
                continue
            gold_facts.append(fact)
            doc_id = title_to_doc_id.get(_norm(title))
            if doc_id is None:
                continue
            gold_doc_ids.add(doc_id)
            fact_norm = _norm(fact)
            if not fact_norm:
                continue
            for chunk in by_doc.get(doc_id, []):
                if fact_norm in _norm(chunk.text):
                    gold_para_ids.add(chunk.id)
                    break  # first matching paragraph per evidence
        out.append(
            MultiHopQuery(
                qid=str(i),
                question=rec["query"],
                answer=str(rec.get("answer", "")),
                question_type=rec.get("question_type", ""),
                gold_paragraph_ids=gold_para_ids,
                gold_doc_ids=gold_doc_ids,
                gold_facts=gold_facts,
            )
        )
    return out


def split_train_heldout(
    queries: list[MultiHopQuery],
    n_heldout: int = 500,
    seed: int = 42,
) -> tuple[list[MultiHopQuery], list[MultiHopQuery]]:
    """Deterministic split: n_heldout evaluation queries, then training rows."""
    import random

    rng = random.Random(seed)
    idx = list(range(len(queries)))
    rng.shuffle(idx)
    heldout_idx = set(idx[:n_heldout])
    train = [q for i, q in enumerate(queries) if i not in heldout_idx]
    heldout = [q for i, q in enumerate(queries) if i in heldout_idx]
    return train, heldout
