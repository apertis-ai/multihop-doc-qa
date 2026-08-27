"""Build DocHop-QA training triples for STALI.

Emits one JSONL row per query:
    {
      "query_id": "...",
      "query": "...",
      "positives": [{"pmc_id","section","subsection","text","section_type_idx"}, ...],
      "hard_negatives": [{"pmc_id","section","subsection","text","section_type_idx","jaccard"}, ...],
      "random_negatives": [{"pmc_id","section","subsection","text","section_type_idx"}, ...],
    }

Strict no-leak contract:
  - Query IDs in data/eval/query_ids_500.json are EXCLUDED from training.
  - The resulting triples never reference a query used for Paper 2's n=500 eval.
  - A manifest file records the final train/eval split sizes.

Usage (W2):
    python -m src.data.build_triples \
        --dochop-json /path/to/'[F] DocHopQA_Dataset.json' \
        --eval-ids data/eval/query_ids_500.json \
        --out data/train/dochop_qa_triples.jsonl \
        --hard-neg-k 10 --random-neg-k 5 --jaccard-threshold 0.05
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import click

# Tolerate being invoked as a standalone script as well as a package
try:
    from src.data.hard_negatives import HardNegativeMiner, Section, tokenise, jaccard
    from src.data.section_classifier import classify_composite_key
    from src.model.stali import SECTION_TYPE_TO_IDX
except ImportError:  # running as __main__ without pythonpath set
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.hard_negatives import HardNegativeMiner, Section, tokenise, jaccard
    from src.data.section_classifier import classify_composite_key
    from src.model.stali import SECTION_TYPE_TO_IDX

log = logging.getLogger("build_triples")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _ctx_to_section(ctx: dict) -> Section:
    return Section(
        pmc_id=str(ctx.get("pmc_id") or ""),
        section_title=str(ctx.get("section") or ""),
        subsection_title=(str(ctx["subsection"]) if ctx.get("subsection") else None),
        text=str(ctx.get("content") or ""),
    )


def _section_record(sec: Section, extra: dict | None = None) -> dict:
    stype = classify_composite_key(sec.pmc_id, sec.section_title, sec.subsection_title)
    rec = {
        "pmc_id": sec.pmc_id,
        "section": sec.section_title,
        "subsection": sec.subsection_title,
        "text": sec.text[:4000],  # match DocHop-QA gold-context truncation
        "section_type": stype,
        "section_type_idx": SECTION_TYPE_TO_IDX[stype],
    }
    if extra:
        rec.update(extra)
    return rec


@click.command()
@click.option("--dochop-json", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to DocHopQA_Dataset.json (HF download cached)")
@click.option("--eval-ids", type=click.Path(exists=True, path_type=Path), required=True,
              help="JSON file with the n=500 eval query IDs to exclude from training")
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True,
              help="Output JSONL path")
@click.option("--hard-neg-k", type=int, default=10)
@click.option("--random-neg-k", type=int, default=5)
@click.option("--jaccard-threshold", type=float, default=0.05)
@click.option("--seed", type=int, default=42)
@click.option("--max-queries", type=int, default=0, help="0 = all")
def main(
    dochop_json: Path,
    eval_ids: Path,
    out_path: Path,
    hard_neg_k: int,
    random_neg_k: int,
    jaccard_threshold: float,
    seed: int,
    max_queries: int,
) -> int:
    rng = random.Random(seed)
    log.info("loading DocHop-QA from %s", dochop_json)
    with dochop_json.open() as fh:
        raw = json.load(fh)
    log.info("  total records: %d", len(raw))

    # Load eval IDs to exclude
    eval_data = json.loads(eval_ids.read_text())
    eval_id_set = set(str(x) for x in eval_data.get("query_ids", []))
    log.info("  eval_ids to exclude: %d", len(eval_id_set))

    # Build per-paper section index — every section across ALL records (train + eval)
    # is a candidate for hard negatives, but queries whose ID is in eval_id_set
    # are never used as training query anchors.
    paper_sections: dict[str, list[Section]] = defaultdict(list)
    seen_keys: set[tuple[str, str, str | None]] = set()
    for rec in raw:
        for ctx in rec.get("context_list") or []:
            sec = _ctx_to_section(ctx)
            if not sec.pmc_id or not sec.text:
                continue
            key = (sec.pmc_id, sec.section_title, sec.subsection_title)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            paper_sections[sec.pmc_id].append(sec)
    all_papers = list(paper_sections.keys())
    log.info("  unique papers: %d  unique sections: %d", len(all_papers), len(seen_keys))

    miner = HardNegativeMiner(threshold=jaccard_threshold, top_k=hard_neg_k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_eval = 0
    n_skipped_no_gold = 0
    with out_path.open("w") as fh:
        for rec in raw:
            qid = str(rec.get("id", ""))
            if qid in eval_id_set:
                n_skipped_eval += 1
                continue
            query = rec.get("question") or ""
            ctx_list = rec.get("context_list") or []
            if not query or not ctx_list:
                n_skipped_no_gold += 1
                continue

            gold_sections = [_ctx_to_section(c) for c in ctx_list if c.get("pmc_id")]
            gold_keys = {s.composite_key for s in gold_sections}
            if not gold_sections:
                n_skipped_no_gold += 1
                continue

            # Hard negatives: same-paper high-Jaccard non-gold
            correct_pmcs = {s.pmc_id for s in gold_sections}
            candidate_sections: list[Section] = []
            for pmc in correct_pmcs:
                candidate_sections.extend(paper_sections.get(pmc, []))
            hard_negs = miner.mine_for_query(
                query_id=qid,
                query_text=query,
                gold_section_keys=gold_keys,
                candidate_sections=candidate_sections,
            )

            # Random negatives: sampled from wrong papers
            wrong_papers = [p for p in all_papers if p not in correct_pmcs]
            random_pool: list[Section] = []
            rng_sample = rng.sample(wrong_papers, min(40, len(wrong_papers)))
            for p in rng_sample:
                random_pool.extend(paper_sections.get(p, []))
            random_negs = miner.mine_random_negatives(
                query_id=qid,
                wrong_paper_sections=random_pool,
                n=random_neg_k,
                seed=seed,
            )

            out_rec = {
                "query_id": qid,
                "query": query,
                "positives": [_section_record(s) for s in gold_sections],
                "hard_negatives": [
                    _section_record(n.section, {"jaccard": n.jaccard_score, "reason": n.reason})
                    for n in hard_negs
                ],
                "random_negatives": [
                    _section_record(n.section, {"reason": n.reason})
                    for n in random_negs
                ],
            }
            fh.write(json.dumps(out_rec) + "\n")
            n_written += 1
            if max_queries and n_written >= max_queries:
                log.info("stopping at --max-queries=%d", max_queries)
                break

    log.info(
        "done. wrote=%d skipped_eval=%d skipped_no_gold=%d out=%s",
        n_written, n_skipped_eval, n_skipped_no_gold, out_path,
    )

    # Manifest for reproducibility.
    manifest = {
        "n_train_triples": n_written,
        "n_eval_excluded": n_skipped_eval,
        "n_skipped_no_gold": n_skipped_no_gold,
        "total_records": len(raw),
        "unique_papers": len(all_papers),
        "unique_sections": len(seen_keys),
        "hard_neg_k": hard_neg_k,
        "random_neg_k": random_neg_k,
        "jaccard_threshold": jaccard_threshold,
        "seed": seed,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("manifest at %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(standalone_mode=False) or 0)
