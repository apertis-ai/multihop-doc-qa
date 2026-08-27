#!/usr/bin/env python3
"""Correct top-5 metrics in legacy DocHop-QA outputs with missing subsections."""

import argparse
import hashlib
import json
from pathlib import Path


GOLD_SHA256 = "7a35f83e0a38ac86da125db8ad3705295186619588ef9827752712d65ca5470d"


def correct_results(data: dict, gold: dict[str, dict]) -> dict:
    rows = data.get("per_query")
    if not rows or len(rows) != 500:
        raise ValueError("result file must contain 500 per_query rows")

    old_scores = []
    new_scores = []
    queries_changed = rescued_sections = 0
    seen = set()
    for row in rows:
        qid = str(row.get("qid", ""))
        if qid in seen or qid not in gold:
            raise ValueError(f"invalid or duplicate query ID: {qid!r}")
        seen.add(qid)
        if "top5_sids" not in row or "section_recall_at_5" not in row:
            raise ValueError(f"query {qid} lacks retained top-5 section IDs")

        gold_ids = set(gold[qid]["relevant_section_ids"])
        corrected_ids = []
        for section_id in row["top5_sids"]:
            corrected = section_id
            if corrected not in gold_ids and corrected.endswith("/N/A"):
                without_placeholder = corrected[:-4]
                if without_placeholder in gold_ids:
                    corrected = without_placeholder
                    rescued_sections += 1
            corrected_ids.append(corrected)

        old_score = float(row["section_recall_at_5"])
        new_score = len(gold_ids & set(corrected_ids)) / len(gold_ids)
        queries_changed += abs(old_score - new_score) > 1e-12
        old_scores.append(old_score)
        new_scores.append(new_score)
        row["top5_sids"] = corrected_ids
        row["section_recall_at_5"] = new_score
        row["section_hit_at_5"] = new_score > 0

    old_mean = sum(old_scores) / len(old_scores)
    new_mean = sum(new_scores) / len(new_scores)
    summary = data.get("summary") or {}
    summary["section_recall_at_5_mean"] = new_mean
    summary["section_hit_at_5_rate"] = sum(score > 0 for score in new_scores) / len(new_scores)
    data["summary"] = summary
    data["section_id_correction"] = {
        "reason": "legacy evaluator rendered a missing subsection as /N/A",
        "scope": "top-5 section IDs, recall, and hit only; top-10 fields are unchanged because top-10 IDs were not retained",
        "queries_changed": queries_changed,
        "rescued_gold_sections": rescued_sections,
        "historical_section_recall_at_5_mean": old_mean,
        "corrected_section_recall_at_5_mean": new_mean,
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gold", type=Path, default=Path(__file__).with_name("gold_n500.json"))
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("refusing to overwrite the historical input")
    if hashlib.sha256(args.gold.read_bytes()).hexdigest() != GOLD_SHA256:
        raise ValueError("gold file does not match the released digest")

    data = json.loads(args.input.read_text())
    gold = json.loads(args.gold.read_text())
    corrected = correct_results(data, gold)
    args.output.write_text(json.dumps(corrected, indent=2) + "\n")


if __name__ == "__main__":
    main()
