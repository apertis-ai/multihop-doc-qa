#!/usr/bin/env python3
"""Rebuild the exact Paper 2 DocHop-QA gold file from its fixed query IDs."""

import argparse
import json
from pathlib import Path


def build_gold(source_path: Path, ids_path: Path) -> dict[str, dict]:
    source = json.loads(source_path.read_text())
    ids_doc = json.loads(ids_path.read_text())
    query_ids = ids_doc["query_ids"]
    if ids_doc.get("n") != len(query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError("query_ids file has an invalid count or duplicate IDs")

    by_id = {str(record["id"]): record for record in source}
    if len(by_id) != len(source):
        raise ValueError("source dataset contains duplicate IDs")

    gold = {}
    for query_id in query_ids:
        if query_id not in by_id:
            raise ValueError(f"source dataset is missing query ID {query_id}")
        record = by_id[query_id]
        used_doc = record.get("used_doc")
        task_type = record.get("task_type")
        contexts = record.get("context_list")
        if used_doc not in {"single", "multi"}:
            raise ValueError(f"query {query_id} has invalid used_doc={used_doc!r}")
        if task_type not in {"Paragraph-Oriented", "Table-Oriented"}:
            raise ValueError(f"query {query_id} has invalid task_type={task_type!r}")
        if not record.get("question") or not contexts:
            raise ValueError(f"query {query_id} has no question or gold contexts")

        doc_ids = []
        section_ids = []
        for context in contexts:
            pmc_id = str(context.get("pmc_id") or "")
            if not pmc_id:
                raise ValueError(f"query {query_id} has a context without pmc_id")
            doc_ids.append(pmc_id)
            section_ids.append(
                f'{pmc_id}||{context.get("section", "")}/{context.get("subsection", "")}'.rstrip("/")
            )

        gold[query_id] = {
            "query": record["question"],
            "hop": 1 if used_doc == "single" else 2,
            "type": "table" if task_type == "Table-Oriented" else "text",
            "relevant_doc_ids": sorted(set(doc_ids)),
            "relevant_section_ids": sorted(set(section_ids)),
        }
    return gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="[F] DocHopQA_Dataset.json")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path(__file__).with_name("query_ids_500.json"),
    )
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_gold(args.source, args.ids), indent=2))


if __name__ == "__main__":
    main()
