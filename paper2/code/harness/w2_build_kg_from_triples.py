"""Build HippoRAG 2 knowledge graph from extracted triples.

Processes triple JSONL output from w2_hipporag_llm_extractor.py:
  - Entity deduplication (lowercase, strip)
  - Edge merging (same subject + relation + object = 1 edge)
  - Undirected entity linking (for retrieval)
  - Output kg.json (HippoRAG format)

Output schema:
{
  "entities": {
    "entity_name": {
      "aliases": [...],
      "relations": [...],
      "doc_source": [...]  # passage IDs where entity appears
    },
    ...
  },
  "relations": [
    {
      "subject": str,
      "relation": str,
      "object": str,
      "doc_source": [...]
    },
    ...
  ]
}

Usage:
    python w2_build_kg_from_triples.py \
      --triples /path/to/triples.jsonl \
      --out /path/to/kg.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import DefaultDict
from collections import defaultdict


def normalize_entity(entity: str) -> str:
    """Normalize entity name for deduplication."""
    return entity.lower().strip()


def build_kg_from_triples(triples_path: Path) -> dict:
    """Build KG from triples JSONL file."""
    entities: dict[str, dict] = {}
    relations: list[dict] = []
    edge_set: set = set()  # (subject, relation, object) for dedup

    if not triples_path.exists():
        print(f"ERROR: triples file not found: {triples_path}", file=sys.stderr)
        sys.exit(1)

    with triples_path.open("r") as fh:
        for line_num, line in enumerate(fh, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: invalid JSON at line {line_num}. Skipping.", file=sys.stderr)
                continue

            passage_id = record.get("passage_id", f"passage_{line_num}")
            triples = record.get("triples", [])

            for triple in triples:
                if not isinstance(triple, dict):
                    continue

                subject = (triple.get("subject") or "").strip()
                relation = (triple.get("relation") or "").strip()
                obj = (triple.get("object") or "").strip()

                # Skip incomplete triples
                if not subject or not relation or not obj:
                    continue

                # Normalize and deduplicate
                subj_norm = normalize_entity(subject)
                obj_norm = normalize_entity(obj)
                rel_norm = relation.lower().strip()

                edge_key = (subj_norm, rel_norm, obj_norm)
                if edge_key in edge_set:
                    # Update doc_source for existing edge
                    for rel_record in relations:
                        if (
                            normalize_entity(rel_record["subject"]) == subj_norm
                            and rel_record["relation"].lower() == rel_norm
                            and normalize_entity(rel_record["object"]) == obj_norm
                        ):
                            if passage_id not in rel_record.get("doc_source", []):
                                rel_record["doc_source"].append(passage_id)
                            break
                    continue

                edge_set.add(edge_key)

                # Record entities
                for ent_name in [subj_norm, obj_norm]:
                    if ent_name not in entities:
                        entities[ent_name] = {
                            "aliases": [ent_name],
                            "relations": [],
                            "doc_source": [],
                        }
                    if passage_id not in entities[ent_name]["doc_source"]:
                        entities[ent_name]["doc_source"].append(passage_id)

                # Record relation
                relation_record = {
                    "subject": subj_norm,
                    "relation": rel_norm,
                    "object": obj_norm,
                    "doc_source": [passage_id],
                }
                relations.append(relation_record)

                # Update entity relations
                entities[subj_norm]["relations"].append(rel_norm)
                entities[obj_norm]["relations"].append(rel_norm)

    # Deduplicate relation lists in entities
    for entity in entities.values():
        entity["relations"] = list(set(entity["relations"]))

    kg = {
        "entities": entities,
        "relations": relations,
    }
    return kg


def main():
    parser = argparse.ArgumentParser(
        description="Build HippoRAG 2 knowledge graph from extracted triples.",
    )
    parser.add_argument(
        "--triples",
        required=True,
        help="Path to triples JSONL file (output from w2_hipporag_llm_extractor.py).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for kg.json",
    )

    args = parser.parse_args()

    triples_path = Path(args.triples)
    out_path = Path(args.out)

    # Create output directory
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build KG
    print(f"Building KG from {triples_path}...", file=sys.stderr)
    kg = build_kg_from_triples(triples_path)

    # Write output
    out_path.write_text(json.dumps(kg, indent=2))

    print(
        json.dumps({
            "num_entities": len(kg["entities"]),
            "num_relations": len(kg["relations"]),
            "output_path": str(out_path),
        }, indent=2),
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
