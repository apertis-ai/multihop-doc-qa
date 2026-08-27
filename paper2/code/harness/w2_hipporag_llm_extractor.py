"""HippoRAG 2 triple extractor using LLM provider API (claude-opus-4-7, gpt-5.5, deepseek-v4-pro).

Per W2 task: extract KGs from passages using three frontier LLMs via LLM provider endpoint.

Prompt template follows paper §2.6 (entity, relation, entity triples).
Temperature: 0.0 (deterministic).
Retry: exponential backoff up to 5 attempts.
Checkpoint: skip passages already in output file.

Usage:
    python w2_hipporag_llm_extractor.py \
      --dataset dochop \
      --extractor claude-opus-4-7 \
      --passages /path/to/passages.jsonl \
      --out /path/to/triples.jsonl

Output JSONL schema:
    {
        "passage_id": str,
        "text": str,
        "triples": [{"subject": str, "relation": str, "object": str}, ...]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI, APIError, RateLimitError
except ImportError:
    print("ERROR: openai SDK not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)


EXTRACTION_PROMPT_TEMPLATE = """Extract structured triples (subject, relation, object) from the following passage.
Each triple should represent a fact or relationship in the passage.
Return ONLY a JSON array of objects with keys "subject", "relation", "object".
If no triples can be extracted, return an empty array [].

Passage:
{passage_text}

JSON array of triples:"""


class LLMProviderTripleExtractor:
    def __init__(
        self,
        model: str,
        max_retries: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY environment variable not set. "
                "Set it before running: export LLM_API_KEY=..."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL", "https://api.apertis.ai/v1"),
        )
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, passage: str) -> list[dict[str, str]]:
        """Extract triples from a passage. Returns list of {subject, relation, object}."""
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(passage_text=passage)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting structured knowledge from text. "
                            "Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content.strip()
                triples = json.loads(content)
                if not isinstance(triples, list):
                    triples = []
                return triples
            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(
                        f"[{self.model}] Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}",
                        file=sys.stderr,
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Rate limit exceeded after {self.max_retries} retries: {e}")
            except APIError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(
                        f"[{self.model}] API error: {e}. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}",
                        file=sys.stderr,
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API call failed after {self.max_retries} retries: {e}")
            except json.JSONDecodeError:
                # If extraction fails to parse JSON, return empty
                print(
                    f"[{self.model}] Failed to parse JSON response. Returning empty triple list.",
                    file=sys.stderr,
                )
                return []
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(
                        f"[{self.model}] Unexpected error: {e}. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}",
                        file=sys.stderr,
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Extraction failed after {self.max_retries} retries: {e}")

        return []


def load_processed_ids(out_path: Path) -> set[str]:
    """Load passage IDs already in the output file (checkpoint mechanism)."""
    if not out_path.exists():
        return set()

    processed = set()
    with out_path.open("r") as fh:
        for line in fh:
            if line.strip():
                try:
                    record = json.loads(line)
                    if "passage_id" in record:
                        processed.add(record["passage_id"])
                except json.JSONDecodeError:
                    pass
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Extract triples from passages using LLM provider LLM.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["dochop", "multihoprag", "hotpotqa"],
        help="Dataset name (used for context only).",
    )
    parser.add_argument(
        "--extractor",
        required=True,
        choices=["claude-opus-4-7", "gpt-5.5", "deepseek-v4-pro"],
        help="LLM model ID on LLM provider.",
    )
    parser.add_argument(
        "--passages",
        required=True,
        help="Path to passages JSONL file (schema: {id, text, ...}).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path for triples.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per passage (exponential backoff).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per response.",
    )

    args = parser.parse_args()

    passages_path = Path(args.passages)
    out_path = Path(args.out)

    if not passages_path.exists():
        print(f"ERROR: passages file not found: {passages_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    processed_ids = load_processed_ids(out_path)
    print(
        f"Loaded {len(processed_ids)} previously processed passages from {out_path}",
        file=sys.stderr,
    )

    # Initialize extractor
    extractor = LLMProviderTripleExtractor(
        model=args.extractor,
        max_retries=args.max_retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Process passages
    total = 0
    skipped = 0
    processed = 0

    with passages_path.open("r") as in_fh, out_path.open("a") as out_fh:
        for line_num, line in enumerate(in_fh, 1):
            if not line.strip():
                continue

            try:
                passage = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"ERROR at line {line_num}: invalid JSON. Skipping.",
                    file=sys.stderr,
                )
                continue

            total += 1
            passage_id = passage.get("id") or passage.get("passage_id") or str(total)

            # Checkpoint: skip if already processed
            if passage_id in processed_ids:
                skipped += 1
                continue

            text = passage.get("text", "").strip()
            if not text:
                print(
                    f"WARNING: passage {passage_id} has no text. Skipping.",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            # Extract triples
            try:
                triples = extractor.extract(text)
            except Exception as e:
                print(
                    f"ERROR processing passage {passage_id}: {e}",
                    file=sys.stderr,
                )
                triples = []

            # Write output record
            record = {
                "passage_id": passage_id,
                "text": text,
                "triples": triples,
            }
            out_fh.write(json.dumps(record) + "\n")
            processed += 1

            if processed % 10 == 0:
                print(
                    f"[{args.extractor}] Processed {processed} passages "
                    f"(total={total}, skipped={skipped})",
                    file=sys.stderr,
                )

    print(
        json.dumps({
            "dataset": args.dataset,
            "extractor": args.extractor,
            "total_passages": total,
            "newly_processed": processed,
            "skipped_checkpoint": skipped,
            "output_path": str(out_path),
        }, indent=2),
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
