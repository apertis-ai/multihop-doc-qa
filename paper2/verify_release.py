#!/usr/bin/env python3
"""Verify Paper 2 release integrity and publication exclusions."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GOLD_SHA256 = "7a35f83e0a38ac86da125db8ad3705295186619588ef9827752712d65ca5470d"
MAX_FILE_BYTES = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".bin", ".log", ".pt", ".pth", ".safetensors"}
FORBIDDEN_FILE_NAMES = {
    "agents" + ".md",
    "claude" + ".md",
    "decision" + ".md",
    "decisions" + ".md",
    "plan" + ".md",
    "plans" + ".md",
    "proposal" + ".md",
    "tasks" + ".md",
}
FORBIDDEN_DIR_NAMES = {"decision" + "s", "open" + "spec", "plan" + "s"}
SECRET_PATTERNS = {
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "RunPod API key": re.compile(rb"rpa_[A-Za-z0-9]{30,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(rb"-----BEGIN (?:EC |OPENSSH |RSA )?PRIVATE KEY-----"),
    "generic API credential": re.compile(
        rb"(?i)\b(?:api[_-]?key|access[_-]?token|secret)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
}
INTERNAL_PATTERNS = {
    "agent instruction reference": re.compile(b"agents" + rb"\.md|claude" + rb"\.md", re.I),
    "internal workflow reference": re.compile(b"open" + b"spec|exp_" + b"plan", re.I),
    "internal run path": re.compile(b"rc-" + rb"run/|stage" + rb"_[0-9]+", re.I),
    "private machine path": re.compile(
        b"/" + b"Users/|/" + b"home/|/work" + b"space/|/runpod-" + b"volume/"
    ),
    "private planning reference": re.compile(b"pre-" + b"reg|pre-" + b"registration", re.I),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_negative_sampling() -> None:
    code = """
from src.data.hard_negatives import HardNegativeMiner, Section, filter_by_jaccard
sections = [Section(str(i), "s", None, str(i)) for i in range(10)]
sample = HardNegativeMiner().mine_random_negatives("query-1", sections, n=4, seed=42)
assert len(filter_by_jaccard([{"jaccard": 0.02}, {"jaccard": 0.05}], 0.05)) == 1
print(",".join(item.section.pmc_id for item in sample))
"""
    outputs = []
    for hash_seed in ("1", "2"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                cwd=ROOT / "code/experiment",
                env=env,
                text=True,
            )
        )
    if outputs[0] != outputs[1]:
        raise SystemExit("random-negative sampling changes with PYTHONHASHSEED")


def verify_correction_logic() -> None:
    def load(relative: str) -> dict:
        path = ROOT / relative
        namespace = {"__name__": "release_check"}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        return namespace

    correction = load("code/experiment/data/eval/correct_legacy_section_ids.py")
    gold = {str(i): {"relevant_section_ids": [f"PMC{i}||"]} for i in range(500)}
    rows = [
        {
            "qid": str(i),
            "top5_sids": [f"PMC{i}||/N/A" if i == 0 else f"PMC{i}||"],
            "section_recall_at_5": 0.0 if i == 0 else 1.0,
        }
        for i in range(500)
    ]
    corrected = correction["correct_results"]({"summary": {}, "per_query": rows}, gold)
    if corrected["section_id_correction"]["queries_changed"] != 1:
        raise SystemExit("legacy section-ID correction self-check failed")

    prep = load("code/harness/prep_dochop_for_hipporag.py")
    if prep["chunk_title"]("PMC1", "", "") != "PMC1||":
        raise SystemExit("HippoRAG section IDs do not match the gold builder")

    stats = load("code/harness/run_e46_stats.py")
    sample = {
        "gold_doc_ids": ["doc1", "doc2"],
        "results": [
            {"chunk_id": "doc1||p0000"},
            {"chunk_id": "doc1||p0001"},
            {"chunk_id": "doc2||p0000"},
        ],
    }
    if stats["metrics_stali"](sample, k=2) != (1.0, 1.0):
        raise SystemExit("STALI chunk-to-document metric self-check failed")
    if stats["metrics_hipporag"](["doc1", "doc1", "doc2"], ["doc1", "doc2"], 2) != (1.0, 1.0):
        raise SystemExit("HippoRAG unique-document metric self-check failed")


def main() -> None:
    manifest = ROOT / "SHA256SUMS"
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path in listed:
            raise SystemExit(f"invalid manifest path: {relative}")
        listed[path] = digest

    actual = {
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path != manifest
    }
    if set(listed) != actual:
        missing = sorted(str(path) for path in actual - set(listed))
        stale = sorted(str(path) for path in set(listed) - actual)
        raise SystemExit(f"manifest coverage mismatch: unlisted={missing}, missing={stale}")

    for relative, expected in listed.items():
        path = ROOT / relative
        if (
            relative.name.casefold() in FORBIDDEN_FILE_NAMES
            or any(part.casefold() in FORBIDDEN_DIR_NAMES for part in relative.parts)
        ):
            raise SystemExit(f"internal planning file found in release: {relative}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            raise SystemExit(f"forbidden release file: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise SystemExit(f"oversized release file: {relative}")
        if sha256(path) != expected:
            raise SystemExit(f"checksum mismatch: {relative}")
        content = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                raise SystemExit(f"{label} found in {relative}")
        for label, pattern in INTERNAL_PATTERNS.items():
            if pattern.search(content):
                raise SystemExit(f"{label} found in {relative}")

    ids_path = ROOT / "code/experiment/data/eval/query_ids_500.json"
    gold_path = ROOT / "code/experiment/data/eval/gold_n500.json"
    ids_doc = json.loads(ids_path.read_text())
    query_ids = ids_doc["query_ids"]
    gold = json.loads(gold_path.read_text())
    if ids_doc.get("n") != 500 or len(query_ids) != 500 or len(set(query_ids)) != 500:
        raise SystemExit("query_ids_500.json is not 500 unique IDs")
    if list(gold) != query_ids:
        raise SystemExit("gold_n500.json keys do not match the fixed query-ID order")
    if sha256(gold_path) != GOLD_SHA256:
        raise SystemExit("gold_n500.json does not match the published digest")

    verify_negative_sampling()
    verify_correction_logic()
    print(f"verified {len(listed)} files; n=500; gold={GOLD_SHA256}")


if __name__ == "__main__":
    main()
