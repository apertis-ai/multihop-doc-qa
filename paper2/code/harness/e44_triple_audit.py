"""E4.4 Triple-extraction audit — 2-annotator precision/recall on HippoRAG 2 OpenIE output.

Samples n triples from the HotpotQA calibration graph (claude-sonnet-4-6 + NV-Embed-v2),
runs two independent LLM annotators (temperature 0 vs temperature 0.3 as annotator drift proxy),
computes precision, Cohen's kappa, and 95% bootstrap CI. Also samples n' passages for a
recall proxy audit (missing-triple elicitation).

Usage:
    LLM_API_KEY=... python3 e44_triple_audit.py \\
        --input  ../e4_hipporag2_calib/outputs/openie_results_ner_claude-sonnet-4-6.json \\
        --outdir ../e4_hipporag2_calib/triple_audit \\
        --n-precision 50 --n-recall 25 --seed 20260421
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from readers.llm_reader import LLMReader  # type: ignore

ANNOTATOR_MODEL = "claude-sonnet-4-6"
PRECISION_LABELS = ("CORRECT", "PARTIAL", "INCORRECT")
LABEL_SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def _stable_id(*parts: Any) -> str:
    h = hashlib.sha1("||".join(map(str, parts)).encode()).hexdigest()
    return h[:12]


def sample_triples(docs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Flatten all (doc, triple) pairs, then sample n without replacement."""
    pool: list[dict] = []
    for doc in docs:
        passage = doc.get("passage", "")
        for triple in doc.get("extracted_triples", []) or []:
            if not (isinstance(triple, list) and len(triple) == 3):
                continue
            pool.append({
                "doc_idx": doc.get("idx"),
                "passage": passage,
                "triple": triple,
                "triple_id": _stable_id(doc.get("idx"), *triple),
            })
    rng.shuffle(pool)
    return pool[:n]


def sample_passages(docs: list[dict], n: int, rng: random.Random) -> list[dict]:
    pool = [doc for doc in docs if (doc.get("extracted_triples") and len(doc.get("passage", "")) > 200)]
    rng.shuffle(pool)
    return pool[:n]


PRECISION_SYSTEM = (
    "You are an expert knowledge-graph annotator auditing OpenIE triple extractions. "
    "You will be shown a source passage and one extracted triple [subject, predicate, object]. "
    "Judge whether the passage *directly supports* the triple as-is. "
    "Return a strict JSON object with fields: "
    '{"label": "CORRECT" | "PARTIAL" | "INCORRECT", '
    '"evidence": "<quote from passage>", '
    '"rationale": "<one sentence>"}. '
    "Use CORRECT only if every element of the triple is unambiguously supported by the passage. "
    "Use PARTIAL if the relation is gist-correct but one element is paraphrased, inexact, or missing a qualifier. "
    "Use INCORRECT if the passage does not support the claim, or it contradicts the passage, or the triple is malformed."
)


def build_precision_user(passage: str, triple: list[str]) -> str:
    s, p, o = triple
    return (
        f"Passage:\n{passage.strip()}\n\n"
        f"Triple to audit: [{s!r}, {p!r}, {o!r}]\n\n"
        "Return JSON only."
    )


RECALL_SYSTEM = (
    "You are an expert knowledge-graph annotator. You are given a passage and a list of triples "
    "extracted from it. Identify any important factual triples the extractor *missed*. "
    "Focus on salient, passage-supported facts a multi-hop QA system would need. "
    "Return a strict JSON object: "
    '{"missing_triples": [[s, p, o], ...], "rationale": "<one sentence per missing triple, concise>"}. '
    "If you think the extractor covered everything important, return an empty list. Do not list triples that are already present."
)


def build_recall_user(passage: str, triples: list[list[str]]) -> str:
    existing = "\n".join(f"- [{t[0]!r}, {t[1]!r}, {t[2]!r}]" for t in triples if len(t) == 3)
    return (
        f"Passage:\n{passage.strip()}\n\n"
        f"Already extracted triples:\n{existing or '(none)'}\n\n"
        "Return JSON only."
    )


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_json_reply(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def normalize_label(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    up = raw.strip().upper()
    for lab in PRECISION_LABELS:
        if lab in up:
            return lab
    return None


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Unweighted Cohen's kappa over PRECISION_LABELS."""
    assert len(labels_a) == len(labels_b) and labels_a
    n = len(labels_a)
    cats = PRECISION_LABELS
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pe = 0.0
    for c in cats:
        pa_c = sum(1 for x in labels_a if x == c) / n
        pb_c = sum(1 for x in labels_b if x == c) / n
        pe += pa_c * pb_c
    if pe >= 1.0:
        return 1.0
    return (agree - pe) / (1 - pe)


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 13) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def run_precision_audit(
    samples: list[dict],
    annotators: list[tuple[str, LLMReader]],
    out_jsonl: Path,
) -> list[dict]:
    results: list[dict] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fh = out_jsonl.open("w")
    try:
        for i, sample in enumerate(samples, 1):
            row = {"triple_id": sample["triple_id"], "doc_idx": sample["doc_idx"],
                   "triple": sample["triple"]}
            for name, reader in annotators:
                try:
                    resp = reader.chat(
                        PRECISION_SYSTEM,
                        build_precision_user(sample["passage"], sample["triple"]),
                    )
                    text = resp["content"]
                    parsed = parse_json_reply(text) or {}
                    label = normalize_label(parsed.get("label"))
                    row[f"{name}_label"] = label or "INCORRECT"
                    row[f"{name}_evidence"] = parsed.get("evidence", "")
                    row[f"{name}_rationale"] = parsed.get("rationale", "")
                    row[f"{name}_raw"] = text[:400]
                except Exception as e:
                    row[f"{name}_label"] = "INCORRECT"
                    row[f"{name}_error"] = str(e)[:200]
            results.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[precision {i}/{len(samples)}] "
                  f"A={row.get('A_label')} B={row.get('B_label')} "
                  f"triple={sample['triple']}", flush=True)
    finally:
        fh.close()
    return results


def run_recall_audit(
    samples: list[dict],
    annotators: list[tuple[str, LLMReader]],
    out_jsonl: Path,
) -> list[dict]:
    results: list[dict] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fh = out_jsonl.open("w")
    try:
        for i, doc in enumerate(samples, 1):
            row = {"doc_idx": doc.get("idx"),
                   "n_existing": len(doc.get("extracted_triples", []) or [])}
            for name, reader in annotators:
                try:
                    resp = reader.chat(
                        RECALL_SYSTEM,
                        build_recall_user(doc["passage"], doc.get("extracted_triples") or []),
                    )
                    text = resp["content"]
                    parsed = parse_json_reply(text) or {}
                    missing = parsed.get("missing_triples") or []
                    missing = [m for m in missing if isinstance(m, list) and len(m) == 3]
                    row[f"{name}_missing"] = missing
                    row[f"{name}_n_missing"] = len(missing)
                    row[f"{name}_rationale"] = parsed.get("rationale", "")[:400]
                except Exception as e:
                    row[f"{name}_missing"] = []
                    row[f"{name}_n_missing"] = 0
                    row[f"{name}_error"] = str(e)[:200]
            results.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[recall {i}/{len(samples)}] existing={row['n_existing']} "
                  f"A_miss={row.get('A_n_missing')} B_miss={row.get('B_n_missing')}", flush=True)
    finally:
        fh.close()
    return results


def aggregate_precision(rows: list[dict]) -> dict:
    la = [r.get("A_label", "INCORRECT") for r in rows]
    lb = [r.get("B_label", "INCORRECT") for r in rows]
    pa = [LABEL_SCORE[x] for x in la]
    pb = [LABEL_SCORE[x] for x in lb]
    merged = [(a + b) / 2 for a, b in zip(pa, pb)]
    mean_a = statistics.mean(pa)
    mean_b = statistics.mean(pb)
    mean_merged = statistics.mean(merged)
    ci = bootstrap_ci(merged)
    kappa = cohens_kappa(la, lb)
    return {
        "n": len(rows),
        "precision_A": mean_a,
        "precision_B": mean_b,
        "precision_merged": mean_merged,
        "precision_95ci": ci,
        "cohens_kappa": kappa,
        "label_hist_A": {lab: la.count(lab) for lab in PRECISION_LABELS},
        "label_hist_B": {lab: lb.count(lab) for lab in PRECISION_LABELS},
    }


def aggregate_recall(rows: list[dict]) -> dict:
    # recall proxy: per passage, recall = n_existing / (n_existing + n_missing_avg)
    per_doc = []
    for r in rows:
        ma = r.get("A_n_missing", 0)
        mb = r.get("B_n_missing", 0)
        missed = (ma + mb) / 2
        gold = r["n_existing"]
        per_doc.append(gold / (gold + missed) if (gold + missed) else 1.0)
    mean = statistics.mean(per_doc) if per_doc else float("nan")
    ci = bootstrap_ci(per_doc) if per_doc else (float("nan"), float("nan"))
    return {
        "n_passages": len(rows),
        "recall_mean": mean,
        "recall_95ci": ci,
        "avg_missing_A": statistics.mean(r.get("A_n_missing", 0) for r in rows) if rows else 0,
        "avg_missing_B": statistics.mean(r.get("B_n_missing", 0) for r in rows) if rows else 0,
    }


def write_report(outdir: Path, prec: dict, rec: dict, meta: dict) -> Path:
    md = [
        "# E4.4 Triple-extraction audit",
        "",
        f"- Generated: {meta['timestamp']}",
        f"- Source: `{meta['input']}`",
        f"- Annotator model: `{ANNOTATOR_MODEL}` via LLM provider",
        f"- Annotator A: temperature={meta['temp_a']} / Annotator B: temperature={meta['temp_b']}",
        f"- Seed: {meta['seed']}",
        "",
        "## Precision",
        f"- n = {prec['n']} triples",
        f"- precision (merged, mean of A/B): **{prec['precision_merged']:.3f}**  (95% CI {prec['precision_95ci'][0]:.3f}–{prec['precision_95ci'][1]:.3f})",
        f"- precision A: {prec['precision_A']:.3f}  |  precision B: {prec['precision_B']:.3f}",
        f"- Cohen's κ (A vs B): **{prec['cohens_kappa']:.3f}**",
        "",
        "Label histogram:",
        f"- A: {prec['label_hist_A']}",
        f"- B: {prec['label_hist_B']}",
        "",
        "## Recall (missing-triple audit)",
        f"- n = {rec['n_passages']} passages",
        f"- recall proxy (mean): **{rec['recall_mean']:.3f}**  (95% CI {rec['recall_95ci'][0]:.3f}–{rec['recall_95ci'][1]:.3f})",
        f"- avg missing per passage: A={rec['avg_missing_A']:.2f}  B={rec['avg_missing_B']:.2f}",
        "",
        "## Interpretation",
        "κ > 0.6 → substantial agreement. Report merged precision & recall proxy in paper §E4.4. "
        "If precision < 0.80, flag triple-extraction quality as a confound for HippoRAG 2 retrieval claims.",
    ]
    out = outdir / "audit_results.md"
    out.write_text("\n".join(md))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-precision", type=int, default=50)
    ap.add_argument("--n-recall", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260421)
    ap.add_argument("--temp-a", type=float, default=0.0)
    ap.add_argument("--temp-b", type=float, default=0.3)
    ap.add_argument("--model", default=ANNOTATOR_MODEL)
    args = ap.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    data = json.loads(args.input.read_text())
    docs = data["docs"]

    triples = sample_triples(docs, args.n_precision, rng)
    passages = sample_passages(docs, args.n_recall, rng)

    (args.outdir / "samples_precision.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in triples)
    )
    (args.outdir / "samples_recall.jsonl").write_text(
        "\n".join(json.dumps({"doc_idx": d.get("idx"),
                              "n_triples": len(d.get("extracted_triples", []) or []),
                              "passage_len": len(d.get("passage", ""))},
                             ensure_ascii=False) for d in passages)
    )

    cache = args.outdir / "_cache"
    annotators = [
        ("A", LLMReader(model=args.model, temperature=args.temp_a,
                            max_tokens=512, cache_dir=cache / "A")),
        ("B", LLMReader(model=args.model, temperature=args.temp_b,
                            max_tokens=512, cache_dir=cache / "B")),
    ]

    prec_rows = run_precision_audit(triples, annotators, args.outdir / "precision.jsonl")
    rec_rows = run_recall_audit(passages, annotators, args.outdir / "recall.jsonl")

    prec = aggregate_precision(prec_rows)
    rec = aggregate_recall(rec_rows)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "input": str(args.input),
        "temp_a": args.temp_a,
        "temp_b": args.temp_b,
        "seed": args.seed,
    }
    report = write_report(args.outdir, prec, rec, meta)

    summary = {"precision": prec, "recall": rec, "meta": meta}
    (args.outdir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✅ Report written: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
