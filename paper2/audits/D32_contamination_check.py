#!/usr/bin/env python3
"""D32 contamination control for the W6 reader-QA table.

Checks whether STALI's W6 EM/F1 numbers reflect retriever quality vs.
reader memorization, by comparing three reader settings on MultiHop-RAG
(news, n=200 sample of n=500 eval set):
  (A) NO-CONTEXT  — just the question, no passages
  (B) SHUFFLED-CONTEXT — question + 5 random passages from the corpus
                         (no overlap with the gold passages)
  (C) STALI-TOP5  — already-computed values from anonymous_repo_build/
                    data/e6_results/eval.json (per-query EM/F1)

Reader: gpt-5.5 via Apertis (env APERTIS_API_KEY, APERTIS_BASE_URL).
Resumable: appends to ./contamination_results.jsonl, skips qids already done.

Aggregate output: ./contamination_summary.json with per-setting EM/F1
and the NO-CONTEXT vs STALI-TOP5 paired delta.
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from openai import OpenAI

# ──── config
N_SAMPLE = 200            # contamination-check subsample
SEED = 42
MODEL = "gpt-5.5"
MAX_TOKENS = 80
N_SHUFFLE = 5             # 5 random other-corpus passages

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DATA = RELEASE_ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent
MHRAG_CORPUS = Path(os.environ.get("MHRAG_CORPUS_JSON", DATA / "hipporag_results/mhrag_corpus.json"))
RESULTS = OUT_DIR / "contamination_results.jsonl"
SUMMARY = OUT_DIR / "contamination_summary.json"
LOG = OUT_DIR / "contamination_run.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def normalize(s):
    s = (s or "").lower().strip()
    for c in [".", ",", "?", "!", "'", '"', "(", ")"]:
        s = s.replace(c, "")
    return " ".join(s.split())


def em_f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    em = 1.0 if p == g else 0.0
    pset = set(p.split()); gset = set(g.split())
    if not pset or not gset:
        return em, 0.0
    common = pset & gset
    if not common:
        return em, 0.0
    prec = len(common) / len(pset); rec = len(common) / len(gset)
    return em, 2 * prec * rec / (prec + rec)


def build_prompt(setting, question, contexts=None):
    if setting == "no_context":
        return [
            {"role": "system", "content": "You are a precise QA assistant. Answer the question with a short factual span (one or two words / a single name / yes-no). If you do not know, answer 'unknown'. Do not explain."},
            {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
        ]
    elif setting == "shuffled_context":
        ctx_block = "\n\n".join(f"[Passage {i+1}]\n{c[:600]}" for i, c in enumerate(contexts))
        return [
            {"role": "system", "content": "You are a precise QA assistant. Answer the question using only information from the provided passages. Use a short factual span (one or two words / a single name / yes-no). If the passages do not contain the answer, output 'insufficient evidence'."},
            {"role": "user", "content": f"{ctx_block}\n\nQuestion: {question}\n\nAnswer:"},
        ]
    raise ValueError(setting)


def main():
    api_key = os.environ.get("APERTIS_API_KEY")
    base_url = os.environ.get("APERTIS_BASE_URL", "https://api.apertis.ai/v1")
    if not api_key:
        sys.exit("missing APERTIS_API_KEY")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load MHRAG eval (queries + STALI top-5 baseline EM/F1)
    eval_data = json.loads((DATA / "e6_results/eval.json").read_text())
    rows = eval_data["rows"]              # 500 entries with qid, gold, pred, em, f1
    retr = json.loads((DATA / "e6_results/retrieval.json").read_text())
    queries = {q["qid"]: q for q in retr["queries"]}

    # Load corpus for shuffle-context construction
    corpus = json.loads(MHRAG_CORPUS.read_text())
    bodies = [c.get("body", "") for c in corpus if c.get("body")]
    log(f"Loaded {len(rows)} eval rows, {len(corpus)} corpus docs ({len(bodies)} with body)")

    # Sample qids
    rng = random.Random(SEED)
    qids = [r["qid"] for r in rows]
    rng.shuffle(qids)
    sample_qids = set(qids[:N_SAMPLE])

    # Resume: load completed (qid, setting) tuples
    done = set()
    if RESULTS.exists():
        for ln in RESULTS.read_text().splitlines():
            try:
                d = json.loads(ln)
                done.add((d["qid"], d["setting"]))
            except Exception:
                pass
    log(f"Resuming: {len(done)} (qid,setting) entries already in {RESULTS.name}")

    settings = ["no_context", "shuffled_context"]
    qid_to_row = {r["qid"]: r for r in rows}
    fail = 0
    written = 0

    with open(RESULTS, "a") as out:
        for qid in qids[:N_SAMPLE]:
            if qid not in qid_to_row:
                continue
            row = qid_to_row[qid]
            question = queries[qid]["question"]
            gold = row["gold"]

            # Pre-pick shuffled passages so they're stable per-qid
            rng_q = random.Random(int(qid) * 1000003 + 7)
            gold_doc_set = set(queries[qid]["gold_doc_ids"])
            cand = [b for c, b in zip(corpus, bodies) if c.get("url", "") not in gold_doc_set]
            shuffle_ctx = rng_q.sample(cand, min(N_SHUFFLE, len(cand)))

            for setting in settings:
                if (qid, setting) in done:
                    continue
                try:
                    msgs = build_prompt(setting, question, shuffle_ctx if setting == "shuffled_context" else None)
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=msgs,
                        max_completion_tokens=MAX_TOKENS,
                        timeout=30,
                    )
                    pred = (resp.choices[0].message.content or "").strip()
                    em, f1 = em_f1(pred, gold)
                    rec = {"qid": qid, "setting": setting, "pred": pred, "gold": gold, "em": em, "f1": f1}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    written += 1
                    if written % 25 == 0:
                        log(f"wrote {written} results (last qid={qid} setting={setting} em={em:.0f} f1={f1:.2f})")
                except Exception as e:
                    fail += 1
                    log(f"FAIL qid={qid} setting={setting}: {e}")
                    time.sleep(2)
                    if fail > 30:
                        log("too many failures; aborting")
                        break

    # Summarise
    if not RESULTS.exists():
        log("no results to summarise")
        return
    by = {"no_context": [], "shuffled_context": [], "stali_top5_baseline": []}
    for ln in RESULTS.read_text().splitlines():
        try:
            d = json.loads(ln)
            by[d["setting"]].append((d["qid"], d["em"], d["f1"]))
        except Exception:
            continue
    # baseline from eval.json restricted to sample
    for r in rows:
        if r["qid"] in sample_qids:
            by["stali_top5_baseline"].append((r["qid"], r["em"], r["f1"]))

    summary = {"n_sample": N_SAMPLE, "model": MODEL}
    for k, v in by.items():
        if not v:
            summary[k] = None
            continue
        ems = [x[1] for x in v]
        f1s = [x[2] for x in v]
        summary[k] = {"n": len(v), "em": sum(ems)/len(ems), "f1": sum(f1s)/len(f1s)}

    # Paired delta no_context vs stali_top5 (only on overlap)
    base = {q: (e, f) for q, e, f in by["stali_top5_baseline"]}
    nc = {q: (e, f) for q, e, f in by["no_context"]}
    sh = {q: (e, f) for q, e, f in by["shuffled_context"]}
    overlap_nc = sorted(base.keys() & nc.keys())
    overlap_sh = sorted(base.keys() & sh.keys())
    if overlap_nc:
        delta_em = sum((base[q][0] - nc[q][0]) for q in overlap_nc) / len(overlap_nc)
        delta_f1 = sum((base[q][1] - nc[q][1]) for q in overlap_nc) / len(overlap_nc)
        summary["paired_delta_top5_minus_nocontext"] = {
            "n": len(overlap_nc), "em": delta_em, "f1": delta_f1,
            "interpretation": (
                "If delta is small (≲0.05), reader is largely answering from memory; "
                "if delta is large (≳0.20), retrieval is doing the work."
            ),
        }
    if overlap_sh:
        delta_em = sum((base[q][0] - sh[q][0]) for q in overlap_sh) / len(overlap_sh)
        delta_f1 = sum((base[q][1] - sh[q][1]) for q in overlap_sh) / len(overlap_sh)
        summary["paired_delta_top5_minus_shuffled"] = {
            "n": len(overlap_sh), "em": delta_em, "f1": delta_f1,
            "interpretation": "If delta is large, real retrieval beats random context (expected).",
        }

    SUMMARY.write_text(json.dumps(summary, indent=2))
    log(f"summary -> {SUMMARY}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
