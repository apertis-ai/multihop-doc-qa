#!/usr/bin/env python3
"""D32 second-reader sanity for the W6 reader-QA table.

Defuses Codex's `single-judge LLM` critique by re-running the W6 reader-QA
under a second frontier reader (claude-opus-4-7) on the SAME 200-query
MultiHop-RAG sample used by D32_contamination_check.py. We use no_context
prompting only (cheapest, addresses memorization concern directly):

  (A1) gpt-5.5 no_context   — already collected by D32_contamination_check
  (A2) claude-opus-4-7 no_context — collected here

Cross-LLM EM/F1 agreement is the rebuttal evidence:
  - If A1.em ≈ A2.em → single-judge concern is mostly cosmetic; the
    measured retrieval gap is robust to reader choice.
  - If A1.em - A2.em is large → reader-LLM choice matters and W6 numbers
    should be averaged or reported per-LLM.

Resumable: appends to ./second_reader_results.jsonl, skips done qids.
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from openai import OpenAI

N_SAMPLE = 200
SEED = 42
MODEL = "claude-opus-4-7"
MAX_TOKENS = 80

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DATA = RELEASE_ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent
RESULTS = OUT_DIR / "second_reader_results.jsonl"
SUMMARY = OUT_DIR / "second_reader_summary.json"
LOG = OUT_DIR / "second_reader_run.log"
PEER = OUT_DIR / "contamination_results.jsonl"  # gpt-5.5 ground truth


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


def main():
    api_key = os.environ.get("APERTIS_API_KEY")
    base_url = os.environ.get("APERTIS_BASE_URL", "https://api.apertis.ai/v1")
    if not api_key:
        sys.exit("missing APERTIS_API_KEY")
    client = OpenAI(api_key=api_key, base_url=base_url)

    eval_data = json.loads((DATA / "e6_results/eval.json").read_text())
    rows = eval_data["rows"]
    retr = json.loads((DATA / "e6_results/retrieval.json").read_text())
    queries = {q["qid"]: q for q in retr["queries"]}

    rng = random.Random(SEED)
    qids = [r["qid"] for r in rows]
    rng.shuffle(qids)
    sample_qids = qids[:N_SAMPLE]

    done = set()
    if RESULTS.exists():
        for ln in RESULTS.read_text().splitlines():
            try:
                done.add(json.loads(ln)["qid"])
            except Exception:
                pass
    log(f"Resuming: {len(done)}/{N_SAMPLE} done")

    qid_to_row = {r["qid"]: r for r in rows}
    fail = 0

    with open(RESULTS, "a") as out:
        for qid in sample_qids:
            if qid in done or qid not in qid_to_row:
                continue
            row = qid_to_row[qid]
            question = queries[qid]["question"]
            gold = row["gold"]

            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a precise QA assistant. Answer the question with a short factual span (one or two words / a single name / yes-no). If you do not know, answer 'unknown'. Do not explain."},
                        {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
                    ],
                    max_completion_tokens=MAX_TOKENS,
                    timeout=30,
                )
                pred = (resp.choices[0].message.content or "").strip()
                em, f1 = em_f1(pred, gold)
                out.write(json.dumps({"qid": qid, "setting": "no_context", "model": MODEL,
                                      "pred": pred, "gold": gold, "em": em, "f1": f1}) + "\n")
                out.flush()
                done.add(qid)
                if len(done) % 25 == 0:
                    log(f"done {len(done)}/{N_SAMPLE} (last qid={qid} em={em:.0f})")
            except Exception as e:
                fail += 1
                log(f"FAIL qid={qid}: {e}")
                time.sleep(2)
                if fail > 30:
                    log("too many failures; aborting")
                    break

    # Summary + cross-LLM agreement
    if not RESULTS.exists():
        return
    opus_recs = [json.loads(ln) for ln in RESULTS.read_text().splitlines() if ln.strip()]
    opus_em = sum(r["em"] for r in opus_recs) / max(len(opus_recs), 1)
    opus_f1 = sum(r["f1"] for r in opus_recs) / max(len(opus_recs), 1)

    # Compare to gpt-5.5 no_context from peer file
    gpt_lookup = {}
    if PEER.exists():
        for ln in PEER.read_text().splitlines():
            try:
                d = json.loads(ln)
                if d.get("setting") == "no_context":
                    gpt_lookup[d["qid"]] = d
            except Exception:
                continue
    overlap = [r for r in opus_recs if r["qid"] in gpt_lookup]
    summary = {
        "n_sample": N_SAMPLE,
        "model_a": "gpt-5.5", "model_b": MODEL,
        "model_b_no_context": {"n": len(opus_recs), "em": opus_em, "f1": opus_f1},
    }
    if overlap:
        gpt_em = sum(gpt_lookup[r["qid"]]["em"] for r in overlap) / len(overlap)
        gpt_f1 = sum(gpt_lookup[r["qid"]]["f1"] for r in overlap) / len(overlap)
        agree = sum(1 for r in overlap if r["em"] == gpt_lookup[r["qid"]]["em"]) / len(overlap)
        summary["model_a_no_context"] = {"n": len(overlap), "em": gpt_em, "f1": gpt_f1}
        summary["cross_llm_agreement_em"] = {
            "n": len(overlap),
            "exact_match_pair_agree_rate": agree,
            "delta_em_a_minus_b": gpt_em - opus_em,
            "delta_f1_a_minus_b": gpt_f1 - opus_f1,
            "interpretation": (
                "If both models agree on >=85% of binary EM and the delta is <0.05, "
                "single-judge-LLM concern for the W6 retrieval ranking is small."
            ),
        }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    log(f"summary -> {SUMMARY}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
