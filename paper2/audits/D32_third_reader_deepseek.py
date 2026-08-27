#!/usr/bin/env python3
"""D32 Phase 1C: deepseek-v4-pro 3rd reader cross-LLM sanity.

Adds deepseek-v4-pro no_context evaluations on:
  (1) The same MultiHop-RAG n=200 sample from D32_contamination_check.py
  (2) The new HotpotQA n=200 sample from D32_contamination_hotpotqa.py
      (must run after that script populates contamination_hotpotqa_results.jsonl)

Combined with gpt-5.5 (#6) and claude-opus-4-7 (#5), this gives a 3-LLM
panel for cross-judge agreement, raising the cross-LLM sanity from
n=2 (78.7% agreement) to n=3.

Resumable jsonl. Output ./second_reader_deepseek_results.jsonl.
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
MODEL = "deepseek-v4-pro"
MAX_TOKENS = 80

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DATA = RELEASE_ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent
RESULTS = OUT_DIR / "second_reader_deepseek_results.jsonl"
SUMMARY = OUT_DIR / "second_reader_deepseek_summary.json"
LOG = OUT_DIR / "second_reader_deepseek_run.log"
PEER_GPT = OUT_DIR / "contamination_results.jsonl"
PEER_OPUS = OUT_DIR / "second_reader_results.jsonl"
PEER_HOTPOT = OUT_DIR / "contamination_hotpotqa_results.jsonl"


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


def call(client, question, gold):
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
    return pred, em, f1


def main():
    api_key = os.environ.get("APERTIS_API_KEY")
    base_url = os.environ.get("APERTIS_BASE_URL", "https://api.apertis.ai/v1")
    if not api_key:
        sys.exit("missing APERTIS_API_KEY")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load MHRAG sample
    eval_data = json.loads((DATA / "e6_results/eval.json").read_text())
    rows = eval_data["rows"]
    retr = json.loads((DATA / "e6_results/retrieval.json").read_text())
    mhrag_q = {q["qid"]: q for q in retr["queries"]}
    mhrag_gold = {r["qid"]: r["gold"] for r in rows}
    rng = random.Random(SEED)
    qids = [r["qid"] for r in rows]
    rng.shuffle(qids)
    mhrag_sample = [(qid, mhrag_q[qid]["question"], mhrag_gold[qid]) for qid in qids[:N_SAMPLE]]
    log(f"MHRAG sample: {len(mhrag_sample)}")

    # Load HotpotQA sample IF the HotpotQA contamination script has populated
    hotpot_sample = []
    if PEER_HOTPOT.exists():
        seen = {}
        for ln in PEER_HOTPOT.read_text().splitlines():
            try:
                d = json.loads(ln)
                if d["setting"] == "no_context":
                    seen[d["qid"]] = d["gold"]
            except Exception:
                continue
        # Need question text - re-derive from HotpotQA dataset
        try:
            from datasets import load_dataset
            ds = load_dataset("hotpot_qa", "distractor", split="validation")
            id_to_q = {f"hotpot_{i}": ds[i]["question"] for i in range(len(ds))}
            for qid, gold in seen.items():
                if qid in id_to_q:
                    hotpot_sample.append((qid, id_to_q[qid], gold))
        except Exception as e:
            log(f"WARN: couldn't load HotpotQA dataset to re-derive questions: {e}")
    log(f"HotpotQA sample (from peer file): {len(hotpot_sample)}")

    combined = [("mhrag", q, g, qq) for q, qq, g in [(s[0], s[1], s[2]) for s in mhrag_sample]] + \
               [("hotpot", q, g, qq) for q, qq, g in [(s[0], s[1], s[2]) for s in hotpot_sample]]
    # rebuild as (dataset, qid, question, gold)
    combined = []
    for q, qq, g in mhrag_sample: combined.append(("mhrag", q, qq, g))
    for q, qq, g in hotpot_sample: combined.append(("hotpot", q, qq, g))

    done = set()
    if RESULTS.exists():
        for ln in RESULTS.read_text().splitlines():
            try:
                d = json.loads(ln)
                done.add(d["qid"])
            except Exception:
                pass
    log(f"Resuming: {len(done)} done. Total to do: {len(combined)}")

    fail = 0
    with open(RESULTS, "a") as out:
        for dataset, qid, question, gold in combined:
            if qid in done:
                continue
            try:
                pred, em, f1 = call(client, question, gold)
                out.write(json.dumps({"dataset": dataset, "qid": qid, "model": MODEL,
                                       "setting": "no_context", "pred": pred, "gold": gold,
                                       "em": em, "f1": f1}) + "\n")
                out.flush()
                done.add(qid)
                if len(done) % 25 == 0:
                    log(f"done {len(done)}/{len(combined)} (last qid={qid} em={em:.0f})")
            except Exception as e:
                fail += 1
                log(f"FAIL qid={qid}: {e}")
                time.sleep(2)
                if fail > 30:
                    log("too many failures; aborting")
                    break

    # 3-LLM agreement on no_context (MHRAG portion)
    rows = [json.loads(ln) for ln in RESULTS.read_text().splitlines() if ln.strip()]
    by_dataset = {"mhrag": [], "hotpot": []}
    for r in rows:
        by_dataset[r["dataset"]].append(r)

    summary = {"n_sample_per_dataset": N_SAMPLE, "model": MODEL}
    for ds_name, drows in by_dataset.items():
        if not drows:
            summary[ds_name] = None; continue
        em = sum(r["em"] for r in drows) / len(drows)
        f1 = sum(r["f1"] for r in drows) / len(drows)
        summary[ds_name] = {"n": len(drows), "em": em, "f1": f1}

    # cross-LLM 3-way agreement on MHRAG
    if PEER_GPT.exists() and PEER_OPUS.exists():
        gpt_lookup = {json.loads(ln)["qid"]: json.loads(ln) for ln in PEER_GPT.read_text().splitlines() if json.loads(ln).get("setting") == "no_context"}
        opus_lookup = {json.loads(ln)["qid"]: json.loads(ln) for ln in PEER_OPUS.read_text().splitlines()}
        ds_lookup = {r["qid"]: r for r in by_dataset["mhrag"]}
        overlap = sorted(gpt_lookup.keys() & opus_lookup.keys() & ds_lookup.keys())
        if overlap:
            triple_agree = sum(1 for q in overlap if (gpt_lookup[q]["em"] == opus_lookup[q]["em"] == ds_lookup[q]["em"])) / len(overlap)
            summary["cross_llm_3way_agreement_mhrag"] = {
                "n_overlap": len(overlap),
                "all_three_agree_em": triple_agree,
                "interpretation": "3-LLM panel agreement >=80% defuses single-judge concern."
            }

    SUMMARY.write_text(json.dumps(summary, indent=2))
    log(f"summary -> {SUMMARY}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
