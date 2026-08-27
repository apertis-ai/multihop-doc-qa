#!/usr/bin/env python3
"""D32 Phase 1A: HotpotQA contamination control.

Mirror of D32_contamination_check.py but for HotpotQA distractor dev set.
HotpotQA is a 2018 Wikipedia QA benchmark — expected to be heavily in
frontier-LLM training corpora, so contamination signal should be even
stronger than MHRAG.

Sample: n=200 random from HotpotQA distractor validation (n=7405 total).
Settings:
  (A) NO-CONTEXT  — just question
  (B) SHUFFLED-CONTEXT — question + 5 random Wikipedia paragraphs from
                         the HotpotQA dev pool (excluding gold)
Reader: gpt-5.5 (paper-default).  Resumable jsonl.
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
MODEL = "gpt-5.5"
MAX_TOKENS = 80
N_SHUFFLE = 5

OUT_DIR = Path(__file__).resolve().parent
RESULTS = OUT_DIR / "contamination_hotpotqa_results.jsonl"
SUMMARY = OUT_DIR / "contamination_hotpotqa_summary.json"
LOG = OUT_DIR / "contamination_hotpotqa_run.log"


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
    if setting == "shuffled_context":
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

    # Load HotpotQA distractor validation (already cached by HF datasets)
    from datasets import load_dataset
    log("Loading HotpotQA distractor validation set (HF cache)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    log(f"Loaded {len(ds)} HotpotQA validation queries")

    # Sample n=200 by SEED
    rng = random.Random(SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    sample_idx = indices[:N_SAMPLE]

    # Build sample dict
    sample = []
    for i in sample_idx:
        ex = ds[i]
        sample.append({
            "qid": f"hotpot_{i}",
            "question": ex["question"],
            "answer": ex["answer"],
        })

    # Pre-build a pool of random paragraphs for shuffled-context.
    # Take the FIRST paragraph of each context-document of every dev query.
    # This gives us ~7400 doc-paragraphs to sample from, drawn from the same
    # Wikipedia distribution — but no overlap with the gold supporting facts
    # (we exclude same-question docs at sampling time).
    log("Building shuffle-context paragraph pool from dev contexts...")
    pool = []  # list of (orig_idx, title, paragraph_text)
    for i, ex in enumerate(ds):
        ctx = ex.get("context", {})
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        for t, sents in zip(titles, sentences):
            text = " ".join(sents) if isinstance(sents, list) else str(sents)
            if text and len(text) > 50:
                pool.append((i, t, text))
        if i % 1000 == 0:
            log(f"  pool built from {i} examples ({len(pool)} para)")
    log(f"Pool size: {len(pool)} paragraphs")

    # Resume
    done = set()
    if RESULTS.exists():
        for ln in RESULTS.read_text().splitlines():
            try:
                d = json.loads(ln)
                done.add((d["qid"], d["setting"]))
            except Exception:
                pass
    log(f"Resuming: {len(done)}/{N_SAMPLE * 2} (qid,setting) entries already done")

    settings = ["no_context", "shuffled_context"]
    fail = 0
    written = 0

    with open(RESULTS, "a") as out:
        for s in sample:
            qid = s["qid"]
            question = s["question"]
            gold = s["answer"]
            # Pre-pick shuffled paragraphs (excluding same-source docs)
            rng_q = random.Random(int(qid.split("_")[1]) * 1000003 + 7)
            cand = [p for p in pool if p[0] != int(qid.split("_")[1])]
            shuffle_para = rng_q.sample(cand, min(N_SHUFFLE, len(cand)))
            shuffle_ctx = [p[2] for p in shuffle_para]

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
                    out.write(json.dumps({"qid": qid, "setting": setting, "model": MODEL,
                                           "pred": pred, "gold": gold, "em": em, "f1": f1}) + "\n")
                    out.flush()
                    written += 1
                    if written % 25 == 0:
                        log(f"wrote {written} results (qid={qid} setting={setting} em={em:.0f})")
                except Exception as e:
                    fail += 1
                    log(f"FAIL qid={qid} setting={setting}: {e}")
                    time.sleep(2)
                    if fail > 30:
                        log("too many failures; aborting")
                        break

    # Summarise
    by = {"no_context": [], "shuffled_context": []}
    for ln in RESULTS.read_text().splitlines():
        try:
            d = json.loads(ln)
            by[d["setting"]].append((d["qid"], d["em"], d["f1"]))
        except Exception:
            continue
    summary = {"n_sample": N_SAMPLE, "model": MODEL, "dataset": "hotpotqa-distractor-validation"}
    for k, v in by.items():
        if not v:
            summary[k] = None
            continue
        ems = [x[1] for x in v]; f1s = [x[2] for x in v]
        summary[k] = {"n": len(v), "em": sum(ems)/len(ems), "f1": sum(f1s)/len(f1s)}
    # paper W6 baseline reference
    summary["paper_w6_baseline_for_reference"] = {"em": 0.568, "f1": 0.691, "n": 500, "reader": "gpt-5.5"}
    if summary.get("no_context"):
        summary["contamination_signal_em"] = {
            "no_context_em": summary["no_context"]["em"],
            "stali_top5_em (paper, gpt-5.5)": 0.568,
            "delta_no_context_minus_stali": summary["no_context"]["em"] - 0.568,
            "shuffled_em": summary["shuffled_context"]["em"] if summary.get("shuffled_context") else None,
            "stali_lift_over_shuffled": (0.568 - summary["shuffled_context"]["em"]) if summary.get("shuffled_context") else None,
        }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    log(f"summary -> {SUMMARY}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
