#!/usr/bin/env python3
"""D33 combined audit summary: extends D32_combined to include
HotpotQA contamination (1A) + 3-LLM cross-reader panel (1C)
+ 6-cell Llama-3.1-8B matched-reader table (1B).

Inputs (jsonl + json):
  contamination_results.jsonl                — gpt-5.5 MHRAG no/shuffled (D32 #6)
  second_reader_results.jsonl                — claude-opus-4-7 MHRAG no_context (D32 #5)
  contamination_hotpotqa_results.jsonl       — gpt-5.5 HotpotQA no/shuffled (D33 1A)
  second_reader_deepseek_results.jsonl       — deepseek-v4-pro MHRAG+HotpotQA no_context (D33 1C, partial)

Plus eval.json from runs/<sys>/ for Llama-8B 6 cells (D33 1B).

Output: D33_combined_audit_summary.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
INPUTS = ROOT / "inputs"

PAPER_W6 = {
    # gpt-5.5 reader, n=500
    "mhrag":  {"stali": 0.524, "hipporag2": 0.462, "linearrag": 0.352},
    "hotpot": {"stali": 0.568, "hipporag2": 0.550, "linearrag": 0.398},
}


def load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def avg(rows, k):
    if not rows:
        return None
    return sum(r[k] for r in rows) / len(rows)


# ---- 1A + 1C contamination & cross-LLM data ----
gpt_mh_cont    = load_jsonl(ROOT / "contamination_results.jsonl")
gpt_mh_nc      = {r["qid"]: r for r in gpt_mh_cont if r["setting"] == "no_context"}
gpt_mh_sh      = {r["qid"]: r for r in gpt_mh_cont if r["setting"] == "shuffled_context"}

opus_mh_nc_rows = load_jsonl(ROOT / "second_reader_results.jsonl")
opus_mh_nc      = {r["qid"]: r for r in opus_mh_nc_rows}

ds_all          = load_jsonl(ROOT / "second_reader_deepseek_results.jsonl")
ds_mh_nc        = {r["qid"]: r for r in ds_all if r["dataset"] == "mhrag"}
ds_hp_nc        = {r["qid"]: r for r in ds_all if r["dataset"] == "hotpot"}

gpt_hp_cont     = load_jsonl(ROOT / "contamination_hotpotqa_results.jsonl")
gpt_hp_nc       = {r["qid"]: r for r in gpt_hp_cont if r["setting"] == "no_context"}
gpt_hp_sh       = {r["qid"]: r for r in gpt_hp_cont if r["setting"] == "shuffled_context"}


def per_llm(rows):
    return {"n": len(rows),
            "em": avg(rows, "em"),
            "f1": avg(rows, "f1")}


summary = {
    "meta": {"n_sample_target": 200,
             "datasets": ["MultiHop-RAG", "HotpotQA-distractor-validation"],
             "readers": ["gpt-5.5", "claude-opus-4-7", "deepseek-v4-pro"]},

    "mhrag": {
        "no_context": {
            "gpt-5.5":          per_llm(list(gpt_mh_nc.values())),
            "claude-opus-4-7":  per_llm(list(opus_mh_nc.values())),
            "deepseek-v4-pro":  per_llm(list(ds_mh_nc.values())),
        },
        "shuffled_context_gpt-5.5": per_llm(list(gpt_mh_sh.values())),
        "paper_w6_gpt-5.5_top5":   PAPER_W6["mhrag"],
        "retrieval_lift_at_5_gpt-5.5":
            PAPER_W6["mhrag"]["stali"] - avg(list(gpt_mh_sh.values()), "em"),
    },

    "hotpotqa": {
        "no_context": {
            "gpt-5.5":         per_llm(list(gpt_hp_nc.values())),
            "deepseek-v4-pro": per_llm(list(ds_hp_nc.values())),
        },
        "shuffled_context_gpt-5.5": per_llm(list(gpt_hp_sh.values())),
        "paper_w6_gpt-5.5_top5":   PAPER_W6["hotpot"],
        "retrieval_lift_at_5_gpt-5.5":
            PAPER_W6["hotpot"]["stali"] - avg(list(gpt_hp_sh.values()), "em"),
    },
}


# ---- pairwise + triple agreement on no_context EM ----
def agreement(*lookups):
    keys = set.intersection(*[set(d.keys()) for d in lookups])
    if not keys:
        return None
    keys = sorted(keys)
    em_lists = [[d[q]["em"] for q in keys] for d in lookups]
    n = len(keys)
    # all agree (binary EM)
    all_agree = sum(1 for i in range(n)
                    if all(em_lists[0][i] == em_lists[j][i] for j in range(1, len(em_lists)))) / n
    return {"n_overlap": n, "all_agree_em": all_agree}


# pairwise
mh_pair_gpt_opus = agreement(gpt_mh_nc, opus_mh_nc)
mh_pair_gpt_ds   = agreement(gpt_mh_nc, ds_mh_nc)
mh_pair_opus_ds  = agreement(opus_mh_nc, ds_mh_nc)
mh_triple        = agreement(gpt_mh_nc, opus_mh_nc, ds_mh_nc)
hp_pair_gpt_ds   = agreement(gpt_hp_nc, ds_hp_nc)

summary["cross_llm_agreement"] = {
    "mhrag": {
        "gpt-5.5_vs_claude-opus":   mh_pair_gpt_opus,
        "gpt-5.5_vs_deepseek":      mh_pair_gpt_ds,
        "claude-opus_vs_deepseek":  mh_pair_opus_ds,
        "3-way_all_agree":          mh_triple,
    },
    "hotpotqa": {
        "gpt-5.5_vs_deepseek": hp_pair_gpt_ds,
    },
}


# ---- 1B 6-cell Llama-3.1-8B matched-reader table ----
def llama_em_f1(eval_path):
    if not eval_path.exists():
        return None
    d = json.loads(eval_path.read_text())
    s = d["summary"]
    return {"em": s["em"], "f1": s["f1"], "para_r5": s["para_r5"], "n": d.get("n_queries")}

# MHRAG: existing 2 cells in hipporag_llama8b_eval.json + new LinearRAG
ll_mh = json.loads((INPUTS / "hipporag_llama8b_eval.json").read_text())
mh_stali_ll  = {"em": ll_mh["stali_e6_llama8b"]["em"],     "f1": ll_mh["stali_e6_llama8b"]["f1"], "n": ll_mh["n_eval"]}
mh_hippo_ll  = {"em": ll_mh["hipporag_llama8b"]["em"],     "f1": ll_mh["hipporag_llama8b"]["f1"], "n": ll_mh["n_eval"]}
mh_linear_ll = llama_em_f1(INPUTS / "mhrag_linearrag_eval.json")

# HotpotQA: 3 new cells (cell 2 raw eval.json valid; cells 3,4 use eval_v2.json)
hp_stali_ll  = llama_em_f1(INPUTS / "hotpotqa_stali_eval.json")
hp_hippo_ll  = llama_em_f1(INPUTS / "hotpotqa_hipporag2_eval.json")
hp_linear_ll = llama_em_f1(INPUTS / "hotpotqa_linearrag_eval.json")

summary["llama_3_1_8b_matched_reader"] = {
    "mhrag":    {"stali": mh_stali_ll,  "hipporag2": mh_hippo_ll,  "linearrag": mh_linear_ll},
    "hotpotqa": {"stali": hp_stali_ll,  "hipporag2": hp_hippo_ll,  "linearrag": hp_linear_ll},
    "paper_w6_gpt-5.5": PAPER_W6,
    "ranking_under_llama_8b": {
        "mhrag":   "hipporag2 > stali > linearrag (MATCHED-READER FLIP vs gpt-5.5)",
        "hotpotqa": "stali > hipporag2 > linearrag (PRESERVED vs gpt-5.5)",
    },
    "matched_reader_paired_test_mhrag": {
        # paired delta + bootstrap CI from D32 hipporag_llama8b_eval.json
        "delta_em_hipporag_minus_stali": ll_mh["paired_delta"]["em"]["mean"],
        "ci": ll_mh["paired_delta"]["em"]["ci"],
        "p_value": ll_mh["paired_delta"]["em"]["p"],
        "cohen_d": ll_mh["paired_delta"]["em"]["cohen_d"],
    },
}


OUT = ROOT / "D33_combined_audit_summary.json"
OUT.write_text(json.dumps(summary, indent=2))
print(f"wrote {OUT}")
print(json.dumps(summary, indent=2))
