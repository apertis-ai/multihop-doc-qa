---
license: mit
library_name: peft
base_model: lightonai/GTE-ModernColBERT-v1
tags:
- colbert
- lora
- information-retrieval
---

# STALI Paper 2 — cross-dataset adapters

Public multi-seed LoRA checkpoints for *A Late-Interaction Retriever Matches Graph-RAG on Multi-Hop Document Retrieval: A Matched-Reader Re-Evaluation*.

Canonical code and data release: https://github.com/apertis-ai/multihop-doc-qa/tree/main/paper2

## Contents

- `a2_seed{1,2,3}/`: DocHop-QA A2 runs.
- `dochop_multiseed/{s1,s2,s2_pure}_seed{1,2,3}/`: additional DocHop-QA trained-system adapters, run metadata, and evaluations.
- `dochop_eval_results/`: retained per-query outputs for the remaining DocHop-QA baselines and context-length runs.
- `dochop_eval_corrected/`: corrected top-5 section IDs and metrics for all 25 retained outputs that contain per-query top-5 IDs. Historical source files are preserved unchanged.
- `e6_multihoprag_seed{1,2,3,4,42}/`: MultiHop-RAG runs.
- `e6_hotpotqa_seed{1,2,3}/`: HotpotQA runs.

Each run contains a canonical `adapter/` directory and retained training checkpoints. Load the `adapter/` subdirectory with PEFT on top of `lightonai/GTE-ModernColBERT-v1`. Per-query evaluations associated with the runs are included where available; the GitHub release contains the compact consolidated result package and integrity manifest.

The adapters target ModernBERT attention modules `Wqkv` and `Wo` with LoRA rank 16 and alpha 32.

The preserved S2-pure metadata records 6,343 training samples. This corrects the paper text's 6,310 count; see the canonical release for the full cohort and historical RNG notes.

The corrected DocHop-QA files repair a legacy `/N/A` suffix mismatch and embed both historical and corrected section R@5 values plus the number of affected queries. Top-10 section metrics remain historical because top-10 IDs were not retained. Three metric-only outputs (`e1_bge_m3`, `e1_qwen3_emb_4b`, and `e2_rerank_bge_v2m3`) cannot be corrected without rerunning retrieval. The S3 context-labelled outputs are also retained as historical records: the old runners did not apply the YAML token limits, so they are not evidence for a 512–4096-token ablation.

## License

Adapter weights and Apertis AI-authored evaluation outputs are released under MIT.
