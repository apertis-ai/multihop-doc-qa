# Paper 2 reproducibility package

Artifacts for *A Late-Interaction Retriever Matches Graph-RAG on Multi-Hop Document Retrieval: A Matched-Reader Re-Evaluation* (EMNLP 2026 submission).

## Released artifacts

| Paper promise | Public location |
|---|---|
| Fixed DocHop-QA `n=500` IDs and gold annotations | [`code/experiment/data/eval/`](code/experiment/data/eval/) |
| Exact training triples | [Hugging Face dataset](https://huggingface.co/datasets/apertis-ai/DocHopQA-triples/tree/17290da76b0463374d47dddc36439b7a796f4e54) with manifests in [`code/experiment/data/train/`](code/experiment/data/train/) |
| Per-query retrieval and reader results | [`results/`](results/) and the public adapter repository, including `dochop_eval_corrected/` |
| Reader-contamination audit JSONLs and scripts | [`audits/`](audits/) |
| Evaluation and training implementation | [`code/experiment/`](code/experiment/) and [`code/harness/`](code/harness/) |
| Training configurations | [`code/experiment/configs/`](code/experiment/configs/) and [`configs/training_yamls/`](configs/training_yamls/) |
| Adapter checkpoints | [DocHop-QA + ablations](https://huggingface.co/theQuert/STALI-paper2-adapters/tree/a29d7ab280edb08ef24a9fa06a7161c08be3e3e7) and [cross-dataset + multi-seed](https://huggingface.co/apertis-ai/stali-paper2-adapters/tree/3a394b14426975076dd432cf1fef5b3374894b8f) |

The training dataset is pinned to Hub revision `17290da76b0463374d47dddc36439b7a796f4e54` (tag `paper2-v1`). The two adapter links are pinned to revisions `a29d7ab280edb08ef24a9fa06a7161c08be3e3e7` and `3a394b14426975076dd432cf1fef5b3374894b8f` (the latter tagged `paper2-v1.1`); their release cards are kept in [`model-cards/`](model-cards/). The latter also contains the retained S1/S2/S2-pure multi-seed adapters, the remaining historical DocHop-QA per-query outputs, and their corrected top-5 counterparts. Public access does not depend on the original RunPod volume.

## Verify the release

The verifier uses only the Python standard library. It checks every file against `SHA256SUMS`, validates the exact 500-query split, and rejects checkpoint binaries, logs, oversized files, and credential-like strings.

```bash
python3 verify_release.py
```

Expected gold digest:

```text
7a35f83e0a38ac86da125db8ad3705295186619588ef9827752712d65ca5470d  code/experiment/data/eval/gold_n500.json
```

## Rebuild `gold_n500.json`

Download the upstream DocHop-QA file at the pinned revision, then run the deterministic builder:

```bash
hf download anonymousaaai123/DocHopQA_Dataset \
  '[F] DocHopQA_Dataset.json' \
  --repo-type dataset \
  --revision bfd2c7c9cf77e27dc93c57fe9f7defb091b3edbe \
  --local-dir /tmp/dochopqa

python3 code/experiment/data/eval/build_n500.py \
  '/tmp/dochopqa/[F] DocHopQA_Dataset.json' \
  /tmp/gold_n500.json
```

The generated file must match the digest above byte-for-byte.

## Result layout

- `results/e1_nv_embed_v2_results/`: three 500-query zero-shot retrieval outputs.
- `results/e6_results/`: STALI MultiHop-RAG retrieval, reader evaluation, and aggregate statistics.
- `results/hipporag_results/`: paired DocHop-QA comparison data and fixed query-ID maps; the generated MultiHop-RAG corpus is intentionally omitted.
- `results/e4_hipporag2_calib_summary/`: triple-extraction calibration summaries.
- `audits/`: the five scripts, four per-query JSONLs, final combined summary, and compact inputs named in the paper appendix.

Run `python3 audits/D33_recompute_summary.py` to regenerate `audits/D33_combined_audit_summary.json` from the released per-query files. The LLM-generating audit scripts require `APERTIS_API_KEY`; `D32_contamination_check.py` additionally accepts `MHRAG_CORPUS_JSON` for a separately obtained MultiHop-RAG corpus.

## Training and evaluation

Download `dochop_qa_triples_t02.jsonl` into `code/experiment/data/train/`, then run the commands from the paper from inside `code/experiment/`:

```bash
python -m src.training.train_stali \
  --config configs/a2_random_only.yaml \
  --no-section-prefix \
  --triples data/train/dochop_qa_triples_t02.jsonl \
  --seed 42

python -m src.eval.run_n500 \
  --config configs/a2_random_only.yaml \
  --adapter runs/a2_random_only/adapter \
  --dochop-json /path/to/'[F] DocHopQA_Dataset.json' \
  --gold-path data/eval/gold_n500.json \
  --output runs/a2_random_only/eval_n500.json
```

See `pyproject.toml` and the YAML files for dependencies and full hyperparameters.
The A2 multi-seed runs reuse `a2_random_only.yaml` with `--seed 1`, `--seed 2`, or `--seed 3`; duplicate seed-only YAML files are unnecessary.

## Reproducibility corrections

The paper reports 6,310 training triples for the S2-pure and A2-small matched cohort. The preserved `dochop_qa_triples_t02.jsonl` contains 6,343 rows with at least one hard negative at its recorded threshold of 0.02, and preserved S2-pure run metadata also records `n_train_samples: 6343`. The historical trainer filtered only on a non-empty hard-negative list and did not apply the YAML threshold. This release therefore corrects both configs to the preserved 6,343-row, 0.02-threshold cohort and makes the trainer enforce those fields. No 6,310-row cohort can be reconstructed from the retained artifacts; at threshold 0.05 the released data contains 5,309 covered rows.

The paper's S2-pure-full threshold of 0.005 also selects 6,343 rows because the retained t02 file contains no positive Jaccard scores below 0.02. Its config is released, but no distinct S2-pure-full checkpoint or per-query result file was found in the local archive or either public adapter repository. The reported aggregate for that row is therefore not independently reproducible from the currently released artifacts.

The historical triples builder also used Python's process-salted `hash()` when selecting random negatives. The two published JSONL files and their SHA-256 digests are the canonical historical artifacts, but their random choices cannot be recreated byte-for-byte from the recorded seed alone. The released builder now uses SHA-256-derived per-query seeds so newly built files are deterministic across processes; those new random selections are not claimed to match the historical JSONL bytes.

The historical DocHop-QA evaluator represented an absent subsection as `/N/A`, while the gold builder omitted that suffix. This understated section recall for affected queries. The original outputs remain unchanged; corrected top-5 outputs for the 25 runs that retained `top5_sids` are in `dochop_eval_corrected/` in the public adapter repository. For example, A2 random-only changes from 0.597333 to 0.607733 section R@5 (20 of 500 queries). Top-10 section metrics cannot be corrected from the retained files because they do not contain top-10 IDs. The metric-only `e1_bge_m3`, `e1_qwen3_emb_4b`, and `e2_rerank_bge_v2m3` files cannot be corrected without rerunning retrieval.

The historical training and evaluation runners also ignored the YAML `max_query_length` and `max_doc_length` fields and truncated section text by characters before PyLate tokenization. Their retained metadata does not establish the claimed context lengths, so the S3 512/1024/2048/4096 outputs are historical records, not a valid context-length ablation. The released runners now pass token limits directly to PyLate and leave token truncation to the model; obtaining corrected context-ablation numbers requires rerunning those systems, and this release does not substitute guessed values.

## Provenance and scope

The fixed DocHop-QA `n=500` ID list predates Paper 2 and is shared with Paper 1; Paper 2 reused it for every paired comparison. The Paper 2 evaluation package and training triples were generated on 2026-04-15, with E6 results and reader audits added during the May 2026 Paper 2 revision cycle.

The historical code that originally selected the 500 IDs was not preserved. Consequently, this release treats `query_ids_500.json` as the authoritative split and reconstructs its annotations exactly from the public 11,379-record DocHop-QA source. It does not claim to recreate the paper draft's described “571 minus 71” selection procedure.

Large upstream corpora, checkpoints, private training-service state, API responses, and operational logs are not duplicated in Git. See [NOTICE.md](NOTICE.md) for licensing and attribution.
