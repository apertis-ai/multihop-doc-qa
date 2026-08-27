---
license: cc-by-4.0
task_categories:
- sentence-similarity
- text-retrieval
language:
- en
pretty_name: DocHop-QA STALI Training Triples
---

# DocHop-QA STALI training triples

Training triples used by Paper 2, derived from the public [DocHopQA_Dataset](https://huggingface.co/datasets/anonymousaaai123/DocHopQA_Dataset) at revision `bfd2c7c9cf77e27dc93c57fe9f7defb091b3edbe`, by Jiwon Park, Seohyun Pyeon, Jinwoo Kim, Rina Carines Cabal, Yihao Ding, and Soyeon Caren Han.

| File | Rows | Jaccard threshold | SHA-256 |
|---|---:|---:|---|
| `dochop_qa_triples.jsonl` | 10,879 | 0.05 | `e2dd00af3c053c620b8d17b08b5364e8b159e8dcf1c3c5f978215ccaee98607d` |
| `dochop_qa_triples_t02.jsonl` | 10,879 | 0.02 | `a0c0116f212ffac14397f43a1d402e90fde7042d5a35c6008487afb27bb55588` |

Both files exclude all 500 evaluation query IDs. Their generation parameters are recorded in the adjacent manifest files. The Paper 2 A2 command uses `dochop_qa_triples_t02.jsonl`.

The JSONL files are hosted at `https://huggingface.co/datasets/apertis-ai/DocHopQA-triples`, tagged `paper2-v1`; download them into this directory before training:

```bash
hf download apertis-ai/DocHopQA-triples --repo-type dataset \
  --revision paper2-v1 --local-dir .
```

Each row contains the query, positive sections, mined hard negatives, and seeded random negatives. Because source questions and passages are included, this derived dataset is released under CC BY 4.0 with attribution to the DocHop-QA authors.

These checksummed JSONL files are the canonical historical training artifacts. Their original random-negative builder used Python's process-salted `hash()`, so the recorded seed alone does not reproduce the same random choices byte-for-byte. The released builder now uses stable SHA-256-derived per-query seeds for deterministic future builds; see the canonical GitHub release's reproducibility-corrections section for the associated cohort correction.

Please cite Park et al., “DocHop-QA: Towards Multi-Hop Reasoning over Multimodal Document Collections,” arXiv:2508.15851 (2025).
