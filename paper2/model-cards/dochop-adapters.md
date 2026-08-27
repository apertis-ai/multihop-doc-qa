---
license: mit
library_name: peft
base_model: lightonai/GTE-ModernColBERT-v1
tags:
- colbert
- lora
- information-retrieval
---

# STALI Paper 2 — DocHop-QA adapters

LoRA adapters and per-query DocHop-QA evaluations for *A Late-Interaction Retriever Matches Graph-RAG on Multi-Hop Document Retrieval: A Matched-Reader Re-Evaluation*.

Canonical repository: https://github.com/apertis-ai/multihop-doc-qa/tree/main/paper2

## Contents

- `adapters/a2_random_only/`: primary A2 adapter, seed 42.
- `adapters/a2_seed{1,2,3}/`: additional A2 seeds.
- `adapters/s1_stali/`, `s2_plain_lora/`, `s2_pure_hard_neg/`, `a2_small_control/`: ablations.
- `eval_results/`: 500-query per-system evaluation outputs.
- `eval/gold_n500.json`: convenience copy of the GitHub gold file; SHA-256 `7a35f83e0a38ac86da125db8ad3705295186619588ef9827752712d65ca5470d`.

## Load an adapter

```python
from peft import PeftModel
from pylate.models import ColBERT

model = ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1")
model[0].auto_model = PeftModel.from_pretrained(
    model[0].auto_model,
    "theQuert/STALI-paper2-adapters",
    subfolder="adapters/a2_random_only",
)
```

The adapters target ModernBERT attention modules `Wqkv` and `Wo` with LoRA rank 16 and alpha 32. See the GitHub release for code, configurations, exact data provenance, and limitations.

## License

Adapter weights and authored evaluation outputs are MIT licensed. The convenience gold file is derived from CC BY 4.0 DocHop-QA data and retains that upstream license and attribution.
