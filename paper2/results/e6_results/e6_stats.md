# E6 MultiHop-RAG — Stats Summary

n_common_qids = 500; bonferroni α = 0.0056 (9 tests)

## Per-system summary

| system | para_r@5 | para_r@10 | para_hit@5 | doc_r@5 | EM | F1 |
|---|---|---|---|---|---|---|
| stali | 0.3907 | 0.4985 | 0.6920 | 0.5312 | 0.3440 | 0.4163 |
| bge_m3 | 0.3157 | 0.4313 | 0.5940 | 0.4647 | 0.3420 | 0.4029 |
| jina_v3 | 0.2837 | 0.3898 | 0.5500 | 0.4402 | 0.3540 | 0.4050 |
| qwen3_emb_4b | 0.4073 | 0.5310 | 0.6940 | 0.5375 | 0.3800 | 0.4444 |

The table above is the historical all-query output: 71 empty-gold `null_query` rows score 0, and `doc_r@5` was computed from the first five paragraph hits before document deduplication. Only ten paragraph hits were retained, leaving fewer than five unique documents for 117 queries. The available STALI ranking therefore bounds document R@5 at 0.5982–0.6335 over all 500 queries and 0.6972–0.7383 over the 429 answerable queries; see `doc_metric_correction.json`. Baseline retrieval files were not retained, so their document-metric bounds are unavailable.

## STALI vs baselines (paired bootstrap 10k, Bonferroni)

| metric | vs | Δ (stali − base) | 95% CI | p | d | class | Bonf. sig |
|---|---|---|---|---|---|---|---|
| para_r5 | bge_m3 | +0.0750 | [+0.0490, +0.1007] | 0.0000 | +0.25 | wins | **YES** |
| para_r5 | jina_v3 | +0.1070 | [+0.0795, +0.1348] | 0.0000 | +0.34 | wins | **YES** |
| para_r5 | qwen3_emb_4b | -0.0167 | [-0.0425, +0.0088] | 0.2056 | -0.06 | n.s. | no |
| em | bge_m3 | +0.0020 | [-0.0340, +0.0360] | 0.9674 | +0.01 | n.s. | no |
| em | jina_v3 | -0.0100 | [-0.0460, +0.0240] | 0.6096 | -0.03 | n.s. | no |
| em | qwen3_emb_4b | -0.0360 | [-0.0740, +0.0000] | 0.0572 | -0.09 | n.s. | no |
| f1 | bge_m3 | +0.0134 | [-0.0202, +0.0473] | 0.4360 | +0.03 | n.s. | no |
| f1 | jina_v3 | +0.0113 | [-0.0228, +0.0447] | 0.5054 | +0.03 | n.s. | no |
| f1 | qwen3_emb_4b | -0.0280 | [-0.0644, +0.0072] | 0.1264 | -0.07 | n.s. | no |
