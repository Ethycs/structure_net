# Llama 3.2 3B: partial lossless archive result

Date: 2026-10-10. Hypothesis `llama32-3b-lossless-component-compression-v1`. Profile: **EXPLORATORY**.
Status: closed partial after a runtime fault and user redirection to deployment.
Classification: **`original_archive_verified_factor_stage_failed`**. The planned full component census did not complete.

Aggregate `data/experiments/llama_lossless_compression/20261010_v1/campaign_results.json`; SHA-256 `ed89d143f6625f7e14946337a95d68360c7225b28d90a8f60fbe8e382851f045`.
[Original plan](../07%20-%20Status%20Reports/2026-10-10_llama32-lossless-compression-plan.md).

The complete original model compressed from **6.425 GB** of tensors to **4.588 GB**, a **28.60%** storage reduction. All 197 matrices and 57 normalization vectors decoded bit for bit after persistence. The archive includes model metadata and can be decoded without the original checkpoint. The tied embedding/output matrix is stored once.

This does **not** meet the clarified two-for-one deployment goal. Two independent archives alone occupy **9.176 GB**, **1.428×** one original model's raw parameter budget, before decoding and runtime memory. Dense inference still needs the original weights.

The selected original layout was adjacent-value XOR plus bit shuffle, then Zstandard level 3, in 1 MiB chunks. Encode/write/hash took **367.3 seconds**; decode/read/hash/direct verification took **395.5 seconds**. These are descriptive CPU costs, not inference measurements.

| Original component | Raw GB | Encoded payload GB |
| --- | ---: | ---: |
| attention | 1.409286 | 1.006748 |
| embedding | 0.788005 | 0.560574 |
| mlp | 4.227858 | 3.018477 |
| norms | 0.000350 | 0.000165 |

The total includes manifest bytes; family rows show payload only. Experiment logs, source snapshots and filesystem block rounding are excluded.

Of 199 planned archives, **1 completed, 1 failed, and 197 were not run**. The factor archive stopped on a non-contiguous SVD core. Its partial payload and `failure.json` remain unchanged. A later implementation correction canonicalizes tensor storage before byte encoding; logical values, shapes and dtypes are preserved, while physical strides are not an archive contract. A focused regression test covers this case. No full factors-plus-residuals total is claimed.

The screen in `screen.json` remains exploratory evidence only. The original lowest three FP16 bit planes were nearly empty; factor bit planes were much more evenly occupied. Full-model factor/residual savings cannot be inferred from those slices. The original archive result is unaffected by the later factor-stage failure.

Input hashes were verified against the previous exact-residual study. Producing source and plan copies are retained under `producer_source/`. Finalization verified the original archive manifest/payload hashes and frozen source fingerprint. The archive preserves original accuracy; the earlier 10/12 validation result and unmeasured 95% application target remain unchanged.

The user clarified that success means resident GPU memory and execution speed. The next study is [the exact deployment experiment](../07%20-%20Status%20Reports/2026-10-10_llama32-lossless-deployment-plan.md), with no archival-only result counted as deployment success.

Decode the verified archive with:

```bash
env PYTHONPATH=.:src data/experiments/moba_paper/environment/bin/python \
  -m experiments.structure_net.run_llama_lossless_compression \
  --decode data/experiments/llama_lossless_compression/20261010_v1/original --output /tmp/restored_llama.pt
```

Finalize this partial record with `experiments.neural_architecture_lab.finalize_llama_lossless_compression --campaign data/experiments/llama_lossless_compression/20261010_v1 --partial --register`. A new complete archival run would require a new output root.
