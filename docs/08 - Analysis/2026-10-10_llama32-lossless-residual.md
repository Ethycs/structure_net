# Llama 3.2 3B: lossless tensorization with exact residual correction

Date: 2026-10-10. Hypothesis `llama32-3b-lossless-residual-cotenancy-v1`. Status **completed**. Classification **`lossless_accuracy_assay_invalid`**.

Aggregate `data/experiments/llama_tensorized_cotenancy/20261010_lossless_v1/campaign_results.json`. SHA-256 `4be53fe363bdd889919260583776e89a56f614e606a6a0581efdf2741800331a`.

The exact correction preserved the source weights and the measured model behavior. It retained the original dense operator to compute the required output residual, so it did not eliminate that operator's storage or work. This is an exact execution control, not evidence that a smaller exact residual network was discovered.

All **197/197 matrices** reconstructed bit for bit after serialization and decoding; all 57 normalization vectors also matched. The codec needed **2,201 exceptional-entry repairs** in addition to FP32 residuals and FP16 factors. The measured prefill/cached-decode logits matched the original FP16 reference bit for bit: **True**. Generated token sequences were also checked pairwise.

## Accuracy

The task copies three exact final-decision sentences from notes containing superseded proposals. A fixed worked example demonstrates the format. These are synthetic fixtures, not application data.

| Split / model | Exact fragments | Supported lines | Complete summaries | Reference NLL |
| --- | ---: | ---: | ---: | ---: |
| validation / original | 10/12 | 10/12 | 3/4 | 0.035892 |
| validation / residual | 10/12 | 10/12 | 3/4 | 0.035892 |

The original FP16 implementation was previously checked against Hugging Face with FP16 numerical tolerances. Bitwise equality here is between that frozen FP16 implementation and its exact residual counterpart, not a claim of bitwise equality between different runtimes.

The sealed ten-example final test was not run because the original model failed the fixed validation competence gate. Exact correction preserves its validation answers, including errors; it cannot increase source-model accuracy while also preserving the same function. The 95% application-accuracy target is unmeasured.

## Complete-model speed and storage

Paired wall-time medians, five alternating samples after warmup, same physical model and GPU. Both execution alternatives use the same dense weights; this is a comparison of one model's two execution paths, not shared weights between independent replicas. The residual path also executes rank-215 factors, subtraction, addition and bitwise rounding repair.

| Workload | Original ms | Exact residual ms | Original / residual speed |
| --- | ---: | ---: | ---: |
| Prefill 128, decode 0 | 57.52 | 142.74 | 0.403× |
| Prefill 512, decode 0 | 154.60 | 265.07 | 0.583× |
| Prefill 128, decode 32 | 1799.78 | 4394.81 | 0.410× |

| Representation | Parameter GB / model | Relative to original FP16 |
| --- | ---: | ---: |
| exact_residual_network | 7.135 | 1.110× |
| factor_only | 0.710 | 0.111× |
| original_dense | 6.425 | 1.000× |
| stored_matrix_residual | 13.561 | 2.110× |

Device memory: **12.486 GB**, with **11.986 GB** free at the storage audit. The capacity table in the aggregate records exact 1/2/4-copy storage bounds. KV caches, activations and runtime allocations require additional space.

The stored matrix residual uses P = U·V in FP32, R = W − P, and exact repairs for any entries where FP16(P + R) differs from W. Decoding materializes the original dense matrix; doing so preserves its arithmetic but keeps dense inference cost. Directly adding factor and matrix-residual outputs is a separate floating-point reassociation and is not automatically bitwise lossless.

Primary timings come from `data/experiments/llama_tensorized_cotenancy/20261010_lossless_timing_v1/timing/result.json` (SHA-256 `2a19d32a3f44e8e85672476f69f739929e06a37c8b1d347cd91bfddff42ee475`). A preregistered timing amendment corrected a helper whose two-arm rotation/reversal accidentally repeated the same order. Original fixed-order results remain in the aggregate as diagnostics; the primary table uses explicitly alternating orders.

| Execution | Maximum measured allocator peak GB |
| --- | ---: |
| original | 6.602 |
| exact_residual | 7.384 |

## Independent-model packing diagnostic

Two physical first-query projections, 128 unequal input tokens each. Exact residual correction versus original packed dense output was bitwise equal: **True**. The direct factor-plus-stored-matrix-residual sum was bitwise equal: **False**. Its measured relative error was 5.28999e-05; it is not promoted as a lossless path.

| Projection execution | Wall median ms |
| --- | ---: |
| exact_packed | 0.6012 |
| exact_separate | 1.1867 |
| original_dense | 0.3221 |

## Scope and prior work

The user explicitly redirected the parent lossy screen to exact correction. That screen was stopped and remains incomplete; its completed artifacts were preserved. Its original Q4 quality assay failed its competence gate (8/12 fragments), while original FP16 scored 12/12. A prospectively declared worked-example correction supplied the fresh fixtures used here. No claim is made that the original invalid assay became valid.

The preserved parent runtime did fit four independent copies of the deployed Q4_K_M artifact on this GPU. Q4 uses quantized weights and a different runtime; it is not a lossless representation of the original FP16 model. Its exact identity, GPU offload and private contexts are recorded in `20261010_campaign_v2/gguf_benchmark_m4/`.

- The exact residual execution control retains and evaluates the full dense operator; it is not a compressed correction network.
- The stored FP32 matrix residual construction exceeds device capacity before activations; capacity is a byte lower bound, not an attempted OOM.
- Four validation and ten final synthetic extraction examples are not production summary accuracy.
- One RTX 3060 and one Llama checkpoint; these constructions do not rule out other lossless representations.

Residual construction, verification and checkpoint writing took 276.8 seconds. Full residual shards and the decoded checkpoint remain under the campaign's `weights/` directory; every shard and producing source is hashed.

## Reproduction

```bash
data/experiments/moba_paper/environment/bin/python -m experiments.structure_net.campaign_llama_cotenancy \
  --lossless --control data/experiments/llama_tensorized_cotenancy/20261010_control_v1 \
  --output data/experiments/llama_tensorized_cotenancy/NEW_LOSSLESS_ROOT
```

Then run `campaign_llama_cotenancy --lossless-timing <new-root>/weights/decoded_model.pt --output <new-timing-root> --control data/experiments/llama_tensorized_cotenancy/20261010_control_v1` through the same Python module invocation. Finalize with `experiments.neural_architecture_lab.finalize_llama_lossless --campaign <new-root> --timing <new-timing-root>/timing/result.json --register`.

The launcher selects GPU 0 before CUDA initialization and runs one producer at a time. Fixtures, original checkpoint, fixed factors, protocol and dependency hashes are recorded in each producer. No model training was performed.
