# Qwen3B native-feature coordinate audit

The large native-feature discrepancy does **not** indicate a discovered cache-normalization or token-indexing bug. Boundary 36 contains native hidden states before final RMSNorm, and the diagnostic compares the matching last valid input token. Native-projected coordinates are nevertheless substantially different from the accepted parent's actual compact coordinates. Substituting them would change the recovery target, rather than repair a demonstrated indexing error.

This is a read-only audit of the existing 64 development cases and saved tensors. No model forward, GPU operation, training, new test, or held-out access was performed. All conclusions remain exploratory; `confirmed=false`.

## Capture and coordinate semantics

- [jspace_full_capture.py](../../../experiments/structure_net/jspace_full_capture.py), line 215, registers a **forward pre-hook on the final norm** and assigns its input to `states[36]`. Lines 232–234 separately write logits from `output = model(...).last_hidden_state` and write the unnormalized `states[36]` to the boundary bank. Lines 218–219 select four logit positions ending at `len(input_ids)-1`; line 255 saves matching offsets, stops and positions.
- Installed Transformers 4.53.3 `models/qwen2/modeling_qwen2.py`, lines 198–202, computes RMSNorm; line 449 applies it before returning `last_hidden_state`. Thus the native logits and boundary 36 intentionally lie on opposite sides of final RMSNorm.
- [jspace_qwen_parallel_train.py](../../../experiments/structure_net/jspace_qwen_parallel_train.py), lines 135–155, reads `boundary[offset:stop]`, reconstructs valid lengths and positions, and applies the explicit `original_indices` mapping when loading original logits. All 576 mappings in this diagnostic are identity mappings. Development source rows 512–575 contain 8,985 valid tokens; the saved trace row lengths sum to exactly that count. The answer row is `native_stop-1`, or cumulative trace offset plus `length-1`, with no padding included.
- [jspace_3b_hidden_feature_diagnostic.py](../../../experiments/structure_net/jspace_3b_hidden_feature_diagnostic.py), line 104, projects native features as `h36 @ U35`; lines 114–116 reconstruct with the stored basis and positional anchor. [jspace_3b_hidden_feature_run.py](../../../experiments/structure_net/jspace_3b_hidden_feature_run.py), lines 281–291, keeps native projection, actual-exit roundtrip, native-projected roundtrip, and native reconstruction as separate comparisons.

The exit reconstruction is approximately `z @ U.T + a`, not an origin-centered orthogonal projection. The anchor is `mean - (mean @ U) @ U.T` ([jspace_qwen_affine_block.py](../../../experiments/structure_net/jspace_qwen_affine_block.py), line 79). Its mean comes from **layer 35 inputs on training cases**, not native boundary 36 outputs ([jspace_qwen_progressive_calibration.py](../../../experiments/structure_net/jspace_qwen_progressive_calibration.py), lines 232–234). The inherited layer 35 fit is `jspace_qwen_parallel/20261114_v2/groups/group_00/block_35`; its interface records that input-boundary provenance. Consequently, native affine-reconstruction relative error above 1 is not a violation of an orthogonal-projection error bound.

## Quantitative audit

All energies below are coordinate means of squared values at the answer position, then averaged equally across 64 cases. Parent energies were read directly from saved BF16 rows; native projected energy/cosine/scaling were inferred from the saved **per-case** error statistics, as detailed below.

| Quantity | Value |
|---|---:|
| Parent compact answer energy | 156.138839 |
| Native projected compact answer energy, inferred | 15.048847 |
| Pooled parent/native compact RMS ratio | 3.221097 |
| Mean per-case cosine, inferred | 0.352098 |
| Median per-case cosine, inferred | 0.355141 |
| Best single native→parent scalar, inferred | 1.120402 |
| Residual RMS after that scaling, divided by parent RMS | 0.937557 |
| Residual with separate nonnegative scalar per case / parent RMS | 0.936622 |
| Parent reconstructed native-width answer energy | 119.133921 |
| Original native boundary36 answer energy | 14.606940 |
| Fixed exit-anchor answer energy | 41.076080 |

The originally reported answer relative RMSE is 3.046622 for parent coordinates versus native projection, and 1.786408 for reconstructed native projection versus original native features. These use the diagnostic's equal-case normalization. They are **not** the pooled residual ratios in the table. Native projected roundtrip relative RMSE is only 0.003365; the parent's own exit roundtrip is 0.002466. Rounding is therefore much smaller than the native-target discrepancy.

RMSNorm largely removes positive uniform hidden-state scaling, so a large unnormalized Euclidean error need not imply an equivalent logit error. However, amplitude alone does not account for this result: optimal scalar correction still leaves about 93.8% of parent feature RMS unexplained. The parent's learned norm gains differ from native in only 2 of 2,048 coordinates, with maximum absolute change 0.0029296875; their mean ratio is 0.999960. The large hidden-state amplitude difference is not a large learned global norm-gain change. This does not establish which remaining directions matter for the task.

## Derivation and reproduction

For case i, let `p_i = mean(parent_z²)`, `e_i = answer_mse`, and `r_i = answer_relative_mse` from `parent_packed/versus_native_projected_final/per_case`. The diagnostic computes `r_i = e_i / mean(native_z²)` in FP32. Therefore:

```text
n_i ≈ e_i / r_i
d_i = mean(parent_z * native_z) ≈ (p_i + n_i - e_i) / 2
cos_i ≈ d_i / sqrt(p_i*n_i)
c_global ≈ sum(d_i) / sum(n_i)
residual_fraction ≈ sqrt(sum(p_i - 2*c_global*d_i + c_global²*n_i) / sum(p_i))
```

These are approximate reconstructions from rounded **per-case FP32 MSE and ratios**, not direct dot products with a newly recomputed native projection. Using only the aggregate 3.046622 error would not recover these quantities. The following CPU-only excerpt reproduces them from existing files; it performs no basis matrix multiplication or model execution:

```python
import json
from pathlib import Path
import numpy as np

root = Path("data/experiments/jspace_3b_hidden_feature/20261115_v1/run")
result = json.loads((root / "result.json").read_text())
rows = json.loads((root / "parent_development/rows.json").read_text())["rows"]
bank = np.load(root / "parent_development/coordinates.npy", mmap_mode="r")
errors = result["feature_metrics"]["parent_packed/versus_native_projected_final"]["per_case"]
values, offset = [], 0
for row, error in zip(rows, errors):
    assert row["identity"] == error["identity"]
    offset += row["length"]
    bits = np.asarray(bank[offset - 1, -1], dtype=np.uint16).astype(np.uint32)
    z = (bits << 16).view(np.float32).astype(np.float64)
    p = np.mean(z*z)
    n = error["answer_mse"] / error["answer_relative_mse"]
    dot = (p + n - error["answer_mse"]) / 2
    values.append((p, n, dot))
p, n, dot = np.asarray(values).T
c = dot.sum() / n.sum()
print("RMS ratio", np.sqrt(p.sum()/n.sum()))
print("mean cosine", np.mean(dot/np.sqrt(p*n)))
print("global scale", c)
print("residual fraction", np.sqrt(np.maximum(p-2*c*dot+c*c*n, 0).sum()/p.sum()))
```

Anchor energy was read from parent checkpoint `stack['blocks.11.anchor']` at the 64 recorded `answer_position` rows. Native-width energies use `parent_development/output.npy` and original `boundary_36.npy` at the row mappings above. Norm-gain comparison reads `checkpoint['norm']['weight']` and native safetensor `model.norm.weight`; no native model is instantiated.

## Evidence identifiers

Diagnostic root: `data/experiments/jspace_3b_hidden_feature/20261115_v1`.
Fingerprint: `51225002a2e9993131b90d9dae0e525a271ccfef04ff1188e9c8dbc434786512`.
The frozen plan contains complete source hashes; its producer archive retains the exact source versions used for the line references above. Data hashes below are the existing authenticated manifest/plan digests; this audit reread small metadata and sampled tensor rows, rather than rehashing all large banks.

| File, relative to diagnostic root unless specified | SHA256 |
|---|---|
| `plan.json` | `3edf52ba5e669ec3cdc12bc93eb8070434e43e5cc929c468ec524b0df060d887` |
| `run/manifest.json` | `e7781dad96944f93640270bae25fa37000114838f9a7e46f9caf7e1e21cee135` |
| `run/result.json` | `d203042c5c7b0101a821ba6d38dadb63fe492eae3186d4d37ba4f9f21a1b904f` |
| `run/parent_development/rows.json` | `bd52b38742c07e3672335cacf975e7a9c68e019199f627d1a2c7e5edd8a4a72d` |
| `run/parent_development/coordinates.npy` | `4a0b6d14a2618af6601981927c98dac3ff2ef8a49bf751f2b92c2bd5de00f6d9` |
| `run/parent_development/output.npy` | `291ba647cf32f2f1fa599b435c91cddd29fd0fd9101aea9b424fbde2b17dd52b` |
| `jspace_full_compact/20261113_capture_v1/capture/items.json` under `data/experiments` | `b89252aab3291f42b45ab1b9ff962c6e41c6ac6f74fbe8925240e91329c66351` |
| Same native capture: `boundary_36.npy` | `3289fd3e4a066226750a36cefbf6b72f8ad8ec6a8f617f1afce8642be9ae820f` |
| Accepted parent: `jspace_3b_staged_campaign/20261115_v1/stages/block_24/train/selected.pt` under `data/experiments` | `db53db32597ebb4947e38f7324e31e6eb9c7686f491a9e8b9745b0c43bca8292` |

Installed Transformers Qwen2 source SHA256: `35847fd8ee493d79fefd799d1bb4e8b634633ead0d22df338ed70cdb5329b834` at `.pixi/envs/ncm/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py`.

The native norm reference is Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, tensor `model.norm.weight` in `model-00002-of-00002.safetensors`. Its previously authenticated shard SHA256 is `a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d`; the campaign plan retains the local snapshot path and complete model inventory.

The parent packed inference and input-autograd coordinate/logit controls were exact. Arm A can therefore use the parent's actual captured features without relying on native-feature equivalence. A native-feature arm remains a separate teacher/representation intervention and needs its own prospective comparison.
