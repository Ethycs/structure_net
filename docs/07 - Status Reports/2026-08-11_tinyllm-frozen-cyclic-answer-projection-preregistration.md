# TinyLLM frozen cyclic answer-projection preregistration

**Status:** FROZEN BEFORE PROJECTED-HEAD OUTPUTS  
**Date:** 2026-08-11  
**Hypothesis ID:** `tinyllm-frozen-cyclic-answer-projection-v1`  
**Schema:** `nal.tinyllm-frozen-cyclic-answer-projection.v1`  
**Evidence role:** `registered_outcome_informed_artifact_only_answer_head_diagnostic`

## Evidence boundary

This is an outcome-informed, no-fit causal decomposition of one exceptional
checkpoint. The source scalar-domain study established that d10 learned seed
29 never gives answer bins `0`, `1`, `6`, or `15` a winning scalar region on a
`1/2048` grid through radius 8. It rejected encoder range as the explanation
but left the frozen scalar embedding, transformer continuation, and answer
rows bundled together.

The new study reuses stored posterior curves and targets. It does not load a
checkpoint or execute TinyLLM.

| Artifact | Frozen identity |
| --- | --- |
| scalar-domain result | `e0bbb6120272627de59acf639e886d434ab53ec7ed0fc11874d77864a4dd1312` |
| scalar-domain diagnostics | `95334b9475bd73111f20f07747ec716fe0bf3c5ff44c14c2a91b78c825218d69` |
| scalar-interface source cell | `16deb857f4449c66a22eac347c2c7e2b2cf23fbd5a8a1ad0daaf39660bf910f0` |
| scalar-interface diagnostics | `3bfbf496e1cb85d79548849b533b4139c0737e7079a53e4e5bb2f974de7f9476` |
| DVC source root | `daa4b990ea92500228d068b68fd0d127.dir` |
| lakeFS source commit | `626e234fd1f316f643b7a3c2149c065dd01c9bf8f05e103fe87514cddb11737b` |

## Question

Is the missing-bin defect causally removable by imposing the known cyclic
answer-group structure on the frozen task logits alone, without changing the
scalar coordinate or transformer continuation?

If a fixed cyclic projection restores a usable sixteen-bin chart, answer-row
geometry is sufficient to explain the hole and a typed readout is licensed for
prospective testing. If it does not restore coverage, readout structure alone
is insufficient and the scalar-conditioned continuation trajectory lacks the
required semantic winding.

## Registered intervention

For every stored posterior curve `p(s)` recover centered answer logits up to
their irrelevant additive gauge:

```text
z(s) = log p(s) - mean_bin(log p(s)).
```

Apply the fixed real `C16` Fourier projector along the answer-bin axis. For
order `m`, retain frequencies `0..m` in `rfft(z)` and zero the rest before
`irfft`. The nested orders are:

```text
m = 0, 1, 2, 4, 8.
```

- `m=0` is the uniform-output negative control.
- `m=1,2,4` are the strict analytic cyclic-head interventions.
- `m=8` is the exact identity/replay control for sixteen real logits.

This transformation is fixed by the declared answer-bin group and uses no
examples or labels to estimate parameters. It is equivalent on the registered
answer subspace to replacing the answer rows by a fixed linear mixture. It is
not a deployable tied-embedding model and is treated only as a causal head
counterfactual.

For specificity, cyclically shift each strict projected logit vector by four
answer bins. This preserves coverage and spectral order while changing the
task semantics.

## Endpoints

For each shift and order, measure on both the admissible radius-1 curve and the
full registered radius-8 curve:

1. reachable and missing answer bins;
2. target-bin reachability;
3. minimum-full-target-cross-entropy oracle accuracy and loss;
4. natural-scalar exact accuracy, circular error, and cross entropy;
5. the same natural metrics after the four-bin target-changing shift.

Natural scalar values come from the frozen scalar-interface diagnostics and
must occur exactly in the stored candidate set. No interpolation is allowed.

## Locked decision rule

A strict order has **deployable chart repair** only if, in both shifts:

- all sixteen bins are reachable on the radius-1 curve;
- the minimum-cross-entropy oracle passes the inherited task floor; and
- natural-scalar accuracy passes the inherited task floor.

The four-bin-shift control must fail at least one inherited floor in both
shifts. The smallest strict order satisfying the complete gate is reported.

| Outcome | Locked classification |
| --- | --- |
| a strict order passes the complete gate | `cyclic_answer_projection_repairs_deployable_chart` |
| all bins and oracle pass at radius 1, but natural accuracy fails | `cyclic_answer_projection_repairs_chart_not_natural_calibration` |
| all bins appear only by radius 8 | `cyclic_answer_projection_repairs_only_out_of_range` |
| at least one but not all source-missing bins appears | `partial_cyclic_answer_projection_repair` |
| no strict order recovers any source-missing bin by radius 8 | `cyclic_answer_projection_does_not_repair_coverage` |

If a result fits multiple rows, use the earliest row in the table. A strict
order that passes only because its shifted control also passes is invalid for
mechanistic interpretation and receives `specificity_failed`.

## Validity gates

1. all source hashes and schemas match;
2. stored curves and targets are finite and normalized;
3. the `m=8` posterior replay error is at most `2e-6` everywhere;
4. `m=0` is uniform within `2e-6`;
5. natural scalars match stored candidates exactly;
6. original radius-1 and radius-8 bin sets replay the source result;
7. all projections preserve sixteen output coordinates and finite values;
8. exact resume leaves output bytes unchanged.

Failure yields `invalid`. No population claim is licensed from this one
outcome-selected checkpoint.

## Artifact and stop contract

The runner must pass unit tests and a separate reduced-order lifecycle before
the primary artifact-only run. Primary root:

```text
data/experiments/tinyllm_frozen_cyclic_answer_projection/
  20260811_d10_learned_seed29_registered/
```

After interpretation, write the dated analysis, store and read back the
conservative meta-hypothesis record, refresh DVC, push and commit lakeFS, and
verify immutable root/result/meta checksums.

Stop after the declared orders. Do not change the shift, add Fourier modes,
fit answer rows, or retrain a readout in response to the outcome. Any trained
typed interface requires a new multi-seed prospective preregistration.
