# TinyLLM frozen cyclic answer projection

**Status:** VALID REGISTERED OUTCOME-INFORMED NEGATIVE; PARTIAL HEAD
SENSITIVITY  
**Date:** 2026-08-11  
**Hypothesis ID:** `tinyllm-frozen-cyclic-answer-projection-v1`  
**Evidence role:** `registered_outcome_informed_artifact_only_answer_head_diagnostic`  
**Preregistration:**
[frozen cyclic answer projection](../07%20-%20Status%20Reports/2026-08-11_tinyllm-frozen-cyclic-answer-projection-preregistration.md)  
**Primary artifact:**
`data/experiments/tinyllm_frozen_cyclic_answer_projection/20260811_d10_learned_seed29_registered/result.json`

## Verdict

A fixed cyclic Fourier projection of the frozen answer logits does not produce
a complete or naturally useful answer chart. Orders 1, 2, and 4 recover some
source-missing bins, but each simultaneously removes other winning regions.
No strict order makes all sixteen bins reachable at radius 1 or radius 8, and
no strict order passes natural task adequacy in either held-out shift.

The locked classification is:

```text
partial_cyclic_answer_projection_repair
```

Simple cyclic answer-row geometry is therefore insufficient to repair this
checkpoint. The hole is sensitive to the answer head, but it is not merely an
irregular set of answer rows that becomes complete under a fixed low-order
`C16` projection. The scalar-conditioned logit trajectory and the answer chart
are co-adapted.

## Primary result

The strict projections retain answer-bin Fourier orders 1, 2, or 4. Order 8
is the exact identity control and order 0 is the uniform negative control.

| Order | Radius-1 winning bins | Source-missing bins recovered | New missing bins | Natural comp accuracy | Natural extrap accuracy |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 9/16 | 6 | 2, 3, 13, 14 | 0.2393 | 0.2178 |
| 2 | 12/16 | 6 | 14 | 0.4229 | 0.3291 |
| 4 | 11/16 | 1, 6 | 2, 3, 4 | 0.4141 | 0.3535 |
| 8 identity | 12/16 | none | none | 0.3691 | 0.3184 |

Bins `0` and `15` remain absent in every strict arm. Widening each projected
curve to radius 8 does not complete it. Order 1 gains one additional existing
bin at radius 8, but still reaches only 10/16; orders 2 and 4 have identical
radius-1 and radius-8 bin sets.

Every strict four-bin-shift control has exact accuracy `0.0` in both shifts.
The partial changes are therefore aligned with the declared answer order, not
a generic coverage-preserving transform that also solves the shifted task.

## Oracle versus natural use

Order 2 is the strongest structural partial result. Its minimum-full-target-
cross-entropy oracle passes both inherited task floors:

| Shift | Order-2 reachability | Oracle accuracy | Source floor | Natural accuracy |
| --- | ---: | ---: | ---: | ---: |
| composition | 0.8613 | 0.7637 | 0.7014 | 0.4229 |
| extrapolation | 0.8418 | 0.7305 | 0.5208 | 0.3291 |

This is not a repair. The oracle uses the hidden target to choose a scalar,
and the natural learned scalar remains badly miscalibrated. It shows only that
a smoother answer chart can improve the best available posterior match while
still omitting four winning bins.

Order 4 recovers two of the original missing rows but worsens the composition
oracle to `0.5801`; order 1 reaches `0.6445`. More retained harmonic capacity
is therefore not monotonic in task adequacy.

## What this settles

The sequence now separates three increasingly specific explanations:

1. **Encoder range:** rejected through radius 8; wider scalar values create no
   source-missing regions.
2. **Fixed cyclic answer-row geometry:** insufficient; analytic projections
   trade one set of holes for another and fail natural task utility.
3. **Co-adapted scalar/continuation/readout chart:** retained. The model has a
   usable quotient coordinate in representation and causal tests, but this
   checkpoint does not implement a complete portable coordinate-to-answer
   interface.

This strengthens the practical boundary on the theory. Symmetry typing of a
front end or answer head alone does not guarantee that their gauges agree with
the intervening computation. A future positive architecture must type the
whole interface—coordinate convention, continuation coupling, and answer
coverage—prospectively.

## Boundaries and next action

- This is one outcome-selected checkpoint and no population frequency is
  estimated.
- The Fourier transform is an analytic counterfactual over stored logits, not
  a deployable tied-embedding model.
- No checkpoint was loaded and no model, head, probe, observer, or map was fit.
- A successful target-using oracle is not an observable repair.

The registered stop rule closes further post-outcome head projections. Do not
add modes, tune mixtures, or fit answer rows to this checkpoint. The next
licensed engineering study is a prospective multi-seed typed-interface test
with matched frozen-backbone readout-only and jointly typed-interface arms.

## Controls and validity

All gates pass:

- source result and diagnostic hashes match both upstream studies;
- order-8 posterior replay error is `1.7881e-7`, below `2e-6`;
- order 0 is exactly uniform;
- every natural scalar is an exact member of its stored candidate set;
- source radius-1 and radius-8 bin sets replay exactly;
- all arrays and metrics are finite;
- checkpoint load count, fitted parameters, and trained parameters are zero;
- exact resume leaves the result and diagnostics byte-identical.

The first lifecycle is preserved as an invalid systems artifact because curve
subsampling removed narrow winning regions. A second lifecycle exposed a
reduced-order classifier assumption before writing a result. The corrected
complete-curve lifecycle passed and is systems evidence only. Neither failed
nor valid lifecycle output contributes to the primary classification.

## Execution and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_frozen_cyclic_answer_projection \
  --output data/experiments/tinyllm_frozen_cyclic_answer_projection/20260811_d10_learned_seed29_registered
```

The primary artifact-only computation completed in `3.789` seconds.

| Artifact | SHA-256 |
| --- | --- |
| result | `e76cbc3747211ececa209a16202c43b672f41a00c72ea473ed43202c97b374e6` |
| diagnostics | `fe47fb505aa3ee64542afa5a3bc7da051c3d9ba23c800f863b0d9dfb83e0e54e` |
| runner | `0fbb88ade2cef3a8439983519c766e56ddb1ba354ce6078ee7aab75c3c610fd0` |
| preregistration | `40a24744e956664b1c67950067021f45c54738f7d7dc0ae27e945e48f01ea05d` |
| combined implementation | `f8fccad4898322aadcf542b69fae7a5cefbfd7bf01b11ba718610ee70f01d8b5` |
| scientific fingerprint | `080693343e1d7c3c8fcd420cf4131a66aca93a4cedba67cf12cc1508309ff237` |
| source scalar-domain result | `e0bbb6120272627de59acf639e886d434ab53ec7ed0fc11874d77864a4dd1312` |
| source scalar-domain diagnostics | `95334b9475bd73111f20f07747ec716fe0bf3c5ff44c14c2a91b78c825218d69` |

## Data and evidence backup

The complete data tree, including the primary artifact-only result, preserved
lifecycle evidence, and Chroma-verified meta-hypothesis record, is tracked by
DVC root `84b6baa7f749551c62815d33130c35d6.dir`:
`49,029,055,228` logical bytes across 3,441 files. DVC reports the local cache
and configured `lakefs` remote in sync. The uploaded objects were sealed as
lakeFS commit
`7a3d981dfc09d56075903db96a07fa328aaeb857687bfcd91c2abeacf35c3713`,
and the branch has no uncommitted `structure-net` diff.

Direct lakeFS metadata readback at that immutable commit verified the DVC root
checksum `84b6baa7f749551c62815d33130c35d6`, result checksum
`1fbda3aee46b23ef6ea19a76b087a4e9`, and meta-hypothesis checksum
`b5167d575df914a830932b5072789be1`. Chroma readback verified the stable
hypothesis ID and its one direct experiment record.
