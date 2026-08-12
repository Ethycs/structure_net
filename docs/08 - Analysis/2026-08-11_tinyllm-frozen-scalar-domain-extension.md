# TinyLLM frozen scalar-domain extension

**Status:** VALID REGISTERED OUTCOME-INFORMED NEGATIVE; RANGE EXPLANATION
REJECTED  
**Date:** 2026-08-11  
**Hypothesis ID:** `tinyllm-frozen-scalar-domain-extension-v1`  
**Evidence role:** `registered_outcome_informed_frozen_scalar_domain_diagnostic`  
**Preregistration:**
[frozen scalar-domain extension](../07%20-%20Status%20Reports/2026-08-11_tinyllm-frozen-scalar-domain-extension-preregistration.md)  
**Primary artifact:**
`data/experiments/tinyllm_frozen_scalar_domain_extension/20260811_d10_learned_seed29_registered/result.json`

## Verdict

The four answer bins missing from d10 learned seed 29 are not recovered by
extending the frozen scalar input from the encoder's `tanh` image `[-1,1]` to
the registered resolved domain `[-8,8]`. Bins `0`, `1`, `6`, and `15` never
become posterior argmax at any of 32,769 fixed grid points, in either held-out
shift. The winning-bin set remains exactly the same twelve bins at radii 1, 2,
4, and 8.

The locked classification is:

```text
continuation_answer_curve_hole_persists_to_radius_8
```

This rejects scalar saturation/range as the explanation within the declared
domain. The defect lies in the frozen scalar-embedding/transformer/answer-row
curve: widening the scalar presented to that continuation does not create the
missing decision regions. The experiment does not distinguish which frozen
component of that composite curve is responsible.

## Primary result

The candidate set at radius 8 contains the registered 32,769 fixed points plus
the stored source scalar values, for 34,305 unique candidates per shift. Only
behavior at `|s| > 1` is new evidence.

| Radius | Reachable bins | Missing bins | Composition reachability | Extrapolation reachability |
| ---: | --- | --- | ---: | ---: |
| 1 | 12/16 | 0, 1, 6, 15 | 0.845703 | 0.857422 |
| 2 | 12/16 | 0, 1, 6, 15 | 0.845703 | 0.857422 |
| 4 | 12/16 | 0, 1, 6, 15 | 0.845703 | 0.857422 |
| 8 | 12/16 | 0, 1, 6, 15 | 0.845703 | 0.857422 |

The common count of source-missing bins discovered outside the encoder image
is therefore `0/4`. At `s=-8`, answer bin 14 still wins; at `s=+8`, bin 2
still wins. The two tails extend existing winning regions rather than opening
new ones.

## Secondary posterior-shape endpoint

Widening the domain slightly lowers the best achievable soft-target cross
entropy but does not repair the registered composition oracle:

| Shift | Radius | Min-CE selected accuracy | Min-CE cross entropy | Source floor | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| composition | 1 | 0.669922 | 1.496225 | 0.701445 | no |
| composition | 8 | 0.666016 | 1.490942 | 0.701445 | no |
| extrapolation | 1 | 0.703125 | 1.488401 | 0.520781 | yes |
| extrapolation | 8 | 0.703125 | 1.484494 | 0.520781 | yes |

The wider values improve posterior shape by only `0.00528` composition CE and
`0.00391` extrapolation CE. They do not change target-bin reachability, and
composition exact accuracy is slightly lower under the full-posterior
selection rule. This reinforces the earlier distinction between argmax
reachability and posterior calibration.

## What this settles

The previous scalar-interface diagnostic left two possibilities for its sole
exceptional checkpoint:

1. the learned encoder's bounded image excludes otherwise valid answer
   regions; or
2. the frozen downstream scalar-to-posterior map has no such regions.

The first explanation is not supported through radius 8 at resolution
`1/2048`. This is a real continuation-chart hole, not merely a natural scalar
that stops too early at `tanh` saturation.

The result sharpens the architecture-family finding. Architectural invariance
can preserve a valid quotient coordinate and still leave the learned task
interface incomplete. Information sufficiency, coordinate gauge, and coverage
of the answer chart are three separate engineering obligations.

## Boundaries

- This is one outcome-selected checkpoint and cannot estimate population
  frequency.
- Scalars outside `[-1,1]` are an oracle intervention that the retained encoder
  cannot produce.
- A finite scan cannot prove absence beyond radius 8 or between grid points.
- The intervention tests the composite scalar embedding, transformer
  continuation, and answer rows; it does not attribute the hole to one module.
- No model, front end, probe, observer, map, or parameter was trained or fit.

The registered stop rule closes further domain widening. A future engineering
study should instead make scalar gauge and full answer-bin coverage explicit
in a prospectively typed interface, with a frozen-backbone readout-only arm as
the cheapest positive test. It must be a new multi-seed preregistration, not a
post-outcome repair of this checkpoint.

The cheapest analytic readout falsifier is complete in the
[frozen cyclic answer projection](2026-08-11_tinyllm-frozen-cyclic-answer-projection.md).
Fixed `C16` Fourier answer projections recover some missing bins but remove
others; no strict order completes the chart or passes natural task adequacy.
This closes post-outcome analytic head projection before any trained readout.

## Controls and validity

All validity gates pass:

- source campaign, result, diagnostics, model, and front-end identities match;
- fresh posterior replay over every stored source candidate has maximum
  absolute error `2.0862e-7`, below `2e-6`;
- the radius-1 scan exactly reproduces the twelve source reachable bins under
  both shifts;
- composition and extrapolation dataset hashes match;
- model and complete-system state hashes are unchanged;
- all stored arrays and JSON values are finite;
- CUDA execution is valid with peak allocation `0.439137 GB`;
- exact resume reports the completed artifact and leaves both output hashes
  byte-identical.

The reduced 64-example/radius-2 CUDA lifecycle is preserved in a separate
root as systems evidence only. It passed the same execution and validity path
and is not pooled with the primary result.

## Execution and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_frozen_scalar_domain_extension \
  --device cuda:0 \
  --output data/experiments/tinyllm_frozen_scalar_domain_extension/20260811_d10_learned_seed29_registered
```

The primary scan completed in `24.976` seconds. Its artifact tree is
approximately 3.1 MB.

| Artifact | SHA-256 |
| --- | --- |
| result | `e0bbb6120272627de59acf639e886d434ab53ec7ed0fc11874d77864a4dd1312` |
| diagnostics | `95334b9475bd73111f20f07747ec716fe0bf3c5ff44c14c2a91b78c825218d69` |
| runner | `d252cff4522c2908b38446cb1d9cf32d83333801660951066753bd8ec49f856e` |
| preregistration | `1eec9eec7832124d5e4fd8d66632e2bede641cba1f4559f2ae677d909670a96e` |
| combined implementation | `494a6de53e14ebff9d285aa0d86af74ab4a6808ba5549c78b452dbf42e77bc4a` |
| scientific fingerprint | `d4b0c2ab9fe0dfd0c1b1b30911e429e6c9d4fdf38456ca44ca8b3553a11e4c55` |
| source scalar-interface campaign | `f71b85bc51f9694346fa23482bc63e1631e0918d55f5cae4f3a0d5ce09a47ba2` |
| source cell result | `16deb857f4449c66a22eac347c2c7e2b2cf23fbd5a8a1ad0daaf39660bf910f0` |
| composition dataset | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation dataset | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |

## Data and evidence backup

The complete data tree, including this primary scan, its lifecycle artifact,
and the Chroma-verified meta-hypothesis record, is tracked by DVC root
`daa4b990ea92500228d068b68fd0d127.dir`: `48,993,132,115` logical bytes
across 3,433 files. DVC reports the local cache and configured `lakefs` remote
in sync. The uploaded objects were sealed as lakeFS commit
`626e234fd1f316f643b7a3c2149c065dd01c9bf8f05e103fe87514cddb11737b`,
and the branch has no uncommitted `structure-net` diff.

Direct lakeFS metadata readback at that immutable commit verified the DVC root
checksum `daa4b990ea92500228d068b68fd0d127`, result checksum
`ebab3e416f1423ff2a3e423dea54134c`, and meta-hypothesis checksum
`2d63099f9725c98aa9564387036e9721`. Chroma readback verified the stable
hypothesis ID and its one direct experiment record.
