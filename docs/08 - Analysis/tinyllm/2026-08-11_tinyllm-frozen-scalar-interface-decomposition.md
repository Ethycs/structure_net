# TinyLLM frozen scalar-interface decomposition

**Status:** VALID REGISTERED POST-OUTCOME DIAGNOSTIC; PRIMARY UPSTREAM
HYPOTHESIS REJECTED  
**Date:** 2026-08-11  
**Evidence role:** `registered_post_outcome_frozen_scalar_causal_diagnostic`  
**Primary artifact:** `data/experiments/tinyllm_frozen_scalar_interface_decomposition/20260811_d6_d10_preregistered/campaign_results.json`

## Verdict

The architecture-family task failures are not uniformly caused by inaccurate
front-end estimation of the absolute cosine. Exact latent cosine repairs only
`1/10` source-failed checkpoints. A label-using search through the existing
one-dimensional scalar channel repairs `9/10` under the registered minimum-
target-cross-entropy selection rule. The remaining d10 learned seed 29 cannot
make four target bins win anywhere on the admissible scalar interval.

The locked classification is:

```text
continuation_capacity_failure_present
```

That label needs one qualification. The preregistered target-bin reachability
fraction itself exceeds the original exact-accuracy floor in all `10/10`
source-failed cells. Seed 29's registered oracle fails because the scalar that
best matches the full soft target posterior still has the wrong argmax too
often, not because too few examples have any correct-bin scalar available.
The result therefore establishes a localized posterior-shape/answer-row
restriction, not a universal inability of the scalar path to meet the old
argmax floor.

No parameter was fit or changed. All 20 source checkpoints were run, including
the ten source-passing resolution controls.

## Primary localization

| Source-failed stratum | Failed cells | Exact cosine repairs | Registered 1-D oracle repairs |
| --- | ---: | ---: | ---: |
| d6 learned equivariant | 4 | 0 | 4 |
| d10 analytic | 2 | 1 | 2 |
| d10 learned equivariant | 4 | 0 | 3 |
| **Total** | **10** | **1** | **9** |

The resulting cell classifications are:

| Classification | Count | Interpretation |
| --- | ---: | --- |
| `sensor_scalar_estimation_failure` | 1 | exact cosine is sufficient for the existing scalar embedding and continuation |
| `scalar_coordinate_or_boundary_failure` | 8 | exact cosine in the embedding's current coordinate fails, but another admissible scalar passes |
| `continuation_or_answer_row_failure` | 1 | the registered full-posterior scalar oracle still misses the source floor |
| source-passing positive control | 10 | source task gate already passed and the search recovered it |

The sole exact-cosine repair is d10 analytic seed 29. D10 analytic seed 17
improves from `0.6660` to `0.7109` on composition but remains below its
`0.7278` floor; its registered scalar oracle reaches `0.8574`.

For the learned arms, directly injecting numerical cosine is not a canonical
positive control for the learned scalar gauge. The scalar map and embedding
were trained jointly and may choose sign, scale, and offset conventions while
remaining structurally invariant. Consistent with that boundary, exact cosine
also fails both learned source-passing checkpoints, while all eight analytic
source-passing checkpoints retain exact-cosine adequacy. The causal result is
therefore that an admissible one-dimensional coordinate usually suffices—not
that learned systems should consume raw cosine in the analytic convention.

## The exceptional d10 learned checkpoint

D10 learned seed 29 is the only failed cell whose registered minimum-cross-
entropy oracle misses a source floor:

| Regime | Natural accuracy | Exact-cosine accuracy | Registered oracle accuracy | Target-bin reachability | Source floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| composition | 0.3691 | 0.0293 | 0.6699 | 0.8457 | 0.7014 |
| extrapolation | 0.3184 | 0.0234 | 0.7031 | 0.8574 | 0.5208 |

Across the complete candidate set, answer bins `0`, `1`, `6`, and `15` never
become the posterior argmax in either shift. They account for 158 composition
examples and 146 extrapolation examples. The other twelve answer bins are
reachable. Thus the frozen scalar-to-answer curve has genuine holes, even
though its maximum possible correct-bin fraction remains above the original
exact-accuracy floor.

This reconciles the two registered measurements:

- **argmax reachability:** enough examples have a correct-bin scalar to clear
  the old accuracy floor;
- **minimum soft-target cross-entropy selection:** the posterior-shape optimum
  often chooses a neighboring bin, leaving composition below the floor.

The locked campaign label remains unchanged. The descriptive reachability
measurement prevents interpreting that label as a stronger global
impossibility result.

## Failed-cell task table

| Preset/arm/seed | Natural comp | Exact comp | Oracle comp | Natural extrap | Exact extrap | Oracle extrap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d10 analytic 17 | 0.6660 | 0.7109 | 0.8574 | 0.5850 | 0.7168 | 0.8418 |
| d10 analytic 29 | 0.7324 | 0.8359 | 0.8828 | 0.6445 | 0.8555 | 0.8984 |
| d6 learned 7 | 0.7676 | 0.0273 | 0.9258 | 0.4453 | 0.0332 | 0.9082 |
| d6 learned 17 | 0.7402 | 0.0918 | 0.9473 | 0.4678 | 0.1172 | 0.9453 |
| d6 learned 41 | 0.8438 | 0.0430 | 0.9434 | 0.3838 | 0.0469 | 0.9531 |
| d6 learned 53 | 0.7139 | 0.0469 | 0.9004 | 0.4990 | 0.0293 | 0.9023 |
| d10 learned 7 | 0.7617 | 0.0332 | 0.8770 | 0.2881 | 0.0391 | 0.8555 |
| d10 learned 29 | 0.3691 | 0.0293 | 0.6699 | 0.3184 | 0.0234 | 0.7031 |
| d10 learned 41 | 0.7588 | 0.0156 | 0.9141 | 0.3037 | 0.0137 | 0.9238 |
| d10 learned 53 | 0.7705 | 0.0039 | 0.9023 | 0.2949 | 0.0059 | 0.8828 |

The apparently catastrophic learned exact-cosine column is evidence about
coordinate convention, not lost cosine information: the earlier frozen probes
already establish high recoverability of cosine from every learned front-end
output. An invariant scalar can still have an arbitrary checkpoint-local
gauge.

## Controls and validity

All preregistered controls pass:

- `20/20` cells are valid;
- direct natural forward versus explicit natural-scalar injection has maximum
  posterior error exactly `0.0`;
- maximum stored-task-metric replay error is `2.3842e-7`, below `2e-6`;
- all 20 model and complete-system state records are unchanged;
- all scalars and arrays are finite and in the declared range;
- the 1-D oracle recovers all `10/10` source-passing controls;
- negative cosine passes `0/10` source-failed cells;
- shuffled cosine passes `0/10` source-failed cells;
- the two cohort hashes exactly match the source campaign.

The dense search includes 4,097 fixed values plus natural and exact scalar
values for the one observed `(BOS=1, query=2)` context. This makes natural and
exact replay members of the searched set and protects the capacity result from
grid-resolution artifacts at those controls.

## What this pays for

The previous architecture-family result separated valid quotient geometry
from natural task utility. This experiment localizes that separation:

1. **The quotient coordinate exists.** Every structured checkpoint retained
   the earlier representation and causal-closure result.
2. **A scalar channel is usually enough.** Nine of ten failed systems pass
   under the registered one-dimensional oracle; target-bin reachability clears
   the source floor in all ten.
3. **The scalar gauge is not fixed by invariance.** Learned front ends and their
   embeddings co-adapt sign, scale, offset, and decision boundaries.
4. **One continuation has real holes.** D10 learned seed 29 omits four answer
   bins over the entire admissible scalar interval.

Architectural invariance therefore pays rent by constraining the information
and transformation law. It does not by itself fix the coordinate gauge or
guarantee a complete scalar-to-posterior chart.

## Shortest next diagnostic

Do not retrain yet. For d10 learned seed 29 only, extend the frozen scalar
domain outside the encoder's `tanh` image while retaining the same continuation
and answer rows. This is an out-of-class oracle, not a repair.

- If bins `0`, `1`, `6`, and `15` appear just outside `[-1,1]`, the defect is
  scalar saturation/range.
- If they remain absent over a declared wider interval, the defect lies in the
  continuation/answer-row curve rather than the encoder's bounded output.

That single-checkpoint scan is cheaper and more decisive than fitting an
affine map, retraining the equivariant encoder, or changing the answer head.

This follow-up is complete in the
[frozen scalar-domain extension](2026-08-11_tinyllm-frozen-scalar-domain-extension.md).
None of the four missing bins appears through radius 8 at resolution `1/2048`;
the locked result is `continuation_answer_curve_hole_persists_to_radius_8`.

## Execution and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_frozen_scalar_interface_decomposition \
  --device cuda:0 \
  --output data/experiments/tinyllm_frozen_scalar_interface_decomposition/20260811_d6_d10_preregistered
```

The 20 cells consumed `282.28` aggregate GPU-seconds, with maximum allocated
CUDA memory `0.4392` GB. The primary artifact tree is approximately 28 MB.
No model, front end, probe, observer, map, or parameter was trained or fit.

| Artifact | SHA-256 |
| --- | --- |
| campaign | `f71b85bc51f9694346fa23482bc63e1631e0918d55f5cae4f3a0d5ce09a47ba2` |
| 20-result manifest | `201f6bb780e5a92938df1596e61f9f75164bb0e3bfcd0554f6ad03ba74f177b4` |
| 20-diagnostics manifest | `bd9adc8e1103835af3c8264465fc0fe1723f06b769b8245ba1d914fe7de6145a` |
| campaign fingerprint | `f64e6c9e6440037c922e83cf4c81603aafef336cd0c8a3e938d787bf69a049e9` |
| implementation | `605798a7966691dfa10dbdd3af1fccee818850c141d4a797f78b11d57a816b4a` |
| runner | `0228836047e99c8af865e377e8cae329ddb56c70a4de22207087b8d10f2da128` |
| preregistration | `157d5b1c0a41f3d89f79530a4f3a92123b2c1c0da265a9cd87c5945f0882ea40` |

The reduced d6 analytic lifecycle root is excluded from science. It passed
CUDA execution, direct replay, scalar range, state immutability, semantic
controls, oracle resolution, and byte-identical exact resume before primary
launch. Re-running the complete primary command likewise left the campaign
SHA-256 byte-identical.

## Data and evidence backup

The complete data tree, including the 20-cell campaign and its Chroma-verified
meta-hypothesis record, is tracked by DVC root
`70142aadaf9ec5f5f45236f4f85e3472.dir`: `48,989,118,877` logical bytes
across 3,427 files. DVC reports the local cache and configured `lakefs` remote
in sync. The uploaded objects were sealed as lakeFS commit
`53462c21f889423c467bad0a003555496ec987b9ef9f42bab3839a10bef7b5c3`,
and the branch has no uncommitted `structure-net` diff.

Direct lakeFS metadata readback at that immutable commit verified the DVC root
checksum `70142aadaf9ec5f5f45236f4f85e3472`, campaign checksum
`3ab6a648091f78a7256e54af0f2e39e4`, and meta-hypothesis checksum
`68dae372fccad00b719ad822c1c4c36d`. Chroma readback verified the stable
hypothesis ID and all 20 direct experiment records.
