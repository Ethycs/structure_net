# TinyLLM calibrated front-end causal closure

**Status:** VALID PREREGISTERED RESULT — FRONT-END CAUSAL QUOTIENT CONFIRMED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-system,
no-fit activation intervention  
**Hypothesis:** `tinyllm-calibrated-frontend-causal-closure-v1`  
**Schema:** `nal.tinyllm-calibrated-frontend-causal-closure.v1`  
**Preregistration:** [calibrated front-end causal-closure preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-calibrated-frontend-causal-closure-preregistration.md)

## Verdict

The calibrated analytic and learned-equivariant front ends already expose a
causally sufficient task quotient before block-0 attention. Replacing both
exact target-equivalent sheets with their activation barycenter preserves the
unchanged frozen TinyLLM task on composition and extrapolation in **5/5**
checkpoints for each structured arm. The result passes at every later cut as
well.

This is not a threshold-only pass. Across all 80 structured
checkpoint-by-shift-by-cut cells, orbit averaging improves exact-bin accuracy
by `1.27` to `17.48` percentage points, reduces circular error by `0.00499` to
`0.10009` radians, and reduces target cross-entropy by `0.00747` to `0.19119`
nats. By contrast, permuting whole barycenters among semantic fibers fails all
80 cells and reduces exact-bin accuracy by `27.44` to `71.48` points.

The locked classification is:

```text
frontend_causal_quotient_closed
```

No model, front end, embedding, task head, probe, or observer was trained or
fit.

## Primary endpoint

A seed passes a cut only if the orbit-average intervention simultaneously
stays within all three task-loss ceilings on both held-out shifts. An arm
requires four of five seeds. Both structured arms pass more strongly than
required.

| Frozen arm | Pre-block | Post-attention 0 | Post-MLP 0 | Full | Pre-block shuffled control |
| --- | ---: | ---: | ---: | ---: | ---: |
| analytic canonicalizer | **5/5** | **5/5** | **5/5** | **5/5** | **0/5** |
| learned equivariant encoder | **5/5** | **5/5** | **5/5** | **5/5** | **0/5** |

The shuffled control also fails `0/5` at every later cut in both arms. It
preserves the activation-barycenter marginal and exact paired-sheet identity,
but assigns each barycenter to a different semantic fiber. The contrast shows
that success depends on the correct task-orbit barycenter, not merely on
collapsing two states or injecting an average-looking residual.

## Task effect at the first causal cut

The table reports five-checkpoint medians at `pre_block`. Accuracy gain and
cross-entropy reduction are relative to the unchanged system, so positive
values favor exact orbit averaging.

| Front end | Shift | Baseline accuracy | Averaged accuracy | Accuracy gain | CE reduction | Posterior JS | Shuffled accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| analytic | composition | `0.7461` | `0.7930` | `+5.08` pp | `0.0160` | `0.00397` | `0.0723` |
| analytic | extrapolation | `0.6152` | `0.6953` | `+8.01` pp | `0.0627` | `0.01504` | `0.0723` |
| learned equivariant | composition | `0.7314` | `0.7500` | `+1.46` pp | `0.0091` | `0.00235` | `0.0742` |
| learned equivariant | extrapolation | `0.5254` | `0.5957` | `+10.74` pp | `0.0747` | `0.01791` | `0.0703` |

Orbit averaging therefore acts as task-aligned denoising inside the retained
cohort. The two sheets have the same declared target, but quantization and
front-end approximation leave small sheet-dependent activation errors.
Averaging cancels part of that error. This interpretation is supported by the
simultaneous improvements in accuracy, circular error, and cross-entropy; it
does not rely on a single permissive endpoint.

The raw-calibrated comparator also passes the relative intervention gate, but
is not primary evidence: its median unchanged accuracy is only `0.3945` on
composition and `0.1250` on extrapolation. Preserving or modestly improving an
already inadequate computation does not establish a useful quotient.

## Where the quotient is formed

For both block-0 sublayers, the propagated barycenter `F(mean h)` and actual
next barycenter `mean F(h)` pass in all five checkpoints and both shifts:

| Frozen arm | Attention already closed | MLP already closed |
| --- | ---: | ---: |
| analytic canonicalizer | **5/5** | **5/5** |
| learned equivariant encoder | **5/5** | **5/5** |

The correct four-regime label is `quotient_already_closed`, not `invariant_synthesis`.
Across structured cells, the largest relative Reynolds/Jensen residual defect
at either transition is `0.02485`, and the largest downstream posterior
Jensen--Shannon divergence between actual and propagated barycenters is
`0.0009651`. Exact affine closure is not claimed; task-relevant closure is.

This localizes the causal front before attention. The first transformer block
does not need branch-bearing cover variation to manufacture the invariant
used by the task. That result distinguishes the calibrated architecture from
the earlier unconstrained residual-penalty experiments, where internal probe
scores could improve without a stable extrapolating quotient.

## Sufficiency is not literal invariance

The natural residuals do not collapse to identical sheet states. Median paired
RMS difference relative to state RMS grows with depth:

| Front end | Shift | Pre-block | Full depth |
| --- | --- | ---: | ---: |
| analytic | composition | `0.0345` | `0.0815` |
| analytic | extrapolation | `0.0657` | `0.1525` |
| learned equivariant | composition | `0.0123` | `0.0396` |
| learned equivariant | extrapolation | `0.0370` | `0.1486` |

The causal conclusion is therefore precise:

> The exact Reynolds projection is a sufficient state for the frozen task,
> even though the unprojected activation continues to carry non-task fiber
> variation.

This resolves an ambiguity left by observational activation probes. A quotient
need not mean that every coordinate of the natural residual is invariant. It
means that an invariant projection exists which the unchanged continuation
can use without task loss. The branch-bearing complement may persist and even
grow while remaining causally unnecessary for this task.

## Relation to the representation result

The source calibrated-identifiability campaign had already shown that both
structured front ends retain the cosine base, suppress conditional branch
decodability, and solve the held-out task across five seeds. That established
representation and prediction endpoints. The present prospective intervention
adds the missing causal statement: the exact task-orbit barycenter can replace
the natural activation before the transformer and the existing answer decoder
still works.

The combined evidence now supports this chain within the retained d8/N3
cohort:

```text
observed calibration fixes the gauge
    -> analytic or equivariant front end exposes the absolute-cosine quotient
    -> exact fiber projection is causally sufficient before attention
    -> block-0 attention and MLP are already task-closed on that projection
    -> later residuals may retain fiber variation without needing it for task output
```

## Controls and integrity

- all 15 source systems were state-validated before the first intervention;
- the source campaign, 15-result manifest, task configuration, checkpoint
  paths, model states, front-end states, and held-out cohorts are locked;
- continuation from every unmodified cut reproduces its posterior exactly
  (`0.0` maximum absolute error);
- source task metrics replay within `2.10e-7`;
- every repeated orbit-average and shuffled state is pair-identical exactly
  (`0.0` maximum error);
- all source model and system hashes remain unchanged;
- every numeric result is finite;
- 15/15 requested cells complete with no retry, failure, or exclusion;
- all 80 structured semantic-shuffle cells fail;
- trained or fitted parameters: **zero**; and
- rerunning the completed command leaves both the primary and shakedown trees
  byte-identical.

| Item | SHA-256 / value |
| --- | --- |
| campaign | `1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14` |
| implementation | `5060b45674430351dabb6cd67af5e41a215f883d09b9702edd3d36b3d1d51260` |
| 15-result manifest | `baed34a16dca206536b2e9cd221fd9f7556f4c063f85ee857352522e770844f4` |
| exact-resume primary tree | `39ade9d4dcf0c9d0ee2dc2d4129232318a6e4f32b3af7580c939e6edb0a72e58` |
| exact-resume shakedown tree | `ee9f28f3c2f75ce199432d86505dc0044a324b10280774c385b845d7afecca4f` |
| source campaign | `80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501` |
| source-result manifest | `34bf25feb896abc9b9e06386b474fc6795c94566a23cd97a06795435fba64d68` |
| source preflight manifest | `80a2c89e1dc1e24f7f3bb1174867cb18d68493741db23716d86a20bf9dbc3b25` |
| composition cohort | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation cohort | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |
| meta-hypothesis record | `713aacb261dbb6cde0ea02824402e26b611e2931eaee290b35d0cb15481061f4` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA allocation | `1,257,699,328` bytes |
| analysis time | `330.19` seconds |
| DVC data root | `3c59c86bc4342cf9416b4ecc9c361fb7.dir` (`2,783` files; `40,110,611,863` bytes) |
| lakeFS commit | `99d98edb56ec6851fe6cb1ffa09f2154f3a46eeafd83e6e250a78608fa4c9292` |

The DVC cache and configured `lakefs` remote are synchronized. The lakeFS
branch has no uncommitted object diff. The root directory object, campaign
blob, and meta-hypothesis blob were read back by checksum at the immutable
commit above without exposing signed URLs.

## Program decision

Close the calibrated front-end causal-sufficiency branch for the retained
cohort. Do not add a residual penalty, adversary, observer, probe, topology
scan, or model retraining to strengthen this same claim. The answer is already
available from frozen activations and is stronger than the preregistered gate.

The next genuinely new question must change scope: a richer nuisance group,
an observation model whose orbit membership is not oracle-known, a different
task family, or an architecture-population replication. Within the present
synthetic `C2`, calibrated d8/N3 setting, more optimization would be redundant.

## Artifacts and reproduction

- primary campaign:
  `data/experiments/tinyllm_calibrated_frontend_causal_closure/20260810_d15_preregistered/campaign_results.json`
- per-system records and posterior diagnostics:
  `data/experiments/tinyllm_calibrated_frontend_causal_closure/20260810_d15_preregistered/runs/*/seed_*/`
- disjoint raw lifecycle shakedown:
  `data/experiments/tinyllm_calibrated_frontend_causal_closure/20260810_shakedown_raw_cuda/`
- runner and tests:
  `experiments/structure_net/tinyllm_calibrated_frontend_causal_closure.py`,
  `tests/structure_net/test_tinyllm_calibrated_frontend_causal_closure.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-causal-closure \
pixi run python -m \
  experiments.structure_net.tinyllm_calibrated_frontend_causal_closure \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_calibrated_frontend_causal_closure/20260810_d15_preregistered
```

## Scope boundary

Exact orbit averaging uses oracle synthetic fiber membership and the existing
answer decoder. The result covers five retained checkpoints per structured
front end, one calibrated `C2` generator, two declared held-out shifts, and the
d8/N3 TinyLLM task. It does not show that a deployable system can infer orbit
membership, that every residual coordinate is invariant, or that the result
generalizes to other groups, natural language, unrelated tasks, or an
architecture population.
