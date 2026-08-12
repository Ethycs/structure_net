# TinyLLM observed cyclic-deck twirl preregistration

**Status:** PREREGISTERED — no outcomes inspected  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-checkpoint
causal replication  
**Hypothesis:** `tinyllm-observed-cyclic-deck-twirl-front-v1`

## Question and prediction

Can the exact `C2` and `C3` task fibers used in the d6 degree-ladder causal
front be generated from one observed calibrated input, without latent phase,
target labels, branch labels, generator-defined orbit membership, or another
nuisance/noise draw?

The directional prediction is that the observed cyclic Reynolds twirl will:

1. destroy the frozen task before the previously measured causal front;
2. preserve the task at full depth for both `k=2` and `k=3`; and
3. reproduce each checkpoint's earlier exact-orbit front within one recorded
   cut.

A matched half-task-turn orbit must remain valid for its own shifted target but
fail when scored against the source target. This prevents generic rotation,
averaging, or smoothing from satisfying the primary claim.

## Frozen sources and replication unit

Reuse without training or fitting all ten d6 analytic-carrier checkpoints from:

```text
data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered/
```

The conditions are `k=2` and `k=3`; seeds are `7,17,29,41,53`. One frozen
checkpoint is the replication unit. The prior generator-defined causal
reference is:

```text
data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered/
```

Before evaluation, validate the source schemas, campaign digests, task
configuration, condition and seed, checkpoint and front-end files, final model
state digest, and recorded per-run result digests. No checkpoint, front end,
probe, observer, action, or task head may be optimized.

## Observed action

For one decoded observation, subtract its observed planar offset and drift.
For declared angle `alpha`, rotate that corrected planar history by

```text
R(alpha) = [[cos(alpha), -sin(alpha)],
            [sin(alpha),  cos(alpha)]]
```

and restore the same observed offset and drift. Calibration, signed speed,
amplitude, and the third sensor channel are unchanged. The action is applied to
continuous decoded values and is not re-quantized.

The correct observed deck orbit is

```text
alpha_j = 2 pi j / k,       j = 0,...,k-1.
```

The matched semantic-control orbit is

```text
alpha'_j = pi/k + 2 pi j/k.
```

It has the same cardinality, norm, nuisance, calibration, and cyclic spacing as
the correct orbit, but its degree-`k` target is shifted by `pi`. The constructor
may read only the decoded sensor, calibration packet, and the declared group
element. Latent phase, target posterior/bin, branch, quotient phase, fiber ID,
and independently generated orbit rows are forbidden inputs.

## Cohorts and cuts

For each checkpoint and held-out regime, generate 256 source anchors using the
same nuisance definitions and seed streams as the prior causal study:

- composition: source seed `model_seed + 211`;
- extrapolation: source seed `model_seed + 307`.

Only branch-zero rows select the anchors; all other action rows are constructed
from those anchors. A separate fixed-nuisance circular grid supplies 192 total
map points per regime. Cohorts are regenerated deterministically and hashed.

Retain full residual sequences at:

- analytic carrier output;
- block-0 pre-attention, post-attention, and post-MLP;
- block-1 post-attention and post-MLP;
- block-2 post-attention and post-MLP; and
- full depth.

At every cut, average the `k` action-generated activation sequences within each
source orbit and patch the result into the frozen continuation. Evaluate the
unpatched correct orbit, correct twirl, unpatched control orbit, and control
twirl. No learned probe participates in a primary endpoint.

## Pre-outcome validity and integrity gates

All requested cells must satisfy these gates before their causal outcomes are
eligible for interpretation:

| Gate | Threshold |
| --- | ---: |
| applying the generator action `k` times recovers the sensor | maximum absolute error `<= 2e-6` |
| action leaves the calibration packet unchanged | maximum absolute error `<= 1e-7` |
| corrected planar norm is preserved | maximum absolute error `<= 1e-6` |
| analytic carrier obeys `c(g_alpha x)=R(alpha)c(x)` | maximum absolute error `<= 1e-6` |
| correct deck elements preserve the degree-`k` character `c^k` | maximum absolute error `<= 1e-6` |
| half-task-turn controls negate the degree-`k` character | maximum absolute error `<= 1e-6` |
| transformed planar observations remain within the declared decoded support | maximum absolute value `<= 2.0` |
| continuation from every unpatched cut replays its captured posterior | maximum absolute error `<= 2e-6` |
| source model/front-end states before and after analysis | exact digest equality |
| numeric artifacts | finite strict JSON/NPZ |

The runner and all outcome-relevant local dependencies must be content-hashed.
Completed reuse requires matching schema, scientific fingerprint,
implementation digest, terminal status, and required artifacts. Exact resume
must leave the complete artifact tree byte-identical.

If any validity gate fails, stop that cell, report the failure, and do not
inspect or promote its primary causal endpoints. Do not relax a gate in place.

## Primary endpoints

Use the prior exact-orbit classification unchanged. A twirl is **preserved** at
a cut when, on the fixed circular map and held-out task cohort:

- circular alignment is at least `0.90`;
- the map is sampling-resolved;
- winding differs from `k` by at most `0.10`; and
- exact-bin accuracy loss versus the corresponding unpatched action population
  is at most `0.03`.

It is **destroyed** when alignment is below `0.50`, a resolved winding differs
from `k`, or exact-bin accuracy loss exceeds `0.20`. Otherwise it is partial or
unresolved.

For each `k`, success requires jointly:

1. **mature observed twirl:** correct twirling at full depth is preserved on
   composition and extrapolation in at least four of five seeds;
2. **computational cover:** correct twirling at the analytic carrier and
   block-0 pre-attention is destroyed on both shifts in at least four of five
   seeds;
3. **front replication:** in at least four of five seeds, the first preserved
   correct-twirl cut lies within one position of that same seed and shift's
   locked generator-defined reference front, simultaneously on both shifts;
4. **control validity:** the control twirl is preserved for its own shifted
   target at full depth on both shifts in at least four of five seeds; and
5. **semantic specificity:** when that same control twirl is scored against the
   original source target, no more than one of five seeds is classified as
   preserved on both shifts.

The full hypothesis is supported only if all five population gates pass for
both `k=2` and `k=3`. Marginal pass counts from different seeds may not be
combined into a joint pass.

## Planned secondary measurements

Report, without allowing them to rescue a failed primary gate:

- posterior Jensen--Shannon dispersion across individual group elements;
- state relative RMS across action elements and cuts;
- observed-action distance from the separately quantized generator orbit;
- exact-bin accuracy and circular error for every intervention and shift;
- per-seed first-preserved cuts and distance from the locked reference front;
- correct/control twirl displacement norms; and
- CUDA allocation and analysis time.

## Outcome meanings and stop rules

| Outcome | Interpretation | Decision |
| --- | --- | --- |
| all gates pass for `C2` and `C3` | the d6 causal quotient fronts are realizable from one observed calibrated input and are not artifacts of oracle orbit membership | close the current finite-cyclic membership branch; do not retrain |
| `C2` passes and `C3` fails | one-observation construction does not extend cleanly to the richer tested group | localize the first failed `C3` validity or causal gate before any architecture change |
| mature twirls pass but front replication fails | deployable quotient sufficiency is real, but the reported depth front depends on quantization or cohort realization | retain mature closure and reject universal front localization |
| control is invalid for its own shifted target | the semantic control or continuation is broken | invalidate the campaign |
| correct mature twirl fails | prior generator-defined averaging does not transfer to an observed action | stop; do not train a descrambler from this result |
| an input/action validity gate fails | the declared action is not an eligible observed deck action | stop before causal interpretation |

No result licenses a representation penalty, post-hoc probe sweep, larger
writer, or TinyLLM retraining. A subsequent experiment must change group-action
observability, sensor symmetry, or the architecture population.

## Execution and artifacts

Run focused contract tests, then a one-checkpoint CUDA lifecycle shakedown marked
`systems_lifecycle_only_not_quality_evidence`. The primary campaign root is:

```text
data/experiments/tinyllm_observed_cyclic_deck_twirl/20260810_d6_preregistered/
```

Required artifacts are a strict per-checkpoint `result.json`, causal diagnostic
NPZ, aggregate `campaign_results.json`, result manifest, environment and source
digests, exact configuration, raw per-cut metrics, and an exact-resume tree
digest. A measured report and read-back-verified meta-hypothesis record are
required before promotion. Preserve the complete `data/` snapshot in DVC and
commit the remote objects to lakeFS.

## Boundaries

This study tests an analytic calibrated carrier, d6 three-block TinyLLMs, the
finite cyclic groups `C2` and `C3`, two synthetic held-out nuisance regimes, and
one exact continuous observed action. It does not test learned/raw front ends,
unknown calibration, stochastic group inference, non-cyclic groups,
anisotropic acquisition laws, natural language, real sensors, or a sampled
architecture family.
