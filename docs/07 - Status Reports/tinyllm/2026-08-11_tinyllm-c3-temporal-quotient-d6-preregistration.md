# TinyLLM observable C3 temporal-quotient d6 preregistration

**Status:** FROZEN BEFORE PRIMARY IMPLEMENTATION OR OUTCOMES

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `CONFIRMATORY / CONDITIONAL STAGE 1`

**Hypothesis:** `tinyllm-c3-temporal-quotient-training-v1`

**Depends on:** [training design](2026-08-11_tinyllm-c3-temporal-quotient-training-design.md); [passed Stage 0](../08%20-%20Analysis/2026-08-11_tinyllm-c3-temporal-quotient-stage0.md)

## Question and prediction

On a token-observable `C3` temporal task, does restricting the front-end
function class to exact invariants allow task-only training to produce a
task-useful and causally sufficient quotient under held-out nuisance composition
and outside-range extrapolation?

Compare three matched d6 arms:

```text
raw three-channel sequence
vs fixed analytic C3-invariant sequence
vs learned exact-C3-invariant sequence.
```

The analytic arm is the positive control. The learned arm is the scientific
test. The prediction is that both structured populations pass their complete
joint gates in at least four of five seeds. Raw behavior is measured as a
baseline and mechanistic comparator; it is not required to fail.

## Conditions and fixed controls

| Field | Frozen value |
| --- | --- |
| preset | TinyLLM d6: 6 blocks, 6 heads, width 384 |
| arms | `raw`, `analytic`, `learned_c3` |
| seeds | `7,17,29,41,53` |
| training examples | 4,096 observed examples / 2,048 paired latent histories |
| steps | 600 |
| batch size | 64, comprising 32 complete two-sheet pairs |
| optimizer | AdamW, learning rate `3e-4`, weight decay `.01` |
| gradient clip | global norm `1.0` |
| task | ordered sixteen-bin likelihood over `cos(3(theta_0+8v))` |
| loss | task cross-entropy only |
| sequence | BOS, eight temporal feature tokens, query |
| primary shifts | composition and extrapolation |
| required seeds | at least four of five per structured arm |

Within a seed, all arms receive byte-identical latent histories, calibration,
deck sheets, targets, pair indices, and minibatch indices. TinyLLM initializes
identically across arms. Analytic and learned sequence injections initialize
identically. The learned shared scalar map and complex mixer are the only extra
front-end parameters. No analytic-carrier regression, adversary, contrastive
term, representation loss, or branch penalty is permitted.

The d6 counts established during Stage 0 are `29,950,080` TinyLLM parameters,
`1,152` structured-injection parameters, and `184` additional learned-encoder
parameters. The primary runner MUST reproduce exact component counts.

## Data and split contract

### Training

Use the Stage-0 support definition. Phase is uniform; signed speed magnitude is
in `[.04,.12]`; deck is uniform. Each latent history activates at most one of
amplitude `[.7,1.8]`, offset `[-.4,.4]`, or drift `[-.06,.06]`; inactive values
are `A=1.2`, `o=0`, and `d=0`.

Each latent history is emitted under two distinct random deck elements. Both
members have exactly the same target and calibration. The training seed is the
model seed plus `1,001`; the minibatch seed is the model seed plus `6,013`.

### Probe and final evaluation

Final evaluation data MUST be disjoint from the no-training preflight, Stage-0
shakedown, training data, and probe fitting. Generate from the same pinned
observation law with these shared seeds:

| Role | Regime | Latent histories | Seed |
| --- | --- | ---: | ---: |
| probe train | composition | 2,048 | `231003` |
| probe validation | composition | 512 | `231021` |
| final composition | composition | 1,024 | `331003` |
| final extrapolation | extrapolation | 1,024 | `331021` |

For representation and causal measurements, emit all three exact deck sheets
for each latent history. Split by latent history before sheet expansion. A
fixed target-changing derangement is constructed between latent histories with
different target bins and is identical across arms and seeds.

Require zero quantizer saturation, exact token group laws, target invariance at
most `1e-12`, no overlap in latent fingerprints across splits, and matching
scientific hashes across arms. Any failure makes the campaign invalid.

## Primary natural-task gate

Measure the fixed sixteen-answer posterior on one deterministic observed sheet
per final latent history. A structured seed must meet every threshold in both
shifts:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation with target | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14/16` | `>=12/16` |

In addition, structured exact-bin accuracy may not be more than `.03` below
the matched raw arm for the same seed and shift. If raw itself misses an
absolute task floor, the relative comparison remains descriptive and does not
invalidate an otherwise passing structured cell; this prevents a broken raw
control from defining the positive threshold.

The Stage-0 64-step analytic values do not enter this gate. In particular, its
`.71176` extrapolation correlation and `.11328` exact accuracy are below the
registered 600-step requirement.

## Primary representation gate

At the front-end sequence and full-depth query residual, fit fresh estimators
using only the registered probe-train/validation splits.

The semantic estimator is a two-layer nonlinear regressor from the complete
temporal representation to the continuous target. The deck estimator is a
two-layer nonlinear three-class classifier conditioned on the exact analytic
invariant history and continuous target. Compare it to a condition-only null.

A structured seed must meet all three endpoints at both cuts and both shifts:

```text
target correlation                         >= .90
conditional deck balanced accuracy         <= .3834
conditional deck log-loss gain over null   <= .02
```

The learned front end must also satisfy maximum exact deck-action error
`<=1e-5` at initialization, after training, and at every measured residual cut.
The analytic arm must satisfy the same action gate. Probe chance means only
non-recoverability by these registered estimators on these splits.

Block-0 post-attention and post-MLP measurements are preregistered mechanistic
secondary endpoints. They cannot rescue a failed front-end/full-depth gate.

## Primary frozen causal gate

For every final latent history, compute the full residual sequence for all
three exact deck sheets. At the front end, block-0 post-attention, block-0
post-MLP, and full depth, compare:

1. natural continuation of each sheet;
2. continuation from the exact orbit-barycenter residual;
3. continuation from a target-changing deranged orbit barycenter; and
4. identity replay of the unmodified residual.

For each structured seed, orbit-barycenter patching must stay within all of
these tolerances in both shifts and at every cut:

```text
exact-bin accuracy loss                 <= .03
target cross-entropy increase           <= .10
mean triple-angle error increase        <= pi/16
```

Identity replay must change answer logits by at most `2e-6`. The
target-changing derangement may pass the three preservation tolerances in at
most one of five seeds per structured arm and shift. All patch values and
posteriors must be finite.

The structured front-end cut is the expected positive-control front. A later
first passing cut cannot rescue its failure.

## Raw-arm mechanism

Raw models receive the same natural-task evaluation, representation probes,
and causal interventions, but do not enter the structured population success
count. At every attention and MLP sublayer compute the exact defect

```text
chi_l = mean_j F_l(h_j) - F_l(mean_j h_j).
```

Classify whether the propagated barycenter and actual next barycenter pass the
frozen continuation. At the first synthesis front, decompose exact `C3`
characters and estimate quadratic then cubic contributions to the task-relevant
defect. The exact defect is primary mechanistic evidence; Taylor truncations are
secondary. No cubic-dominance claim is preregistered.

## Population decisions

A structured seed passes only when source/data validity, natural task,
front-end/full representation, exact action, all-cut causal preservation,
identity replay, and relative-utility applicability pass jointly on both
shifts. Marginal seeds cannot be combined.

| Population outcome | Classification | Decision |
| --- | --- | --- |
| analytic and learned each pass `>=4/5`; controls valid | `c3_d6_structured_quotient_supported` | report support; freeze a separate conditional d10 preregistration |
| analytic passes; learned fails | `c3_architectural_invariance_not_learned_useful` | freeze checkpoints; run no-training sensor-versus-continuation decomposition; no d10 |
| analytic fails | `c3_positive_control_task_failure` | no learned interpretation; repair only a demonstrated task/system defect |
| representation passes but task or causal gate fails | `c3_representation_without_causal_utility` | localize frozen interface; no loss tuning |
| action, data, replay, finiteness, source, or control contract fails | `invalid` | preserve all artifacts and repair systems under a new implementation hash |

No post-outcome threshold, extra step, seed, loss, width, endpoint map, or d10
result may rescue a failed d6 stage.

## Required artifacts

Store under a new DVC experiment root:

```text
data/experiments/tinyllm_c3_temporal_quotient/<dated-d6-run>/
  campaign_results.json
  runs/<arm>/seed_<seed>/
    result.json
    model.pt
    frontend.pt
    diagnostics.pt
```

Every run records source/config/scientific fingerprints, data and minibatch
hashes, initial/final component digests, complete training history, exact
parameter counts, task metrics, probe diagnostics, causal cells, control
outcomes, wall time, and artifact hashes. Resume requires a matching successful
fingerprint and every artifact hash. Failed cells and retries remain visible.

After execution, write a dated analysis report, add the material verdict to the
meta-hypothesis system with conservative confirmation status, DVC-push the
artifacts, create a lakeFS commit, and verify remote object hashes before making
the result part of the active frontier.

## Execution authorization

This document freezes the numeric d6 design. Primary execution remains
unauthorized until a separate campaign runner:

1. imports the unchanged Stage-0 runner by its sealed SHA;
2. pins this preregistration SHA;
3. passes focused unit tests for split integrity, gates, causal continuation,
   checkpoint round trip, and exact resume; and
4. passes a one-cell two-step CUDA lifecycle outside the evidence root.

D10 is not authorized by this document.
