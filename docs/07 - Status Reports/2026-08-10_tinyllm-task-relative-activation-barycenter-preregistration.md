# TinyLLM task-relative activation barycenter preregistration

**Status:** PREREGISTERED — NO ACTIVATION-PATCH OUTCOME GENERATED OR INSPECTED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`,
frozen-checkpoint causal bridge  
**Hypothesis:** `tinyllm-task-relative-activation-barycenter-v1`

## Prior evidence and unresolved question

The matched task-quotient experiment trained phase-circle and cosine-interval
TinyLLMs from identical initializations, inputs, minibatches, optimizer, and
budget; only the supervised target changed. Its layer atlas found that:

- conditional cosine-branch probes fell to chance after block 1's MLP in both
  retained model classes;
- the complete observational cosine-quotient tuple first passed at d6 block 5
  post-attention and d8 block 3 post-attention; and
- the phase carrier followed a different layerwise geometry.

Those results are observational. They do not establish that the frozen cosine
continuation can actually use a phase-fiber barycenter, or that the same
intervention selectively destroys the phase task.

The new question is:

> Does the supervised task determine when an exact task-fiber activation
> barycenter becomes causally sufficient for the frozen continuation?

The directional prediction is that the cosine model acquires a mature causal
front near its previously measured complete quotient geometry, whereas the
same opposite-phase barycenter remains insufficient for the matched phase
model.

## Frozen sources

Reuse the four retained trained seed-7 checkpoints without fitting:

| Preset | Task | Checkpoint SHA-256 |
| --- | --- | --- |
| d6 | cosine interval | `5343fa57d88b356c26138b763d23e8defd0b1860239d6aabca718d09516181e4` |
| d6 | phase circle | `5c84701e0022f03af90204b43620b227faee65beedf768187c0546c946d0f854` |
| d8 | cosine interval | `8170856da5e6b1f8b7a7f1c2c2121ffd4c53edaaf2ea5aabe01fa14d2f063f3f` |
| d8 | phase circle | `3e2053273075d2404a05d9097823814d39b8682875c0e7f4052df7d37d70de4d` |

Validate the source task-contrast result SHA-256
`129a7a47757d2ecdde7138ca7729c0052f4c4653d5afc38106949f477e87652d`
and source atlas SHA-256
`da6dad8c00738e5fc364de45b970245fa0667731aeedbcb462b85aa94bf1c3d6`.
Checkpoint metadata must declare the registered schema, trained condition,
seed 7, preset, and task. Model state and system-state digests must be unchanged
after analysis.

## Exact task-fiber cohorts

Generate `512` two-sheet fibers for each of two fixed regimes:

- `training_support`, dataset seed `850011`; and
- `outside_range`, dataset seed `850021`.

For fiber coordinate `u` in `[-0.95, 0.95]`, use future phases

```text
phi+ = arccos(u),
phi- = 2 pi - arccos(u).
```

The two sheets have exactly the same cosine target. Hold direction, amplitude,
orientation, offset, harmonic strength, angular speed, and the complete
pre-quantization sensor-noise array fixed within each pair. Change only the
phase sheet before serializing both observations through the original fixed
tokenizer.

The phase model receives its original circular targets for `phi+` and `phi-`;
the cosine model receives its original interval targets for `u`. No latent
quantity enters a model or continuation. Latent phase is used only to construct
the registered synthetic pair and its target, exactly as in the source task.

Before model evaluation, require:

1. exactly two rows per fiber;
2. cosine-target posterior maximum within-pair difference at most `1e-6`;
3. phase targets differ in every fiber;
4. within-pair nuisance and pre-quantization noise are byte-identical by
   construction;
5. serialized sheets differ in every fiber;
6. all values and targets are finite; and
7. a semantic-control matching, defined below, changes cosine by at least
   `0.50` in every fiber.

If any data or source contract fails, stop before activation outcomes.

## Activation cuts and intervention

Retain the complete residual sequence at:

```text
query embedding,
every block post-attention,
every block post-MLP.
```

At each cut, for the two activations in fiber `i`, compute the exact
barycenter

```text
b_i = (h_i,+ + h_i,-) / 2.
```

Patch `b_i` into both rows and run the actual frozen remainder of the model.
The patch changes the full token sequence, not only the query vector. Evaluate
the recipient model's native answer head and native task target.

For every cut, continue the unmodified activation as an exact replay control.
Maximum posterior replay error must be at most `2e-6`.

## Matched controls

### Phase-task specificity

Apply the same exact opposite-phase barycenter operation to the matched phase
model. Because the two sheets have different phase targets, successful
preservation here would refute task specificity.

### Semantic reassignment

Sort the unique cosine fibers by `u` and assign each fiber the barycenter half
the sorted list away. This is fixed before model outcomes and guarantees a
minimum absolute target-coordinate change of `0.50`. Patch the reassigned
barycenter into the original rows. This control preserves tensor shape,
intervention rank, and barycenter construction while changing the semantic
target.

No fitted alignment, probe, decoder, activation map, or threshold selection is
allowed.

## Task-sufficiency gate

For a patched posterior relative to exact replay, define simultaneous
sufficiency as:

1. exact-bin accuracy loss at most `0.03`;
2. mean target cross-entropy increase at most `0.05`; and
3. mean posterior Jensen--Shannon divergence from replay at most `0.02`.

The replay baseline itself must have exact-bin accuracy at least `0.15` in
each model/regime cell. This is a validity floor, not a candidate endpoint.

For each model class, define the **mature causal front** as the first cut whose
correct cosine-fiber barycenter passes all three gates in both regimes and for
which every later cut also passes in both regimes.

The previously observed complete-geometry fronts are fixed references:

```text
d6: block 5 post-attention,
d8: block 3 post-attention.
```

Cut distance counts adjacent attention/MLP cuts; distance at most one passes.

## Primary gates

The complete hypothesis passes only if all of the following hold:

1. both cosine models have a mature causal front;
2. each causal front lies within one sublayer of its fixed observational
   complete-geometry front;
3. at the selected front and final cut, the correct barycenter passes in both
   regimes;
4. at those same cuts, the matched phase-model barycenter fails sufficiency in
   both regimes;
5. at those same cuts, the cosine semantic-reassignment control fails
   sufficiency in both regimes; and
6. replay, data, source, state, finite, and exact-resume contracts pass.

Both d6 and d8 must pass. These two architectures are mechanistic replications,
not independent training seeds; the evidence remains explicitly underpowered.

## Secondary measurements

Report without rescue authority:

- baseline and patched accuracy, target cross-entropy, and task-map score;
- per-cut posterior JS and activation pair RMS;
- the first isolated passing cut even if mature monotonicity fails;
- whether causal sufficiency appears at the earlier block-1 MLP probe-collapse
  cut;
- front distance from both branch collapse and complete quotient geometry; and
- any recovery or loss after an intermediate failure.

## Interpretation table

| Outcome | Interpretation |
| --- | --- |
| cosine fronts match complete geometry; phase and semantic controls fail | strong causal support that task-relative activation geometry reflects a usable quotient |
| cosine barycenters pass but fronts differ | causal quotient sufficiency is real, but the observational atlas mislocalizes it |
| branch probe collapses before barycenter sufficiency | information suppression and causal quotient use are distinct events |
| cosine barycenters never pass | the observed interval geometry is not a sufficient causal state for the frozen continuation |
| phase barycenters also pass | the intervention is not task-specific or the phase task is insensitive under the declared endpoint |
| semantic reassignment passes | barycenter preservation is nonspecific and cannot identify quotient use |

## Boundaries and stopping rule

This study covers four retained seed-7 raw TinyLLMs, two model depths, one
synthetic phase-versus-cosine target contrast, two nuisance regimes, exact
two-sheet fibers, and off-manifold activation barycenters. It does not establish
population prevalence, calibrated/equivariant front-end behavior, natural
language task switching, or literal information erasure from the full
residual.

Regardless of outcome, do not train another model, align cross-model residual
bases, fit a probe, or add a topology scan in this branch. A negative result
closes the causal interpretation of the old task atlas; a positive result
completes its missing intervention bridge.
