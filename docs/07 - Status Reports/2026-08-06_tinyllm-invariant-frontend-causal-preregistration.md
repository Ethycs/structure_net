# TinyLLM invariant front-end causal test preregistration

**Status:** PREREGISTERED — PRIMARY OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Profile:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-invariant-frontend-stable-cosine-quotient-v1`  
**Schema:** `nal.tinyllm-invariant-frontend-causal.v1`

## Question

Does restricting the sensor front end to respect the known nuisance symmetry
produce a stable internal cosine quotient, where residual penalties on an
unrestricted transformer did not?

## Three matched arms

The d8/N3 campaign compares seeds 7, 17, 29, 41, and 53:

1. `raw_n3`: the retained ordinary N3 TinyLLM checkpoints;
2. `analytic_invariant`: a fixed observation-only canonicalizer followed by a
   learned scalar embedding and the same d8 transformer;
3. `learned_equivariant`: a task-trained sensor encoder whose vector map is
   equivariant by construction, followed by a fixed scalar projection and the
   same d8 transformer.

The raw checkpoints are admissible controls only after schema, configuration,
state, training-data, and minibatch-schedule checks. The two new arms use the
same paired examples, model initialization seeds, paired minibatches, 600
updates, batch size 64, AdamW learning rate `3e-4`, weight decay `0.01`, and
gradient clipping at `1.0`. The only task loss is the ordinary cosine-interval
cross-entropy. No residual, adversarial, contrastive, or equivariance penalty
is used.

## Observation contract and identifiability boundary

Every front end receives only the quantized sensor values in the ordinary N3
input. It receives neither latent phase, nuisance parameters, branch, nor task
targets.

The current generator has an exact gauge ambiguity. For harmonic order `k`,

```text
phase' = phase + alpha
orientation' = orientation - alpha
harmonic_phase' = harmonic_phase - k * alpha
```

leaves the observed sequence unchanged while generally changing absolute
cosine. Consequently no observation-only map can be both invariant to the full
orientation/harmonic-phase gauge and retain absolute cosine exactly. This is a
property of the current quotient specification, not a trainable-encoder defect.

The declared architectural relation therefore separates identifiable and
gauge components. Both structured arms remove constant offset, affine drift,
positive global amplitude, and the nuisance harmonic channel. They retain a
planar vector that is equivariant to sensor orientation:

```text
E(a R x + c + d t, z') = R E(x, z)
```

for positive `a`, planar rotation `R`, constant offset `c`, affine drift `d t`,
and arbitrary discarded harmonic channel `z'`. Their scalar output is the
laboratory-frame first coordinate of the estimated future vector. It is not
claimed to be exactly orientation-invariant. Passing the operational 0.90 gate
despite the bounded gauge ambiguity is an empirical question.

## Analytic canonicalizer

The positive control operates on decoded observed bins:

1. discard the third harmonic channel;
2. fit each planar history over a fixed positive angular-speed grid;
3. regress constant offset, affine drift, and sinusoidal coefficients;
4. choose the speed with minimum planar reconstruction error;
5. determine direction from the rotation-consistency constraints between the
   fitted x/y sine and cosine coefficients;
6. normalize the recovered endpoint vector and advance it by the known task
   horizon;
7. emit only its first coordinate.

This removes amplitude, offset, affine drift, speed, and specified harmonic
content without phase or label access. Quantization and observation noise make
it an estimator rather than an oracle.

## Learned equivariant encoder

The learned encoder:

1. applies a fixed temporal projection that annihilates constant and affine
   trends;
2. normalizes positive global amplitude;
3. ignores the harmonic channel;
4. forms learned temporal linear combinations with weights shared between x
   and y;
5. applies only radial gates and scalar mixing to vector channels;
6. normalizes the resulting future-vector estimate and emits its first
   coordinate.

Before the final coordinate projection, every learned operation commutes with
planar rotations. Translation/drift removal and amplitude normalization are
structural. The relation is covered by numerical contract tests on transformations
not used for training the check.

## Representation cuts

Frozen post-training probes evaluate:

- `frontend`: flattened decoded raw input for `raw_n3`, and the emitted scalar
  for the two structured arms;
- `post_attention`: final query residual after block-1 attention;
- `post_mlp`: final query residual after block-1 MLP;
- `full`: final residual after all eight blocks.

The structured arms present `[BOS, scalar feature, query]` embeddings to the
same transformer. This sequence change is the declared front-end intervention;
the transformer and task head remain the same d8 architecture.

## Primary endpoint

At every declared cut, a seed passes a regime only when both hold:

```text
cosine Pearson correlation >= 0.90
conditional branch balanced accuracy <= 0.55
```

The cosine-conditioned nonlinear probe uses disjoint train, validation, and
test samples. Conditional log-loss gain over a cosine-only null remains a
reported secondary diagnostic but is not part of this two-dimensional gate.

An arm succeeds only if the same four of five seeds pass all four cuts on both
held-out composition and outside-range extrapolation. In-distribution results
and task exact-bin accuracy are controls, not alternative success paths.

## Interpretation

| Outcome | Meaning |
| --- | --- |
| analytic and learned pass | architectural symmetry makes the quotient stable; learned encoder approaches the positive control |
| analytic passes, learned fails | the quotient is operationally attainable but the declared learned equivariant family or optimization is insufficient |
| analytic passes only composition | canonicalization remains support-relative under quantization/noise or the unresolved orientation gauge |
| both structured arms erase branch but lose cosine | the full quotient is not identifiable from current observations; invariance creates compression |
| learned preserves cosine but leaks branch | equivariance retained a nuisance/fiber coordinate and the scalar bottleneck or downstream routing is insufficient |
| neither improves over raw | stop treating the current N3 observation as supporting an absolute-cosine quotient; add an observed gauge reference or change the target |

Secondary geometry and task accuracy cannot rescue a failed joint endpoint.

## Fixed execution plan

- Training examples: 4,096 paired N3 examples.
- Probe train/validation/test: 2,048 / 512 / 1,024.
- Seeds: 7, 17, 29, 41, 53.
- Learned equivariant vector channels: 16.
- Analytic positive-speed grid: 91 values from 0.10 to 0.60 radians/step.
- Scheduler: at most two CUDA workers after a representative memory pilot.
- Artifact root:
  `data/experiments/tinyllm_invariant_frontend_causal/20260806_d8_preregistered`.

The producing implementation and its digest will be frozen before primary
workers launch. Any design change after primary outcomes are visible is a new
exploratory study or a dated amendment, not completion of this preregistration.
