# TinyLLM observed-twirl noise-law preregistration

**Status:** FROZEN BEFORE PRIMARY OUTCOME INSPECTION  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`; frozen-system,
no-fit sensor intervention  
**Hypothesis:** `tinyllm-noise-law-observed-twirl-v1`  
**Schema:** `nal.tinyllm-noise-law-observed-twirl.v1`

## Question

Does the validated observed `C2` action and Reynolds twirl remain causally
task-sufficient when the planar sensor-noise law is no longer invariant under
reflection about the observed calibration axis?

The predecessor established closure under the existing isotropic,
reflection-compatible generator. This study changes only the distribution of
an additional decoded planar measurement error. It does not train or fit a
model, front end, action, observer, probe, head, threshold, or denoiser.

## Prediction

The strong prediction is structural rather than distribution-relative:

> Both the exact analytic canonicalizer and the learned equivariant front end
> retain natural task utility, correct-action sufficiency, and correct-twirl
> sufficiency under matched-energy isotropic, lab-anisotropic, and lab-biased
> planar noise in at least four of five frozen checkpoints per arm.

The analytic arm is an exact positive-control mechanism. The learned arm is
the substantive test of whether the learned scalar interface generalizes when
the observation distribution no longer respects the deck action.

## Frozen source lineage

The following existing evidence and code are fixed inputs:

| Source | SHA-256 / value |
| --- | --- |
| observed-deck campaign | `79c3e27374d8b6f4611552595de5852ace940204bda825e64cf80eff6ab2050d` |
| observed-deck result manifest | `b91af38162fbf45e29348fbdf583cb676660d68cf22e5a795b438fd8cd015db3` |
| observed-deck implementation | `c970fe8801524f5248a9314e821b6783127596d05a2f206325ed85deb42f9629` |
| observed-deck runner file | `6468a7af23cd1ae11cdc6cae3cf553252c2cb19352ed89e44c053a1ade60d213` |
| causal-closure campaign | `1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14` |
| causal-closure result manifest | `baed34a16dca206536b2e9cd221fd9f7556f4c063f85ee857352522e770844f4` |
| causal-closure implementation | `5060b45674430351dabb6cd67af5e41a215f883d09b9702edd3d36b3d1d51260` |
| calibrated front-end runner file | `73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77` |
| composition cohort | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation cohort | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |

The retained systems are the five d8/N3 checkpoints with seeds
`(7, 17, 29, 41, 53)` in each of:

- `analytic_calibrated`; and
- `learned_calibrated_equivariant`.

All model, scalar-embedding, analytic-front-end, learned-encoder, and answer
head parameters remain frozen. Source state hashes and clean posterior replay
must pass before a noisy condition is interpreted.

## Data and noise laws

The exact 512-example composition and 512-example extrapolation cohorts from
the predecessor are reused intentionally. Reuse isolates the new noise-law
intervention; the additional noise arrays are new, deterministic, and shared
across every checkpoint and arm.

Decode each cohort's tokenized three-channel history once. Add error only to
the two planar channels after decoding and before the structured front end.
The harmonic channel, calibration packet, labels, examples, task posterior,
and input boundary tokens are unchanged.

Let `z_x,z_y ~ N(0,1)` be common draws and fix `sigma = 0.05`. At every
example and time step, apply one of:

| Law | Added planar error | Expected squared norm |
| --- | --- | ---: |
| `isotropic` | `sigma (z_x, z_y)` | `2 sigma^2` |
| `lab_anisotropic` | `sigma (sqrt(1.8) z_x, sqrt(0.2) z_y)` | `2 sigma^2` |
| `lab_biased` | `sigma (1 + z_x/sqrt(2), z_y/sqrt(2))` | `2 sigma^2` |

The lab axes are fixed and are not rotated with each example's calibration
orientation. Thus the last two laws are generally not invariant under the
example-specific deck reflection. The common draws create a paired
matched-energy comparison without selecting a favorable realization.

Noise seeds are `861001` for composition and `861002` for extrapolation. The
observed empirical RMS of each law must lie within 5% of
`sqrt(2) * sigma`. The analytic population contracts must additionally show:

- zero population mean/covariance reflection defect for `isotropic` to
  numerical tolerance;
- median normalized covariance reflection defect at least `0.10` for
  `lab_anisotropic` in each regime; and
- median normalized mean reflection defect at least `0.05` for `lab_biased`
  in each regime.

These are generator validity checks, not learned outcomes.

## Interventions and fixed controls

For every arm, checkpoint, regime, and law, compute:

1. the natural noisy observation;
2. the declared observed deck action from decoded planar history and the
   calibration packet;
3. the matched target-changing orthogonal-axis action;
4. the correct Reynolds twirl, formed by averaging natural and correct-action
   activations; and
5. the orthogonal control twirl.

Evaluate at exactly two cuts:

- `pre_block`, the structured front-end output plus fixed embeddings before
  block-0 attention; and
- `full`, the final post-MLP residual before the fixed answer map.

The architecture, examples, calibration packets, labels, checkpoints, answer
rows, batch size (`256`), continuation, and task metric implementation remain
fixed. The only scientific intervention is the added noise law.

## Primary endpoints

Task metrics are exact-bin accuracy, circular error, and target cross-entropy.
For each seed, all gates are joint within the same system.

### Natural noisy utility

Relative to the same checkpoint's clean posterior on the identical cohort,
the natural noisy input must satisfy:

- accuracy loss at most `0.05`;
- circular-error increase at most `pi/16`; and
- cross-entropy increase at most `0.10`.

This prevents preservation of a noise-destroyed computation from counting as
quotient robustness.

### Correct action and twirl

Relative to the natural noisy posterior, both the correct action and the
correct twirl must satisfy, at `pre_block` and `full`:

- accuracy loss at most `0.03`;
- circular-error increase at most `pi/16`; and
- cross-entropy increase at most `0.10`.

A seed passes one law only if natural utility plus action and twirl sufficiency
pass simultaneously on composition and extrapolation at both cuts.

An arm/law cell passes at `>=4/5` seeds. The primary hypothesis passes only if
all six arm/law cells pass. Separate marginal gates on different seeds cannot
be combined.

### Negative-control specificity

For each arm/law cell, a seed counts as a control pass only if its
orthogonal-axis action or twirl passes the same task gate in both regimes at
either declared cut. At most one of five seeds may pass. Correct intervention
success without this specificity gate does not confirm the hypothesis.

### Positive-control validity

The campaign is invalid for primary interpretation if either structured arm
fails the `isotropic` population gate, if the analytic feature is not invariant
under the correct action to maximum absolute error `<=1e-6`, if the analytic
arm fails the correct action/twirl gate under any law, or if source replay,
noise-law, finite-number, state-identity, or artifact-integrity contracts fail.

## Secondary measurements

These measurements explain but cannot rescue a failed primary gate:

- identity/action/twirl posterior Jensen--Shannon divergence;
- front-end scalar difference under the correct action;
- activation relative RMS at both cuts;
- per-law accuracy, circular error, and cross-entropy changes;
- law mean/covariance reflection defects;
- transformed planar range;
- per-seed and population differences between isotropic and asymmetric laws;
- exact model/system state hashes before and after analysis; and
- CUDA memory and analysis time.

No probe, observer, decoder, carrier, representation loss, or topology
estimator is licensed.

## Outcome interpretation

| Outcome | Interpretation |
| --- | --- |
| Both arms pass all laws | Observed quotient closure is structurally robust to these matched-energy symmetry-breaking laws at the registered dose. Observation-distribution symmetry is not necessary for frozen functional closure here. |
| Analytic passes all; learned fails an asymmetric law | Exact canonicalization is law-robust, while the learned scalar quotient is support-relative despite its equivariant vector construction. |
| Isotropic passes; both arms fail asymmetric laws | Reflection compatibility of the observation law is necessary for the current system-level quotient claim at this dose. |
| Natural utility fails before action/twirl | The law breaks the task interface before quotient closure can be isolated; report failure without attributing it to the action. |
| Isotropic or integrity control fails | The campaign is invalid; repair implementation or dose contract before scientific interpretation. |
| Orthogonal controls pass too often | Success is nonspecific to the target-preserving deck action. |

## Stop rule

Run one systems-only shakedown, then one primary campaign. Do not tune `sigma`,
anisotropy, bias, thresholds, draws, cuts, or laws after observing the primary.
Do not retrain a front end or TinyLLM regardless of outcome. A failure selects
either an exact architectural scalar symmetrization or a genuinely new sensor
model; a pass closes additive lab-frame noise-law symmetry at this dose.

## Artifacts and execution

Planned artifacts:

```text
data/experiments/tinyllm_noise_law_observed_twirl/<run>/
├── campaign_results.json
├── noise_law_arrays.npz
└── runs/<condition>/seed_<seed>/
    ├── result.json
    └── noise_law_diagnostics.npz
```

The shakedown must be labeled
`systems_lifecycle_only_not_quality_evidence`, use a disjoint output root, and
must never enter the scientific aggregate. Completed-result reuse requires
schema, implementation, scientific fingerprint, and artifact-digest matches.

## Boundaries

This design tests one added error scale, three additive lab-frame planar laws,
two reused synthetic shifts, one calibrated `C2` task, and ten retained d8/N3
systems. It does not test temporal correlation, errors in the calibration
packet, saturation/requantization, learned denoising, unknown groups, other
noise scales, architecture populations, natural language, or real sensors.
The transformed asymmetric observations may be off the natural support of
their declared law; action/twirl success is frozen functional robustness, not
proof that the asymmetric observation distribution itself has a `C2` symmetry.
