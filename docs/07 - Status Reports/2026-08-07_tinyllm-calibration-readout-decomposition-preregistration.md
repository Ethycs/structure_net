# TinyLLM calibration readout decomposition preregistration

**Status:** COMPLETED — CORRECTIVE V2 VALID; V1 QUARANTINED

**Date:** 2026-08-07

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, existing-checkpoint
outcome-directed mechanistic diagnostic

**Hypothesis:** `tinyllm-calibration-readout-decomposition-v1`

**Schema:** `nal.tinyllm-calibration-readout-decomposition.v1`

**Post-outcome lifecycle amendment (2026-08-07):** The first complete root,
`20260807_d8_existing_checkpoints`, was produced with implementation digest
`edb368b293f4d5673cadc7236cf3ec55051f5f760e0afcf64366f87b5e92724e`.
After its outcomes existed, the runner changed to digest
`0649a40c17384266f360daf33b68519e8f6d068aeec03893cfe9229f4f8d222d`
and no longer accepted exact resume of that root. The original result is
therefore retained but quarantined from evidence promotion.

Run one same-protocol corrective replay under
`20260807_d8_corrective_v2`. The sources, checkpoints, examples, noise arrays,
observer family, readouts, causal patches, controls, levels, thresholds,
classification order, and seed rule below remain unchanged. The corrective
root is explicitly post-outcome replication, not fresh confirmation. Promote
it only if all ten cells are valid, the registered classification reproduces,
the producing implementation digest remains stable through completion, and an
exact resume leaves every result, observer, and aggregate byte unchanged.

## Decision question

The calibrated-reference robustness curve rejected its end-to-end target, but
the failure separated two measurements. At every noise level through
`sigma=0.20`, both structured arms retained the declared probe-defined quotient
at the front end and full depth; frozen exact-bin task utility failed much
earlier. Correlation alone can conceal coordinate error, so this does not yet
show that the downstream decoder is the bottleneck.

The next question is:

> At the same frozen full-depth activation, is calibration-noise failure caused
> primarily by posterior/bin calibration, by a mismatch between the available
> cosine coordinate and the trained task decoder, or by insufficient precision
> in the reference-derived cosine coordinate itself?

No TinyLLM model, sensor front end, or task head is trained or fine-tuned. The
only fitted object is a diagnostic cosine observer trained on the predecessor's
clean observer split and frozen across all perturbations.

## Locked source and data

Reuse exactly:

```text
source campaign
data/experiments/tinyllm_calibration_degradation_causal/
    20260807_d8_existing_checkpoints/campaign_results.json

source campaign SHA-256
170d99553058ab544d95d6abf4d26ea1062bfe1b79fda2deefc531dfaebd0f6e

source implementation SHA-256
f273abf83231e4fc37583154c24158c81bba15a29e11ca3f96daf0b579be0f70

source calibrated checkpoint manifest SHA-256
23598aaef5a0d16825ff9a928de57857e7bd58b10feab8853739448acfd983fc

source common-noise SHA-256
1896c852644f724f5a2d214d5e69380eef3e46abb2e7b295eb0d4bace51a0c82
```

Use `analytic_calibrated` and `learned_calibrated_equivariant`, seeds `7`,
`17`, `29`, `41`, and `53`, and the exact composition/extrapolation datasets,
calibration perturbations, calibration packets, targets, and checkpoint states
from the source runner. The primary levels are `0.00`, `0.05`, `0.10`, and
`0.20`. Larger source levels and field ablations are outside this diagnostic.

Every source result, model, front end, and state digest must replay. The source
campaign hash, manifest hash, data hashes, common-noise hash, and source result
hashes are hard validity gates.

## Fixed readouts

Let the 16 cosine-interval answer centers be

```text
q_j = -1 + 2j/15,  j=0,...,15.
```

For each example, evaluate these fixed readouts:

1. **Model argmax.** The unchanged TinyLLM answer-token argmax. This must replay
   the source exact-bin accuracy exactly.
2. **Model posterior mean.** Compute `m=sum_j p_j q_j` from the unchanged model
   posterior and select the nearest center. This is a no-fit posterior-shape
   diagnostic.
3. **Front-end coordinate.** Clamp the structured front-end scalar to `[-1,1]`
   and select the nearest center. This is a no-fit sensor-coordinate readout.
4. **Frozen full-depth cosine observer.** Reproduce the predecessor's clean-fit
   nonlinear conditional suite at full depth, retain its cosine output, freeze
   it, clamp to `[-1,1]`, and select the nearest center at every perturbation.

The diagnostic observer must use the exact predecessor train/validation splits,
architecture, initialization seed, optimizer, minibatches, early stopping, and
normalization. Its clean cosine correlation and RMSE must replay the source
campaign to `1e-5`. Store its weights and normalization statistics so this
campaign never needs to refit them for resume or downstream use.

For every coordinate readout, also construct the fixed interval posterior

```text
p_j(u) proportional to exp(-0.5 * ((q_j-u)/(2/15))^2)
```

and report target cross-entropy, cosine MAE/RMSE/correlation, exact-bin
accuracy, and accuracy drop from that same readout's clean value.

## Causal task-gradient intervention

At the full-depth residual `h`, define the unchanged model posterior mean
`m(h)` and its per-example task gradient `g=grad_h m(h)`. Apply one locked
first-order scalar write:

```text
h' = h + ((u_hat - m(h)) / (||g||^2 + 1e-8)) g,
```

where `u_hat` is the frozen clean observer's cosine estimate. Recompute the
unchanged model logits from `h'`. This is the primary causal diagnostic: it
asks whether writing the already decoded coordinate through the current local
task covector restores the frozen task output.

Retain four controls:

- **target oracle:** substitute the true target cosine for `u_hat`; this is a
  positive-control ceiling, not a deployable result;
- **flipped:** negate the primary scalar coefficient;
- **shuffled:** permute `u_hat` across examples using a fixed seed; and
- **kernel:** replace `g` by a deterministic random direction projected
  orthogonally to `g`, with the same patch norm.

Record task-gradient norms, patch norms, finite fractions, first-order target
error, post-patch posterior-mean error, exact-bin accuracy, and target
cross-entropy. Do not iterate Newton updates, fit patch amplitudes, use noisy
targets to train an observer, or tune a trust radius after outcomes.

## Readout utility gate

For each readout, arm, and seed:

1. clean adequacy requires its clean exact-bin accuracy to be no more than
   `0.03` below the unchanged model's clean accuracy in each shift;
2. noisy utility at a level requires its own exact-bin accuracy drop from clean
   to be at most `0.03` in each shift; and
3. the seed passes only when clean adequacy and noisy utility hold on both
   composition and extrapolation.

An arm passes a readout/level in at least `4/5` seeds. No mean can rescue a
different set of failing seeds. The primary decision is at `sigma=0.20`; the
lower levels establish the failure order.

Apply the same gate to the observer-gradient patch, using the unchanged model
as its clean baseline. The target-oracle patch must pass at least `4/5` seeds
in both arms for a strong causal localization. Flipped, shuffled, and kernel
controls must each pass at most `1/5` seeds per arm at `0.20` for specificity.

## Fixed classifications

Apply the first matching label:

| Outcome at `sigma=0.20` | Classification |
| --- | --- |
| Provenance, state, data identity, clean replay, finite, observer replay, or baseline replay fails | `invalid` |
| Model posterior mean passes both arms while argmax fails | `posterior_shape_calibration_limited` |
| Frozen observer and observer-gradient patch pass both arms, with causal controls, while model-moment fails | `decoder_relation_limited` |
| Frozen observer passes both arms but the causal patch or its controls fail | `coordinate_decodable_but_causal_write_unresolved` |
| Frozen observer fails both arms, regardless of decoder diagnostics | `reference_coordinate_precision_limited` |
| Only the learned arm's frozen observer passes | `analytic_reference_precision_limited` |
| Only the analytic arm's frozen observer passes | `learned_reference_precision_limited` |
| Otherwise | `mixed_readout_reference_limit` |

The front-end coordinate readout distinguishes whether transformer depth
improves the noisy reference estimate, but it does not override the
classification order.

## Outcome-directed decisions

| Classification | Next action |
| --- | --- |
| posterior-shape calibration limited | replace only the bin/posterior calibration and retest frozen systems |
| decoder-relation limited | train the smallest prospective calibration-aware task readout; keep the front end and transformer frozen first |
| coordinate decodable but causal write unresolved | audit the local task covector/nonlinearity; do not train yet |
| reference-coordinate precision limited | improve or denoise orientation/speed/offset measurement before model optimization |
| one arm reference-limited | localize the arm-specific front-end formula/learned dependency |
| mixed | preserve per-arm/seed stratification and design the narrowest follow-up |
| invalid | repair only the digital contract under a new root |

No topology scan, representation loss, writer sidecar, full TinyLLM retraining,
or calibration-noise augmentation is licensed before this decision.

## Fixed artifacts

- runner: `experiments/structure_net/tinyllm_calibration_readout_decomposition.py`
- tests: `tests/structure_net/test_tinyllm_calibration_readout_decomposition.py`
- quarantined first root: `data/experiments/tinyllm_calibration_readout_decomposition/20260807_invalid_prelock_runner`
- unpromoted same-protocol operational replay: `data/experiments/tinyllm_calibration_readout_decomposition/20260807_d8_existing_checkpoints`
- corrective result root: `data/experiments/tinyllm_calibration_readout_decomposition/20260807_d8_corrective_v2`
- report: `docs/08 - Analysis/2026-08-07_tinyllm-calibration-readout-decomposition.md`
- meta hypothesis: `tinyllm-calibration-readout-decomposition-v1`

The runner must use strict JSON, atomic writes, deterministic fingerprints,
per-result and observer hashes, exact resume, and explicit counts of trained
models (`0`), trained front ends (`0`), trained task heads (`0`), and fitted
diagnostic observers (`10`).

## Method boundaries

This is an outcome-directed mechanistic follow-up to an inspected source
campaign, not an independent confirmation. The clean nonlinear observer is a
tested estimator, not proof of information absence. The target oracle uses
labels and is only a positive control. The synthetic calibration errors are
not a real-instrument uncertainty model. External nearest-center readouts show
recoverability; only the task-gradient patch is a causal intervention on the
frozen task computation.

## Registered disposition

The locked corrective replay completed under `20260807_d8_corrective_v2`.
All ten cells passed provenance, state, input/target identity, clean replay,
and finite gates. The producing implementation digest stayed fixed, and an
exact completed-result resume preserved the aggregate, all results, and all
observer bytes.

At `sigma=0.20`, every deployable fixed readout and the clean-observer
task-gradient write passed `0/5` seeds per arm. The true-target-coordinate
positive-control write passed `5/5` per arm; flipped, shuffled, and kernel
controls passed `0/5` per arm. The first matching registered classification is
`reference_coordinate_precision_limited`.

The corrective campaign SHA-256 is
`833392ad7956ddcf715a20211431586f371edee280f167bf5a4fa51437cdc6c6`;
the implementation SHA-256 is
`0649a40c17384266f360daf33b68519e8f6d068aeec03893cfe9229f4f8d222d`;
the exact-resume tree-manifest SHA-256 is
`ec2a7338655b3cb3e02ab3808665d3b0947318abf06b0a17e0fbedab6798cc1c`.
See
`docs/08 - Analysis/2026-08-07_tinyllm-calibration-readout-decomposition.md`.
