# TinyLLM calibration-orientation noise causal titration

**Status:** CONFIRMED `reference_precision_critical` — COMPLETE GATE RADIUS `0°`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-checkpoint causal intervention  
**Hypothesis:** `tinyllm-calibrated-orientation-noise-radius-v1`  
**Schema:** `nal.tinyllm-calibrated-orientation-noise.v1`  
**Preregistration:** [calibration-orientation noise preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-calibration-orientation-noise-preregistration.md)

## Verdict

The complete joint robustness radius is zero for both the analytic calibrated
canonicalizer and the learned calibrated-equivariant encoder. Clean replay
passes in `5/5` checkpoints for both arms. At the first nonzero registered
level, `sigma=0.035` radians (about `2°`), the analytic arm passes `0/5`
checkpoint gates and the learned arm passes `2/5`; four were required. The
preregistered classification is therefore `reference_precision_critical`.

The mechanism is more informative than the headline radius. Every analytic
and learned checkpoint retains the representation gate through `10°`:

```text
cosine correlation >= 0.90
conditional branch balanced accuracy <= 0.55
conditional log-loss gain <= 0.02.
```

The `2°` complete-gate failure is entirely due to the separate exact-bin task
accuracy guard. The internal quotient remains base-preserving and
branch-contracted while the fixed downstream interval decoder becomes
miscalibrated.
At `20°`, both arms finally fail the representation gate in all checkpoints.

Thus orientation error first breaks **chart-to-bin calibration**, not quotient
formation. This reproduces, under a direct input intervention, the program's
earlier separation between a robust continuous carrier and a brittle discrete
readout.

## Registered curve

Counts are checkpoint passes out of five. `Rep` is the four-cell
representation gate, `Task` is the two-regime no-more-than-three-point
exact-bin accuracy-loss gate, and `Joint` requires both.

| Orientation sigma | Analytic Rep / Task / Joint | Learned Rep / Task / Joint |
| ---: | ---: | ---: |
| `0°` | `5 / 5 / 5` | `5 / 5 / 5` |
| `2°` | `5 / 0 / 0` | `5 / 2 / 2` |
| `5°` | `5 / 0 / 0` | `5 / 0 / 0` |
| `10°` | `5 / 0 / 0` | `5 / 0 / 0` |
| `20°` | `0 / 0 / 0` | `0 / 0 / 0` |
| `30°` | `0 / 0 / 0` | `0 / 0 / 0` |
| `45°` | `0 / 0 / 0` | `0 / 0 / 0` |

Both curves are monotone under their registered gates. No smoothing,
interpolation, or post-outcome threshold selection was used.

## Mechanistic measurements

At `2°`, all representation cells still pass. Across checkpoints:

| Arm | Minimum front-end cosine correlation | Maximum conditional branch accuracy | Maximum conditional log-loss gain | Mean composition accuracy loss | Mean extrapolation accuracy loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| analytic | `0.9711` | `0.5225` | `0.00057` | `3.77` points | `3.42` points |
| learned equivariant | `0.9394` | `0.5254` | `0.00112` | `4.39` points | `0.70` points |

The learned arm's `2/5` joint passes are seeds `7` and `29`. This is a small
checkpoint-stratified robustness advantage, not a population-level pass.

Representation degradation appears between `10°` and `20°`. At `10°`, the
worst front-end cosine correlation is `0.9330` for the analytic arm and
`0.9171` for the learned arm; all ten checkpoints still satisfy the full
representation gate. At `20°`, the minima fall to `0.8526` and `0.8421`, and
both arms pass `0/5`. Conditional branch accuracy remains near chance even at
`45°` (worst observed `0.5234` analytic, `0.5205` learned), with conditional
log-loss gains always far below `0.02`. Noise destroys the semantic base before
it exposes the branch fiber.

Mean exact-bin accuracy loss rises much earlier. For the analytic arm it is
`3.77/3.42` points on composition/extrapolation at `2°`, `17.89/12.95` at
`5°`, and `35.55/27.21` at `10°`. The learned arm is somewhat more robust on
extrapolation (`0.70`, `6.54`, and `17.89` points respectively) but not enough
to meet the joint four-of-five gate.

## Interpretation

The causal sequence is now:

```text
clean gauge reference
  -> stable quotient carrier and calibrated bins

small orientation error (~2°)
  -> quotient carrier still stable
  -> branch still contracted
  -> fixed discrete task calibration already brittle

large orientation error (between 10° and 20°)
  -> semantic base itself falls below the quotient gate
  -> branch remains contracted rather than reappearing.
```

This sharpens “identifiability is necessary.” A reference can make the target
identifiable in principle while finite reference precision still limits the
usable decoder. Architectural equivariance protects the representation over a
moderate interval, but does not by itself calibrate a fixed output boundary
against reference uncertainty.

The next shortest causal test is therefore not another invariant encoder or
internal probe. Freeze the same front ends and recalibrate only the final
ordered-interval readout on noisy-reference training data, with a clean/noisy
cross-condition matrix and a low-capacity affine scalar calibration control.
If readout-only calibration restores the task gate while the representation
gate already passes, the quotient mechanism is vindicated and the remaining
defect is localized to the decoder interface. Full TinyLLM retraining is not
yet justified.

## Validity and lifecycle

- source campaign, result, model-state, system-state, and front-end files
  replayed for all ten frozen systems;
- clean fresh-cohort replay passed both arms in `5/5` checkpoints;
- orientation vectors remained unit norm within `1e-6`;
- both C2 sheets received bit-identical angular errors;
- corruptions were common across arms and checkpoints;
- all features and metrics were finite;
- TinyLLM and front-end parameters remained frozen;
- `210` diagnostic probe sets were fit; they are measurement instruments, not
  deployed modules; and
- exact resume preserved campaign and noise-array hashes.

An initial launch targeted a host/PyTorch-mismatched GPU index and failed with
OOM before loading the first checkpoint. It produced no result and is not part
of the campaign. The authoritative replay used PyTorch `cuda:1`.

## Campaign integrity and reproduction

| Item | Value |
| --- | --- |
| frozen systems requested/completed | `10/10` (two arms × five seeds) |
| noise levels | `7` |
| primary representation cells | `280` |
| TinyLLM models / front ends trained | `0 / 0` |
| diagnostic probe sets fit | `210` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA allocation | `325,561,344` bytes |
| analysis time | `466.09` seconds |
| implementation SHA-256 | `990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70` |
| campaign SHA-256 | `876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f` |
| noise arrays SHA-256 | `b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7` |
| source campaign SHA-256 | `80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501` |
| meta-hypothesis JSON SHA-256 | `31e398d4f7f6a7b27743aa413b35b306f4bddc96be0ff60bf319f275f1e73285` |
| DVC data root | `c47addb9c3dbb2fea3e51a0940e46d69.dir` (`2,119` files; `39,892,387,779` bytes) |
| lakeFS snapshot | `d53a11c89bd23da70894c0701a1f7d0f468cd1eb47f7a35c76aeb35546518772` |

- campaign:
  `data/experiments/tinyllm_calibration_orientation_noise/20260807_d8_preregistered/campaign_results.json`
- per-arm/checkpoint records:
  `data/experiments/tinyllm_calibration_orientation_noise/20260807_d8_preregistered/runs/*/seed_*/result.json`
- deterministic corruptions:
  `data/experiments/tinyllm_calibration_orientation_noise/20260807_d8_preregistered/orientation_noise_arrays.npz`
- runner:
  `experiments/structure_net/tinyllm_calibration_orientation_noise.py`
- tests:
  `tests/structure_net/test_tinyllm_calibration_orientation_noise.py`
- meta-hypothesis record:
  `data/meta_hypotheses/tinyllm-calibrated-orientation-noise-radius-v1.json`

The meta-hypothesis store read back the hypothesis and all `10` direct
checkpoint records. DVC reports the local cache and configured `lakefs` remote
in sync. The exact root object is present at
`lakefs://artifacts/d53a11c89bd23da70894c0701a1f7d0f468cd1eb47f7a35c76aeb35546518772/structure-net/files/md5/c4/7addb9c3dbb2fea3e51a0940e46d69.dir`,
and the lakeFS branch has no uncommitted object diff.

```bash
MPLCONFIGDIR=/tmp/matplotlib-calibration-noise \
pixi run python -m \
  experiments.structure_net.tinyllm_calibration_orientation_noise \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_calibration_orientation_noise/20260807_d8_preregistered
```

## Scope boundary

This is a frozen-checkpoint robustness curve on a controlled synthetic
orientation reference. It does not estimate real calibration cost, robustness
to missing amplitude/offset/drift/speed fields, or natural-language model
behavior. The five checkpoints were selected from the preceding successful
calibrated campaign; they are replication units but not a random population.
