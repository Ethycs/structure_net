# TinyLLM continuous semantic carrier and scalar readout calibration

**Status:** SUPPORTED IN POST-OUTCOME CORRECTIVE REPLICATION — NOT FRESH CONFIRMATION  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED` WITH POST-OUTCOME PRODUCER CORRECTION, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-semantic-carrier-readout-separation-v1`  
**Preregistration:** [`2026-08-06_tinyllm-continuous-readout-calibration-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-continuous-readout-calibration-preregistration.md)

## Verdict

The frozen three-checkpoint cohort supports a clean separation between a
small continuous semantic carrier and the 16-bin decoder boundary. A
source-only rule selected geometric ranks `2,3,3` for seeds `7,29,53`. Those
ranks passed the registered continuous endpoint in every held-out composition
and extrapolation cell. One source-fitted scalar boundary rotation per
checkpoint then passed every held-out discrete endpoint without adding an
activation dimension.

The mechanism is reproduced exactly but is not fresh confirmation. A
concurrent primary producer exposed all outcomes while runner pedigree and
lifecycle guards were changing. The authoritative schema-v1.1 campaign is
therefore labelled `post_outcome_corrective_replication_evidence`. Its source
selection, held-out payloads, gates, and headline metrics are exactly equal to
the original numerical run in all three seeds.

The supported architectural account is:

```text
distributed symmetry-typed synthesis
  -> 2--3 dimensional continuous semantic carrier
  -> checkpoint-local scalar boundary calibration
  -> 16-bin task decision.
```

This does not show that one common basis works across checkpoints, or that the
source-fitted carrier can be computed independently from raw observations.

## Campaign integrity

The study reused three frozen 29,956,608-parameter TinyLLM checkpoints and the
source-fitted block-0 `C2` defect bases. It trained no model, adapter, probe, or
nonlinear readout. It fit two declared scalar rotations per checkpoint: one
for the selected-rank state and one exact-state positive control.

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 |
| trained models / predictive observers | 0 / 0 |
| fitted scalar calibrators | 6 |
| source / held-out cells per checkpoint | 2 / 4 |
| rank grid | 1, 2, 3, 4, 5 |
| calibration grid | 4,097 points in `[-0.5,0.5]` bin widths |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| summed analysis time | 11.8 seconds |
| implementation SHA-256 | `6a88480cf37e0f03819a73d03685d0bb5ab7cd0c6554b84194d2fac9b9a127d6` |
| campaign SHA-256 | `af3b4a650c0f17b750e5072066a6461e6934b09d5eac38d5daa7ce16e7edc4ac` |
| DVC data root | `5365b60d75515697a16a57e7433161c6.dir` |
| lakeFS commit | `86e8896337f8db92b1791bdf8a70bd7c09388992c3c66430e939fc656fdb8673` |

Checkpoint hashes, character-source hashes, and all rank, boundary-audit, and
boundary-basis predecessor hashes are stored per seed. The loaded schema,
evidence role, and producer digest were checked from an empty alternate Python
bytecode cache before the correction. A fingerprint-matched resume reused all
three results and left the campaign bytes unchanged.

An eight-orbit CUDA lifecycle completed separately as systems-only evidence.
Its topology gates were under-resolved and were not pooled.

## Evidence correction history

Three quality roots are preserved:

1. `20260806_d6_preregistered` contains the concurrent numerical primary run.
   It used the registered science and authoritative predecessor paths, but its
   producer state was superseded while it executed.
2. `20260806_d6_preregistered_v2` numerically reproduced the result but is
   excluded. Its records declare schema v1 and preregistered evidence while
   its stored implementation digest identifies the contemporaneous raw v1.1
   source, consistent with a stale-bytecode/source race.
3. `20260806_d6_corrective_v3_6a88480c` is authoritative. It declares schema
   v1.1 and `post_outcome_corrective_replication_evidence`, matches the loaded
   producer, reproduces the original numerical payload exactly, and passes
   immutable resume.

No rank, threshold, cohort, seed, calibration grid, or endpoint was changed
after outcomes became visible.

## Primary gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| source rank selected without fallback | **3/3** | 3/3 | pass |
| selected rank at most three | **3/3** | 3/3 | pass |
| held-out continuous sufficiency | **3/3** | 3/3 | pass |
| exact-state scalar positive control | **3/3** | 3/3 | pass |
| selected-rank scalar calibration | **3/3** | 3/3 | pass |
| seed-29 held-out-B extrapolation repair | **pass** | at least 1/64 | pass |

Had the producer identity remained stable, the registered endpoint would have
passed. The scientifically correct phrasing after correction is that the
endpoint is reproduced and supported, not freshly confirmed.

## Continuous carrier

Ranks were selected on the two source shifts before held-out evaluation.

| Seed | Selected rank | Worst held-out mean shift, bins | Worst held-out p95 shift, bins | Registered ceilings |
| ---: | ---: | ---: | ---: | --- |
| 7 | **2** | 0.0615 | 0.1400 | 0.125 / 0.50 |
| 29 | **3** | 0.0469 | 0.1544 | 0.125 / 0.50 |
| 53 | **3** | 0.0178 | 0.0445 | 0.125 / 0.50 |

All twelve held-out cells also passed winding, sampling, and circular-
alignment requirements. Minimum first-moment magnitudes remained about
`0.915`, so the angles were not being extracted from nearly vanishing moments.
The result strengthens the earlier rank interpretation: seed 53's third
direction is semantic, while seed 29 does not need its fifth direction for the
continuous map.

## Scalar boundary calibration

The source-fitted rotations were checkpoint-local:

| Seed | Selected-rank rotation, bins | Source accuracy | Worst held-out loss from untouched baseline |
| ---: | ---: | ---: | ---: |
| 7 | +0.1580 | 0.7969 | 0.0000 |
| 29 | -0.1343 | 0.8281 | -0.0312 |
| 53 | -0.0588 | 0.8750 | -0.0391 |

Negative loss means the calibrated quotient readout exceeded the untouched
baseline. The exact-state positive control passed all twelve cells; its worst
loss was only `0.0078`, below the registered `0.03` ceiling. This rules out the
main control failure: first-moment quantization itself is adequate for these
frozen posteriors.

Seed 29 is the decisive margin case. On held-out-B extrapolation, the
uncalibrated rank-3 moment readout scored `0.6875`; the frozen source rotation
raised it to `0.7500`, a repair of four of 64 quotient representatives, above
the untouched `0.7109` baseline. This is a broader readout comparison than the
earlier one-example rank-4-versus-exact argmax discrepancy, so the two counts
should not be conflated.

## Interpretation and next action

The result supports a three-dimensional typed carrier as the smallest common
architecture target for the stable `C2` fronts. Extra exact-bin rank should
not automatically be treated as extra semantic dimension. Decoder calibration
can absorb at least the tested checkpoint-local margin offsets.

The next shortest frozen diagnostic is cross-checkpoint carrier alignment:
canonicalize each selected source basis under the known `C2` action, fit the
alignment on source data only, and test held-out subspace angles and transported
continuous readouts. Success would justify a shared group-typed three-channel
interface; failure would keep the dimension claim but make the coordinates
checkpoint-local. Do this before training an equivariant sidecar.

## Artifacts and reproduction

- Authoritative aggregate:
  `data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_corrective_v3_6a88480c/campaign_results.json`
- Authoritative per-seed records:
  `data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_corrective_v3_6a88480c/runs/seed_*/result.json`
- Preserved concurrent primary:
  `data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_preregistered/`
- Excluded stale-bytecode attempt:
  `data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_preregistered_v2/`
- Systems-only lifecycles:
  `data/experiments/tinyllm_continuous_readout_calibration/shakedown_20260806/`
  and `20260806_shakedown_cuda/`
- Meta-hypothesis record:
  `data/meta_hypotheses/tinyllm-c2-semantic-carrier-readout-separation-v1.json`
- Runner: `experiments/structure_net/tinyllm_continuous_readout_calibration.py`
- Tests: `tests/structure_net/test_tinyllm_continuous_readout_calibration.py`

The named hypothesis and all three experiment records passed authoritative
Chroma readback after ledger storage. The legacy Chroma transport emitted
known NumPy-2.0 consumer and telemetry warnings; the readback gate passed and
the JSON record remains the portable source artifact. Fifteen focused tests
across the runner, rank predecessor, and meta-hypothesis passed.

```bash
PYTHONPYCACHEPREFIX=/tmp/structure-net-continuous-v3-run-pyc-20260806 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_continuous_readout_calibration \
  --device cuda:0 \
  --post-outcome-corrective-replication \
  --output \
  data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_corrective_v3_6a88480c
```

## Method boundaries

This is a decoder-conditioned causal sufficiency result, not an intrinsic
dimension theorem. Held-out states use exact held-out defects projected into a
source-fitted basis, so they do not establish independent computation from raw
observations. Scalar rotations use source target bins and are supervised
readout calibration. Bases and rotations remain checkpoint-local. Three
selected stable checkpoints are underpowered for a population-prevalence
claim.
