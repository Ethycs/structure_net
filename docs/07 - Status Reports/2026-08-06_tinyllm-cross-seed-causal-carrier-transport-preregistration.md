# TinyLLM cross-seed causal-carrier transport preregistration

**Status:** PREREGISTERED — PRIMARY OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-cross-seed-causal-carrier-transport-v1`  
**Schema:** `nal.tinyllm-cross-seed-causal-carrier-transport.v1`

## Question

Do the checkpoint-local three-dimensional `C2` quotient-sufficient carriers
implement the same function of the input orbit up to a label-free linear change
of coordinates?

The post-attention Reynolds defect is already invariant under sheet exchange,
so merely assigning it the trivial `C2` representation would be vacuous. The
test must transport one checkpoint's carrier coordinates into another
checkpoint and causally substitute the transported write into the target
model's frozen continuation.

## Prediction

A whitened orthogonal map fitted on paired, label-free source orbits will:

1. explain at least `0.80` of target carrier variance on every held-out cell;
2. preserve the target continuous degree-two map;
3. pass the target's previously frozen scalar-calibrated discrete endpoint; and
4. outperform a regime-preserving shuffled-pair map in every directed
   checkpoint pair.

The full hypothesis requires all six directed maps among seeds 7, 29, and 53.

## Fixed checkpoints and carrier coordinates

- d6 calibrated degree-two checkpoints, seeds 7, 29, and 53;
- frozen block-0 post-attention causal front in every checkpoint;
- three leading right-singular defect directions per checkpoint;
- each basis refit exactly from that checkpoint's predecessor
  `source_selection` composition and extrapolation defects;
- target scalar rotations frozen from the authoritative continuous-readout
  campaign;
- no transformer, frontend, probe, decoder, or calibration training.

Rank three is fixed for all models. Seed 7's preceding minimum was rank two,
but three is the conservative common carrier budget and avoids a rectangular
transport advantage for either direction.

## Paired data

The generator creates exactly the same latent/nuisance orbits for every model
in a cell. These new deterministic seeds are disjoint from the predecessor
selection and held-out cohorts:

| Cohort | Composition seed | Extrapolation seed | Role |
| --- | ---: | ---: | --- |
| alignment fit | 130007 | 130008 | map fitting only |
| held-out A | 230007 | 230008 | primary evaluation |
| held-out B | 330007 | 330008 | primary evaluation |

Each cell contains 64 exact `C2` orbits. Phase, nuisance, target, and branch
labels are not used to fit coordinate maps.

## Alignment intervention

For checkpoint `s`, let `B_s` be its three-row orthonormal defect basis and

```text
C_s(x) = defect_s(x) B_s^T.
```

On the two alignment-fit regimes, center and whiten `C_s` and `C_t`, solve the
orthogonal Procrustes map between whitened coordinates, and retain the implied
affine source-to-target map. On held-out input `x`, patch target checkpoint `t`
with

```text
propagated_t(x) + transport_s_to_t(C_s(x)) B_t.
```

The target model and its continuation remain frozen.

## Controls

| Control | Purpose |
| --- | --- |
| target zero defect | verifies the frozen causal front is active on each new cell |
| target exact full defect | upper endpoint and continuous reference |
| target direct rank-3 defect | confirms the declared carrier remains sufficient on the new cohorts |
| regime-preserving shuffled target pairing | tests whether marginal coordinate distributions suffice without example correspondence |
| unconstrained affine ridge map | descriptive ceiling; cannot rescue failure of whitened orthogonal transport |

The shuffled permutation is deterministic, fixed by the directed seed pair,
and applied separately within composition and extrapolation before map fitting.

## Primary endpoints

For every directed pair and each of four held-out cohort/shift cells:

1. **coordinate transport:** held-out variance explained in the target
   three-coordinate carrier is at least `0.80`;
2. **continuous task map:** alignment loss from the target exact defect is at
   most `0.005`, mean circular-moment shift at most `0.125` bins, p95 shift at
   most `0.50` bins, winding within `0.10` of degree two, and sampling resolved;
3. **discrete task map:** the target's frozen scalar rotation loses no more than
   `0.03` exact-bin accuracy relative to the untouched target continuation.

Target zero must fail and target exact plus direct rank three must pass the
continuous/discrete conjunction in every target cell. A target cell that fails
these controls makes the campaign fail rather than being silently excluded.

For specificity, each of the six shuffled maps must fail at least one primary
endpoint in at least one held-out cell. The paired map's worst held-out variance
explained must also exceed the shuffled map's worst value by at least `0.20`.

The campaign is confirmed only if all target controls, all 24 paired transport
cells, and all six specificity gates pass.

## Secondary measurements

- coordinate normalized RMSE and mean row cosine;
- source-fit Procrustes residual;
- singular values of the whitened cross-covariance;
- unconstrained affine-ridge held-out variance explained and causal endpoints;
- paired versus shuffled task-posterior Fisher-effect preservation.

Secondary measurements cannot rescue a failed primary gate.

## Outcome interpretation

| Outcome | Interpretation |
| --- | --- |
| all gates pass | the selected checkpoints share a functionally transportable causal carrier type despite unrelated neuron/head coordinates |
| coordinates transfer but causal task fails | geometric correspondence is not decoder-compatible causal correspondence |
| task transfers but coordinate R2 fails | a lower-dimensional task statistic transfers, not the full declared three-coordinate carrier |
| paired and shuffled both pass | marginal distributions, not examplewise representation correspondence, explain the result |
| only some directed pairs pass | the carrier atlas is checkpoint-stratified rather than one common chart |
| no paired map passes | keep causal charts checkpoint-local and proceed directly to an architecturally fixed sidecar |

## Evidence and integrity

This is an underpowered three-checkpoint mechanistic study because only three
block-0 fronts have independent cross-cohort stability evidence. It makes no
population-prevalence claim. All predecessor campaign/result hashes,
checkpoint hashes, implementation digest, scientific fingerprints, new data
seeds, maps, controls, and per-cell endpoints must be stored in strict JSON.

A CUDA shakedown may reduce orbit count and directed pairs only with
`systems_lifecycle_only_not_quality_evidence`; it cannot enter the scientific
aggregate. Any implementation change during execution invalidates the root and
requires a clean rerun.

## Planned artifacts

- runner:
  `experiments/structure_net/tinyllm_cross_seed_causal_carrier_transport.py`
- tests:
  `tests/structure_net/test_tinyllm_cross_seed_causal_carrier_transport.py`
- primary root:
  `data/experiments/tinyllm_cross_seed_causal_carrier_transport/20260806_d6_preregistered`
- report:
  `docs/08 - Analysis/2026-08-06_tinyllm-cross-seed-causal-carrier-transport.md`
- meta hypothesis:
  `tinyllm-c2-cross-seed-causal-carrier-transport-v1`

## Amendment A — pre-outcome provenance and lifecycle contracts

Recorded before any shakedown or primary transport artifact existed. The
readout predecessor now must be the authoritative schema-v1.1 post-outcome
corrective campaign and must match its declared evidence role, implementation,
checkpoint, scientific fingerprint, and result hash. Completed campaigns are
byte-immutable on fingerprint/hash-matched resume; incompatible aggregates or
pair records fail closed, while compatible partial pair records may be reused.

Six focused tests cover fixed configuration, whitened transport recovery,
regime-preserving shuffles, the full conjunctive pair gate, authoritative
predecessor loading, and completed-campaign hash validation. These changes do
not alter any seed, cohort, map, threshold, control, or endpoint.
