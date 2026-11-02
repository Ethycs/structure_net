# TinyLLM C3 switching-law corruption fixed-decoder preregistration

**Status:** FROZEN BEFORE PRIMARY COHORT GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE NO-TRAINING IDENTIFIABILITY AND FIXED-CEILING PREFLIGHT`

**Hypothesis:** `tinyllm-c3-switching-corruption-fixed-decoder-v1`

## Decision question

The nested constant-speed/constant-acceleration branch is closed by one robust
quadratic chart. The shortest genuinely new dynamics question is an unknown
law change *inside* one sequence, still under one unmarked frame substitution.

This study asks two ordered questions:

1. Does the target remain identifiable when a velocity change may occur after
   frames `2`, `3`, `4`, or `5` and one of eight frames may be replaced?
2. After repairing that acquisition contract by requiring at least three
   post-change observations, does a frozen exhaustive change-point decoder
   already reach the oracle ceiling?

No TinyLLM, learned selector, checkpoint, optimizer, or reusable fitted
parameter is permitted. Training is considered only if the repaired target is
identifiable, the oracle passes, and the strongest registered fixed decoder
fails.

## Declared switching law

For frames `t=0,...,8`, define

```text
theta_t = phi + v*t + delta_v*max(t-s, 0).
```

The velocity changes after observed frame `s`; frames `s+1,...,7` are the
post-change observations and frame `8` is the forecast. The exact invariant
carrier is

```text
q_t = exp(i*3*theta_t),
target = cos(3*theta_8).
```

The observation, three-channel `C3` deck action, quantizer, calibration packet,
and physical sixteen-bin decoder are inherited unchanged. The switch and
corruption indices are hidden from the primary decoder.

## Exact identifiability contract

For switch `s`, let the unwrapped carrier-phase design row be

```text
X_s(t) = [1, t, max(t-s, 0)].
```

For every pair of allowed switches and every subset of the eight observed
times, compute over exact rational arithmetic the nullspace of

```text
[X_s(t), -X_u(t)].
```

Among pairs whose phase at `t=8` differs, record the maximum number of equal
observed coordinates. The induced clean-trajectory Hamming distance is

```text
d_min = 8 - maximum_equal_coordinates.
```

Correcting one arbitrary coordinate requires `d_min >= 3`.

Two supports are locked:

| Support | Required exact result | Meaning |
| --- | ---: | --- |
| late-inclusive `{2,3,4,5}` | `d_min = 2` plus an explicit target-changing collision | target does not descend through one-frame corruption |
| primary `{2,3,4}` | `d_min = 3` | one arbitrary corrupted frame is uniquely correctable at the future-target level |

The late-inclusive witness must lie inside the composition parameter support.
Use common `v=.08`, switches `s_A=4`, `s_B=5`, and changes
`delta_v_A=.05`, `delta_v_B=.10`. Choose `phi` so
`3*theta_A(8)=pi/2`. The trajectories agree at frames
`0,1,2,3,4,6` and differ only at `5,7`. Construct one observed sequence using
trajectory B at frame `5` and trajectory A at frame `7`. It is obtainable from
A by one substitution at `5` and from B by one substitution at `7`, while the
two future cosine targets must differ by at least `.25`.

Failure of either exact contract invalidates the study. The late-inclusive
scope can never license training under the declared observation relation.

## Fresh primary population

The repaired primary support is `s in {2,3,4}`. Use five fresh seeds, disjoint
from all predecessor primary and pilot cohorts:

```text
337, 347, 359, 373, 389
```

Generate `4,096` examples per seed and shift from:

| Stream | Composition | Extrapolation |
| --- | ---: | ---: |
| base data and switch | `941107 + seed` | `943107 + seed` |
| donor derangement | `951107 + seed` | `953107 + seed` |
| corrupted frame | `955107 + seed` | `957107 + seed` |
| target derangement | `961107 + seed` | `963107 + seed` |

Sample `phi`, amplitude, offset, drift, deck element, and signed pre-change
velocity from the inherited constant-speed composition/extrapolation ranges.
Sample the signed velocity-change magnitude independently from:

| Shift | `abs(delta_v)` |
| --- | ---: |
| composition | `[.05, .12]` |
| extrapolation | `[.10, .18]` |

Sample the primary switch uniformly from `{2,3,4}`. Sample one corrupted frame
uniformly from `0,...,7` and replace all three quantized channels with the
same-time frame from a Sattolo-deranged donor within the same seed and shift.
Every switch count must be at least `1,200`; every frame count at least `400`.

Implementation tests may use `64` examples from lifecycle seed `997` with
separate base streams `971107 + seed` and `973107 + seed`. Lifecycle metrics
are never interpreted against scientific gates.

## Frozen estimators

All fits operate independently per observed sequence, use no target, and retain
no parameter across examples.

1. `clean_known_switch`: fit `X_s` to all eight clean carrier phases using the
   true switch; positive control.
2. `clean_global_quadratic`: apply the predecessor robust delete-one quadratic
   chart to the clean switching sequence; tests whether switching changes the
   old function class.
3. `corrupted_global_quadratic`: apply that same old robust chart after the
   hidden substitution.
4. `corrupted_known_switch_no_drop`: fit the true switch to all eight corrupted
   observations; isolates corruption materiality.
5. `oracle_switch_drop`: fit `X_s` after deleting the true corrupted frame and
   using the true switch.
6. `fixed_switch_drop`: enumerate all `3 x 8 = 24` switch/deletion pairs and
   select minimum retained mean-square phase residual.

Phase unwrapping uses the principal carrier-phase increment. After true-frame
deletion, the minimum clean chart margin must be at least `.20`. Latent switch
or corruption-index recovery is descriptive only: alternative candidates may
represent the same future, so validity is defined by exact future equivalence,
not hidden-label recovery.

## Primary endpoints

Retain the inherited complete task gate:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14` | `>=12` |

The strong fixed ceiling additionally requires:

```text
scalar RMSE <= .020
exact-bin accuracy >= .90
complete task gate passes.
```

For a seed to pass an endpoint, both composition and extrapolation must pass.
Require at least `4/5` seeds jointly.

| Endpoint | Seed-level requirement |
| --- | --- |
| dynamics materiality | `clean_known_switch` passes and `clean_global_quadratic` fails the strong ceiling |
| corruption materiality | `clean_known_switch` passes and `corrupted_known_switch_no_drop` fails |
| oracle recoverability | `oracle_switch_drop` passes |
| fixed closure | `fixed_switch_drop` passes |
| Pareto repair | fixed/global-corrupted RMSE ratio `<=.50`, accuracy delta `>=-.005`, cross-entropy delta `<=-.10` |
| oracle fidelity | fixed RMSE excess `<=.002`, accuracy delta `>=-.01`, cross-entropy excess `<=.005` |

## Controls and validity

- Base, corruption, and target-shuffle streams must regenerate exactly.
- Quantizer saturation must be zero.
- Exact `C3` identity, composition, order, stored-action, latent-regeneration,
  and target-invariance contracts must pass.
- Donor and target derangements must have zero fixed points.
- Every switch and corruption-frame count must pass its declared floor.
- Corruption must commute exactly with the deck action.
- Every estimator must be deck invariant within `2e-12`.
- On continuous corrupted carriers, oracle and fixed predictions must equal the
  exact time-8 carrier within `1e-10` for every primary example.
- The exact late-inclusive collision must survive continuous observation,
  quantization, and token-level substitution, with observed token equality and
  future-target separation `>=.25`.
- Every shuffled-target arm must have absolute scalar correlation `<=.10`,
  scalar RMSE `>=.80`, and fail the complete task gate.
- All values must be finite strict JSON.
- Law-label reads by the fixed decoder, reusable fits, target-using fits,
  models, checkpoints, optimizer steps, and changed parameters must be zero.

Any validity failure prevents a scientific conclusion.

## Locked classifications

Use the first matching valid row:

| Outcome | Classification | Program decision |
| --- | --- | --- |
| late support has distance 2; primary support has distance 3; dynamics materiality, corruption materiality, oracle, fixed closure, Pareto repair, and fidelity each pass `>=4/5` | `fixed_change_point_decoder_closes_identifiable_switching_corruption` | close TinyLLM on the repaired scope; retain the late-switch impossibility |
| either global quadratic arm passes `>=4/5` | `switching_corruption_scope_not_material` | do not train |
| oracle passes `<4/5` | `identifiable_switching_not_recoverable_at_required_ceiling` | repair observation precision; do not train |
| oracle passes `>=4/5` but fixed closure passes `<4/5` | `recoverable_switching_exceeds_fixed_change_point_decoder` | license one compact typed continuation comparison, not unrestricted TinyLLM |
| any other valid combination | `inconclusive_switching_corruption_preflight` | inspect the joint gate without tuning on these cohorts |
| any validity failure | `invalid_switching_corruption_preflight` | repair infrastructure only |

`tinyllm_training_licensed` is always false. Only the recoverable-fixed-failure
row may set `compact_typed_continuation_comparison_licensed=true`.

## Disclosed pre-registration pilot

Before this document was frozen, two excluded `1,024`-example execution pilots
used base seeds `77811` and `77813` plus corruption seed `991`. They showed that
late switch support produced ambiguous minimum-residual candidates and that the
three-post-change-frame support was executable. They are not pooled, do not
set thresholds, and cannot satisfy a primary gate. Every numeric threshold
above is inherited from predecessor fixed-operator/corruption studies; the
support boundary is fixed by the exact rational distance calculation.

## Frozen source lineage

| Source | SHA-256 |
| --- | --- |
| fresh corrective runner | `6d8e408f2347b86440ba6307345b86cad9f2740ee0945c358dea122d1c8d2789` |
| fresh corrective result | `bd5af6550e65c0b9048030a8f6d01cf80e06e9f027edb89066d8338bb8b47c2a` |
| single-frame corruption runner | `8600081fc813ae6da5a49c39222bf3a94286127d7238899d61ec9acd1c6d31f8` |
| C3 observation/action generator | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| physical interval likelihood | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |

The implementation must pin this preregistration and all listed sources before
primary generation.

## Expected artifact and accounting

```text
data/experiments/tinyllm_c3_switching_corruption_fixed_decoder/
  20260811_preregistered/result.json
```

```text
fresh base examples:                         40,960
fresh corrupted evaluations:                 40,960
observation-only candidate fits:           1,720,320
continuous validation candidate fits:        983,040
pre-registration pilot examples pooled:            0
models / checkpoints / optimizer steps:       0 / 0 / 0
changed parameters / target-using fits:        0 / 0
```
