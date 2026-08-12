# TinyLLM C3 gauge-jump joint typed-score result

**Status:** VALID FRESH RESULT — BOTH FIXED CONNECTION SCORES CLOSE THE NEW POPULATION

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-gauge-jump-joint-typed-score-v1`

**Classification:** `both_fixed_connection_scores_close_fresh_scope`

**Preregistration:** [joint typed score](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_gauge_jump_joint_typed_score/20260811_preregistered/result.json`

## Verdict

The hidden within-sequence `C3` gauge jump does not create learned-model rent
under the declared observation law. On five wholly fresh seeds:

```text
invariant oracle:                         5/5
fixed invariant:                          5/5
charged connection oracle:                5/5
original fixed charged connection:        4/5
joint invariant/charged fixed score:       5/5
joint score oracle fidelity:               5/5
charged without connection:                0/5
required:                                >=4/5
```

The preregistered classification is conservative. The joint score passes all
ten fresh cells, but the original charged-only selector independently reaches
the population gate at `4/5`. Therefore the experiment does **not** confirm a
unique causal tail repair by the added invariant residual. It confirms the
broader result that fixed symmetry-typed estimators close the time-varying
gauge scope. Neither the compact chart-mixture model nor TinyLLM training is
licensed.

## Relation to the predecessor

The preceding valid primary found original fixed charged closure in only `1/5`
seeds, while both oracles and the invariant fixed comparator passed `5/5`.
That was a real result for its sealed population and remains preserved. A
post-outcome diagnostic proposed the no-fit score

```text
R_joint = R_charged + R_invariant / 9,
```

where the factor nine converts squared cubic-phase residuals to the local
charged-phase scale. The score repaired all ten predecessor cells, but those
examples contributed zero evidence here.

On fresh data, the original fixed charged selector fails only seed 821 because
its composition RMSE is `.020163`, narrowly above the `.020` ceiling. Its
extrapolation cell passes, so no seed satisfies the strict two-shift
material-repair conjunction. The typed score reduces that composition RMSE to
`.004291`, but the registered typed-repair count is consequently `0/5`, not
`4/5`.

The correct cross-study conclusion is:

> The charged-only hard selector has a sampling-sensitive rare-error tail.
> An independently normalized invariant term removes the observed fresh tail,
> but this population does not establish that the term is necessary.

## Fresh results

Per-cell scalar RMSE:

| Seed | Shift | Original charged | Joint typed | Joint pass |
| ---: | --- | ---: | ---: | ---: |
| 773 | composition | `.004117` | `.004117` | pass |
| 773 | extrapolation | `.004408` | `.004408` | pass |
| 821 | composition | `.020163` | `.004291` | pass |
| 821 | extrapolation | `.004361` | `.004361` | pass |
| 1003 | composition | `.004268` | `.004268` | pass |
| 1003 | extrapolation | `.004290` | `.004290` | pass |
| 1031 | composition | `.004091` | `.004091` | pass |
| 1031 | extrapolation | `.004392` | `.004392` | pass |
| 1039 | composition | `.004185` | `.004184` | pass |
| 1039 | extrapolation | `.004542` | `.004542` | pass |

Joint typed exact-bin accuracy ranges from `.9758` to `.9810`, and target
cross-entropy ranges from `1.2753` to `1.2856`. All ten cells pass the complete
task gate and the stronger fixed ceiling.

## Mechanistic lesson

The physical and gauge decisions transform differently:

```text
invariant cubic carrier       -> physical switch/deletion evidence
charged first character       -> connection and phase evidence
```

Combining their normalized residuals is a legitimate typed construction, not
an unstructured feature fit. Its perfect fresh closure shows that the extra
information can be used without breaking exact deck invariance. Its failure to
meet the strict material-repair gate prevents the stronger statement that the
combined score is required.

This sharpens the program's representation-ordering rule:

```text
retain charge through gauge inference,
use invariant evidence for gauge-immune physical decisions,
project to the invariant target only at the output.
```

## Contracts and accounting

| Contract | Result |
| --- | ---: |
| fresh requested / completed / invalid cells | `10 / 10 / 0` |
| fresh base / corrupted examples | `40,960 / 40,960` |
| predecessor or diagnostic examples pooled | `0` |
| exact connection integer-action errors | `0` |
| maximum all-arm deck-action error | `1.692e-12` |
| joint typed deck-action error | `0.0` in every cell |
| maximum continuous forecast error | `4.871e-12` |
| maximum stabilization displacement | `7.050e-13` |
| minimum invariant / connected chart margin | `.87098 / 2.38472` |
| maximum shuffled absolute correlation | `.02842` |
| minimum shuffled RMSE | `.98385` |
| models / checkpoints / optimizer steps | `0 / 0 / 0` |
| reusable or target-using fits | `0 / 0` |

All exact regeneration, coverage, derangement, jump inverse, global action,
corruption commutation, task, shuffle, strict-JSON, and finite-value contracts
pass. The candidate bank is shared by both fixed connection scores, so adding
the typed score adds no candidate fits.

## Program decision

Close both learned branches for this generator:

- `compact_typed_chart_mixture_licensed=false`;
- `tinyllm_training_licensed=false`.

The time-varying gauge experiment therefore strengthens the theory without
creating a model requirement. A new study must change scope—for example to
multiple jumps, an unknown group, partial group observations, or a connection
whose admissible estimator is not analytically enumerated—and must repeat the
identifiability and fixed-estimator audits before training.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-gauge-typed-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_gauge_jump_joint_typed_score
```

| Artifact | SHA-256 |
| --- | --- |
| fresh result | `f52ce2103a07086a7118975d69f49b7cbeca01ac0ca7c5a15fd6d2a96fbc51fa` |
| runner | `6a9b1b849c97fc30bef7292ed0bbc097c4829db2920d3abe8f698e7546489944` |
| preregistration | `e4629a11cac991b1bd64d641f3276b4517296ee31ac3b9e0a3837e5cb5ce4663` |
| predecessor result | `16f98f5c3cbf09fedfc18f12eca24a5fe69da46411d587c48c5d9072c912aca7` |

## Intermediate evidence backup checkpoint

Before the final report-hash provenance record was written, the repository
data tree was checkpointed at DVC root
`52be9186ed44f5c631491cec02a20800.dir` (`54,464,856,385` logical bytes,
`4,143` files). DVC pushed ten new objects and an immediate replay reported
`Everything is up to date`.

lakeFS commit
`1d399a19bb4032b0926f356a44f98b37859c6177def8bf835ff19c2e3fcc00b8`
seals the object graph on `artifacts/main`, with parent
`d96febecfa829fc860965429d33173c98cbfa6d97bc160ba079a3614175e14bc`.
The branch diff is empty after commit. Direct object inspection recovers the
DVC root checksum `52be9186ed44f5c631491cec02a20800` and a `666,418`-byte
directory manifest.

The final meta-hypothesis-inclusive DVC root and subsequent clean lakeFS commit
are recorded separately in the [evidence backup receipt](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-gauge-jump-evidence-backup.md), avoiding a self-referential report-hash/data-root cycle.
