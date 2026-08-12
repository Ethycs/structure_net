# TinyLLM C3 hidden gauge-jump fixed-decoder result

**Status:** VALID FRESH RESULT — ORACLE CONNECTION RECOVERABLE; JOINT FIXED CONNECTION FAILS

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-gauge-jump-corruption-fixed-decoder-v1`

**Classification:** `recoverable_time_varying_gauge_exceeds_fixed_connection_decoder`

**Preregistration:** [hidden gauge-jump fixed decoder](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-gauge-jump-corruption-fixed-decoder-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_gauge_jump_corruption_fixed_decoder/20260811_preregistered/result.json`

## Verdict

One hidden within-sequence `C3` frame jump creates a real charged-connection
inference problem, but it does not make the physical invariant target hard.
Across five wholly fresh seeds:

```text
invariant oracle recoverability:       5/5
fixed invariant closure:               5/5
charged connection oracle recovery:    5/5
fixed charged connection closure:      1/5
charged connection oracle fidelity:    1/5
charged decoder without connection:    0/5
required:                             >=4/5
```

The result is valid and the registered classification licenses one compact
typed gauge/physical chart comparison. It never licenses unrestricted
TinyLLM training.

The scientific distinction is important. Framewise cubing removes the hidden
deck jump exactly, so the invariant 24-way switch/deletion selector remains a
complete fixed solution. The charged carrier instead requires a connection to
compare phases across the jump. Supplying the true connection works in every
seed, while selecting the connection jointly with the physical chart from 288
candidates produces rare large forecast errors.

## Primary results

Means over five seeds:

| Arm | Composition RMSE / accuracy | Extrapolation RMSE / accuracy | Joint seeds |
| --- | ---: | ---: | ---: |
| fixed invariant switch/drop | `.006391 / .9786` | `.004723 / .9793` | `5/5` |
| fixed charged, no connection | `.874068 / .2003` | `.866700 / .2010` | `0/5` |
| fixed charged connection | `.012524 / .9786` | `.013289 / .9793` | `1/5` |
| oracle charged connection | approximately `.0041 / .9786` | approximately `.0045 / .9793` | `5/5` |

The fixed charged connection misses the `.020` scalar-RMSE ceiling in four
cells:

| Seed | Shift | Fixed connection RMSE | Exact-bin accuracy |
| ---: | --- | ---: | ---: |
| 673 | extrapolation | `.026572` | `.9802` |
| 691 | composition | `.025864` | `.9812` |
| 709 | composition | `.024107` | `.9802` |
| 751 | extrapolation | `.026324` | `.9773` |

All four still have high exact-bin accuracy and pass the broader task gate.
The registered fixed ceiling deliberately catches the rare continuous tails
that bin accuracy hides. Seed 727 is the only seed whose fixed connection
passes both shifts.

## Mechanistic interpretation

The 288-way residual score asks one charged coordinate system to choose two
conceptually different objects at once:

```text
physical chart:       corrupted frame and dynamics switch
gauge connection:     jump time and deck element
```

Exact hidden-tuple recovery is not required for the physical endpoint and is
only about `.78-.79`. Nevertheless, the no-connection control has RMSE around
`.87`, while the oracle connection has ordinary quantization-scale error.
The connection is therefore causally material even though its latent label is
not the endpoint.

The invariant comparator resolves the physical chart with selected-oracle
agreement around `.953` and passes all ten cells. That suggests the failure is
not absent physical information or insufficient observations. It is a score-
factorization problem: charged residual alone occasionally trades a better
connection fit against the wrong physical chart.

## Validity and controls

| Contract | Result |
| --- | ---: |
| fresh requested / completed / invalid cells | `10 / 10 / 0` |
| fresh base / corrupted examples | `40,960 / 40,960` |
| predecessor or audit examples pooled | `0` |
| exact connection integer-action errors | `0` |
| maximum all-arm deck-action error | `1.750e-12` |
| maximum continuous forecast error | `5.224e-12` |
| maximum stabilization displacement | `7.051e-13` |
| minimum invariant / connected chart margin | `.86713 / 2.38344` |
| maximum shuffled absolute correlation | `.03963` |
| minimum shuffled RMSE | `.98501` |
| models / checkpoints / optimizer steps | `0 / 0 / 0` |
| reusable or target-using fits | `0 / 0` |

All dataset replay, jump inverse, global action, corruption commutation,
coverage, derangement, task, shuffle, strict-JSON, and finite-value contracts
pass.

## Shortest next test

Do not train the licensed compact continuation first. A cheaper typed score is
available from the same observations:

```text
score(connection, physical chart)
    = charged phase residual
    + invariant physical-chart residual / 9.
```

Cubing triples local phase error, so dividing its squared residual by nine
places the invariant term on the charged phase-variance scale. This keeps the
charged carrier responsible for the connection while giving the gauge-immune
carrier an independent vote on physical dynamics.

As a disclosed post-outcome diagnostic, this fixed score repaired all ten
sealed predecessor cells, reducing every RMSE to `.00407-.00460`. Those cells
are not confirmatory evidence and must not be pooled. Freeze the score and
repeat it on fresh seeds and streams. If it closes at least four of five seeds,
close the learned comparison; if it fails while the connection oracle remains
recoverable, execute exactly one compact typed chart-mixture model.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-gauge-jump-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_gauge_jump_corruption_fixed_decoder
```

| Artifact | SHA-256 |
| --- | --- |
| result | `16f98f5c3cbf09fedfc18f12eca24a5fe69da46411d587c48c5d9072c912aca7` |
| runner | `5b35658103481645aba809f5575d38159dcddc9dc7330ebfa6764ad65ba170a4` |
| preregistration | `caab4704bb5c11a3f7353c31e20a80da33f14e1a925402df055b652916de10d9` |

