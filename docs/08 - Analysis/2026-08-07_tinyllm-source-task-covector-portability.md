# TinyLLM source task-covector portability

**Status:** NOT CONFIRMED — COVECTOR PORTABLE, SIGNED AMPLITUDE NOT PORTABLE  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, fresh-cohort post-outcome mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-source-task-covector-portability-v1`  
**Schema:** `nal.tinyllm-c2-source-task-covector-portability.v1`  
**Preregistration:** [source task-covector portability preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-source-task-covector-portability-preregistration.md)

## Verdict

The full source-only correction hypothesis is **not confirmed**: the
preregistered portability gate passed `0/3` checkpoints. The result nevertheless
cleanly separates two parts of the previously observed local mechanism.

The task covector is portable. A nine-feature phase map fit only on source
cohorts A/B predicted the signed fresh-cohort decoder gradient with
zero-referenced R2 `0.989--0.996` and mean row cosine `0.9969--0.9995`.
Supplying that source-predicted covector with the fresh signed output error
repaired all six cohort-C composition and extrapolation cells, just as the
fully local oracle did.

The signed correction amplitude is not portable under the tested phase-only
map. Its fresh zero-referenced R2 was `-0.053--0.041`, with only
`54.8--59.8%` sign agreement. Consequently, the completely source-predicted
correction passed only `1/6` cells, the same cell count as the uncorrected
order-4 baseline. The fixed classification was therefore
`source_covector_portable_scalar_not` in all three checkpoints.

This is a genuine narrowing of the mechanism: TinyLLM's decoder-sensitive
local metric field is a stable function of quotient phase on a new cohort, but
the example-specific displacement along that field is not. The next shortest
test is an observable scalar residual sensor; another richer covector or global
writer is not justified by these data.

## Campaign integrity

| Item | Measured value |
| --- | --- |
| frozen checkpoints requested/completed | `3/3` (`7`, `29`, `53`) |
| failures, exclusions, retries | `0`, `0`, `0` |
| TinyLLM models trained | `0` |
| writers trained | `0` |
| small predictive maps fit | `6` (`g_hat` and `y_hat` per checkpoint) |
| source-map fit examples | `768` orbit examples across A/B |
| fresh primary cells | `6` (composition and extrapolation per checkpoint) |
| fresh seeds | composition `430007`; extrapolation `430008` |
| device | NVIDIA GeForce RTX 3060 |
| PyTorch / CUDA build | `2.5.1+cu121` |
| peak allocated CUDA memory | `282,743,808` bytes |
| aggregate analysis time | `13.96` seconds |
| implementation SHA-256 | `6716b909d0c245059a1ed1310f20f4d9e56deb8c49a7d3a031972542fccb3046` |

The primary root is distinct from the one-checkpoint systems-only CUDA
shakedown. All results use one producing implementation. Strict JSON parsing
passed, every aggregate result hash matches its referenced run, and an exact
resume left the campaign and all three result files byte-identical.

The checkpoints are the replication units. Three selected checkpoints remain
underpowered and were chosen after earlier outcomes; the six fresh cells are
repeated measurements, not six independent model replicates.

## Primary endpoints

Values below are the aggregate mean circular-moment shift from the exact state
across fresh composition and extrapolation; lower is better. A state must also
pass the complete endpoint in both cells, including p95 shift, alignment,
winding, and sampling resolution.

| Seed | order-4 baseline | local oracle | source `g_hat` + fresh `y` | fresh `g` + source `y_hat` | source `g_hat` + source `y_hat` | full gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 0.12497 | 0.02343 | 0.02443 | 0.12254 | 0.12286 | fail |
| 29 | 0.19179 | 0.04254 | 0.03964 | 0.19199 | 0.19022 | fail |
| 53 | 0.15161 | 0.01732 | 0.01934 | 0.14700 | 0.14639 | fail |

The local oracle and source-covector/fresh-error intervention passed `6/6`
fresh cells. The fresh-covector/source-error and fully source-predicted
interventions passed only seed 7 extrapolation, or `1/6` cells each. The
order-4 baseline passed that same single cell.

### Per-cell intervention pass matrix

| Seed | Regime | order 4 | local oracle | source covector + fresh error | fresh covector + source error | fully source predicted |
| ---: | --- | --- | --- | --- | --- | --- |
| 7 | composition | fail | pass | pass | fail | fail |
| 7 | extrapolation | pass | pass | pass | pass | pass |
| 29 | composition | fail | pass | pass | fail | fail |
| 29 | extrapolation | fail | pass | pass | fail | fail |
| 53 | composition | fail | pass | pass | fail | fail |
| 53 | extrapolation | fail | pass | pass | fail | fail |

## Preregistered gates

| Gate | Required | Result | Verdict |
| --- | --- | --- | --- |
| provenance and numerical contracts | all checkpoints | `3/3` | pass |
| source local linearization | all checkpoints | `3/3` | pass |
| fresh local linearization | all checkpoints | `3/3` | pass |
| zero fails; exact and direct rank 3 pass both fresh cells | all checkpoints | `3/3` | pass |
| local oracle passes both fresh cells | all checkpoints | `3/3` | pass |
| source covector with fresh error passes both fresh cells | all checkpoints | `3/3` | pass |
| fresh covector with source error passes both fresh cells | all checkpoints | `0/3` | fail |
| completely source-predicted correction passes both fresh cells | all checkpoints | `0/3` | fail |
| every negative control fails a cell and trails by at least 0.125 bins | all checkpoints | `0/3` | fail |
| complete joint source-covector portability gate | `3/3` | `0/3` | **fail** |

The specificity gate cannot rescue or weaken the failed primary endpoint. All
four controls failed at least one fresh cell in every checkpoint, but their
aggregate margins over the already ineffective source-only intervention were
usually below `0.125` bins. This is expected when `y_hat` supplies little
usable correction signal.

## Mechanistic measurements

### The covector field transports

| Seed | source covector R2 | fresh covector R2 | fresh mean signed cosine | fresh relative L2 |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.9950 | 0.9963 | 0.99948 | 0.0608 |
| 29 | 0.9840 | 0.9887 | 0.99688 | 0.1061 |
| 53 | 0.9903 | 0.9934 | 0.99896 | 0.0815 |

The constant source-mean covector was not an adequate substitute: it failed all
six fresh cells and produced aggregate shifts from `0.209` to `2.366` bins.
This secondary control is consistent with a phase-varying metric field rather
than one global correction axis.

### The signed amplitude does not transport

| Seed | source signed-error R2 | source sign agreement | fresh signed-error R2 | fresh sign agreement |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.1067 | 63.4% | 0.0407 | 59.8% |
| 29 | 0.1021 | 64.5% | -0.0525 | 54.8% |
| 53 | 0.0702 | 58.9% | -0.0043 | 58.4% |

The source-only correction's mean standardized norm was only
`27.5--45.6%` of the corresponding local-oracle norm across fresh cells. This
under-correction is consistent with the near-zero scalar fit, but norm alone
does not explain wrong-sign examples.

### Local causal geometry reproduces

The source and fresh finite-difference derivatives were numerically stable:
fine/coarse cosines exceeded `0.9999991`. The direct residual's first-order
signed-error model had source R2 `0.986--0.997` and fresh R2 `0.993--0.997`.
Thus the scalar-map failure is not attributable to a nonlocal fresh residual or
an unstable derivative estimate. Both local-oracle interventions independently
closed all six cells.

## Interpretation and boundaries

The result rejects a monolithic claim that the whole task correction is a
function of quotient phase. It supports a more precise decomposition:

```text
portable phase-conditioned task covector
  + example-specific signed residual amplitude
  -> successful local causal correction.
```

This explains why the preceding tangent interventions were highly repeatable
yet did not immediately yield a deployable sidecar. The expensive geometric
part—the direction downstream computation reads—appears predictable and
stable. The unresolved part is a one-dimensional signed displacement.

The next experiment should therefore fit the smallest observable scalar sensor
for `y` while keeping `g_hat` frozen. Candidate inputs should be tested in
increasing-cost order: order-4 continuation confidence/residual statistics,
the already observed calibration packet, then a minimal local activation
summary. Each candidate must be trained on A/B and evaluated once on a new D
cohort. If no observable scalar predicts sign and magnitude, stop attempting a
source-only sidecar and retain the local covector as an explanatory diagnostic.

Important boundaries remain:

- `g` is conditioned on the frozen answer-token decoder and circular angle;
- the source map consumes an oracle quotient-phase chart inherited from the
  predecessor, so this is not yet an observable front end;
- fresh C changes generator seeds within known composition/extrapolation
  families rather than adding a new shift family;
- source cohorts and checkpoints were selected after prior results;
- rank-three patches are local off-manifold interventions and do not prove
  TinyLLM naturally computes or uses the fitted map; and
- three selected checkpoints do not establish population prevalence.

The conformance profile is met as an underpowered preregistered diagnostic.
The study does not meet the standard's recommended five-seed power level and
cannot make a population-level confirmatory claim.

## Artifacts and reproduction

- campaign:
  `data/experiments/tinyllm_source_task_covector_portability/20260807_d6_preregistered_fresh_cohort/campaign_results.json`
- per-seed records:
  `data/experiments/tinyllm_source_task_covector_portability/20260807_d6_preregistered_fresh_cohort/runs/seed_*/result.json`
- systems-only shakedown:
  `data/experiments/tinyllm_source_task_covector_portability/20260807_shakedown_cuda/`
- campaign SHA-256:
  `fe153abd0aa1749a862095e577e6e9cf8a0f7054bade2ef3c6833bf32d7f4ac5`
- result SHA-256 values:
  seed 7 `62fad2de528c9a31d79ae95ab179b9e70365212a0463d6f0c8a891e13105460c`;
  seed 29 `b483d94e3305b17afa660e08cd9da0b80bacd26a0c40a5dbec072282dd183b91`;
  seed 53 `a5a466f77c8f1561699f9eb3004103ea7d7b4a2f8af7fc092b6b7d299199d6ad`

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_source_task_covector_portability \
  --device cuda \
  --output \
  data/experiments/tinyllm_source_task_covector_portability/20260807_d6_preregistered_fresh_cohort
```

The final DVC root is `9f9077c17fbbc668805088bf604deafc.dir`, covering
`39,816,811,567` bytes in `1,904` files; `dvc status` reports data and
pipelines up to date. The root was pushed to the configured
`lakefs://artifacts/main/structure-net/` remote and sealed in clean lakeFS
commit `8eccad2c763ea0230fde1e484b2d8c631dbe91524799c21920686bd23d704872`.

The conservative meta-hypothesis record is stored at
`data/meta_hypotheses/tinyllm-c2-source-task-covector-portability-v1.json`.
ChromaDB read-back verified the stable hypothesis ID and all three direct
experiment records. Legacy ChromaDB emitted telemetry and NumPy-compatibility
warnings during the write, but the required read-back contract passed.
