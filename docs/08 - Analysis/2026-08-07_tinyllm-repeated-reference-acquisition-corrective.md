# TinyLLM repeated-reference acquisition corrective replication

**Status:** VALID CORRECTIVE RESULT — ACQUISITION REPAIR SUPPORTED, NOT CONFIRMATORY  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT, frozen-system outcome-informed corrective intervention  
**Hypothesis:** `tinyllm-repeated-reference-acquisition-v1`  
**Schema:** `nal.tinyllm-repeated-reference-acquisition.v1`  
**Specification:** [repeated-reference acquisition provenance record](../07%20-%20Status%20Reports/2026-08-07_tinyllm-repeated-reference-acquisition-preregistration.md)

## Verdict

Repeated independent measurements of the observed orientation reference repair
the unchanged TinyLLM computation. Both the analytic circular mean and a shared
label-free `SO(2)`-equivariant Noise2Noise aggregator pass the registered task
gate in both structured front-end arms. No TinyLLM, front-end, answer-head,
probe, or observer parameter changes.

The locked campaign classification is:

```text
acquisition_variance_causally_sufficient
```

This is **corrective, outcome-informed evidence**, not fresh confirmation. The
analytic one-seed result was inspected before the learned aggregator and
expanded gates were finalized. A separate earlier preregistered campaign
already supports the same direct acquisition mechanism but remains globally
`invalid` because its retained one-step residual-write ceiling failed. The two
campaigns must remain distinct.

## Population result

Each cell is the number of checkpoints staying within three accuracy points of
its own exact-reference clean baseline on both composition and extrapolation.

| Aggregator | Repeats | Analytic front end | Learned equivariant front end |
| --- | ---: | ---: | ---: |
| analytic circular mean | `1` | `0/5` | `0/5` |
| analytic circular mean | `4` | `0/5` | `0/5` |
| analytic circular mean | `16` | `0/5` | `0/5` |
| analytic circular mean | `64` | **`5/5`** | **`4/5`** |
| analytic circular mean | `256` | **`5/5`** | **`5/5`** |
| learned equivariant | `1` | `0/5` | `0/5` |
| learned equivariant | `4` | `0/5` | `0/5` |
| learned equivariant | `16` | `0/5` | `0/5` |
| learned equivariant | `64` | **`5/5`** | **`4/5`** |
| learned equivariant | `256` | **`5/5`** | **`5/5`** |

The population gate is four of five, so both methods and both systems pass at
`m=64`. The learned-front-end seed-53 checkpoint first passes at `m=256`; this
places the `m=64` threshold close to a checkpoint boundary rather than making
64 a universal per-checkpoint constant.

## Causal interpretation

At `sigma=0.175` radians, the analytic angular RMSE follows the expected
standard-error law:

| Shift | measured log-log slope | allowed interval |
| --- | ---: | ---: |
| composition | `-0.4766` | `[-0.60,-0.40]` |
| extrapolation | `-0.4818` | `[-0.60,-0.40]` |

The measurement intervention supplies a dose-response:

```text
m <= 16: reference remains too imprecise; task gate fails 20/20 method-arm populations
m = 64: effective sigma 0.021875 rad; all four populations pass
m = 256: effective sigma 0.0109375 rad; every checkpoint passes
```

The learned aggregator does not reveal a useful non-Gaussian correction. It
matches the analytic circular mean at the population endpoint in all ten
checkpoints. Under independent homoscedastic Gaussian angular error, the
analytic sufficient statistic is already adequate.

This triangulates the earlier decomposition:

```text
quotient representation survives
    -> one noisy observed coordinate is too imprecise
    -> coherent reference acquisition reduces coordinate variance
    -> the unchanged front end and TinyLLM recover the task
```

The result supports a measurement-precision bottleneck, not a claim that the
transformer learned a universally stable quotient or that a post-hoc residual
write is globally valid.

## Controls and integrity

- exact-reference oracle: `5/5` in each arm;
- single noisy observation: `0/5` in each arm;
- fiber-shuffled `m=256` reference: `0/5` in each arm;
- analytic and learned `m=1` packets agree within `5.97e-8`;
- pair-shared angular error agrees within `1.09e-7`;
- maximum orientation-unit-norm error is `9.68e-8`;
- learned aggregator permutation and rotation errors are below `1.8e-7`;
- all ten system-state hashes are unchanged;
- exact resume reports the campaign complete and leaves bytes unchanged; and
- `16` focused runner tests pass.

The runner's implementation-digest guard also stopped earlier partial roots
when concurrent source edits occurred. Those roots are lifecycle debris, not
quality evidence. The authoritative replay is
`20260807_d8_corrective_expanded_v8`: it uses a protocol-scoped digest that
still invalidates scientific changes while excluding non-scientific CLI
default-path edits. Exact resume revalidated every artifact and left the tree
byte-identical. The earlier `20260807_d8_corrective_v4` run produced the same
population result and identical learned parameter state on the same acquisition
array; it is retained as a predecessor execution, not counted as an independent
replication.

## Relationship to the preregistered acquisition campaign

The earlier [preregistered acquisition report](2026-08-07_tinyllm-reference-acquisition-replicates.md)
used acquisition seed `23711`, a different equivariant estimator, and counts
through `64`. Its direct acquisition endpoint passed `5/5` in both arms at
`m=64`, but the full campaign was invalidated by a nonportable one-step
residual-write positive control.

This corrective campaign uses acquisition seed `42700019`, counts through
`256`, and an acquisition-only Noise2Noise set aggregator. It again recovers
both population arms at `m=64`, with one learned-front-end checkpoint requiring
`m=256`. Together the results support the acquisition mechanism across two
stored repeat arrays while showing that the exact per-checkpoint threshold is
draw-sensitive.

## Provenance

| Item | SHA-256 / value |
| --- | --- |
| campaign | `e045849e2c7abb19fc682d826de912760a29c38636c151651f2e81ed363b4832` |
| implementation | `2693f5d42d460a3b43ac2951a895545b7d565a81487b69c9988a36d78da0f40c` |
| runner protocol | `9257bcedcdfd6a10f244f8c373839a51180a8c3070fbf2c61d3f6edb1da6dd44` |
| ten-result manifest | `4cae9415fb7000d35ea8f5f3e6b4ba5453746245e12194de3d13378c2db96b61` |
| exact-resume tree manifest | `d1f8afc3d681ba7618777adbeca15082a25cfe4350e74e6ebac21e969bed09b5` |
| acquisition arrays | `895d398e4ca31611f04f92921e1169704eeba6292813de5b3015415dbd746fec` |
| learned aggregator checkpoint | `6e041c0111667f71f4d3c85f2f62ca29cc2d376f496945a17bd01f96daeb1657` |
| learned aggregator state | `ad4f0d59da7cfc0dd9badee5e50c84dbbf3b61f0de816f5b8ecce4ee3e1c3865` |
| source orientation campaign | `876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f` |
| source calibrated campaign | `80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA allocation | `327,121,920` bytes |
| analysis time | `123.09` seconds |
| DVC data root | `39e8256421b0e4d967003f6358c8b857.dir` (`2,477` files; `39,940,991,836` bytes) |
| lakeFS commit | `89e466d78f86a67a92e161f50fcf12f8fb6bfa7c3decab4e9d29acf50b70ba07` |

The immutable DVC directory object was verified at
`lakefs://artifacts/89e466d78f86a67a92e161f50fcf12f8fb6bfa7c3decab4e9d29acf50b70ba07/structure-net/files/md5/39/e8256421b0e4d967003f6358c8b857.dir`.
Local DVC status is clean, the configured remote is synchronized, and the
lakeFS branch has no uncommitted diff.

## Artifacts and reproduction

- campaign:
  `data/experiments/tinyllm_repeated_reference_acquisition/20260807_d8_corrective_expanded_v8/campaign_results.json`
- per-checkpoint results:
  `data/experiments/tinyllm_repeated_reference_acquisition/20260807_d8_corrective_expanded_v8/runs/*/seed_*/result.json`
- acquisition arrays:
  `data/experiments/tinyllm_repeated_reference_acquisition/20260807_d8_corrective_expanded_v8/acquisition_errors.npz`
- learned aggregator:
  `data/experiments/tinyllm_repeated_reference_acquisition/20260807_d8_corrective_expanded_v8/equivariant_reference_denoiser.pt`
- runner and tests:
  `experiments/structure_net/tinyllm_repeated_reference_acquisition.py`,
  `tests/structure_net/test_tinyllm_repeated_reference_acquisition.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-repeated-reference-v8 \
pixi run python -m \
  experiments.structure_net.tinyllm_repeated_reference_acquisition \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_repeated_reference_acquisition/20260807_d8_corrective_expanded_v8
```

## Next shortest diagnostic

Do not train another representation or denoiser. The shortest independent
check is a frozen acquisition-draw robustness campaign using several new
master repeat seeds, only the analytic circular mean, and only `m=64` plus
`m=256`. It should estimate how often `m=64` clears the checkpoint gate and
whether `m=256` is a stable ceiling.

Separately, the failed residual-write ceiling should be investigated by the
already prescribed frozen reference-path versus residual-tangent transport
audit. That diagnostic asks why coherent input-side correction succeeds where
one large local residual step fails; it should not be conflated with confirming
the acquisition sample-cost threshold.

## Scope boundary

The sensor error is synthetic, independent, unbiased, and Gaussian. The
checkpoints are retained replication units rather than a sampled architecture
population. The result does not establish performance under correlated error,
systematic bias, real sensing cost, natural language, or arbitrary TinyLLM
architectures.
