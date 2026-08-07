# TinyLLM cross-seed causal-carrier transport

**Status:** NOT CONFIRMED — GEOMETRY TRANSFERS, CAUSAL USE DOES NOT  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-cross-seed-causal-carrier-transport-v1`  
**Preregistration:** [`2026-08-06_tinyllm-cross-seed-causal-carrier-transport-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-cross-seed-causal-carrier-transport-preregistration.md)

## Verdict

The three stable block-0 `C2` carriers implement strikingly similar coordinate
functions, but they are not causally interchangeable under the preregistered
label-free linear transport. All six directed checkpoint pairs explained at
least `0.956` of target rank-three coordinate variance on every held-out cell,
and all six defeated the shuffled-pair control. Nevertheless, none of the six
maps preserved the target model's continuous causal endpoint.

The result rejects the proposed shared linear causal chart:

```text
shared statistical carrier geometry
  does not imply
shared checkpoint-independent causal coordinates.
```

The paired transport's mean circular-moment shift was `0.151--0.328` bins over
the 24 held-out cells, above the registered `0.125` ceiling. An unconstrained
affine-ridge ceiling also failed all 24 cells. Direct target rank-three patches
remained close to exact target patches (`0.015--0.055` mean bins), so the
failure is introduced by cross-checkpoint transport rather than rank three
alone.

## Campaign integrity

The study reused three frozen 29,956,608-parameter TinyLLM checkpoints and
their source-fitted block-0 post-attention defect bases. It trained no model,
adapter, probe, or decoder. The fitted objects were six paired whitened
orthogonal maps, six regime-preserving shuffled maps, and six descriptive
affine-ridge maps.

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 6 / 6 / 0 / 0 directed pairs |
| trained models / predictive observers | 0 / 0 |
| fitted coordinate maps | 18 |
| fit / held-out cells per pair | 2 / 4 |
| exact orbits per cell | 64 |
| carrier rank | 3 for every checkpoint |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| summed analysis time | 12.0 seconds |
| implementation SHA-256 | `798a85b50f7dc8489e28cc66a498a5c3a8af193f859b80b43b1c5cc551ba0a89` |
| campaign SHA-256 | `44707fd4bcd810e63614671aa491095fae735ee52359d464ab25abb10a2bc228` |
| DVC data root | `23cc720334f5baf33af2823af0b0f4a1.dir` |
| lakeFS commit | `062cd9cd128adc7fa5798ab7d5700dcff38057d40cea3d5e52abc90847605d14` |

Every result records checkpoint, frontend, predecessor, character-analysis,
readout-campaign, scientific-fingerprint, and result hashes. A
fingerprint-matched completed resume left the aggregate bytes unchanged. The
separate eight-orbit CUDA run is systems-only evidence and was not pooled.

## Primary gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| paired held-out coordinate transport | **6/6** | 6/6 | pass |
| paired causal transport | **0/6** | 6/6 | fail |
| shuffled specificity | **6/6** | 6/6 | pass |
| target control contract | **4/6** | 6/6 | fail |
| complete hypothesis | **not confirmed** | all gates | fail |

The two target-control failures are both directed into seed 7. On the fresh
held-out-A and held-out-B composition cells, even the exact target state failed
the previously frozen discrete calibration-loss ceiling (`0.0781` and
`0.0703`). Direct rank three lost `0.03125` and `0.0703`, respectively. These
failures make the full campaign fail by contract; they do not explain away the
transport result. All four maps into seeds 29 and 53 had valid controls, yet
paired causal transport still passed `0/4`.

## Directed-pair results

| Pair | Worst held-out paired R2 | Worst shuffled R2 | Specificity margin | Paired mean shift range, bins | Causal pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 -> 29 | 0.9670 | -0.1245 | 1.0915 | 0.214--0.295 | no |
| 7 -> 53 | 0.9824 | -1.7573 | 2.7396 | 0.151--0.184 | no |
| 29 -> 7 | 0.9558 | -1.2157 | 2.1714 | 0.278--0.328 | no |
| 29 -> 53 | 0.9745 | -0.5539 | 1.5284 | 0.158--0.231 | no |
| 53 -> 7 | 0.9838 | -0.9014 | 1.8852 | 0.165--0.194 | no |
| 53 -> 29 | 0.9760 | -1.4567 | 2.4327 | 0.161--0.239 | no |

The paired p95 moment shifts span `0.265--0.755` bins. Exact-bin accuracy is
often good (`0.625--0.906` across pair/cell interventions), but that cannot
rescue the registered continuous failure. This is the same reason the prior
study separated semantic geometry from bin-boundary calibration: hard labels
can hide small but systematic phase errors.

## Why ordinary linear flexibility is unlikely to rescue it

The descriptive affine-ridge map improves Euclidean coordinate fit slightly,
but its mean moment shifts remain `0.139--0.290` bins and it passes `0/24`
joint causal cells. Thus the failure is not peculiar to an orthogonality
constraint. Very small coordinate residuals are amplified along directions
that matter to the target continuation.

This identifies a metric mismatch. Ordinary coordinate R2 treats all three
target directions according to activation variance; the frozen continuation
weights them by task sensitivity. The data support a common statistical
carrier atlas, but not a common Euclidean causal chart.

## Symmetry-group consequence

Applying the known deck symmetry is still the right architectural move, but it
must constrain the representation before comparing checkpoints. The
post-attention Reynolds defect is already `C2`-invariant, so merely labelling
all three channels as the trivial representation adds no information. A useful
group construction instead starts from the exact orbit sheets and decomposes
them into deck characters:

```text
c_r = (1 / |G|) sum_g conjugate(chi_r(g)) h(g x).
```

The causal invariant write is then synthesized only from character-neutral
couplings. For `C2`, the lowest nontrivial neutral interaction is `c1 * c1`;
for `C3`, `c1 * c2` is already neutral. This is the selection rule already
supported by the Reynolds-character and irrep-fusion experiments. It should be
used as a fixed coordinate contract, not recovered afterward with generic
Procrustes alignment.

The shortest next frozen diagnostic is a **group-anchored task-metric
transport**:

1. retain the exact `C2` character construction and the registered rank-three
   source bases;
2. evaluate the target continuation Jacobian or Fisher metric on fit orbits;
3. fit a label-free map weighted by that frozen task metric, with no held-out
   adaptation;
4. reuse the same 24 cells, controls, and continuous gates;
5. require improvement over both the Euclidean paired map and the affine-ridge
   ceiling, with shuffled orbit membership still failing.

If this passes, the checkpoints share a symmetry-typed carrier only after the
correct causal metric is supplied. If it fails, the scientifically honest
architecture is a shared group representation with checkpoint-local causal
charts—not one portable three-number interface.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_cross_seed_causal_carrier_transport/20260806_d6_preregistered/campaign_results.json`
- Per-pair records:
  `data/experiments/tinyllm_cross_seed_causal_carrier_transport/20260806_d6_preregistered/runs/source_*_target_*/result.json`
- Systems-only CUDA lifecycle:
  `data/experiments/tinyllm_cross_seed_causal_carrier_transport/20260806_shakedown_cuda/`
- Runner:
  `experiments/structure_net/tinyllm_cross_seed_causal_carrier_transport.py`
- Tests:
  `tests/structure_net/test_tinyllm_cross_seed_causal_carrier_transport.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-cross-seed-causal-carrier-transport-v1.json`

The named hypothesis and all six directed-pair records passed authoritative
Chroma readback. Eleven current runner and meta-ledger tests pass.

```bash
PYTHONPYCACHEPREFIX=/tmp/structure-net-carrier-transport-pyc-20260806 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_cross_seed_causal_carrier_transport/20260806_d6_preregistered
```

## Method boundaries

Only three selected stable block-0 `C2` checkpoints are tested, so this is an
underpowered mechanistic result rather than a population-prevalence estimate.
The coordinate maps use paired source orbits and are not independently
computable encoders. Transport patches are off-manifold interventions into a
frozen continuation. The study aligns post-synthesis carriers, not individual
neurons, heads, or pre-synthesis character modes. A task-metric alignment would
remain a checkpoint-conditioned diagnostic, not by itself a deployable shared
encoder.
