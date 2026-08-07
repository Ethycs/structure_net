# TinyLLM degree-two attention-head synthesis decomposition

**Status:** NOT CONFIRMED — STABLE QUOTIENT SYNTHESIS REQUIRES FOUR OR FIVE HEADS  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-c2-sparse-attention-head-synthesis-v1`  
**Preregistration:** [`2026-08-06_tinyllm-c2-attention-head-decomposition-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-c2-attention-head-decomposition-preregistration.md)

## Verdict

The cross-cohort-stable degree-two quotient fronts do not use a sparse one- or
two-head circuit. Exhaustive enumeration of all 64 head subsets selected four
heads in seeds 7 and 29 and five heads in seed 53. Sparse source selection
therefore failed in all three preregistered primary checkpoints.

The selected four-head circuits transferred strongly for seeds 7 and 29,
retaining at least `0.984` of the exact Fisher effect in all held-out cells.
Seed 53's five-head subset retained at least `0.985`, but failed the hard causal
gate in both held-out composition cohorts. Only seed 29 cleared the declared
same-cardinality specificity margin. Complements remained too effective in
seeds 7 and 29, showing redundancy rather than a unique circuit.

The decisive architectural conclusion is:

> Put the group representation and invariant-fusion rules across multi-head
> attention. A single-head adapter or privileged-head rewrite would remove
> distributed, partially redundant causal contributions.

## Campaign integrity

The campaign reused all five d6 degree-two checkpoints and frozen first
synthesis attention cuts. For each seed it exhaustively evaluated every subset
of six output-projected heads on a source cohort, selected one subset using the
frozen rule, and evaluated that unchanged subset on two disjoint held-out
cohorts under both shifts. No training or probe fitting occurred.

Seeds 7, 29, and 53 formed the frozen primary cohort because independent
radius studies had established their early block-0-attention fronts as the
most cross-cohort stable. Seeds 17 and 41 were prespecified secondary contrasts.

| Item | Value |
| --- | --- |
| requested / completed / failed | 5 / 5 / 0 |
| primary checkpoints | seeds 7, 29, 53 |
| secondary contrasts | seeds 17, 41 |
| cohorts | source selection; held-out A; held-out B |
| exact orbits | 64 per shift, cohort, and seed |
| subsets | all 64 subsets of six heads |
| trained models / fitted observers | 0 / 0 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| total analysis time | 29.8 seconds |
| implementation SHA-256 | `d85887a39cc2ee17cb5f35b73d18e396e9490eed9f6fd3ada479ad2c635aad4a` |
| campaign SHA-256 | `8d44328af6b4b44d3753305c4544e05d7024ad3685ea8c8a0c27c2f7adff2401` |
| DVC data root | `46949f1680cbf3fad63c85686b9ae626.dir` |
| lakeFS commit | `fbd9ab1fdd2ff762510bae8ab4e81b78eca646dd593fd3257e0e5876bc750391` |

Nine focused tests across this campaign and its same-cohort predecessor pass.
Maximum all-head decomposition error across primary cells and cohorts was
`2.35e-7` for seed 7, `1.93e-7` for seed 29, and `2.13e-7` for seed 53, well
below the frozen `1e-6` threshold.

## Preregistered primary gates

Every primary gate required all three stable early-front seeds.

| Gate | Result | Required |
| --- | ---: | ---: |
| exact six-head decomposition | **3/3** | 3/3 |
| source subset has at most two heads | 0/3 | 3/3 |
| zero fails and exact full state passes on held-out cohorts | **3/3** | 3/3 |
| fixed selected subset sufficient on all held-out cells | 2/3 | 3/3 |
| complement retains at most `0.50` Fisher effect | 1/3 | 3/3 |
| selected subset beats same-size median by at least `0.20` | 1/3 | 3/3 |

The full sparse-circuit hypothesis is rejected.

## Primary held-out results

Worst preservation and specificity aggregate all four held-out shift/cohort
cells. Complement range lists Fisher preservation across those cells.

| Seed | Selected heads | Cardinality | Worst selected preservation | Fixed subset passes | Complement preservation | Specificity margin |
| ---: | --- | ---: | ---: | --- | --- | ---: |
| 7 | `[0,1,2,3]` | 4 | `0.994` | yes | `0.536–0.561` | `0.047` |
| 29 | `[0,1,2,5]` | 4 | `0.985` | yes | `0.696–0.716` | `0.263` |
| 53 | `[0,1,2,4,5]` | 5 | `0.985` | no | `0.130–0.250` | `0.039` |

Seed 53's smooth preservation is high, but its selected subset fails the exact
composition task gate in both held-out cohorts. The full six-head endpoint
passes, so this is a head-subset failure rather than front instability.

Seed 29 is the only primary cell with a clearly distinguished selected subset,
yet its two-head complement still retains roughly `0.70` Fisher effect. Seed 7
has a transferable four-head subset, but same-size alternatives are nearly as
good and the excluded two heads retain more than the declared dominance limit.
Those patterns are redundant distributed synthesis, not sparse localization.

## Per-head credit is distributed

Exact Shapley values of downstream Fisher preservation spread task credit over
all six heads. Held-out cohort means were approximately:

| Seed | Heads 0–5, mean held-out Shapley credit |
| ---: | --- |
| 7 | `[0.217, 0.100, 0.149, 0.256, 0.263, 0.016]` |
| 29 | `[0.044, 0.242, 0.121, 0.249, 0.132, 0.213]` |
| 53 | `[0.102, 0.303, 0.244, 0.091, 0.118, 0.143]` |

No head receives more than about `0.31` mean credit. Defect-norm fractions are
also spread rather than concentrated. Shapley credit and residual norm do not
rank heads identically, confirming that downstream task geometry—not component
magnitude alone—determines importance.

The low seed-7 credit for head 5 does not imply a unique four-head circuit:
head 4 has high credit but is also excluded by the selected subset. Several
same-cardinality alternatives trade redundant contributions, which is why the
specificity margin is only `0.047`.

## Secondary later-front contrasts

Seed 17 selected four heads `[0,1,3,4]` and retained at least `0.927` Fisher
effect where the endpoint was eligible, but held-out B composition lost the
full exact endpoint and held-out B extrapolation already passed at zero radius.
This is cohort instability of the frozen later front, consistent with the
radius studies.

Seed 41 selected three heads `[0,1,4]`. Its full endpoint replicated, but the
selected subset's worst preservation was `0.896`, just below `0.90`, and one
held-out composition task gate failed. Later-front cells therefore do not
provide a hidden sparse-circuit rescue.

## Interpretation and next decision

The attention output admits an exact linear head decomposition, yet quotient
sufficiency is distributed across four or five components in the stable
models. The group-theoretic carrier and first harmonic identified by the
preceding experiments should therefore be treated as representation types
shared across heads, not assigned to one privileged head.

The shortest architecture test is now a multi-head equivariant replacement or
sidecar with:

```text
per-head charged Ck carriers
    -> symmetry-respecting attention and output projection
    -> cross-head neutral fusion pool
    -> invariant residual/readout.
```

Before retraining a full TinyLLM, this can be falsified with a frozen low-rank
subspace intervention: project the six exact head defects onto their leading
task-weighted joint directions and test the smallest subspace that transfers
across held-out cohorts. A positive result would justify a compact distributed
sidecar; a negative result would require full multi-head equivariance.

## Boundaries

Subset selection uses an already observed source-cohort family, while both
held-out cohorts provide new head-level measurements. Head indices are local
to each independently trained checkpoint. Output-projected head defects do not
identify unique neurons or independent upstream computation. The interventions
remain off-manifold residual patches and are conditioned on the frozen decoder.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_c2_attention_head_decomposition/20260806_d6_preregistered/campaign_results.json`
- Per-seed records:
  `data/experiments/tinyllm_c2_attention_head_decomposition/20260806_d6_preregistered/runs/seed_*/result.json`
- Disposable CUDA lifecycle:
  `data/experiments/tinyllm_c2_attention_head_decomposition/shakedown_20260806/`
- Runner: `experiments/structure_net/tinyllm_c2_attention_head_decomposition.py`
- Tests: `tests/structure_net/test_tinyllm_c2_attention_head_decomposition.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-sparse-attention-head-synthesis-v1.json`

The meta-hypothesis write completed with authoritative persistent-store
readback of the exact hypothesis ID and all five experiment IDs. Legacy Chroma
telemetry and NumPy compatibility warnings were non-fatal.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_c2_attention_head_decomposition \
  --device cuda:0 \
  --output data/experiments/tinyllm_c2_attention_head_decomposition/20260806_d6_preregistered
```
