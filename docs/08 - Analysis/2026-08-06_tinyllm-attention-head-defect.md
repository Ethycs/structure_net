# TinyLLM attention-head Reynolds-defect localization

**Status:** NOT CONFIRMED — NO ONE- OR TWO-HEAD SYNTHESIS CIRCUIT  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-attention-head-defect-sparsity-v1`  
**Preregistration:** [`2026-08-06_tinyllm-attention-head-defect-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-attention-head-defect-preregistration.md)

## Verdict

The degree-two invariant-synthesis defect is not localized to a stable one- or
two-head attention circuit. Across five frozen checkpoints and both shifts, no
singleton or head pair simultaneously passed the causal task gate and explained
at least `0.90` of the full downstream Fisher effect. Sparse sufficiency and a
shift-stable sparse circuit therefore passed in 0/5 seeds.

Individual leave-one-out necessity appeared in three seeds, below the required
four. Seeds 7, 29, and 53 shared necessary heads across shifts, but their sets
differed and contained two to four heads. Seeds 17 and 41 had no head that was
individually necessary under both shifts.

The supported mechanism is distributed:

> Attention heads contribute exact additive Reynolds-defect components, but
> quotient sufficiency requires a multi-head combination rather than a
> privileged one- or two-head circuit.

This falsifies the shortest proposed adapter strategy. Q/K/V localization
inside one selected head is not warranted; the next diagnostic must preserve
multi-head interactions or test the full head-defect subspace.

## Campaign integrity

The campaign reused the five retained d6 degree-two checkpoints, their frozen
first synthesis attention transitions, and the same 64-orbit predecessor
cohort. It trained no model and fitted no selector. For every shift it evaluated
the empty set, six singletons, fifteen pairs, six leave-one-out subsets, and the
full six-head set.

| Item | Value |
| --- | --- |
| requested / completed / failed | 5 / 5 / 0 |
| trained models | 0 |
| transitions | frozen first synthesis attention cut per checkpoint and shift |
| exact orbits | 64 per shift and seed; predecessor cohort reused |
| interventions | 29 unique subsets per shift |
| device | CPU |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| total analysis time | 111.4 seconds |
| implementation SHA-256 | `4a45c490a6af398ddbf0841ca997de4f39b6c9ae9562100efd4db0874151547b` |
| campaign SHA-256 | `2e8bd1b0f48e1da9b6c7dcc42900bba3bc6bef05413089ef4d0a26530bdb4fbb` |
| DVC data root | `46949f1680cbf3fad63c85686b9ae626.dir` |
| lakeFS commit | `fbd9ab1fdd2ff762510bae8ab4e81b78eca646dd593fd3257e0e5876bc750391` |

The all-head output reconstruction contract passed in every cell. Maximum
ordinary-attention relative error was `2.99e-7`; maximum Reynolds-defect error
was `5.14e-7`, both below `1e-6`. The empty patch failed and the full exact
defect passed under both shifts in all five seeds.

## Preregistered gates

Every gate required at least four of five seeds.

| Gate | Result | Required |
| --- | ---: | ---: |
| exact additive contract and endpoints | **5/5** | 4/5 |
| some singleton or pair sufficient under both shifts | 0/5 | 4/5 |
| identical sparse subset sufficient across shifts | 0/5 | 4/5 |
| identical head individually necessary across shifts | 3/5 | 4/5 |

The full hypothesis is rejected.

## Best sparse interventions

The same best Fisher pair emerged across both shifts within each checkpoint,
but none met the joint sparse-sufficiency endpoint. Effect explained and causal
pass are composition / extrapolation.

| Seed | Frozen target | Best pair | Fisher effect explained | Causal pass |
| ---: | --- | --- | --- | --- |
| 7 | block-0 attention | `[3,4]` | `0.978 / 0.974` | no / no |
| 17 | block-2 attention | `[3,4]` | `0.701 / 0.726` | yes / no |
| 29 | block-0 attention | `[1,4]` | `0.876 / 0.715` | no / no |
| 41 | block-1 attention | `[1,4]` | `0.822 / 0.854` | no / no |
| 53 | block-0 attention | `[0,1]` | `0.890 / 0.884` | no / no |

Seed 7 illustrates why smooth posterior proximity cannot substitute for the
frozen causal conjunction. Its best pair explained about `0.98` of the Fisher
effect yet failed the degree/alignment/accuracy task gate. Seed 17 composition
shows the converse boundary: its best pair passed the coarse task gate but
explained only `0.701`, below the frozen `0.90` sparse-mechanism threshold.

## Individual necessity

Heads are necessary when removing them from the full defect either breaks the
task gate or retains less than `0.70` Fisher effect.

| Seed | Composition necessary heads | Extrapolation necessary heads | Common |
| ---: | --- | --- | --- |
| 7 | `[2,4,5]` | `[2,5]` | `[2,5]` |
| 17 | `[]` | `[]` | `[]` |
| 29 | `[0,1,2,3,4,5]` | `[1,2,5]` | `[1,2,5]` |
| 41 | `[0,4,5]` | `[]` | `[]` |
| 53 | `[0,1,2,4,5]` | `[0,1,2,5]` | `[0,1,2,5]` |

Multiple individually necessary heads can coexist because the head defects add
in residual space while the frozen continuation is nonlinear and thresholded.
The result does not mean every head independently computes the quotient. It
means the full task-effective state is fragile to removal of several distinct
components in the stable early-front seeds.

## Interpretation and boundaries

The exact attention output is additive over output-projected head slices, and
the implementation confirms that identity numerically. Downstream sufficiency
is not additive: partial sums can be close in Fisher geometry while missing
the map-level task gate, or can cross the gate without reproducing most of the
full posterior effect.

This same-cohort experiment localizes the failure of sparse-head explanations;
it is not an independent front-replication study. Head indices have no claimed
alignment across independently trained seeds. Output-projection slices do not
identify unique neurons, Q/K/V paths, or upstream computability. Repeating a
patched barycenter makes within-orbit branch identity constant by construction.

The exhaustive held-out follow-up in
[`2026-08-06_tinyllm-c2-attention-head-decomposition.md`](2026-08-06_tinyllm-c2-attention-head-decomposition.md)
is authoritative about how many heads remain sufficient off the selection
cohort.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_attention_head_defect/20260806_d6_preregistered/campaign_results.json`
- Per-seed records:
  `data/experiments/tinyllm_attention_head_defect/20260806_d6_preregistered/runs/seed_*/result.json`
- Runner: `experiments/structure_net/tinyllm_attention_head_defect.py`
- Tests: `tests/structure_net/test_tinyllm_attention_head_defect.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
pixi run python -m experiments.structure_net.tinyllm_attention_head_defect \
  --output data/experiments/tinyllm_attention_head_defect/20260806_d6_preregistered \
  --device cpu
```
