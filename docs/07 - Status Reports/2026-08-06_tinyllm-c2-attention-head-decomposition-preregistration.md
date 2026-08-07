# TinyLLM degree-two attention-head synthesis decomposition preregistration

**Status:** PREREGISTERED — HEAD-SUBSET OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Profile:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-c2-sparse-attention-head-synthesis-v1`  
**Schema:** `nal.tinyllm-c2-attention-head-decomposition.v1`

## Question and prediction

Do the three cross-cohort-stable degree-two quotient fronts synthesize their
task-effective Reynolds defect through a fixed sparse subset of attention
heads?

Previous campaigns established that all five degree-two first-synthesis events
occur at attention sublayers. Across two independently generated fresh-orbit
campaigns, the block-0-attention fronts in seeds 7, 29, and 53 were monotone and
shift-stable, whereas the later fronts in seeds 17 and 41 were cohort-sensitive.
Those front and endpoint outcomes are already known. No per-head decomposition
has been computed or inspected.

The primary prediction is that each stable early-front checkpoint has one
fixed subset of at most two of its six attention heads that is sufficient and
dominant across two held-out cohorts and both shifts.

## Frozen sources and replication unit

Reuse without training or parameter changes:

- the d6, degree-two checkpoints for seeds `7,17,29,41,53` under
  `data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered`;
- the frozen synthesis transitions under
  `data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered`;
- the previously declared generator and causal output thresholds.

One independently trained checkpoint/seed is the replication unit. Seeds 7,
29, and 53 are the preregistered primary early-front cohort. Seeds 17 and 41
are mechanistic secondary contrasts and cannot promote the primary claim.

Every checkpoint, comparator, schema, implementation, seed, degree, and cut
identity must validate before analysis. Each evaluated cohort contains 64
exact nuisance-matched `C2` orbits per shift.

## Cohorts and selection policy

Use three deterministic, disjoint orbit cohorts:

| Role | Generator seed offset | Use |
| --- | ---: | --- |
| source selection | `+2501 + 101 * regime_index` | choose one subset per checkpoint |
| held-out A | `+4501 + 101 * regime_index` | frozen evaluation only |
| held-out B | `+9101 + 307 * regime_index` | frozen evaluation only |

These reproduce the cohort identities used by the character-coupling,
orbit-radius, and controlled C2-radius campaigns. Subset selection may inspect
both source-cohort shifts but no held-out head-subset outcome.

Enumerate all 64 subsets of the six heads. A source-cohort subset is sufficient
only if, under both composition and extrapolation:

- its repeated barycenter patch passes the frozen causal output conjunction;
- it preserves at least `0.90` of the full exact-head downstream Fisher effect.

Choose the sufficient subset with smallest cardinality. Break ties by largest
minimum Fisher preservation across source shifts, then lexicographically by
head indices. If no subset passes, selection is missing.

## Exact head decomposition

For an attention synthesis sublayer

```text
F(h) = h + W_O concat(A_0(h), ..., A_5(h)) + bias,
```

write the exact orbit sheets as `h_j` with barycenter `b`. Because the output
projection is linear and its bias cancels in a Reynolds difference, define the
per-head defect

```text
chi_q = mean_j W_O,q A_q(h_j) - W_O,q A_q(b).
```

Then

```text
mean_j F(h_j) - F(b) = sum_q chi_q.
```

For every subset `S`, patch

```text
F(b) + sum_(q in S) chi_q
```

at the frozen attention output, repeat it across both sheet identities, and
continue through the unchanged network. The all-head state must reconstruct
the directly evaluated exact barycenter with relative error at most `1e-6`.

Fisher preservation is

```text
1 - d_FR(D(patch_S), D(exact))^2 / d_FR(D(propagated), D(exact))^2.
```

Effects below `1e-6` are degenerate and cannot pass.

## Primary gates

The full hypothesis is confirmed only if all gates pass:

1. **Exact decomposition:** maximum all-head state reconstruction error is at
   most `1e-6` in every source and held-out cell for seeds 7, 29, and 53.
2. **Sparse source selection:** every primary seed selects a source-cohort
   subset of cardinality at most two.
3. **Held-out endpoint replication:** in every primary seed, radius-zero fails
   and the exact all-head state passes in both shifts of both held-out cohorts.
4. **Fixed-subset sufficiency:** the selected source subset passes the causal
   endpoint and preserves at least `0.90` Fisher effect in all four held-out
   cells of every primary seed.
5. **Complement dominance:** the complement of the selected subset preserves
   at most `0.50` Fisher effect in all four held-out cells of every primary
   seed.
6. **Subset specificity:** for each primary seed, the selected subset's worst
   held-out Fisher preservation exceeds the median worst-held-out preservation
   of alternative subsets with the same cardinality by at least `0.20`.

These are exact three-of-three cohort gates, not four-of-five gates, because
the stable early-front cohort was fixed by prior independent evidence.

## Secondary measurements

Report for every seed, cohort, shift, and subset:

- causal pass and component output diagnostics;
- Fisher-effect preservation;
- subset cardinality;
- per-head residual-defect norm;
- exact six-head Shapley values of Fisher preservation;
- complement behavior;
- selected-subset rank among same-cardinality alternatives.

For seeds 17 and 41, report selection and held-out behavior under the identical
protocol but do not use them to rescue a primary gate. A changed propagated or
exact endpoint is reported as cohort instability rather than a head failure.

## Outcome meanings

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| all gates pass | a checkpoint-specific one/two-head circuit generates the stable degree-two quotient defect | ablate the corresponding Q/K/V and output channels before any equivariant retraining |
| sparse source subset fails held-out | head routing is cohort-relative despite a stable aggregate front | preserve group types across all heads instead of selecting heads |
| three or more heads are required | invariant synthesis is distributed across attention | use group-equivariant multi-head attention, not an adapter on one head |
| complement remains sufficient | the circuit is redundant rather than localized | measure redundancy and avoid single-head claims |
| same-size alternatives perform similarly | apparent localization is subset-search degeneracy | reject a privileged-head interpretation |
| exact decomposition fails | the intervention implementation is invalid | stop and repair before interpreting outcomes |

## Boundaries

Head indices are meaningful only within one checkpoint; no cross-seed neuron or
head alignment is claimed. Exhaustive subset patching is exact for the
attention output projection but remains an off-manifold residual intervention.
The downstream Fisher endpoint is decoder-conditioned. Repeated barycenter
states make within-orbit identity constant by construction and do not prove
global absence of sheet information. The source subset is selected on an
already observed cohort family, while both held-out cohorts provide new
head-level outcomes rather than new trained models.

## Artifacts and execution

Primary root:

`data/experiments/tinyllm_c2_attention_head_decomposition/20260806_d6_preregistered`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_c2_attention_head_decomposition \
  --device cuda:0 \
  --output data/experiments/tinyllm_c2_attention_head_decomposition/20260806_d6_preregistered
```

Focused algebraic tests and a disposable eight-orbit CUDA lifecycle must pass
before the primary campaign. Shakedown metrics are systems-only and cannot be
pooled.
