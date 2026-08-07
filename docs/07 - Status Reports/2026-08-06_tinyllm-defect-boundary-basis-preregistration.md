# TinyLLM decoder-boundary defect basis preregistration

**Status:** PREREGISTERED — PRIMARY OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-defect-boundary-basis-v1`  
**Schema:** `nal.tinyllm-defect-boundary-basis.v1`

## Question

The stable block-0 `C2` Reynolds defect has a compact geometric carrier, but
seeds 29 and 53 require more hard-gate rank than their smooth Fisher effect
suggests. Are the additional causal dimensions source-stable corrections along
the frozen decoder's local decision-boundary normals, rather than additional
semantic coordinates?

## Frozen scope

Reuse only the three independently trained checkpoints with replicated
block-0 attention fronts: seeds `7,29,53`. Reuse the exact orbit generator,
the source and two held-out cohorts, and the checkpoint-local geometric bases
from the completed rank titration. Train no model or observer and update no
parameter. This is a selected three-checkpoint mechanistic study and is marked
`UNDERPOWERED`; it is not a five-seed population claim.

The predecessor minimum sufficient ranks were `2,8,4`. Freeze the immediately
preceding geometric ranks before this experiment:

| Seed | Base geometric rank | Predecessor failure cells |
| ---: | ---: | ---: |
| 7 | 1 | 4/4 |
| 29 | 4 | 1/4 |
| 53 | 2 | 2/4 |

No rank, threshold, cohort, or seed may be changed after primary outcomes are
read.

## Source-only boundary construction

For source composition and extrapolation, let `d_i` be the exact attention
Reynolds defect, `a_i` the propagated barycenter, and `G` the frozen leading
geometric basis at the seed's declared base rank. Define

```text
h_i^base = a_i + P_G d_i
e_i      = (I - P_G) d_i.
```

Let `y_i` be the answer-token winner under the exact-defect state. Let `c_i`
be the strongest non-`y_i` answer under `h_i^base`. These are frozen-model
outputs, not target labels. At the base state compute the local logit-margin
normal

```text
n_i = grad_h [z_y_i(h) - z_c_i(h)] evaluated at h_i^base.
```

Project the missing defect onto that normal:

```text
q_i = n_i (n_i^T e_i) / (||n_i||^2 + 1e-12).
```

Remove numerical components in `span(G)`, concatenate `q_i` across the two
source shifts, and take its right-singular basis `B`. The held-out rank-`s`
intervention is

```text
a_test + P_[G,B_s] d_test,
```

where `[G,B_s]` is re-orthonormalized once on source data and then frozen.
Held-out defects are projected but held-out gradients, winners, labels,
posteriors, and gates cannot influence the basis.

## Controls

Evaluate the following at unchanged held-out A/B composition and
extrapolation cells:

1. zero and exact defect endpoints;
2. the frozen base geometric span `G`;
3. `G` plus the next one, two, and four geometric SVD directions;
4. `G` plus one, two, and four paired boundary directions;
5. `G` plus boundary directions fitted after a deterministic permutation of
   source residual/normal membership;
6. `G` plus deterministic random directions drawn inside the remaining source
   defect span.

Every control uses the same total rank as its paired intervention. Random and
permutation seeds are fixed functions of checkpoint seed. Basis
orthogonality must be at most `1e-8`; inherited exact head decomposition error
must remain at most `1e-6`.

## Primary endpoint

An intervention is sufficient in a held-out cell only when it passes the
unchanged degree/alignment/sampling/exact-bin causal conjunction and preserves
at least `0.90` of the exact downstream Fisher effect.

The boundary-correction hypothesis passes a checkpoint only if all conditions
hold:

1. zero fails and the exact defect passes all four held-out cells;
2. `G + B_1` is sufficient in all four held-out cells;
3. the shuffled-pair and random-residual rank-one additions each remain
   insufficient in at least one of that checkpoint's declared predecessor
   failure cells;
4. all source-basis, decomposition, and nondegeneracy contracts pass.

The preregistered mechanism claim requires all three checkpoints. Secondary
directions cannot rescue a failed one-direction primary claim.

## Secondary measurements

Report, without changing the primary verdict:

- the minimum fixed paired-boundary correction count in `1,2,4`;
- the minimum same-total-rank geometric, shuffled, and random correction;
- source boundary-residual singular energy and numerical rank;
- residual/normal cosine and signed margin contribution distributions;
- exact-winner agreement, top-two posterior margin, Fisher preservation, and
  the full causal diagnostics for every held-out intervention;
- whether repaired examples concentrate near the exact decoder boundary.

## Outcome meanings

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| paired `+1` passes and controls fail | extra hard-gate rank is a stable boundary correction | implement a small task-weighted neutral sidecar |
| paired needs `+2/+4` but beats controls | boundary geometry matters but is vector-valued | preserve a small boundary-normal bundle |
| geometric and paired bases tie | the extra rank is ordinary defect geometry, not decoder-specific | stop task-weighted fitting |
| paired basis fails held-out | source boundary normals are support-relative or nonlinear | audit local curvature; do not train an adapter yet |
| contracts fail | diagnostic invalid | repair implementation only; do not interpret outcomes |

## Artifact root and planned command

Primary evidence belongs under:

`data/experiments/tinyllm_defect_boundary_basis/20260806_d6_preregistered/`

Planned command:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_defect_boundary_basis \
  --device cuda:0 \
  --output data/experiments/tinyllm_defect_boundary_basis/20260806_d6_preregistered
```

## Method boundaries

The construction uses the frozen decoder and therefore tests a
decoder-conditioned mechanism, not an intrinsic representation dimension.
The local normal is a first-order object and may miss curvature between the
base and exact states. Held-out patches use the exact held-out defect, so this
tests representational sufficiency rather than independent computability from
raw input. Bases remain checkpoint-local and projected states may be off the
natural activation manifold.

## Amendment A — post-outcome aggregate-resume repair

Recorded after the primary outcomes were inspected. A fingerprint-matched
resume reused all three per-seed records, but the aggregate immutability check
compared in-memory tuples with JSON-reloaded lists and therefore missed the
completed aggregate. It attempted to initialize CUDA before rewriting any
artifact and failed because that audit ran in a GPU-isolated sandbox; the
primary campaign SHA-256 remained unchanged.

The repair canonicalizes the in-memory configuration through strict JSON
before comparison. It changes no source basis, gradient, intervention,
threshold, cohort, seed, or gate. The campaign will be deterministically
replayed under the repaired producer, and its source-basis/held-out/gate
payload must retain SHA-256
`54acd44cd92584e65b557b3d0197a76f7d65232857798c4f10cf75cb0d2b3267`
before the repaired artifact is accepted.
