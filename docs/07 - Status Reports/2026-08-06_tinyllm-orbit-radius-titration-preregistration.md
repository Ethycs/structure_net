# TinyLLM causal orbit-radius titration preregistration

**Status:** PREREGISTERED — Amendment A recorded before primary outcomes  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-causal-orbit-radius-threshold-v1`  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`

## Question and prediction

The prior Reynolds character-coupling experiment established an exact causal
synthesis front for degree-two TinyLLMs but rejected a universal local
quadratic approximation. The shortest remaining diagnostic is to vary only the
amplitude of the already observed deck orbit at that frozen sublayer.

For source sheets `h_j`, barycenter `b`, centered cover directions
`delta_j = h_j - b`, and frozen residual sublayer `F`, define

```text
chi(r) = mean_j F(b + r delta_j) - F(b),  0 <= r <= 1.
```

The primary prediction is that degree-two quotient synthesis is a stable radial
threshold: on new orbits, the exact patch `F(b) + chi(r)` changes from causally
insufficient to sufficient exactly once as `r` increases, and the first passing
radius differs by at most one grid step between composition and extrapolation.

Degree three is retained as a declared secondary diagnostic because the prior
causal regime was support-relative. It cannot promote the primary result.

## Frozen sources and replication unit

Reuse without training or parameter changes:

- d6 degree-ladder checkpoints for `k=2,3` and seeds `7,17,29,41,53`;
- the frozen causal fronts from the deck-action campaign;
- the exact synthesis fronts from the Reynolds character-coupling campaign.

Every source and comparator checkpoint digest MUST match. One checkpoint/seed,
containing both composition and extrapolation, is the independent replication
unit. Evaluation uses 64 newly generated exact nuisance-matched orbits per
shift, with evaluation seeds distinct from all prior campaigns.

## Fixed intervention

At the previously measured first exact synthesis sublayer, evaluate the frozen
radius grid

```text
R = {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0}.
```

For each radius, patch the exact partial-orbit barycenter
`F(b) + chi(r)` at the sublayer output, repeat it across every fiber member,
and run the unchanged downstream TinyLLM. The repetition makes within-orbit
branch chance exact under this evaluation; it does not establish global branch
absence off the tested fibers.

Two deterministic path controls distinguish nonlinear synthesis from merely
moving along the final task-effective direction:

1. linear chord: `F(b) + r chi(1)`;
2. quadratic chord: `F(b) + r^2 chi(1)`.

These are controls, not fitted approximations. No probe, decoder, optimizer, or
threshold is selected from the titration outcomes.

## Causal endpoint

Use the frozen deck-action causal criterion. A patched state passes only when:

- circular alignment is at least `0.90`;
- phase sampling is resolved;
- winding degree lies within `0.10` of `k`;
- exact-bin accuracy falls no more than `0.03` below the untouched model.

For the exact radius curve, define the critical radius as the smallest grid
value that passes. A curve is single-crossing when radius zero fails, radius
one passes, and every grid point at or above the critical radius passes while
every earlier point fails.

## Primary gates

Evaluate each gate jointly within seed and require at least four of five
degree-two seeds:

1. **Endpoint replication:** `r=0` fails and `r=1` passes under both shifts.
2. **Single radial crossing:** the exact-radius causal curve is single-crossing
   under both shifts.
3. **Shift-stable threshold:** the composition and extrapolation critical
   radii differ by at most `0.125`.

The hypothesis is confirmed only if all three campaign gates pass. Marginally
passing different seeds do not count as a joint pass.

## Secondary measurements

At each radius and for all three paths, report:

- the complete causal output diagnostics;
- Fisher--Rao downstream effect explained relative to `r=0` and exact `r=1`;
- residual norm ratio and cosine to `chi(1)`;
- squared residual error relative to the linear and quadratic chords;
- exact/chord critical radii and the earliest radius at which each path passes.

The Fisher-effect degeneracy floor is frozen at `1e-6`.

The chord comparison determines whether partial group-orbit computation changes
the task-effective direction, rather than only its magnitude. Degree-three
curves test whether its previously observed support-relative behavior also
appears as multiple crossings or shift-dependent thresholds.

## Outcome interpretation

| Outcome | Interpretation |
| --- | --- |
| all degree-two gates pass | the exact causal synthesis is reducible, at task level, to a stable one-dimensional orbit-amplitude threshold |
| endpoint replication passes but crossing is non-monotone | quotient sufficiency is path-dependent; stop pursuing low-order scalar expansions |
| crossing is monotone but shift-unstable | the causal mechanism remains support-relative even though its endpoint is stable |
| exact path and chords cross together | downstream use depends mainly on accumulated defect magnitude |
| exact path differs materially from both chords | the sublayer rotates or reshapes the task-effective invariant during synthesis |
| endpoint replication fails | the earlier front is not stable on new exact orbits; do not interpret the titration |

## Boundaries

This intervention identifies causal sufficiency along observed deck-orbit
directions only. It does not recover a global Hessian, certify a neighborhood,
or prove that orbit amplitude is the model's only internal coordinate. The
output gate is decoder-conditioned and task-specific. Degree-three results are
secondary regardless of apparent strength.

## Artifacts and execution

Confirmatory artifacts will be written under
`data/experiments/tinyllm_orbit_radius_titration/20260806_d6_preregistered`.
Shakedowns use a separate root and cannot enter the scientific aggregate.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_orbit_radius_titration \
  --output data/experiments/tinyllm_orbit_radius_titration/20260806_d6_preregistered \
  --device cuda:0
```

## Amendment A — CPU execution

**Timing:** recorded after the disposable eight-orbit lifecycle and before any
64-orbit primary cell was executed or inspected.

The execution environment did not expose an NVIDIA device: both `nvidia-smi`
and PyTorch CUDA initialization failed before producing an experiment result.
The under-resolved eight-orbit CPU lifecycle then validated source loading,
intervention execution, strict JSON, and aggregation; its topology-resolution
failures remain shakedown-only and are not evidence.

The confirmatory campaign will use CPU with two numerical threads. This study
performs frozen forward interventions only, so the scientific design, source
weights, seeds, 64-orbit evaluation sets, radius grid, endpoints, and gates are
unchanged. Environment and per-cell elapsed time will record the deviation.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
pixi run python -m experiments.structure_net.tinyllm_orbit_radius_titration \
  --output data/experiments/tinyllm_orbit_radius_titration/20260806_d6_preregistered \
  --device cpu
```
