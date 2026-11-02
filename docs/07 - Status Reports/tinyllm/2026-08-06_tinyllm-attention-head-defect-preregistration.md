# TinyLLM attention-head Reynolds-defect localization preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-attention-head-defect-sparsity-v1`  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`

## Question and prediction

The exact Reynolds defect is causally necessary at every frozen degree-two
quotient-synthesis front, and the cross-cohort-stable fronts occur in attention.
TinyLLM has six attention heads. Does one stable one- or two-head circuit
generate most of the task-effective invariant defect?

The primary prediction is sparse and falsifiable: in at least four of five
frozen degree-two checkpoints, the same subset of at most two heads is causally
sufficient under composition and extrapolation, and at least one head is
individually necessary under both shifts.

## Frozen sources and replication unit

Reuse without retraining:

- the five d6, degree-two checkpoints for seeds `7,17,29,41,53`;
- the first exact synthesis attention transition frozen by the Reynolds
  character-coupling campaign;
- that campaign's same 64-orbit composition and extrapolation cohorts.

Every degree-two synthesis front is an attention residual sublayer: block 0 for
seeds 7, 29, and 53; block 1 for seed 41; and block 2 for seed 17. One trained
checkpoint/seed, containing both shifts, is the replication unit.

Reusing the predecessor cohorts makes this a causal decomposition of an
already-established event, not an independent front-replication study. The
separate fresh-cohort radius campaigns retain authority over support stability.

## Exact additive head decomposition

For attention input `x`, let `A_h(LN(x))` be head `h` after its corresponding
slice of the shared output projection, excluding the projection bias. Then

```text
A(LN(x)) = bias + sum_h A_h(LN(x)).
```

For exact deck sheets `x_j` with barycenter `b`, the attention-residual Reynolds
defect is

```text
chi = mean_j [x_j + A(LN(x_j))] - [b + A(LN(b))]
    = sum_h chi_h,

chi_h = mean_j A_h(LN(x_j)) - A_h(LN(b)).
```

The residual identity and projection bias cancel exactly. The implementation
MUST verify both ordinary-attention reconstruction and Reynolds-defect
reconstruction to relative error at most `1e-6` before interpreting a cell.

## Frozen causal interventions

At the frozen post-attention target cut, patch

```text
F(b) + sum_(h in S) chi_h
```

and repeat the state across both fiber members before running the unchanged
downstream TinyLLM. Enumerate only the subsets needed to decide the hypothesis:

- empty set;
- all six singleton sets;
- all fifteen head pairs;
- all six leave-one-head-out sets;
- the full six-head set.

This is 29 unique patches per shift, not a fitted selector. No outcome is used
to alter the subset family.

## Endpoints

### Frozen causal task gate

A patch passes only when circular alignment is at least `0.90`, phase sampling
is resolved, winding degree is within `0.10` of two, and exact-bin accuracy
falls no more than `0.03` below the untouched checkpoint.

### Smooth downstream effect

For every patch, report Fisher--Rao effect explained relative to the empty and
full exact-defect posteriors. An effect is degenerate below `1e-6`. Sparse
sufficiency requires both the causal task gate and Fisher-effect explained at
least `0.90`.

Individual necessity holds when removing a head from the full patch either
fails the causal gate or retains less than `0.70` Fisher effect.

## Primary gates

Compute gates jointly within checkpoint and require at least four of five
seeds:

1. **Exact contract and endpoint:** head outputs reconstruct ordinary attention
   and the full Reynolds defect within `1e-6`; the empty patch fails and the
   full patch passes under both shifts.
2. **Sparse sufficiency:** some subset of at most two heads is sufficient under
   each shift.
3. **Shift-stable sparse circuit:** at least one identical subset of at most two
   heads is sufficient under both shifts.
4. **Individual necessity:** at least one identical head is necessary under
   both shifts.

The full hypothesis is confirmed only if every campaign gate passes. Different
sparse subsets on different shifts fail the stable-circuit gate even if both
shifts have a small sufficient set.

## Secondary measurements

Report per head and shift:

- defect norm fraction and cosine to the full defect;
- singleton Fisher effect and task diagnostics;
- leave-one-out Fisher retention and task diagnostics;
- all sufficient singleton/pair subsets;
- best pair by Fisher effect;
- Jaccard overlap of sufficient subsets across shifts.

These measurements can identify distributed, redundant, or antagonistic heads
but cannot rescue a failed primary gate.

## Outcome interpretation

| Outcome | Interpretation | Next shortest action |
| --- | --- | --- |
| all gates pass | quotient synthesis is a stable sparse head circuit | localize Q/K/V versus output-projection causality inside those heads |
| sparse but shift-specific | small circuits exist, but routing is support-relative | compare head attention maps across exact orbit cohorts |
| no pair is sufficient, individual head is necessary | the invariant is distributed with a bottleneck head | test head plus low-rank complement interactions |
| no pair sufficient and no head necessary | redundant distributed synthesis | stop searching for a single circuit; analyze the six-head defect subspace |
| additive contract fails | implementation error | do not interpret outcomes |
| full endpoint fails | predecessor event does not reproduce | preserve the failure and defer head claims |

## Boundaries

Head contributions are defined after the learned output-projection slices;
they do not identify unique neurons within a head. Patching a head's Reynolds
defect tests downstream sufficiency of that component, not whether the model
computes it independently of other heads. Repeated patches make within-orbit
branch identity constant by construction. All task gates remain conditioned on
the frozen decoder and predecessor cohort.

## Artifacts and execution

Primary root:
`data/experiments/tinyllm_attention_head_defect/20260806_d6_preregistered`.

Shakedowns use a separate root and cannot enter the evidence aggregate.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
pixi run python -m experiments.structure_net.tinyllm_attention_head_defect \
  --output data/experiments/tinyllm_attention_head_defect/20260806_d6_preregistered \
  --device cpu
```
