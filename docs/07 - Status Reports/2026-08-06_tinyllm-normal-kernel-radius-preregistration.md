# TinyLLM normal jet-kernel radius preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-normal-jet-kernel-structural-radius-v1`

## Question and prediction

Near the d6 step-15 degree defect, does the reduced normal kernel predict the
minimum parameter intervention required to reach the degree discriminant more
accurately than matched random directions?

## Fixed design

Use the same deterministic seed-7 d6 step-14 state and continuous input lift as
the certification study. The declared edit space is the block-1 MLP output
projection bias, with the Euclidean parameter metric. At the closest sampled
phase, project the two-component posterior moment orthogonally to its phase
derivative, form `C = Jbar Jbar^T`, and compute the pseudoinverse radius and
minimum-norm predicted normal direction.

Directly optimize phase and the edit vector for the minimum-norm perturbation
whose posterior moment is numerically zero. Apply the predicted direction and
32 norm-matched isotropic random directions. Every search uses identical phase
resolution, tolerances, maximum evaluations, and a held-out refined grid.

## Primary endpoints

- predicted radius / directly optimized radius in `[0.75, 1.25]`;
- direct optimized residual norm at most `1e-5`;
- predicted direction reaches residual norm at most `1e-4` at no more than
  `1.25` times the optimized radius;
- predicted direction ranks in the best 10% of the 33 tested directions by
  minimum achieved residual;
- a degree transition occurs across a small bracket around the predicted
  crossing and phase sampling is resolved.

This is a single-transition mechanistic study, not a population claim. Passing
supports local intervention cost only in the declared edit space and metric.
Finite precision and nonconvex direct search remain explicit limitations.

## Artifacts

The append-only root is
`data/experiments/tinyllm_normal_kernel_radius/20260806_d6_step15`, retaining
states by digest, Jacobians, kernel, predicted/direct perturbations, random
controls, refined phase fields, optimizer traces, and `results.json`.
