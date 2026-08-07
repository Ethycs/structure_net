# TinyLLM degree-k quotient ladder preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-degree-k-finite-quotient-ladder-v1`

## Question and prediction

Given an identifiable fixed analytic phase carrier, will matched TinyLLM models
learn the circle maps `q_k(exp(i phi)) = exp(i k phi)` for `k=1,2,3`, while
their internal representations contract the `Z_k` branch index? The prediction
is posterior winding degree `k` for every task, circular output topology for
every task, and branch-probe chance after internal quotient formation for
`k=2,3` even though the input carrier retains the branch.

## Fixed design

Use d6 TinyLLM, five seeds `7,17,29,41,53`, a fixed observation-only analytic
carrier `(cos(phi), sin(phi))`, the same phase examples and minibatches within
seed, 4,096 examples, batch 64, 600 AdamW updates, learning rate `3e-4`, weight
decay `0.01`, and gradient clipping `1.0`. Only `k` changes. Outputs use the
same 16 circular answer bins. The analytic carrier is computed from observed
sensor and calibration packets, not latent phase at model input.

## Primary gates

For every seed and `k` on both composition and extrapolation:

- posterior circular alignment with `k phi` at least `0.90`;
- induced posterior winding within `0.10` of `+k`;
- resolved phase sampling with maximum adjacent phase increment below `pi/2`;
- normalized posterior H1 lifetime at least `0.40`.

For `k=2`, conditional branch balanced accuracy at post-MLP block 1 and full
depth must be at most `0.55`. For `k=3`, it must be at most `0.3834`. The
conditional probe receives `(cos(k phi), sin(k phi))`. The analytic carrier
must decode the branch at least `0.90`, confirming that the negative control
contains the distinction. At least four of five seeds must pass all applicable
cells jointly for each `k`.

## Secondary defect law

For seed 7, trace every optimizer step. Whenever endpoint degree changes,
interpolate the consecutive weight states on the declared straight path and
measure indexed phase/path defect cells. The secondary prediction is total net
charge `k` from degree-zero initialization to the trained endpoint. This grid
identity is numerical and cannot by itself certify root uniqueness.

## Interpretation boundary

Equal H1 across tasks does not imply equal induced maps. A passed degree gate
supports the map-aware ladder; a failed fiber gate means TinyLLM learned the
task output without forming the internal finite-group quotient. Task accuracy
or a circular barcode cannot rescue either failed gate.

## Artifacts

The append-only root is
`data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered`, with one
strict run record and checkpoint per seed/k, a seed-7 degree trajectory and
transition-field supplements, and one `campaign_results.json`.
