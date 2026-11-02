# TinyLLM relative-carrier factorization preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-relative-carrier-fixed-quotient-v1`

## Question and prediction

Can the identifiable relative angle `psi = phi + theta` be recovered first as
the vector carrier `(cos(psi), sin(psi))`, then converted by the fixed quotient
`q_x(v) = v[0]` into a stable interval representation? The prediction is that
direct vector supervision will recover the carrier and that freezing it before
the fixed projection will pass the existing base/fiber gate more often than
end-to-end scalar-only supervision.

## Design

The four arms are:

1. observation-only analytic vector carrier, fixed x-projection, and TinyLLM;
2. learned equivariant vector carrier trained only by vector MSE;
3. the checkpointed learned carrier frozen before fixed x-projection and
   TinyLLM training;
4. the corrected 2-D equivariant encoder trained end-to-end only by the scalar
   task loss.

The confirmatory campaign fixes d8 TinyLLM, seeds `7,17,29,41,53`, N3 support,
4,096 shared training examples, paired batch size 64, 600 carrier updates where
applicable, 600 TinyLLM task updates where applicable, AdamW `3e-4`, weight
decay `0.01`, and gradient clipping `1.0`. Model initialization, examples,
pair minibatches, output bins, and probe cohorts are matched by seed. The
analytic arm is a positive control and has no carrier optimization.

The learned carrier is structurally SO(2)-equivariant. Its direct target is the
complete relative vector, never latent phase as an input. The vector checkpoint
is frozen in arm 3; no task gradient may enter it.

## Primary carrier gate

At the carrier output on both composition and extrapolation, each seed must
satisfy:

- mean vector alignment with the target at least `0.95`;
- mean squared vector-coordinate error at most `0.02`;
- winding degree `+1` on the declared fixed-nuisance phase loop;
- maximum SO(2) equivariance error at most `2e-5`.

The learned carrier succeeds if at least four of five seeds pass every carrier
cell jointly.

## Primary quotient gate

At the front-end output and full depth on both composition and extrapolation:

- cosine correlation at least `0.90`;
- nonlinear conditional branch balanced accuracy at most `0.55`;
- conditional log-loss gain over the cosine-only null at most `0.02`.

A downstream arm succeeds only if at least four of five seeds pass every one
of those cells jointly. Learned-fixed task accuracy may not trail the
scalar-only baseline by more than three percentage points on either primary
shift. Carrier-only arm 2 is not assigned a TinyLLM gate.

## Interpretations

| Outcome | Interpretation |
| --- | --- |
| Analytic carrier fails | the finite observation protocol does not reliably recover the relative coordinate |
| Analytic passes, learned carrier fails | approximation or carrier-optimization failure |
| Learned carrier and learned-fixed quotient pass | the earlier target-side failure was selection caused by scalar-only training |
| Carrier passes, learned-fixed downstream fails | embedding or TinyLLM corrupts a correct quotient |
| Fixed arms pass while scalar-only fails | scalar supervision retains an unnecessary branch cover |

Fresh held-out nonlinear probes measure tested conditional decodability, not
mutual information. The fixed-nuisance loop tests degree on one declared
section and does not establish a global bundle theorem.

## Artifacts

The append-only root is
`data/experiments/tinyllm_relative_carrier_factorization/20260806_d8_preregistered`.
Every seed/arm retains strict `result.json`, carrier weights where applicable,
TinyLLM weights where applicable, fingerprints, and matched tensor/schedule
hashes. The campaign root retains `campaign_results.json`.
