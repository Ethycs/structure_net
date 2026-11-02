# TinyLLM observable C3 temporal-quotient d6 result

**Status:** PREREGISTERED POSITIVE CONTROL FAILED — RAW/LEARNED STAGES STOPPED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-temporal-quotient-training-v1`

**Classification:** `c3_positive_control_task_failure`

**Preregistration:** [observable C3 d6](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-quotient-d6-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_temporal_quotient/20260811_d6_preregistered/campaign_results.json`

## Verdict

The analytic d6 positive-control population passes the exact invariant
representation, registered semantic/deck probes, four-cut causal closure,
identity replay, and target-changing specificity gates in all five seeds. It
passes the complete natural-task gate in only two of five seeds, below the
preregistered four-seed requirement.

The campaign therefore stops with:

```text
c3_positive_control_task_failure
```

The full hypothesis is not confirmed. More importantly, the learned-`C3` arm
was not tested: the stop rule prevented all ten conditional raw and learned
cells from running. This result cannot be interpreted as evidence that the
learned invariant encoder succeeds or fails.

## Campaign integrity

| Item | Result |
| --- | ---: |
| requested grid | 15 cells |
| analytic cells completed | 5/5 |
| raw/learned cells stopped prospectively | 10 |
| failed executions | 0 |
| reused results | 0 |
| optimizer steps executed | 3,000 |
| optimizer steps avoided by stop rule | 6,000 |
| valid checkpoints/artifact triples | 5/5 |
| primary preset | d6, 6 blocks / 6 heads / width 384 |
| trainable parameters per analytic cell | 29,951,232 |
| GPU | NVIDIA GeForce RTX 3060 |
| scheduler | two logical CUDA slots |

Each seed used 4,096 paired observed examples, 600 AdamW steps, batch size 64,
and the frozen task-only objective. Final evaluation used the untouched shared
seeds `331003` and `331021`, disjoint from training, probe fitting, the analytic
preflight, and both systems shakedowns.

Every model, front end, diagnostics file, and result is hashed and reloadable.
The campaign contains 27 files and approximately 573 MiB. There were no retries
or excluded seeds.

## Primary natural-task endpoint

### Per-seed result

| Seed | Composition acc | Extrap acc | Extrap corr | Extrap CE | Joint natural task |
| ---: | ---: | ---: | ---: | ---: | --- |
| 7 | `.84375` | `.42383` | `.96444` | `2.05514` | pass |
| 17 | `.76758` | `.32910` | `.87133` | `3.36108` | fail |
| 29 | `.83398` | `.40918` | `.95101` | `2.32589` | fail |
| 41 | `.79492` | `.34180` | `.96084` | `2.07151` | fail |
| 53 | `.84668` | `.39941` | `.97120` | `2.19308` | pass |

The frozen extrapolation thresholds were accuracy `>=.35`, correlation `>=.90`,
cross-entropy `<=2.20`, and bin coverage `>=12`. All seeds cover all 16 bins.

The failures are specific:

- seed 17 misses extrapolation correlation, accuracy, and cross-entropy;
- seed 29 misses only cross-entropy, by `.12589`;
- seed 41 misses only exact accuracy, by `.00820`.

No threshold was relaxed for these near-boundary cells.

### Population summaries

| Regime | Mean accuracy | Accuracy range | Mean correlation | Mean CE |
| --- | ---: | ---: | ---: | ---: |
| composition | `.81738` | `.76758-.84668` | `.99889` | `1.29525` |
| extrapolation | `.38066` | `.32910-.42383` | `.94376` | `2.40134` |

The trained analytic systems learn the supported temporal map well. Their
outside-speed generalization is not population-reliable under the fixed ordered
decoder.

## Representation and causal endpoints

All five seeds pass the complete representation gate at the front-end sequence
and full-depth query on both shifts.

| Endpoint | Worst registered value | Gate | Pass count |
| --- | ---: | ---: | ---: |
| front-end extrap target correlation | `.90613` | `>=.90` | 5/5 |
| full-depth extrap target correlation | `.92194` | `>=.90` | 5/5 |
| conditional deck accuracy | exactly `.33333` | `<=.3834` | 5/5 |
| maximum conditional log-loss gain | `.00185` | `<=.02` | 5/5 |
| exact action error, every residual cut | `0.0` | `<=1e-5` | 5/5 |

The causal result is equally clean:

- orbit-barycenter preservation passes every front-end, post-attention,
  post-MLP, and full-depth cell under both shifts in 5/5 seeds;
- identity replay changes answer logits by exactly `0.0` in every cell; and
- target-changing derangements pass no seed in either shift.

Thus the analytic architecture really does expose and propagate a
deck-invariant, task-bearing representation. The population failure occurs in
natural extrapolating use of that representation, not in invariance, retained
target information, causal orbit closure, or control specificity.

## No-checkpoint positive-control localization

After the registered failure, a fixed no-fit diagnostic reapplied the analytic
temporal rule to the same untouched final observations. It loaded no checkpoint
and optimized no parameter:

```text
q_next = q(7) * conjugate(q(6)) * q(7)
target = Re(q_next).
```

| Regime | Temporal corr | Temporal RMSE | Fixed interval acc | Fixed interval CE |
| --- | ---: | ---: | ---: | ---: |
| composition | `.999948` | `.00737` | `.95313` | `1.27167` |
| extrapolation | `.999924` | `.00876` | `.95898` | `1.27524` |

This route passes every frozen natural-task floor by a wide margin. Together
with the registered full-depth semantic probes passing 5/5, it yields the
post-outcome localization:

```text
invariant_sensor_valid_trained_continuation_readout_extrapolation_unreliable
```

This is not a rescue. The fixed rule bypasses the trained TinyLLM computation,
and the semantic probe is an estimator rather than a causal readout repair. The
original analytic population remains failed at 2/5.

## Interpretation

### What this supports

- Exact architectural `C3` invariance can create a nontrivial temporal quotient
  without exposing the answer in one frame.
- The quotient representation remains target-bearing and causally closed under
  composition and outside-speed extrapolation in all five analytic systems.
- Structural quotient formation and natural trained utility are separate
  requirements.
- The preregistered positive-control stop rule saved two thirds of the planned
  optimizer work and prevented an uninterpretable learned-arm comparison.

### What this does not support

- It does not establish that a task-only learned invariant encoder discovers
  the analytic temporal carrier.
- It does not compare raw synthesis with architectural invariance.
- It does not license d10.
- It does not show that longer training, a different optimizer, or an auxiliary
  loss would repair extrapolation.
- It does not upgrade probe decodability into natural downstream use.

## Program decision

Do not run the stopped raw/learned cells and do not rescue this campaign with a
step, threshold, seed, or loss sweep. The constructive quotient theory survives
at the representation and causal levels, but the proposed trained-system test
is blocked by its analytic natural-utility control.

The shortest successor is a separately preregistered typed temporal
continuation/readout study. It must decide prospectively whether the scientific
object is:

1. learning the invariant sensor while fixing the known temporal operator and
   ordered physical decoder; or
2. learning temporal dynamics while providing a fixed metric interface that
   prevents the continuation and answer head from inventing a support-relative
   chart.

That successor is a new hypothesis, not a continuation or repair of the failed
population gate. Before any training, use these checkpoints only for a frozen
continuation-versus-answer-head decomposition if a more precise interface
boundary is needed.

## Artifact identity

| Artifact | SHA-256 |
| --- | --- |
| campaign JSON | `e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc` |
| result manifest | `7dfdcf1ff80fe20a975fe6a7d1311dc92e3ff1a396a6da9550c91835a568a0ff` |
| checkpoint/front-end/diagnostic manifest | `a0b90484863346cf2a5e0ef8be65cac3a221cfa50a373f88c9ead07a0cd351a1` |
| campaign fingerprint | `6b74b0acb406943eadd21323477a1d609b49d88df60940b3e943b06fdd8c73e0` |
| no-checkpoint diagnostic | `9e3b635b5de997641fec9722989e13047fdf6409a953e0829e8677aa2f4e1e4d` |
| campaign runner | `9b2cd0e3ce3752b7eea80d5859c11880a9d3732fb48b58306e34eab4f080d5ec` |
| analysis source | `89dacc60d02707678e689c6ce1e8f9c963889af352565a227bb90ed8e367e6a3` |

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl-c3-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_quotient_campaign \
  --execute-primary --gpus 0 --max-parallel 2 \
  --output \
  data/experiments/tinyllm_c3_temporal_quotient/20260811_d6_preregistered

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl-c3-positive \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_quotient_positive_control
```
