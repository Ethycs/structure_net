# TinyLLM final-query semantic-kernel decomposition

**Status:** VALID PREREGISTERED UNDERPOWERED NULL  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`,
frozen-checkpoint no-fit causal decomposition  
**Hypothesis:** `tinyllm-final-query-semantic-kernel-v1`  
**Schema:** `nal.tinyllm-final-query-semantic-kernel.v1`  
**Preregistration:** [final-query semantic-kernel preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-final-query-semantic-kernel-preregistration.md)

## Verdict

The hypothesis is **not confirmed**. Replacing the complete-posterior map with
the model's declared one-dimensional posterior-mean cosine coordinate does not
reveal a hidden final-query quotient.

The rank-1 semantic tangent still contains about half of the exact same-cosine
fiber chord. Removing the complementary local kernel leaves `51.2%` of pair
distance in support and `57.0%` outside range, both above the preregistered
`25%` ceiling. The finite semantic-kernel patch also fails scalar-coordinate
preservation in both regimes. Its support miss is near two individual
thresholds, but outside-range failure is large and its nonlinear attribution
gate fails.

| Primary item | Training support | Outside range | Gate |
| --- | ---: | ---: | --- |
| scalar-coordinate preservation | fail | fail | **fail** |
| remaining pair-distance ratio | `0.5124` | `0.5698` | **fail**; ceiling `0.25` |
| complete-posterior separation | fail (`JS 0.0103`) | pass (`JS 0.1270`) | **fail** |
| nonlinear scalar attribution | pass | fail | **fail** |
| rank-1 random specificity | pass | pass | pass |
| Jacobian / nesting / replay / state validity | pass | pass | pass |
| complete hypothesis |  |  | **fail** |

The raw classification is
`scalar_and_complete_posterior_quotients_not_separated`.

## Declared scalar computation

The study reused the valid retained d8 cosine-interval seed-7 checkpoint, the
same `512` exact shared-nuisance fibers in each regime, and the final post-MLP
query vector. No model parameter, probe, decoder, carrier, metric, or threshold
was fit.

For fixed cosine answer centers `c = linspace(-1, 1, 16)`, the scalar map was

```text
s(h) = c^T softmax(W LN(h)).
```

This is exactly the frozen posterior-mean coordinate used by the registered
cosine task-map correlation and RMSE. At each pair barycenter `b`, its analytic
activation Jacobian was

```text
j_s(b) = sum_k p_k(b) (c_k - s(b)) Dg_k(b),
```

where `Dg` is the preceding centered-answer-logit Jacobian. Each barycenter
displacement was split into the rank-1 row-space component and its
511-dimensional local kernel before actual finite frozen-head replay.

## The semantic kernel does not contain most of the chord

| Regime | Original pair norm | Kernel-patched pair norm | Remaining ratio | Mean semantic-component norm fraction | Mean kernel-component norm fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| training support | `121.65` | `62.33` | `0.5124` | `0.4584` | `0.8604` |
| outside range | `318.01` | `181.21` | `0.5698` | `0.5243` | `0.8013` |

The norm fractions are row-wise means and need not sum to one; the components
are orthogonal per row. The remaining pair ratio is the direct quantity in the
locked gate.

Moving from the rank-15 complete-posterior tangent to the rank-1 semantic
tangent lowers the remaining ratio from `0.8061` to `0.5124` in support and
from `0.8474` to `0.5698` outside range. This is a real dimensional
distinction, but not the predicted quotient: the one-dimensional task tangent
still captures a material, approximately half-chord component.

## Finite scalar preservation fails

The semantic-kernel patch was required to preserve the frozen scalar
computation, not merely improve agreement with the label.

| Regime | Mean absolute scalar change | P95 change | Correlation loss | RMSE increase | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| training support | `0.01007` | `0.04049` | `-0.00465` | `-0.01378` | **fail** |
| outside range | `0.14345` | `0.69565` | `-0.09235` | `-0.06677` | **fail** |
| locked ceilings | `0.01000` | `0.03000` | `0.01000` | `0.01000` |  |

Negative correlation loss and RMSE increase mean that target-facing metrics
improved. They do not rescue the preservation endpoint. The patch can move the
model's scalar computation toward the correct label while still changing that
computation materially. That is useful projection, not autonomous closure.

The support mean misses by only `0.00007`, but its p95 also exceeds the locked
ceiling, exact-bin accuracy falls from `0.4385` to `0.3848`, and the
outside-range coordinate shifts are an order of magnitude larger. No threshold
is amended after observing those values.

## Scalar and posterior quotients do not separate stably

The same semantic-kernel patch changes the complete posterior by:

| Regime | Posterior JS | Locked separation rule | Result |
| --- | ---: | ---: | --- |
| training support | `0.01028` | strictly greater than `0.02` | fail |
| outside range | `0.12700` | strictly greater than `0.02` | pass |

The registered separation therefore fails. In support, the patch is below the
posterior-JS ceiling even though its exact-bin accuracy loss exceeds the
parent's task-sufficiency allowance. Outside range, both scalar preservation
and complete-posterior preservation fail. There is no shift-stable regime in
which the raw final query supplies the predicted scalar quotient while only
posterior shape changes.

## Nonlinear attribution is support-relative

The actual finite scalar change from the rank-1 semantic component explains
the full barycenter change in support but not outside range:

| Regime | Mean semantic/full residual | P95 residual | Sign agreement | Mean kernel/full effect | P95 kernel/full effect | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| training support | `0.0510` | `0.1354` | `1.0000` | `0.1271` | `0.3136` | pass |
| outside range | `0.4769` | `1.1002` | `0.8965` | `0.5273` | `1.2703` | **fail** |

All `1,024` rows per regime had a material full scalar effect. The
outside-range failure shows that the local rank-1 tangent and its kernel
interact nonlinearly over the finite barycenter displacement. First-order
semantic geometry is not a stable finite quotient intervention under the
declared extrapolation.

## Rank-1 specificity

The deterministic random rank-1 control fails scalar preservation in both
regimes. Its mean absolute coordinate change exceeds the semantic-kernel value
by `0.05014` in support and `0.16633` outside range, above the locked `0.01`
margin. The semantic projector is therefore more task-specific than a random
rank-1 direction even though the complete quotient hypothesis fails.

The semantic-only arm also fails coordinate preservation in both regimes, so
the analytic rank-1 component is causally active. These narrower positives
cannot rescue contraction, preservation, or cross-shift attribution.

## Numerical and lifecycle integrity

| Contract | Training support | Outside range |
| --- | ---: | ---: |
| scalar Jacobian rank | `1` in `512/512` fibers | `1` in `512/512` fibers |
| complete-posterior Jacobian rank | `15` in `512/512` | `15` in `512/512` |
| maximum finite-difference relative error | `6.01e-10` | `4.46e-9` |
| maximum reconstruction relative error | `3.01e-17` | `3.17e-17` |
| maximum scalar-kernel leakage | `1.15e-15` | `6.91e-16` |
| maximum scalar-outside-posterior-rowspace leakage | `1.11e-14` | `1.47e-14` |
| all parent and predecessor posterior replays | exactly `0.0` | exactly `0.0` |

The rank-1 semantic tangent is numerically nested inside the rank-15
answer-sensitive row space, as required by the chain rule. Model and system
state hashes are unchanged.

The primary run used Python `3.11.13`, PyTorch `2.5.1+cu121`, and an NVIDIA
GeForce RTX 2060 SUPER on CUDA device `1`. It allocated at most `494,806,528`
CUDA bytes and completed the recorded analysis in `16.37` seconds. A second
invocation left campaign and diagnostics bytes unchanged.

The preceding eight-fiber CUDA shakedown is labeled
`systems_lifecycle_only_not_quality_evidence` and contributes no scientific
evidence.

## Interpretation and scientific decision

The scalar/full-posterior distinction is mathematically real but does not
produce the proposed raw-model quotient. The one-dimensional cosine tangent is
a strict subspace of the 15-dimensional answer tangent, yet it still carries
roughly half of the same-target chord. Local scalar nullness is also not stable
under the finite outside-range intervention.

This sharpens the activation conclusion:

```text
same latent cosine label
    != same frozen scalar computation
    != same frozen answer posterior.
```

The raw model can improve target accuracy when averaged because both its
scalar prediction and posterior are support-relative across exact target
fibers. Neither complete-posterior nor scalar local-kernel projection supplies
autonomous final-query closure.

Close the raw final-query Euclidean kernel branch. Do not relax the near support
threshold, scan earlier layers, select a different scalar after the result,
fit a new carrier, or retrain under this hypothesis. The existing calibrated
analytic and equivariant front ends remain the constructive positive evidence:
they create causal quotient sufficiency before the transformer instead of
revealing a hidden autonomous quotient in the raw final representation.

## Artifacts and reproduction

| Item | SHA-256 / value |
| --- | --- |
| campaign | `e668f2d9d45212334a24d10b420977927e68e280b5241745578eb46bf29558c8` |
| diagnostics | `a2aa5c52f8dbba7027f68012a36ec1d6533e9846813b957ae1797a5025acf394` |
| implementation | `f634e645cff4fa1e43cd727ed11efe040e21c642af7ab61061b3730f85645e13` |
| runner | `34ef7544fe8660a8ddbbfb048595ed61be58df048e8b0f649e82bd75ba53c449` |
| preregistration | `fcd7e6aedb6e7d53de111fb0ffedafda86f8702bb538ad7d06d60e754e3d7e59` |
| predecessor posterior-kernel campaign | `93d9e22d766aa56943f0bd0c41b31ed25dc592e97ae0667863f3963588c38cde` |
| meta-hypothesis record | `ab290e298d120347ee7615d86c3d06b7331c5f6d877ec4adc06d554f1c9af8b9` |
| DVC data root | `641397465b9c65ff0f5860950dc7735f.dir` |
| lakeFS commit | `a0cd293f69b4ddfe05846295649e69d88ce2ca18d6c8c7822938a9112cfaa78b` |

- primary campaign:
  `data/experiments/tinyllm_final_query_semantic_kernel/20260810_d8_seed7_preregistered/`
- systems-only shakedown:
  `data/experiments/tinyllm_final_query_semantic_kernel/20260810_shakedown_cuda/`

```bash
MPLCONFIGDIR=/tmp/matplotlib-semantic-kernel-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_final_query_semantic_kernel \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_final_query_semantic_kernel/20260810_d8_seed7_preregistered
```

## Boundaries

This result covers one retained d8 seed-7 cosine checkpoint, two exact
synthetic regimes, final post-MLP query activations, the fixed posterior-mean
cosine coordinate, and activation-local Euclidean Jacobian geometry. It does
not establish population prevalence, a global kernel, an earlier-layer
mechanism, literal information erasure, or natural-language behavior. It
rejects the declared local scalar-kernel rescue, not every nonlinear semantic
coordinate that could be defined prospectively in another architecture.
