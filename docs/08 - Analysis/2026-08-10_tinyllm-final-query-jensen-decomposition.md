# TinyLLM final-query Jensen decomposition

**Status:** VALID REGISTERED POST-OUTCOME DIAGNOSTIC — PRIMARY GATE PASSED  
**Date:** 2026-08-10  
**Classification:** `generic_jensen_sufficient_with_layernorm_modulation`  
**Hypothesis:** `tinyllm-final-query-jensen-decomposition-v1`  
**Registration:** [final-query Jensen decomposition](../07%20-%20Status%20Reports/2026-08-10_tinyllm-final-query-jensen-decomposition-preregistration.md)

## Verdict

The raw d8 final-query activation barycenter's lower target cross-entropy does
not require favorable quotient geometry. Generic cross-entropy convexity in
answer-logit space supplies at least the entire observed aggregate improvement
on both registered shifts.

On training support, the generic logit-midpoint Jensen gain is `0.118570` nats
and the actual activation-midpoint gain is `0.118152` nats. Final layer
normalization cancels only `0.000418` nats, or `0.35%` of the available Jensen
gain. Outside range, the generic Jensen gain is much larger, `1.458095` nats,
while the actual activation gain is `1.071331` nats. Final layer normalization
therefore cancels `0.386764` nats, or `26.53%`, rather than helping.

The primary gate passes because the generic term supplies `100.35%` and
`136.10%` of the observed gain in the two regimes. The near-complete subtype
passes only on support, so the locked classification retains explicit
layer-normalization modulation.

This resolves the apparent tension in the parent result:

> A same-target activation average can improve accuracy or cross-entropy for a
> generic convexity reason while still failing to preserve the frozen
> posterior. Task-metric improvement is not evidence of autonomous quotient
> closure.

## Exact accounting

For paired final-query states `h+` and `h-`, the study compared

```text
z_log = (z(h+) + z(h-)) / 2
z_act = z((h+ + h-) / 2),
```

where `z` is the unchanged final-layer-normalization and centered answer-logit
map. With endpoint, logit-midpoint, and activation-midpoint cross-entropies
`E`, `L`, and `A`, respectively,

```text
J = E - L
N = A - L
G = E - A = J - N.
```

| Regime | Endpoint CE `E` | Logit midpoint `L` | Activation midpoint `A` | Jensen `J` | LN remainder `N` | Actual gain `G` | `J/G` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training support | `1.626031` | `1.507461` | `1.507879` | `0.118570` | `0.000418` | `0.118152` | `1.0035` |
| outside range | `3.987820` | `2.529725` | `2.916489` | `1.458095` | `0.386764` | `1.071331` | `1.3610` |

The maximum accounting residual is `4.44e-16`. Every one of the 1,024 pairs
satisfies the logit Jensen inequality; the smallest gains are `8.80e-5` and
`2.92e-4` nats. The activation midpoint improves pairwise cross-entropy in
`97.27%` of support pairs and `88.87%` of outside-range pairs.

## What final layer normalization changes

On support, activation and logit midpoints are nearly task-equivalent: their
mean posterior Jensen--Shannon divergence is `0.000566`, and the final-LN logit
remainder is `0.0783` of the endpoint logit-chord norm on average.

Outside range, the nonlinear difference is material. Mean midpoint posterior
JS rises to `0.04444`, and the mean logit-remainder ratio rises to `0.2275`
(`0.3818` at the 95th percentile). The nonlinear map moves the activation
midpoint away from the better generic logit midpoint. It does not create the
improvement.

This is consistent with the previous causal null. Outside range, the
activation barycenter is better than the two endpoints on mean
cross-entropy, yet it remains far enough from their logit-space optimum and
from the natural frozen posterior to fail complete-posterior preservation.

## Target-changing control

The fixed semantic reassignment demonstrates why Jensen improvement is
nonspecific. Convexity still holds for every target-changing control pair.

| Regime | Control Jensen `J` | Control LN remainder `N` | Control activation gain `G` |
| --- | ---: | ---: | ---: |
| training support | `0.118570` | `0.213621` | `-0.095051` |
| outside range | `1.458095` | `1.424934` | `0.033161` |

The Jensen term is essentially the same after the fixed pair permutation,
because the endpoint logits are only reassigned. On support, the actual
target-changing activation midpoint becomes worse even though its generic
Jensen term is positive. Outside range, almost all of the generic gain is
cancelled. The parent semantic-reassignment intervention still fails the task
gate; this diagnostic does not promote it.

Thus `J >= 0` says only that averaging logits with a fixed scoring target is
convex. It says nothing about whether the averaged state belongs to the right
semantic fiber.

## Contracts and integrity

- exact target identity within every natural pair: maximum error `0.0`;
- parent baseline posterior replay: maximum error `0.0`;
- parent correct-barycenter posterior replay: maximum error `0.0`;
- parent semantic-reassignment posterior replay: maximum error `0.0`;
- pairwise natural and control Jensen contracts: pass in both regimes;
- accounting identity: pass to `4.44e-16`;
- all values finite;
- initial and final model state SHA-256:
  `bf20a98e242b72f14c186176951973970cb96f52353a6416c7b4db7b208d02fc`;
- initial and final system state SHA-256:
  `e776446b0778aca4ca22c7f14c2460feda278c4d348dd382b457827bc39aa036`;
- trained or fitted models, heads, probes, carriers, observers, thresholds, or
  decoders: **zero**.

The first complete output root,
`20260810_d8_seed7_registered`, serialized a positive-gain ratio even when a
control gain was negative, producing a meaningless huge secondary value. The
bug affected no tensor, gate, classification, or primary quantity. The runner
was corrected to emit no ratio when `G <= 0`, unit-tested, and fully replayed to
the authoritative `20260810_d8_seed7_registered_v2` root. Its diagnostic NPZ is
byte-identical to the first run. Both roots are retained for audit.

## Scientific decision

Close the raw final-query task-improvement ambiguity. The parent barycenter's
improved cross-entropy should be treated as a generic Jensen effect with
shift-dependent adverse final-LN modulation, not as partial evidence that the
whole residual is a quotient.

For future activation-averaging studies, report the logit-midpoint comparator
alongside the causal posterior/task endpoint. An activation intervention earns
quotient interpretation only when it preserves the unchanged computation and
beats semantic, action, and pairing controls—not merely when average loss
falls.

This result does not alter the calibrated analytic/equivariant positive branch,
where the barycenter passes causal task gates before attention. It only removes
a misleading positive gloss from the raw whole-residual null.

## Evidence limits

- This is a registered post-outcome diagnostic of an already observed result,
  not fresh confirmatory evidence.
- Only one retained raw d8/seed-7 checkpoint is evaluated.
- Only the final post-MLP query and its native final-LN/answer-head map are
  decomposed.
- No population prevalence, earlier-layer closure, hidden scalar quotient, or
  global residual invariance is established.

## Artifacts and reproduction

- authoritative campaign:
  `data/experiments/tinyllm_final_query_jensen_decomposition/20260810_d8_seed7_registered_v2/campaign_results.json`
- diagnostics:
  `data/experiments/tinyllm_final_query_jensen_decomposition/20260810_d8_seed7_registered_v2/diagnostics.npz`
- runner:
  `experiments/structure_net/tinyllm_final_query_jensen_decomposition.py`
- tests:
  `tests/structure_net/test_tinyllm_final_query_jensen_decomposition.py`

Reproduce with:

```bash
pixi run python -m experiments.structure_net.tinyllm_final_query_jensen_decomposition \
  --output data/experiments/tinyllm_final_query_jensen_decomposition/20260810_d8_seed7_registered_v2 \
  --device cuda:1
```

The exact-resume replay leaves completed bytes unchanged.

The complete `data/` tree is tracked by DVC root
`9ab22e64de5723bc06d76e55af39f1cb.dir` (`40,293,402,158` bytes,
`3,141` files). Ten incremental objects were pushed to the configured remote
and committed on lakeFS as
`d3e7f287e5b056a7a4e1f6fbe541e79469b756b57e87b612c27ce86de6d4cc31`,
whose parent is the noisy-pilot commit
`ab65ecfef3f08620775705b96eb09f26b2e13e40380db60433edca20352f2d40`.
The lakeFS branch is clean and DVC reports cache/remote synchronization. Direct
non-presigned reads at the immutable commit reproduce the local campaign,
diagnostics, and meta-JSON SHA-256 values.

| Artifact | SHA-256 |
| --- | --- |
| campaign | `ab47fb645001e84d2b4e198f5bd269011bdc0438d056ea840f183156b98fc88a` |
| diagnostics | `cd8695fa9e0040657a661e7aae4324408a63ad5f3fd2cf858ba8a163ed4a8459` |
| implementation | `a9d06810ef198e6f58654cdde309c0225db75f82b5bd6a65268f7e2baf960353` |
| runner | `a53eaad8509b1bd1b08ad12e03caff396215591f24a2c9dab6a2bf78b9af0f33` |
| registration | `0c8e4d871637007405e8e57868a24a02962f52b04a2e1c87af72de0f71b5df69` |
| scientific fingerprint | `6f2a37301dc208b209473eae540541ba9145274bfa3d2fb0ee0780ddf3dfea1c` |
| meta-hypothesis JSON | `61db54abdd5aab55db3f1a024de9452e54a518cd3d6a022a3e51c11c6c003578` |
| DVC root | `9ab22e64de5723bc06d76e55af39f1cb.dir` |
| lakeFS commit | `d3e7f287e5b056a7a4e1f6fbe541e79469b756b57e87b612c27ce86de6d4cc31` |
