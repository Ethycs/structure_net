# TinyLLM final-query Jensen decomposition — registered diagnostic

**Status:** FROZEN BEFORE DECOMPOSITION EVALUATION  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `POST_OUTCOME_DIAGNOSTIC`  
**Hypothesis:** `tinyllm-final-query-jensen-decomposition-v1`

## Question

Why did exact same-target activation barycentering improve mean target
cross-entropy in the retained raw d8 TinyLLM even though it failed the frozen
posterior-preservation gate at every cut?

The parent result is already known. This study is therefore not fresh
confirmation and does not test population prevalence. It freezes a no-fit
mechanistic decomposition before evaluating the previously unmeasured
logit-midpoint quantities.

## Locked source and scope

- the single valid raw d8/seed-7 cosine-interval checkpoint from the
  task-relative activation-barycenter campaign;
- the exact 512 same-cosine fiber pairs already regenerated under
  `training_support` and `outside_range` with seeds `850011` and `850021`;
- only the final post-MLP query vector, where the continuation is final layer
  normalization followed by the unchanged affine answer head;
- the parent baseline, correct-barycenter, and semantic-reassignment
  posteriors must replay within `2e-6`;
- no model, head, probe, observer, carrier, threshold, or decoder may be fit or
  trained.

The d6 checkpoint remains excluded because it failed the parent's locked
outside-range baseline floor.

## Exact decomposition

For one same-target pair, let `h_+` and `h_-` be the two final-query states,
`t` their identical target distribution, and

```text
z(h) = centered answer logits(head(LN_f(h))).
```

Define

```text
z_+   = z(h_+)
z_-   = z(h_-)
z_log = (z_+ + z_-) / 2
z_act = z((h_+ + h_-) / 2).
```

With `CE(t,z)` denoting soft-target cross-entropy, measure

```text
E = mean_pair (CE(t,z_+) + CE(t,z_-)) / 2
L = mean_pair CE(t,z_log)
A = mean_pair CE(t,z_act)

J = E - L       # generic logit-space Jensen gain
N = A - L       # final-LN/nonlinear midpoint remainder
G = E - A       # observed activation-barycenter gain
```

The exact accounting identity is

```text
G = J - N.
```

Because cross-entropy is convex in logits for a fixed target,

```text
J >= 0
```

for every pair up to a numerical tolerance of `1e-10`. This inequality is a
mathematical contract, not empirical evidence for symmetry or quotient
formation.

## Primary diagnostic

The registered hypothesis is deliberately narrow:

> The sign of the aggregate cross-entropy improvement at the raw final-query
> barycenter does not require favorable final-layer-normalization geometry;
> generic logit-space Jensen averaging is sufficient on both declared shifts.

A regime passes when all of the following hold:

1. source targets are pair-identical to `1e-12`;
2. the parent baseline, correct-barycenter, and semantic-reassignment
   posteriors replay within `2e-6`;
3. the exact accounting residual `|G - (J - N)|` is at most `1e-10`;
4. the minimum pairwise Jensen gain is at least `-1e-10`;
5. the observed aggregate activation-barycenter gain is at least `1e-4` nats;
6. `J >= 0.90 G`, equivalently favorable final-LN assistance can account for
   at most ten percent of the observed gain.

The primary hypothesis passes only when both regimes pass and the frozen model
state remains byte-identical.

## Secondary measurements

For each regime, also report:

- `J / G` when `G > 0`;
- `N / J` when `J > 0`;
- the fraction of pairs with positive `G`;
- the fraction of pairs on which the activation midpoint beats the logit
  midpoint;
- mean and 95th-percentile posterior Jensen--Shannon divergence between the
  two midpoint constructions;
- the centered-logit layer-normalization remainder norm relative to the
  endpoint logit-chord norm;
- the corresponding probability-space midpoint cross-entropy as a second
  convexity comparator.

These measurements describe how final layer normalization modifies the generic
averaging effect. They cannot promote a failed primary gate.

## Target-changing specificity control

Reuse the parent's fixed semantic reassignment: pair barycenter `i` is assigned
to target fiber `i xor 1`. Apply the same logit and activation-midpoint
decomposition to the reassigned source endpoints while scoring against the
current target.

Its Jensen inequality must also hold because convexity does not know whether
the pairing is semantically correct. The control is not expected to preserve
the task. Its purpose is to make explicit that `J >= 0` is nonspecific; only a
causal posterior/task gate can establish quotient sufficiency.

## Locked classifications

| Outcome | Classification | Decision |
| --- | --- | --- |
| both regimes pass and `|N| <= 0.10 J` in both | `generic_jensen_near_complete` | attribute the improvement almost entirely to generic logit convexity |
| both regimes pass, but the near-complete condition fails | `generic_jensen_sufficient_with_layernorm_modulation` | generic Jensen explains the improvement sign; retain the measured LN modulation |
| a valid regime has favorable LN assistance above ten percent | `activation_geometry_materially_assists_jensen` | the gain combines generic convexity with favorable activation-map geometry |
| a valid regime lacks material aggregate gain | `no_material_activation_barycenter_gain` | reject the parent-improvement premise for this replay |
| replay, target, identity, finiteness, or accounting contract fails | `invalid_final_query_jensen_decomposition` | stop without interpretation |

No classification establishes a whole-state quotient, a hidden scalar
quotient, population prevalence, or earlier-layer closure.

## Artifacts

- runner:
  `experiments/structure_net/tinyllm_final_query_jensen_decomposition.py`
- tests:
  `tests/structure_net/test_tinyllm_final_query_jensen_decomposition.py`
- intended result root:
  `data/experiments/tinyllm_final_query_jensen_decomposition/20260810_d8_seed7_registered`
- intended report:
  `docs/08 - Analysis/2026-08-10_tinyllm-final-query-jensen-decomposition.md`
- intended meta-hypothesis:
  `tinyllm-final-query-jensen-decomposition-v1`

