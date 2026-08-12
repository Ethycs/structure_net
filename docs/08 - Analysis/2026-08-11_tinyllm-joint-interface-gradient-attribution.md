# TinyLLM Joint-Interface Gradient Attribution

**Status:** MEASURED NEGATIVE — NO REGISTERED POPULATION-WIDE GRADIENT FAILURE MECHANISM

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `REGISTERED POST-OUTCOME CORRECTIVE`

**Hypothesis:** `tinyllm-joint-interface-gradient-attribution-v2`

**Preregistration:** [v2 numerical-validity correction](../07%20-%20Status%20Reports/2026-08-11_tinyllm-joint-interface-gradient-attribution-v2-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_joint_interface_gradient_attribution/20260811_d6_d10_registered_v2/campaign_results.json`

## Verdict

The saved Stage A objective geometry does not establish one population-wide
gradient mechanism for the failed learned physical interface. The valid locked
classification is:

```text
no_registered_gradient_failure_mechanism
```

Initial pure cross-block starvation passes d6 `5/5` but d10 only `2/5`.
Persistent trained-state objective conflict passes d6 `0/5` and d10 `2/5`.
Both registered population gates require at least four of five seeds in each
architecture, so both fail.

This result rejects the strong explanation that the same registered clipping
or conflict condition accounts for the learned-interface failure across d6 and
d10. It does not make the original optimizer benign. Descriptively, the
zero-initialized final extractor creates a very large head gradient and causes
severe extra encoder suppression in every learned cell. Three d10 cells fail
the stricter initial gate only because their encoder gradient would also be
clipped slightly under a separate block clip.

## Campaign integrity

The corrective replay used all twenty sealed Stage A cells and performed no
fit or optimizer update. For each exact initial and saved final interface
state, it differentiated the sensor MSE, final MSE, and task cross-entropy
separately on the first and last rows of the sealed minibatch schedule.

| Check | Result |
| --- | ---: |
| requested / completed / failed | `20 / 20 / 0` |
| gradient snapshots | `80` |
| optimizer steps / trained parameters | `0 / 0` |
| valid cells | `20/20` |
| maximum absolute additivity error | `1.1444091796875e-5` |
| absolute ceiling | `2e-5` |
| maximum relative additivity error | `1.6847193548889924e-7` |
| relative ceiling | `1e-6` |
| exact campaign resume | byte-stable |

V1 remains preserved and invalid because five cells exceeded its frozen
absolute-only `1e-5` ceiling by `1.444091796875e-6`. V2 changed only numerical
validity: it requires both the corrected absolute and scale-aware relative
checks. The scientific gradients, batches, states, thresholds, and
classification rules are unchanged, and partially exposed v1 outcomes are not
treated as fresh confirmation.

## Registered gates

| stratum | initial starvation | required | final conflict | required |
| --- | ---: | ---: | ---: | ---: |
| d6 learned calibrated equivariant | `5/5` | `4/5` | `0/5` | `4/5` |
| d10 learned calibrated equivariant | `2/5` | `4/5` | `2/5` | `4/5` |

The analytic cells are fixed-sensor controls and do not enter either learned
encoder gate. All ten are valid.

The learned seedwise results are:

| preset | seed | initial starvation | final conflict |
| --- | ---: | --- | --- |
| d6 | 7 | pass | fail |
| d6 | 17 | pass | fail |
| d6 | 29 | pass | fail |
| d6 | 41 | pass | fail |
| d6 | 53 | pass | fail |
| d10 | 7 | fail | fail |
| d10 | 17 | pass | pass |
| d10 | 29 | fail | fail |
| d10 | 41 | pass | pass |
| d10 | 53 | fail | fail |

## What happens at initialization

The final scalar extractor is exactly zero at the registered initial state.
Consequently, final/task gradients do not yet reach the sensor encoder, while
the final-scalar block itself receives gradients with norms in these ranges:

| preset | final-head total norm | global clip coefficient | encoder sensor norm | encoder cross-block suppression |
| --- | ---: | ---: | ---: | ---: |
| d6 | `294.7–423.1` | `.00236–.00339` | `.341–.709` | `.00236–.00339` |
| d10 | `388.5–565.4` | `.00177–.00257` | `.710–1.260` | `.00186–.00258` |

Thus the one global clip multiplies the direct sensor update by roughly
`.18%–.34%` of the corresponding separately clipped block update in all ten
learned cells. D6 satisfies the preregistered pure cross-block gate because
all encoder norms are below `1.0`. D10 seeds 7, 29, and 53 fail because at
least one locked minibatch has an encoder norm slightly above `1.0`, giving a
separate encoder block-clip coefficient between `.794` and `.962` rather than
exactly `1.0`.

That distinction matters. The registered claim was deliberately narrow:
unrelated blocks must be the only source of clipping. The descriptive result
is broader: the final head supplies an enormous additional suppression in
every learned checkpoint. The latter cannot promote the failed population
gate after outcome inspection.

## What happens at the trained state

Final-state downstream gradients are heterogeneous rather than persistently
opposed. D6 passes no seed: even when one locked batch has a negative sensor
descent ratio, the other does not. D10 seeds 17 and 41 are the only cells with
nonpositive sensor descent ratios on both batches. Other d10 seeds change sign
between batches or remain aligned on at least one batch.

This rejects a stable story in which downstream objectives universally force
the learned encoder away from physical cosine. It is consistent with
checkpoint- and minibatch-dependent competition, but local heterogeneity is
not the registered population mechanism.

## Interpretation

The Stage A failure remains exactly what it was: learned front ends pass `0/5`
in both architectures while analytic controls pass and shuffled controls fail.
Gradient attribution neither rescues nor relabels that result.

It establishes three narrower facts:

1. the initialization couples a zero head's very large gradient to the sensor
   through one global clip;
2. the registered pure-starvation condition is architecture-dependent;
3. persistent sensor-versus-downstream opposition is not population-stable.

Therefore unfreezing the whole transformer cannot be justified as a remedy for
one confirmed gradient pathology. It remains licensed by the Stage A stop
rule, but it is not yet the shortest causal intervention.

## Next action

Before a high-dimensional full-interface fine-tune, preregister one prospective
frozen-backbone optimizer intervention: repeat the learned physical and
pair-shuffled arms with parameter-block clipping for encoder, scalar embedding,
and final extractor. Keep source states, examples, schedule, objectives,
updates, learning rate, and endpoints fixed.

This is the direct causal test of the universal descriptive suppression:

- if both learned populations reach `4/5` while shuffled controls remain at
  most `1/5`, global cross-block coupling caused the typed-interface failure;
- if they do not, close the optimizer branch and execute the already licensed
  full-interface stage without a loss-weight or warm-start sweep.

The original global-clip cells remain a sealed external comparator. Do not
retune Stage A or reinterpret this post-outcome diagnostic as prospective
evidence.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_joint_interface_gradient_attribution \
  --gpus auto --max-parallel 3 --slots-per-gpu 1 \
  --output data/experiments/tinyllm_joint_interface_gradient_attribution/20260811_d6_d10_registered_v2
```

| Artifact | SHA-256 |
| --- | --- |
| campaign | `a3540216800a0cccf0d3725cf349f8a5c91bf01b8680d44c814afd8f4fa6ba25` |
| result manifest | `84af30bd069bc62d963090442383330af5c8acc3f935fdd346d855f4600c31b2` |
| diagnostics manifest | `638047a25c77ffba507c59a9b5442d0002a45e98831b36976d7576ed9146daf4` |
| implementation | `550f25b276ecd502f73f8b220af4ac1eff5c1ec73eb8838c1f6d27234a2d7183` |
| producing runner | `75b9222673e543657efd801226149fc154e3c3993f39d05148519c883eebcc23` |
| v2 preregistration | `74aef578a00b4cd71cfdd7a94852cb65ba1bef76e3b13be6e1b12b9acab4cbdd` |
| campaign fingerprint | `e9a3fc91320ce5dada8c2fe5f610b0c45a0310279a7afba26a5c683a7928faad` |

The valid v2 artifact root is `5.3 MB`. The invalid v1 root and valid d6
systems-only shakedown remain separate and are excluded from this aggregate.

## Data and evidence backup

The complete repository data tree is tracked by DVC root
`15b6b2f87023a8f91e154eed8bd0a3cf.dir` (`49,122,341,599` logical bytes,
`3,808` files). DVC pushed 129 new objects and reports the cache and `lakefs`
remote in sync.

lakeFS commit
`6fcd351bd133a26b15229c669166622356cc0ae8e7422b5767f2e420d89160f3`
seals the object graph on `artifacts/main`, with parent
`0891afcfbdabe4dc4f2f8c2ba299ad5e54aa1fd9d37318e88a1991b6f35ad896`.
The branch diff is empty after commit. Direct object checks recover the DVC
root checksum `15b6b2f87023a8f91e154eed8bd0a3cf`, campaign MD5
`86f3962e2e9627f854ea561e957040df`, and meta-record MD5
`dac97e7996ba09cfb7ed0b02b17b048b`.
