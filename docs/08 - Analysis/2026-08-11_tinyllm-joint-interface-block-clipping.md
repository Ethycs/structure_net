# TinyLLM Joint-Interface Parameter-Block Clipping

**Status:** MEASURED NEGATIVE — PARAMETER-BLOCK CLIPPING INSUFFICIENT

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED POST-DIAGNOSTIC INTERVENTION`

**Hypothesis:** `tinyllm-joint-interface-block-clipping-v1`

**Preregistration:** [parameter-block clipping](../07%20-%20Status%20Reports/2026-08-11_tinyllm-joint-interface-block-clipping-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_joint_interface_block_clipping/20260811_d6_d10_preregistered/campaign_results.json`

## Verdict

Removing cross-block gradient-norm coupling does not repair the learned
physical interface. Physical-true and pair-shuffled arms both pass `0/5` in d6
and `0/5` in d10. The valid locked classification is:

```text
parameter_block_clipping_insufficient
```

The intervention was material. On the first physical update, an equivalent
single global clip would have multiplied every gradient by about
`.00194–.00339`, while independent encoder clipping retained coefficients from
`.794` through `1.0`. The learned sensor nevertheless remained compressed or
sign-reversed after 600 updates. Severe cross-block gradient suppression was
therefore real as a raw-gradient fact but was not causally sufficient to
explain the Stage A failure.

## Campaign integrity

All ten learned d6/d10 source cells completed two independent fits from exact
Stage A initial states. Every source tensor, minibatch schedule, target
permutation, held-out cohort, task floor, and initial interface digest replayed.
Only gradient clipping changed.

| Check | Result |
| --- | ---: |
| source cells requested / completed / failed | `10 / 10 / 0` |
| interface fits requested / completed | `20 / 20` |
| valid cells | `10/10` |
| exact Stage A initial-state replay | `20/20` arms |
| frozen backbones | `20/20` arms |
| checkpoint and diagnostics reload | `10/10` cells |
| physical-true passes | d6 `0/5`; d10 `0/5` |
| pair-shuffled passes | d6 `0/5`; d10 `0/5` |
| exact campaign resume | byte-stable |

Each fit used one AdamW optimizer and the same 600 sealed updates as Stage A.
The encoder, scalar embedding, and final scalar extractor were each clipped
independently to norm `1.0`; there was no subsequent global clip. Every TinyLLM
parameter remained frozen.

## Primary endpoints

The table reports five-seed physical-true means. `Pass` counts seeds meeting
correlation, conditional branch accuracy, conditional log-loss gain, and
inherited exact-bin floor simultaneously at both shifts and both cuts.

| preset | cut | shift | corr | RMSE | slope | branch acc | log-loss gain | exact acc | Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d6 | front | composition | `.2000` | `.4312` | `.2101` | `.5025` | `.000052` | `.1109` | `0/5` |
| d6 | front | extrapolation | `.1942` | `.4367` | `.2088` | `.5027` | `.000116` | `.1006` | `0/5` |
| d6 | full | composition | `.9983` | `.0322` | `.9953` | `.5027` | `.000069` | `.8191` | `5/5` cut-only |
| d6 | full | extrapolation | `.9791` | `.1090` | `.9853` | `.5045` | `.000130` | `.4426` | `2/5` cut-only |
| d10 | front | composition | `-.9989` | `.6076` | `-.1249` | `.5027` | `.000018` | `.0822` | `0/5` |
| d10 | front | extrapolation | `-.9805` | `.6100` | `-.1243` | `.5035` | `.000035` | `.0754` | `0/5` |
| d10 | full | composition | `.9990` | `.0329` | `.9995` | `.5008` | `-.000033` | `.8102` | `4/5` cut-only |
| d10 | full | extrapolation | `.9807` | `.1082` | `.9938` | `.5012` | `-.000002` | `.3941` | `1/5` cut-only |

`Pass` in the final column is descriptive at that cut; the registered seed
gate still requires every cut and shift jointly and therefore passes no seed.

D6 again mixes orientations: three seeds are positively oriented and two are
sign-reversed. All five d10 sensors remain sign-reversed. High correlation
magnitude continues to coexist with the wrong absolute physical chart.

## Comparison with Stage A

Block clipping moves the final scalar but does not fix the sensor convention.

| preset | measure | Stage A global clip | block clip |
| --- | --- | ---: | ---: |
| d6 | final logged sensor MSE | `.2027` | `.2146` |
| d10 | final logged sensor MSE | `.3560` | `.3767` |
| d6 | final logged final MSE | `.00129` | `.00108` |
| d10 | final logged final MSE | `.00135` | `.00100` |
| d6 | full composition accuracy | `.8088` | `.8191` |
| d10 | full composition accuracy | `.7324` | `.8102` |
| d6 | full extrapolation accuracy | `.4172` | `.4426` |
| d10 | full extrapolation accuracy | `.4088` | `.3941` |

The intervention modestly improves some full-depth quantities while the sensor
MSE becomes slightly worse in both architectures. This is the same failure
pattern as Stage A: downstream modules learn a useful physical-looking scalar
around a nonphysical front-end gauge.

## Why raw-gradient “starvation” was misleading under AdamW

The attribution diagnostic correctly measured clipped gradients, but its
first-order SGD language overstated their implication for AdamW. At the first
Adam update, scaling a gradient by a positive constant `c` gives approximately

```text
m_hat = c g
v_hat = c^2 g^2
m_hat / (sqrt(v_hat) + epsilon) ~= sign(g).
```

The adaptive denominator cancels most uniform gradient scaling. A global clip
coefficient of `.002` therefore does not imply an Adam parameter update that
is 500 times smaller. Time-varying coefficients can still affect moment
history, and block clipping changes relative histories across modules, but the
raw coefficient is not itself an Adam update-size ratio.

This experiment supplies the causal check: despite replacing the global
coefficient with encoder coefficients near one, the physical front end remains
miscalibrated in every seed. The correct conclusion is not that clipping was
absent, but that optimizer rescaling was not the bottleneck.

## Interpretation

The learned equivariant sensor family can encode an almost perfectly ordered
cosine-like scalar, but neither joint physical losses nor independent
parameter-block clipping force its absolute sign and scale to the declared
chart while the flexible embedding and final extractor coadapt.

The evidence now rejects both inexpensive optimizer explanations:

```text
population-wide persistent objective opposition    rejected
cross-block gradient clipping as sufficient cause  rejected causally
```

The remaining live distinction is structural:

```text
frozen continuation cannot accommodate the declared chart
                         versus
the learned sensor parameterization cannot hold it under joint training.
```

Resolving that distinction requires the already licensed full-interface stage,
not another clipping, loss-weight, or warm-start sweep.

## Next action

Preregister a full-interface causal stage over the same ten learned cells and
matched true/shuffled arms. Unfreeze the TinyLLM residual continuation while
retaining the physical sensor and final scalar endpoints, exact source data,
schedule, optimizer, and gates. The protocol must define the transformer
parameter set and clipping rule before any fit and include a frozen-backbone
block-clipped comparator from this campaign.

If full-interface training still leaves the sensor sign/scale regauged, close
flexible end-to-end physical supervision as an interface-construction method.
The next constructive architecture must fix the physical chart by construction
rather than ask a jointly adaptive path to choose it.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_joint_interface_block_clipping \
  --gpus auto --max-parallel 3 --slots-per-gpu 1 \
  --output data/experiments/tinyllm_joint_interface_block_clipping/20260811_d6_d10_preregistered
```

| Artifact | SHA-256 |
| --- | --- |
| campaign | `2f7c7cdd5494322ff89e20fb55407c6d4d8de66dde852ca9a8ec67fbc22a2349` |
| result manifest | `a78a57e3a2bf0f44946a8b3081a64a3bb915ef1e3426c068a765bba70b0e6d69` |
| diagnostics manifest | `16ec8c5e312b96d1b2880049fcadadc4ee1f466be172a4aa052c1f0b3dd160df` |
| interface manifest | `156144e9f6a1bdbebb90262f5c38417c1c238101f3f471df6c8fab33c146ecfd` |
| implementation | `e69d570234fb727c8ee243d7fb5e72f5a4abaa8ad749f6d9dce0f54794b2a895` |
| producing runner | `7dc67af033523e7819b0f7cadeac9848508ce2cbe9cce201bef67c6e26dcacf6` |
| preregistration | `4264a56a0be6f70fc5c4e812f2fef8aadcdc541450d51993f5f1bbc52fc26f6f` |
| campaign fingerprint | `f25912d98c0e1ad11bd0bf115e093443ecf761cd544abe07fc608d8bad3aeb26` |

The primary root is `16 MB`. The valid systems-only shakedown remains separate
and is excluded from the population result.

## Data and evidence backup

The complete repository data tree is tracked by DVC root
`36b9618c7464926fb4c18197d2b56bc4.dir` (`49,138,566,121` logical bytes,
`3,876` files). DVC pushed 70 new objects and reports the cache and `lakefs`
remote in sync.

lakeFS commit
`e1653d5f4de8c341358666d68ff8c6071b9706322212af9728dc301248700836`
seals the object graph on `artifacts/main`, with parent
`6fcd351bd133a26b15229c669166622356cc0ae8e7422b5767f2e420d89160f3`.
The branch diff is empty after commit. Direct object checks recover the DVC
root checksum `36b9618c7464926fb4c18197d2b56bc4`, campaign MD5
`ebc31dba4d1bfa847002779e79d58ca9`, and meta-record MD5
`1dcd2df540e788092f7d287c119c5ed6`.
