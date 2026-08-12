# TinyLLM Joint Physical Scalar Interface

**Status:** MEASURED NEGATIVE — FROZEN BACKBONE JOINT INTERFACE INSUFFICIENT

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`

**Hypothesis:** `tinyllm-joint-physical-scalar-interface-v1`

**Preregistration:** [joint physical scalar interface](../07%20-%20Status%20Reports/2026-08-11_tinyllm-joint-physical-scalar-interface-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_joint_physical_scalar_interface/20260811_d6_d10_preregistered/campaign_results.json`

## Verdict

Joint supervision around a frozen TinyLLM backbone did **not** create a stable
physical scalar interface in the learned equivariant family. The analytic
positive control passed `d6 5/5` and `d10 4/5`; every pair-shuffled control
passed `0/5`; but the learned arm passed `0/5` in both d6 and d10. The locked
classification is:

```text
frozen_backbone_joint_interface_insufficient
```

The failure is more specific than a general continuation failure. In the
learned conditions, the final scalar was strongly correlated with physical
cosine and often task-adequate, while the declared front-end scalar remained
mis-scaled or sign-reversed. The trainable interface therefore recreated a
checkpoint-specific internal gauge instead of carrying the same physical
coordinate from sensor to answer.

## Campaign integrity

| Check | Result |
| --- | ---: |
| source cells requested/completed/failed | `20 / 20 / 0` |
| matched interface fits completed | `40 / 40` |
| source replay maximum error | `1.979e-7` |
| frozen backbone unchanged | `20/20` cells, both arms |
| interface checkpoint reload | `40/40` |
| diagnostics exact reload | `20/20` |
| finite records and arrays | `20/20` |
| pair-shuffled joint passes | `0/20` |
| maximum CUDA allocation | `0.4754 GB` |
| summed cell wall time | `2,030.41 s` |
| exact campaign resume | byte-stable |

All cells reused the sealed 4,096-example source training tensor and complete
600-step pair schedule. Only the learned encoder when present, the scalar
embedding, and a one-dimensional final extractor were trainable. Every token,
position, attention, MLP, layer-normalization, and LM-head parameter remained
frozen by state digest.

## Preregistered population gates

| stratum | true joint passes | shuffled passes | family gate |
| --- | ---: | ---: | --- |
| d6 analytic | `5/5` | `0/5` | pass |
| d6 learned equivariant | `0/5` | `0/5` | **fail** |
| d10 analytic | `4/5` | `0/5` | pass |
| d10 learned equivariant | `0/5` | `0/5` | **fail** |

The primary architecture-family claim required at least four jointly passing
seeds in every analytic and learned stratum, plus at most one shuffled pass.
The specificity and analytic-control gates passed. The learned-family gate did
not.

## Primary endpoints

The table reports means over five seeds. `Pass` is the number of seeds meeting
the full per-cell endpoint at that cut and shift: direct scalar correlation at
least `.90`, conditional branch balanced accuracy at most `.55`, conditional
log-loss gain at most `.02`, and exact interval-decoder accuracy above that
checkpoint's inherited task floor.

| stratum | cut | shift | corr | RMSE | branch acc | log-loss gain | exact acc | Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d6 analytic | front | comp | `.9979` | `.0350` | `.5008` | `-0.00004` | `.7920` | `5/5` |
| d6 analytic | front | extra | `.9918` | `.0696` | `.4992` | `.00001` | `.6543` | `5/5` |
| d6 analytic | full | comp | `.9977` | `.0393` | `.5018` | `-0.00006` | `.7668` | `5/5` |
| d6 analytic | full | extra | `.9917` | `.0715` | `.4979` | `-0.00004` | `.6320` | `5/5` |
| d6 learned | front | comp | `.1996` | `.4149` | `.5010` | `.00004` | `.1230` | `0/5` |
| d6 learned | front | extra | `.1952` | `.4210` | `.5084` | `.00013` | `.1098` | `0/5` |
| d6 learned | full | comp | `.9984` | `.0332` | `.5002` | `-.00000` | `.8088` | `5/5` |
| d6 learned | full | extra | `.9771` | `.1148` | `.5080` | `.00011` | `.4172` | `2/5` |
| d10 analytic | front | comp | `.9979` | `.0350` | `.5008` | `-0.00004` | `.7920` | `5/5` |
| d10 analytic | front | extra | `.9918` | `.0696` | `.4992` | `.00001` | `.6543` | `5/5` |
| d10 analytic | full | comp | `.9973` | `.0438` | `.5021` | `-0.00006` | `.7377` | `4/5` |
| d10 analytic | full | extra | `.9914` | `.0736` | `.4965` | `-0.00004` | `.6129` | `4/5` |
| d10 learned | front | comp | `-.9989` | `.5910` | `.5020` | `.00003` | `.0914` | `0/5` |
| d10 learned | front | extra | `-.9806` | `.5952` | `.5037` | `.00004` | `.0787` | `0/5` |
| d10 learned | full | comp | `.9989` | `.0417` | `.5041` | `.00009` | `.7324` | `4/5` |
| d10 learned | full | extra | `.9807` | `.1106` | `.5045` | `.00008` | `.4088` | `2/5` |

The d6 learned front-end mean correlation hides seed heterogeneity: seeds 17,
29, and 41 were positively oriented, while seeds 7 and 53 were sign-reversed.
All five d10 learned front ends were sign-reversed. Across learned cells the
front-end scalar was usually almost an affine copy of cosine in magnitude, but
with the wrong sign, slope, or offset. For example, the d10 front-end slopes
ranged from about `-0.075` to `-0.128`, rather than the physically declared
unit slope.

## What changed relative to the frozen comparator

The previous frozen typed endpoint readout passed d6 analytic `5/5`, d6
learned `4/5`, d10 analytic `5/5`, and d10 learned `1/5`. Joint interface
training did improve the **full-depth** typed scalar in many learned cells:

- d6 full-depth composition passed `5/5` and extrapolation `2/5`;
- d10 full-depth composition passed `4/5` and extrapolation `2/5`.

That narrower improvement does not rescue the hypothesis because the physical
front-end coordinate passed `0/5` in both learned strata. It is evidence that
the frozen continuation can support a useful one-dimensional answer scalar,
not that the learned sensor and continuation agree on its physical meaning.

The negative control is decisive for specificity: none of the 20
pair-preserving shuffled fits passed. The true result is not generic fitting
capacity or target-marginal leakage.

## Optimization diagnostic visible in the locked run

The final logged sensor MSE averaged `.203` for d6 learned and `.356` for d10
learned, while final-state MSE averaged only `.00129` and `.00135`. Pre-clip
combined gradient norms averaged `14.6` and `33.5` at the last update against a
global gradient ceiling of `1.0`.

These measurements are consistent with the task/final objectives dominating
the shared clipped update while the learned sensor remains in a compressed or
sign-reversed gauge. They do **not** by themselves prove gradient competition;
the campaign did not record per-objective gradient vectors. This is a
post-outcome mechanistic hypothesis, not a reinterpretation of the failed
primary gate.

## Interpretation

The experiment rejects the idea that merely attaching physical losses at both
ends of a frozen continuation forces one stable coordinate through the entire
learned path. A flexible scalar embedding and final extractor can coadapt
around a nonphysical sensor scalar, even when both are nominally supervised.

The theory has not collapsed. The result strengthens its distinction between:

```text
decodable or task-useful one-dimensional state
                       !=
one declared physical coordinate shared across interfaces
```

It also puts a sharper engineering constraint on typed interfaces: physical
typing must be enforced by construction or by an optimization protocol that
cannot trade the sensor convention against a downstream inverse map.

## Next action

The preregistered stop rule licenses a full-interface extension, but unfreezing
the transformer is not yet the shortest decisive move. The observed failure is
already present at the learned sensor output while the frozen full-depth scalar
often succeeds.

First run an artifact-only per-objective gradient attribution on the saved
source and trained interface states. Measure sensor-loss, final-MSE, and task-CE
gradient norms and cosine conflicts separately for the encoder, scalar
embedding, and final extractor, before and after the global clip. This requires
no retraining and directly tests whether the registered equal-weight/global-
clip optimizer prevented the physical sensor objective from taking effect.

- If strong gradient domination/conflict is absent, preregister the licensed
  full-interface fine-tuning stage.
- If it is present, the more decisive prospective intervention is an
  architecturally fixed physical sensor output or separately normalized
  parameter-block updates, not a larger unrestricted fine-tune.

Do not tune Stage A, add an endpoint-only map, or reinterpret full-depth-only
success as a passing physical interface.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_joint_physical_scalar_interface \
  --gpus auto --max-parallel 3 --slots-per-gpu 1 \
  --output data/experiments/tinyllm_joint_physical_scalar_interface/20260811_d6_d10_preregistered
```

Exact resume leaves the completed campaign bytes unchanged.

| Artifact | SHA-256 |
| --- | --- |
| campaign | `65ab4b4e887212c4754cf918908cd5e3f04727af4d7876de1d3aa749bc50ac51` |
| result manifest | `3299a6cd2edf8816b8bb65ef1ddfb7dfc18f0f6edd41ff09730af632580fc9f3` |
| diagnostics manifest | `1f69d3223ae8eb75ff52235be2db326ccf60ecc7f978f1ee89649c7ddb5ba778` |
| interface-checkpoint manifest | `50dec4731b55d118561e36f0ff35e8bbcadb3fbca856f7cd234151332e09322a` |
| combined implementation | `ac1f5b42e2e8bcfc645de7dba048ae258e692db79f09b20f4e25f8d00940e1a1` |
| producing runner | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |
| preregistration | `1f83fb2802340a51f4dc281f898e99122b99089c48e6c207bd46626c20aab838` |
| composition cohort | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation cohort | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |

The primary artifact root is 29 MB. The two invalid systems-development roots
and the valid systems-only shakedown roots remain separate and are excluded
from every scientific aggregate.

## Data and evidence backup

The complete repository data tree is tracked by DVC root
`1fd820449450751e0692cbd7c38ec5ca.dir` (`49,111,122,412` logical bytes,
`3,661` files). DVC pushed 148 new objects and reports the cache and `lakefs`
remote in sync.

lakeFS commit
`0891afcfbdabe4dc4f2f8c2ba299ad5e54aa1fd9d37318e88a1991b6f35ad896`
seals the object graph on `artifacts/main`, with parent
`9a3fd4f4c462fbc4e4dc9270d4defca53dd3eb3fb333d9e3340b81ec0d824c77`.
The branch diff is empty after commit. Direct object checks recover the DVC
root checksum `1fd820449450751e0692cbd7c38ec5ca`, campaign MD5
`517e05e534efbdec01e184b6e247b0a5`, and meta-record MD5
`f114f0e5a63bf52959a2519447203780`.
