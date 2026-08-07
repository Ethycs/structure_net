# TinyLLM carrier-Jacobian axis audit

**Status:** MIXED — THIRD-AXIS CORRECTION REJECTED; LOCAL TASK METRIC VALIDATED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-carrier-jacobian-axis-audit-v1`  
**Preregistration:** [`2026-08-06_tinyllm-carrier-jacobian-axis-audit-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-carrier-jacobian-axis-audit-preregistration.md)

## Verdict

The failed cross-seed transport is not concentrated in the weak third
canonical axis. Axis 3 was dominant in `0/6` directed checkpoint pairs, far
below the preregistered `5/6` rule for a universal two-dimensional base plus
checkpoint-local scalar correction. Axes 1--2 were strictly dominant in only
`2/6`, below the reciprocal `5/6` rule. The other four pairs were mixed.

The useful positive result is sharper: the frozen target continuation is
extremely well described by its local task Jacobian at the transported scale.
All six directions passed every local-linearization gate. Predicted signed
moment error explained `0.9835--0.9941` of observed zero-referenced error
variance (mean `0.9889`), with `1.000` sign agreement and residual MAE only
`6.2--9.9%` of the observed error.

```text
transport failure
  is locally task-linear and decoder-sensitive,
  but is not a universal weak-third-axis correction.
```

The shortest justified next intervention is therefore a group-anchored map
fitted in the frozen target task metric. A typed two-channel sidecar would be
premature because discarding axis 3 alone does not remove the causal error.

## Preregistered gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| locally adequate directed pairs | **6/6** | 6/6 | pass |
| axis-3-dominant pairs | **0/6** | at least 5/6 | fail |
| axes-1--2-dominant pairs | **2/6** | at least 5/6 | fail |
| mixed pairs | **4/6** | descriptive | — |
| universal 2D base + local scalar | **not supported** | local pass + axis 3 in at least 5/6 | fail |
| shared axes causally misoriented | **not supported by strict reciprocal gate** | local pass + axes 1--2 in at least 5/6 | fail |

The campaign conclusion is
`mixed_checkpoint_stratified_carrier_atlas`. The strict shared-axis alternative
does not pass, but the shared subspace contributes more predicted RMS error
than axis 3 in every direction. That narrower descriptive result motivates a
task-metric map; it does not upgrade the failed gate.

## Directed-pair evidence

| Pair | Local signed-error R2 | Shared predicted RMS fraction | Third predicted RMS fraction | Shared-only error retained | Third-only error retained | Classification |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 7 -> 29 | 0.9871 | 0.667 | 0.333 | 0.854 | 0.409 | mixed |
| 7 -> 53 | 0.9941 | 0.700 | 0.300 | 0.883 | 0.332 | axes 1--2 dominant |
| 29 -> 7 | 0.9835 | 0.542 | 0.458 | 0.710 | 0.565 | mixed |
| 29 -> 53 | 0.9899 | 0.614 | 0.386 | 0.833 | 0.528 | mixed |
| 53 -> 7 | 0.9908 | 0.638 | 0.362 | 0.822 | 0.411 | mixed |
| 53 -> 29 | 0.9880 | 0.772 | 0.228 | 0.892 | 0.264 | axes 1--2 dominant |

The paired signed moment error itself averaged `0.156--0.297` bins across
directions. Retaining only the shared-axis coordinate error preserved
`0.710--0.892` of that error. Retaining only axis 3 preserved `0.264--0.565`.
Thus the weak axis matters in several maps, but it is neither sufficient nor
the common dominant cause.

## What the Jacobian establishes

The fine (`0.025` target standard deviations) and coarse (`0.05`) centered
finite differences agree almost exactly:

- derivative cosine: `0.99999954--0.99999993`;
- derivative relative L2 difference: `0.00037--0.00096`;
- signed-error prediction R2: `0.9835--0.9941`;
- prediction residual MAE fraction: `0.062--0.099`;
- sign agreement above the registered magnitude floor: `1.000` in every pair.

This rules out two easy explanations for the earlier causal-transport failure:
the intervention is not too large for a local analysis, and the frozen
continuation is not behaving as an opaque high-order discontinuity at this
scale. A target task metric is a quantitatively appropriate object for the
next map.

The causal retain-only controls remain essential. A derivative attribution by
itself could mislead through cancellation or curvature. Here the actual
shared-only and third-only patches agree with the broader conclusion: neither
one-axis partition gives a universal explanation, while shared-axis mismatch
is consistently substantial.

## Campaign integrity

The audit reconstructed the same three frozen 29,956,608-parameter TinyLLM
checkpoints, carrier bases, alignment cohorts, and 24 held-out cells used by
the predecessor campaign. It trained and fit nothing.

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 6 / 6 / 0 / 0 directed pairs |
| trained models / observers / coordinate maps | 0 / 0 / 0 |
| predecessor maps reproduced | 6 |
| centered finite-difference continuations | 288 |
| predecessor-map reproduction maximum | `0` in stored double precision |
| axis reconstruction relative error maximum | `2.22e-15` |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| analysis time | 13.64 seconds |
| implementation SHA-256 | `3e1b25510ecf2bbac9ab0424e283f38ac8ec660c2036e2ca77dd0c0671d1cff7` |
| campaign SHA-256 | `d953889c7103bdbececf16b9b2f450d14122509bc8305fb5235c86ad0446a6ee` |
| DVC data root | `25d2be3b471682646ba2fc4404de412a.dir` |
| lakeFS commit | `3f87160c02c0fc051335706c383f682f7f8b7e2a0d4f2123ffc5b4d3d18c5509` |

Seven audit-focused tests cover fixed configuration, canonical-map and
axis-decomposition recovery, zero-referenced linear gates, reciprocal
classification, six-pair aggregation, predecessor identity, and completed
campaign hash validation. The current producer digest exactly matches the
campaign record.

The first systems-only CUDA attempt used an earlier implementation revision
and exposed a numerical replay tolerance before producing primary evidence.
Amendment A recorded a relaxed `1e-6` map-reproduction tolerance. The
consolidated primary producer instead used a stricter `1e-10` threshold and
reproduced every stored map exactly. This conservative implementation
deviation cannot create the reported axis classification. The systems-only
two-pair artifact is not pooled.

## Interpretation for the program

The cross-seed study originally suggested a tempting model:

```text
two shared semantic axes + one checkpoint-local correction.
```

The present causal test rejects that as a universal interface. High canonical
correlation on axes 1--2 does not make their small coordinate residuals
decoder-neutral. Those shared axes account for `54--77%` of the additive
predicted RMS split in every direction, and removing their mismatch is needed
to obtain the low third-only errors seen in the two strict shared-dominant
pairs.

A better unifying description is:

```text
shared statistical carrier geometry
+ checkpoint-conditioned local task metric
= checkpoint-stratified causal atlas.
```

This is real interpretability progress because it predicts a specific frozen
intervention: weight carrier correspondence by target continuation
sensitivity while preserving exact orbit membership and group character. If
that map improves the same held-out causal endpoint, the atlas shares a typed
carrier after supplying the correct metric. If it does not, portable
post-hoc coordinates should be abandoned in favor of architecturally fixed
local charts.

## Artifacts and reproduction

- aggregate:
  `data/experiments/tinyllm_carrier_jacobian_axis_audit/20260806_d6_preregistered_diagnostic/campaign_results.json`
- per-pair records:
  `data/experiments/tinyllm_carrier_jacobian_axis_audit/20260806_d6_preregistered_diagnostic/runs/source_*_target_*/result.json`
- systems-only CUDA artifact:
  `data/experiments/tinyllm_carrier_jacobian_axis_audit/20260806_shakedown_cuda/`
- runner:
  `experiments/structure_net/tinyllm_carrier_jacobian_axis_audit.py`
- tests:
  `tests/structure_net/test_tinyllm_carrier_jacobian_axis_audit.py`

```bash
PYTHONPYCACHEPREFIX=/tmp/structure-net-carrier-jacobian-pyc-20260806 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_carrier_jacobian_axis_audit \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_carrier_jacobian_axis_audit/20260806_d6_preregistered_diagnostic
```

## Method boundaries

This is a post-outcome diagnostic on reused cells, not an independent
replication of transport failure. The Jacobian is conditioned on the frozen
answer-token decoder and does not define an intrinsic representation metric.
Canonical axes are pair-specific, and the residual writes remain off-manifold
interventions. Only three selected checkpoints are included. The result
localizes this causal failure but does not establish a population law or a
deployable shared encoder.
