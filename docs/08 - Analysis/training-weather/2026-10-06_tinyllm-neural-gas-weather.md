# TinyLLM from-scratch training weather: completed exploratory study

**Completed:** 2026-10-06 16:10 PDT / 23:10 UTC.  
**Classification:** `exploratory_tinyllm_weather`.  
**Study:** `tinyllm-neural-gas-weather-v1`.

The requested visual deliverable is the standalone forecast dashboard at
`data/visualizations/tinyllm_weather/20261006_forecast_v2/index.html`, with a
printable `forecast.png` and vector `forecast.svg` beside it. It replays saved
forecasts with issue-time scrubbing, 100/300/600-update outlooks, projected
ensemble paths and spread, layer activity, and an optional observed-outcome
overlay. The visualization reads existing artifacts and does not train or refit
models. The quantitative results below document the source campaign.

All six independently initialized TinyLLM models completed 12,000 actual
optimizer updates. The neural-gas map and ensemble forecasters were fitted to
four training trajectories and evaluated on two held-out model seeds. The
full pipeline completed in approximately 17 hours 40 minutes, with no invalid
forecast records.

The neural-gas forecaster achieved slightly lower pooled probe-loss error than
the global ensemble. Compared with persistence, its point predictions were
better at 300 and 600 updates and worse at 100 updates. Its uncertainty
intervals were too narrow: nominal 90% loss intervals covered approximately
76–77% of observations. These two held-out seeds provide exploratory evidence,
not a general validation of training prediction or an exhaustive fold census.

## Training and artifact verification

Each model has 51,082,752 parameters, eight transformer blocks of width 512,
context 256, and vocabulary 50,257. All weights were initialized from scratch;
initialization hashes differ across all six seeds. The existing BabyLM corpus,
AdamW recipe, 200-update warm-up, cosine learning-rate decay, and effective
batch size 24 were fixed before this campaign. Each model processed 73,728,000
training tokens; the campaign processed **442,368,000 tokens** in total.

| Seed | Forecast role | Final diagnostic-probe loss | Final validation loss |
| --- | --- | ---: | ---: |
| 7 | Fit | 6.3459 | 4.8822 |
| 17 | Fit | 6.4580 | 4.8045 |
| 29 | Fit | 6.6514 | 4.8572 |
| 41 | Fit | 6.4906 | 4.8853 |
| 101 | Held-out | 6.3178 | 4.8034 |
| 131 | Held-out | 6.5962 | 4.8801 |

Losses are cross-entropy in nats. The diagnostic probe is one fixed 32-token
sequence. Broader validation uses 98,304 sampled token positions per evaluation
from the reserved suffix, with fixed per-seed samples. These measurements have
different sampling scopes and are not interchangeable. This experiment does
not include a broad language-generation or task-quality benchmark.

All six final checkpoints passed exact logit replay. Ten AMP overflow attempts
were retried successfully across the campaign; they did not count toward the
72,000 optimizer updates. Completion review verified hashes for all six model
checkpoints, all six telemetry archives, all six fold-audit files, the corpus,
current producing sources, and the retained source copies. The GPU runner has
exited and released its GPU allocation.

The preflight suite passed 26 tests. A separate full-size GPU recovery check
also reproduced final parameters, losses, batch indices, and telemetry bitwise
after resuming at update 8 of a 16-update lifecycle check. These checks establish
pipeline behavior, not forecast skill.

## Held-out forecast results

There are 56 rolling forecast origins per held-out seed, four forecast arms,
and three horizons: **1,344 scored records**, all numerically valid. Each table
entry pools 112 predictions. Overlapping forecast origins are dependent; the
number of independent held-out model seeds is two.

Probe-loss RMSE, in nats, with lower values better:

| Lead in optimizer updates | Neural gas | Global ensemble | Persistence | Recent trend |
| --- | ---: | ---: | ---: | ---: |
| 100 | 0.1171 | 0.1176 | **0.1022** | 0.1587 |
| 300 | **0.1380** | 0.1412 | 0.1422 | 0.3321 |
| 600 | **0.1422** | 0.1473 | 0.1943 | 0.6583 |

At 600 updates the neural-gas loss RMSE is approximately 26.8% lower than
persistence and 3.5% lower than the global ensemble. At 100 updates it is
approximately 14.6% higher than persistence. Gains over the global ensemble
are not uniform across seeds: at 600 updates neural gas is worse on seed 101
and better on seed 131. No significance claim is made.

Loss-distribution CRPS (lower is better) and empirical coverage:

| Lead | Neural-gas CRPS | Global CRPS | Persistence CRPS | Trend CRPS | Neural-gas 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.0662 | 0.0682 | 0.0775 | 0.1225 | 76.8% |
| 300 | 0.0791 | 0.0819 | 0.1098 | 0.2563 | 75.9% |
| 600 | 0.0831 | 0.0850 | 0.1436 | 0.5205 | 76.8% |

Persistence and trend are deterministic forecasts, so their distributions are
point masses; they do not provide uncertainty intervals. Lower neural-gas CRPS
does not establish calibrated uncertainty. Global loss-interval coverage is
77.7%, 77.7%, and 80.4%, also below nominal 90%.

For the whole 22-field state, neural-gas standardized RMSE is 0.2845, 0.3690,
and 0.4197 versus persistence at 0.2363, 0.2909, and 0.3509. Thus the loss-only
improvements do not establish better point prediction of the entire measured
training state. Loss-increase Brier scores for neural gas are 0.2175, 0.2408,
and 0.2333, versus global scores of 0.2189, 0.2386, and 0.2324: there is no
consistent neural-gas advantage on that event score either.

The map and normalization were fitted only on seeds 7, 17, 29, and 41.
Held-out measurements enter causal ensemble assimilation as they become
available; future measurements are not supplied to forecasts. The future
learning-rate schedule is known. This evaluates short-range rolling forecasts,
not an unobserved simulation of all 12,000 updates from initialization.

## A₂ fold coverage

The campaign recorded **726 derivative-audit snapshots** in the declared
two-gain, final-block, last-probe-token loss slice. Bounded multistart searches
ran at updates 0, 6,000, and 12,000 on seed 101. They returned **zero verified
slice A₂ candidates**.

This is not a proof of absence, including within the bounded slice: the search
is not exhaustive. It also does not satisfy an exhaustive census of all
minimum–saddle mergers in the full network. Full-space stationarity and
completeness remain unverified. The diagnostic linear unfolding term is not
part of the pretraining objective.

## Retained outputs

Artifact root, relative to the repository:

```text
data/experiments/tinyllm_neural_gas_weather/20261005_full_from_scratch_v1/
```

It contains the immutable manifest and source copies, six model checkpoints
and their optimizer recovery checkpoints, telemetry and fold audits, fitted
global and neural-gas forecasters, per-seed forecast arrays and scores,
`campaign_results.json`, and `tinyllm_weather.png` / `tinyllm_weather.svg`.
The figure shows seed 101 for the map, example forecasts, and layer-activity
panels; its score panel aggregates both held-out seeds.

Scientific fingerprint:

```text
456e1fdd65eeac4e62edc55050502061675aeff61875541e5e22cfc92dd01fff
```

See the [frozen study design](../../01%20-%20Design/training-weather/tinyllm-neural-gas-weather.md)
and [operational runbook](../../07%20-%20Status%20Reports/training-weather/2026-10-05_tinyllm-neural-gas-weather-running.md).
Producing files and completed evidence were not changed during completion review.
