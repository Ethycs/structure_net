# Weather-inspired training simulation

**Status:** EXPLORATORY prototype design, written before pilot outcomes  
**Date:** 2026-10-05  
**Study ID:** `training-weather-a2-v1`

## Question and interpretation

Can an ensemble dynamical model, periodically corrected by measurements, forecast
neural-network training trajectories and abrupt changes in predictions better
than persistence and recent-trend forecasts on independent training seeds?

This first study is a small-network exploratory calibration pilot. Initial
sandbox checks hid the GPUs; an outside-sandbox check established access to
physical GPU 2, an RTX 2060 SUPER. CPU numerical tests precede CUDA training.
The user's subsequent requirement to track training-landscape A₂ folds changed
the original classifier plan before any outcomes were collected. The final
pilot below uses a neural loss whose complete fold set is analytically known.
It does not alter or reopen any completed TinyLLM campaign. Results are
descriptive: there is no confirmatory pass gate, and negative comparisons remain
valid pilot results.

## Translation from weather forecasting

| Weather concept | Training analogue implemented here |
| --- | --- |
| Observed atmospheric state | Neural weight and transverse parameter at each optimizer update |
| Forecast dynamics | Controlled, second-order polynomial surrogate learned from training runs |
| Ensemble uncertainty | Bootstrap independent seed groups for model coefficients; sample residual process noise |
| Data assimilation | Ensemble Kalman updates using measurements available at the forecast issue time |
| Known external forcing | Planned loss-control coefficient, regression target, and learning rate |
| Hindcast verification | Held-out model seeds; frozen forecasts scored at several lead times |
| Weather-event probability | Probability that the tied neural weight crosses zero within the horizon |

The method borrows the ensemble/assimilation workflow; it does not assert that
SGD obeys atmospheric fluid equations. The fitted surrogate approximates the
weight dynamics and represents momentum through the previous increment.

## Fixed initial pilot

- Two tied scalar linear layers produce `w*w*x`, plus an independent transverse
  output `b`. Inputs are 1. Population loss is
  `L(w,b; alpha,target) = (w*w-target)^2/4 + b*b/2 + alpha*w`.
  The linear regularizer `alpha*w` is an intentional unfolding control; it is
  not a conventional weight-decay penalty.
- Independent RNG streams for initialization, training examples, validation
  (not used in this analytic family), minibatches, assimilation, and forecasts.
- Gaussian minibatch target noise: standard deviation 0.15 on the squared-path
  target and 0.1 on the transverse target. Initialization noise has standard
  deviation 0.08, around `(w,b)=(1.15,0)` for rising control and `(-1.15,0)` for
  falling control. Minibatch realizations are paired across these two schedules
  within a seed; seeds are independent, schedules are not extra replicates.
- SGD, momentum 0.8, learning rate 0.08, batch size 64, 240 optimizer updates,
  measurement every update.
- Control `alpha` sweeps from -0.6 to 0.6 or the reverse. The population target
  is `1 + 0.15*sin(2*pi*step/240)`. Both schedules occur in fit and test runs.
- Fit seeds: 7, 17, 29, 41, 53, 67. Test seeds: 101, 131, 167.
- 32 ensemble members, ridge penalty 1.0, covariance shrinkage 0.1,
  multiplicative anomaly inflation 1.02, observation standard deviation 0.05
  in fit-standardized units. The observation error is a modeling assumption,
  not a measured sensor error or learned calibration.
- Polynomial degree 3, including current-state/control interactions. Dynamics
  regress the next increment on standardized state, previous increment,
  state powers, and known controls; bootstrap units are independent seeds,
  retaining both paired schedules together.
- Forecast horizons: 1, 5, 15 optimizer updates. Issue every 5 updates after
  10 warm-up updates.
- Forecast event: weight `w` changes sign relative to its value at issue time
  during the horizon. This is a branch-change proxy, not itself an A₂ test.
  Dynamic switching may lag behind the landscape's fold crossing.
- Baselines: persistence, last-interval linear trend, and the same ensemble
  propagated without subsequent observations. Deterministic baselines have
  degenerate predictive distributions and 0/1 event probabilities.

Score standardized RMSE and ensemble CRPS (lower is better), empirical coverage
and width of central 90% intervals, and event Brier score. Report each test run
and lead time before averaging equally across equal-length test trajectories.
There are three independent test seeds, each with two paired schedules.
Overlapping forecasts are dependent; do not attach independent-sample
significance claims. Compare
assimilated forecasts to both simple baselines, not just the uncorrected model.

## Data and numerical contract

Standardization, coefficients, and residual covariance use only fit runs.
Controls at row `t` drive the transition from row `t` to `t+1`. Future controls
must be known at issue time; they cannot include future measured loss or
gradients. Test observations are assimilated only through the issue checkpoint.
Forecast RNG streams are separate from assimilation so issuing extra forecasts
cannot change the filtered state. Missing observations can be represented by
NaN during assimilation, but fitting and scoring require finite data.

The state variables are unrestricted weights, so there is no artificial
clipping or positivity projection. Nonfinite rollouts are explicit failures.
Gaussian EnKF updates can poorly represent distributions spanning two branches;
empirical interval coverage and event scores must be reported, not assumed.

## A₂ fold detection and completeness

For a smooth loss `L(theta,alpha)`, solve the augmented equations
`gradient(L)=0`, `Hessian(L)*v=0`, and `v.T*v=1`. Verify:

1. Stationarity residual at most `1e-7`.
2. Exactly one Hessian eigenvalue with magnitude at most `1e-6`.
3. Every other Hessian eigenvalue greater than `1e-6` (minimum/index-one-saddle).
4. Nonzero null-direction cubic `D³L[v,v,v]`, magnitude greater than `1e-5`.
5. Nonzero unfolding coupling `v.T * partial_alpha gradient(L)`, magnitude
   greater than `1e-5`.

The generic detector uses double precision and multistart root solving, retaining
failed attempts, out-of-domain roots, degeneracies, and duplicates. The dense
Hessian budget defaults to 32 parameters and is explicit/configurable. Its
output always says `unverified_multistart_search`: finding candidates cannot
establish global completeness. Autograd compatibility and a smooth loss are
required; ReLU kinks cannot be classified by these smooth A₂ conditions.

For the calibration network, stationarity forces `b=0`; the Hessian is
`diag(3*w*w-target, 1)`. Therefore the **entire** A₂ set is exactly
`w=+-sqrt(target/3)`, `alpha=target*w-w^3`, with cubic `6*w` and coupling 1.
For every checkpoint, enumerate both folds, verify all derivatives, and retain
track identities as the target varies. Independently run multistart search at
the initial, middle, and final target values, checking it recovers both folds.
The fold catalogue is shared across seeds and schedules because they use the
same population-loss family. This proof covers the population loss, not every
random minibatch loss.

Generic temporal matching uses gated minimum-cost coordinate assignment and
reports ambiguous and unmatched tracks. It does not relabel failed detections
as physical births or deaths. Parameter scaling and search domains must be
chosen for each application.

Stop training on nonfinite model values, invalid traces, failed numerical tests,
or checkpoint replay mismatch. Do not rerun a poor result with tuned parameters
under the same output root. Each execution creates a new directory, records
full configuration and producing source hashes, and preserves raw traces,
checkpoints, forecast arrays, per-run metrics, and strict-JSON aggregation.

### Lifecycle amendment, before full pilot

`20261005_shakedown_cuda_v1` trained all six tiny lifecycle trajectories, then
the uncorrected cubic surrogate overflowed. That failed artifact and its source
snapshot remain untouched. The evaluator now stops a failing ensemble rollout,
records an invalid forecast (without dropping any ensemble members), and allows
other arms to be evaluated. If any requested forecast for an arm/lead fails,
its aggregate scores are null and failure counts are displayed; it cannot look
better by averaging only survivors. If a filter's state itself fails, subsequent
forecasts from that filter remain invalid. A failure at the longest requested
lead conservatively invalidates all leads at that issue time. No optimizer,
forecast coefficient, seed, or model hyperparameter was tuned in response.

## What remains beyond this prototype

Applying this to TinyLLM requires independent training histories with aligned
metrics, optimizer/schedule context, and held-out seeds. Tracking every fold of
an arbitrary large network would additionally require an exhaustive search or
proof over a declared loss/control domain; the generic detector does not supply
that certificate. Weight symmetries, nonsmooth activations, and multiple zero
Hessian modes require separate treatment. Adaptive local refinement and
catastrophe-preserving compression remain future experiments.

## Method sources

- [ECMWF: quantifying forecast uncertainty](https://www.ecmwf.int/en/research/modelling-and-prediction/quantifying-forecast-uncertainty)
  explains initial-state and model uncertainty in ensemble forecasts.
- [Mandel, 2009: A Brief Tutorial on the Ensemble Kalman Filter](https://arxiv.org/abs/0901.3725)
  describes sample-covariance data assimilation and its Gaussian approximation.
- [Kuznetsov: Saddle-node bifurcation](https://www.scholarpedia.org/article/Saddle-node_bifurcation)
  gives the null-mode, quadratic, and control nondegeneracy conditions. Applying
  these to the gradient flow `-gradient(L)` gives the cubic-loss test above.

Implementation and reproduction instructions accompany the measured report in
`docs/08 - Analysis/2026-10-05_training-weather.md`.
