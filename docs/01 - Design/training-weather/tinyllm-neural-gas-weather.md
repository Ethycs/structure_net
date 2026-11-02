# TinyLLM neural-gas weather model

**Status:** EXPLORATORY design, before pilot outcomes  
**Date:** 2026-10-05  
**Study:** `tinyllm-neural-gas-weather-v1`

## Question

Does a neural-gas map of measured training states improve short-range ensemble
forecasts of TinyLLM d8 trained from scratch over a global dynamical surrogate,
persistence, and recent-trend extrapolation?

Here neural gas means rank-based competitive vector quantization, not a physical
ideal-gas law. Prototype nodes represent recurring telemetry states; competitive
Hebbian edges connect the two closest prototypes. Soft membership in this map
augments the ensemble's regression features. Kalman updates incorporate new
measurements before forecasts are issued. The map is fitted exclusively to
fitting trajectories, frozen before held-out trajectories are evaluated.

## Full-budget study selected by the user

Before any training outcomes, the user selected the full existing 12,000-update
recipe rather than the initially proposed early-training pilot. All six seeds
therefore use the original cosine schedule; the proposed pulse intervention is
removed to keep the pretraining recipe matched.

- Source: fresh `TinyLLMConfig.from_preset("d8")` initialization for each seed;
  no pretrained weights are loaded. This was corrected at the user's request
  before any TinyLLM run or producing implementation existed.
- Corpus: existing `data/corpora/babylm_10M_bpe16k.tokens.npy`; no downloads.
- All TinyLLM parameters train, with a 50,257-token vocabulary and eight
  512-wide transformer blocks. Context/positional embedding size is 256 tokens.
- AdamW, fresh optimizer per seed, initial peak learning rate `3e-4`, final rate
  `3e-5`, weight decay `0.01`, betas `(0.9,0.95)`, gradient clipping 1, effective
  batch size 24, and **12,000 actual optimizer updates**. The first 200 updates
  warm up linearly; cosine decay follows. CUDA training uses float16 autocast
  with a gradient scaler, matching the original mixed-precision training mode.
- Microbatch size 2 with 12-way accumulation implements the effective batch of
  24 within available memory. This preserves the mean objective but can change
  floating-point rounding compared with one 24-example forward. Overflowed AMP
  attempts retry the same batch and are recorded; skipped attempts do not count
  toward 12,000 optimizer updates.
- 6,144 tokens/update, 73,728,000 training tokens per seed, 442,368,000 total
  training tokens across six seeds. This is a multi-hour sequential campaign.
- Fit training seeds: 7,17,29,41; held-out seeds: 101,131. Different seeds have
  independently initialized models and independent minibatch streams. Each
  seed has one cosine schedule. Bootstrap units are independent model seeds.
- Exclude the original final 262,144-token validation suffix from training.
  A fixed independent probe batch of one 32-token sequence comes from that
  suffix, with seed 20261005. It is diagnostic telemetry, not a broad LM test.
- Measure every 100 updates: probe cross-entropy, output entropy, log global
  gradient norm, log residual RMS and log gradient norm at every transformer
  block, plus three diagnostics from a declared two-parameter loss slice.
- Fit 16 neural-gas prototypes for 20 passes; learning rate decays 0.2 to 0.02,
  rank-neighborhood width decays 8 to 0.3. Initialize from fitting states only.
- Compare neural-gas ensemble and global ensemble with identical 24-member
  bootstrap groups, ridge 1, polynomial degree 1, residual covariance shrinkage
  0.1, observation noise 0.05 standardized units, inflation 1.02.
- Horizons: 1,3,6 telemetry intervals (100,300,600 optimizer updates). Issue every two
  intervals after four warm-up intervals. Future learning rates are known;
  interval-mean scheduled learning rate and progress are the forcing fields.
- Every 500 updates evaluate 16 effective 24-by-256 validation batches, using
  fixed per-seed suffix samples. This broader validation is logged separately
  from the small shared diagnostic probe and does not enter the forecast state.
- Report standardized state RMSE/CRPS, raw probe-loss RMSE/CRPS, marginal 90%
  coverage, and Brier score for any probe-loss increase of at least 0.05 nats
  within the horizon relative to the issue value. Report event prevalence.
- No confirmatory success gate. Compare all fixed arms regardless of outcome;
  failed rollouts remain counted and invalidate the associated arm/lead mean.

## Fold coverage

The full TinyLLM has about 51M parameters. This pilot does not claim exhaustive
A₂ detection in its full parameter space. Every telemetry checkpoint receives
a derivative audit of the last block's attention and MLP projection gains:

`L(z0,z1; alpha) = last_probe_token_cross_entropy(gains=1+z) + alpha*z0`.

All other weights and the input to the final block are frozen at that
checkpoint. The unfolding control is a diagnostic linear regularizer; it is
not applied in the pretraining objective. The gain directions correspond to
simultaneous scaling of projection weights and biases. The last token is used
to bound the cost of double-precision third derivatives.

Record gradient norm, both Hessian eigenvalues, cubic null-direction derivative,
and control coupling at the current gains. These are loss-slice measurements,
not full-model Hessian measurements. On held-out seed 101, also run bounded
multistart searches at initial, middle, and final checkpoints with three initial
guesses, gain perturbations in `[-1,1]^2`, control in `[-2,2]`, and at most 80
root-function evaluations per guess. Only candidates passing the A₂ derivative
tests receive slice track IDs. Even a valid slice fold does not establish full
stationarity or transverse definiteness in all remaining parameters. Report
`full_parameter_space_completeness: unverified` and never convert absent
detections into a proof that the full model has no folds.

## Runtime and artifacts

Use GPU 2 through outside-sandbox execution after CPU tests and a representative
CUDA lifecycle check. Run sequentially to bound memory. The CPU smoke test uses
a small genuine TinyLLM architecture and synthetic token data; it is not quality
evidence. The CUDA shakedown uses freshly initialized d8 models, two fit seeds,
one held-out seed, sixteen updates, one validation batch, and a smaller gas map.
It retains the full d8 architecture and 6,144 training tokens per update, and is not quality
evidence and is kept in a separate root.

Each new output root retains hashes and copies of producing source, complete
configuration, initialization/token hashes, per-run telemetry and batches,
final checkpoints, derivative/search records, forecast arrays, the gas map and
its edges, strict-JSON aggregation, and a standalone figure. Existing completed
campaigns and user worktree changes remain untouched. Stop neural training on
nonfinite loss/gradients, checkpoint replay failure, or incompatible source
metadata. Forecast failures are reported, not silently discarded.

Every 2,000 updates (and at completion), atomic resume checkpoints preserve
weights, optimizer moments, gradient scaler, minibatch RNG, accumulated
telemetry, fold audits, validation history, and track identities. Resume
requires matching source/config/token fingerprints. Completed training runs
are reused only after their checkpoint, trace, and fold-audit hashes pass.
`status.json` and per-seed `progress.json` are explicitly mutable live records;
completed evidence remains immutable. A single long-lived runner trains all
six seeds, then automatically fits and evaluates the forecasting models and
writes the aggregate and figures. Full-training results are not claimed until
that process completes.

## Sources

- [Martinetz, Berkovich & Schulten (1993)](https://www.ks.uiuc.edu/Publications/Papers/PDF/MART93B/MART93B.pdf):
  rank-based neural gas and local time-series prediction.
- [ECMWF ensemble uncertainty](https://www.ecmwf.int/en/research/modelling-and-prediction/quantifying-forecast-uncertainty)
  and [Mandel's EnKF tutorial](https://arxiv.org/abs/0901.3725).
- Local [training-weather calibration](../../08%20-%20Analysis/training-weather/2026-10-05_training-weather.md)
  supplies the numerical forecasting and A₂ checks, not evidence for TinyLLM.
