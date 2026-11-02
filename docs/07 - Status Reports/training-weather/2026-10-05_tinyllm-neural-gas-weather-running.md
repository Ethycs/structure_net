# TinyLLM from-scratch weather campaign

**Status:** completed, including all six training runs and forecast evaluation.  
**Started:** 2026-10-05 22:30 PDT (2026-10-06 05:30 UTC).  
**Completed:** 2026-10-06 16:10 PDT (2026-10-06 23:10 UTC).  
**Study:** `tinyllm-neural-gas-weather-v1`, exploratory.

The user selected the full 12,000-update pretraining recipe. A detached runner
trained six independently initialized TinyLLM d8 models sequentially on
physical GPU 2 (RTX 2060 SUPER). It then fitted the neural-gas and global ensemble
forecasters and evaluated both against persistence and trend controls.

The runner has exited. Completion review verified all six model, trace, and
fold-audit hashes, plus corpus and producing sources. There were no invalid
forecast records. See the [completed analysis](../../08%20-%20Analysis/training-weather/2026-10-06_tinyllm-neural-gas-weather.md)
for forecast comparisons, undercoverage, and the unresolved full-network A₂
scope. The operational instructions and launch-time estimates below are retained
as history; `campaign_results.json` contains the final results.

## Configuration and evidence

- Eight transformer blocks, width 512, context 256, vocabulary 50,257; all model
  parameters train. Each seed starts with fresh weights and a fresh optimizer.
- Existing BabyLM token cache; final 262,144 tokens excluded from training.
- AdamW, peak learning rate `3e-4`, final rate `3e-5`, 200-update warm-up,
  cosine decay, weight decay `0.01`, betas `(0.9, 0.95)`, gradient clip 1.
- Effective batch 24 through twelve microbatches of two; float16 autocast with
  gradient scaling. Skipped overflow attempts do not count as updates.
- **12,000 updates and 73,728,000 training tokens per seed**; 442,368,000 tokens
  across six seeds. Fit seeds: 7, 17, 29, 41. Held-out seeds: 101, 131.
- Record 22 training-state measurements every 100 updates. Forecast leads are
  100, 300, and 600 updates. Broader validation runs every 500 updates.
- Atomic recovery checkpoints every 2,000 updates include optimizer, scaler,
  sampling RNG, measurements, validation history, and fold-tracking state.

The four focused test files passed **26 tests**. A separate CUDA lifecycle
check trained three fresh d8 models for 16 updates each at the full effective
batch size, exercised forecast fitting/evaluation and checkpoint replay, and
used approximately **1.39 GiB peak allocated tensor memory**. This excludes
CUDA runtime overhead and memory used by other services.

A separate GPU recovery check resumed seed 7 at update 8. All final parameters,
losses, batch indices, and telemetry arrays were bitwise identical to the
uninterrupted 16-update run. Neither short check establishes language-model
quality or forecast skill.

## Live files

Paths below are relative to `/home/rabbit/structure_net`:

```text
data/experiments/tinyllm_neural_gas_weather/
  20261005_full_from_scratch_v1.launch.json
  20261005_full_from_scratch_v1.log
  20261005_full_from_scratch_v1/
    manifest.json
    status.json
    producer_source/
    runs/fit/cosine/seed_7/progress.json
  20261005_shakedown_cuda_v1/campaign_results.json
  20261005_resume_cuda_check/verification.json
```

Launch PID: **659636**. The mutable campaign `status.json` reports the current
seed, role, and completed training-run count; each seed's `progress.json`
reports its last measured update. These records are updated every 100 training
updates, so they can lag actual computation. Check process liveness as well as
the recorded status. A failure writes a timestamped failure JSON and traceback
to the log.

The first live report confirmed seed 7 at **100/12,000 updates**, 614,400
training tokens, and no skipped AMP attempts. Fixed-probe loss moved from
10.9387 to 8.6048 nats. This small training-progress observation is not a
forecast evaluation. The first 100 updates took approximately 91 seconds
after initialization; a rough initial estimate is **18–24 hours for all six
seeds**, subject to shared GPU load, validation, checkpointing, and fold-search
cost. The live records supersede this launch-time snapshot.

Scientific fingerprint:

```text
456e1fdd65eeac4e62edc55050502061675aeff61875541e5e22cfc92dd01fff
```

The manifest retains configuration, corpus and producing-source hashes, and
copies of the producing files. Producing sources and the linked design are
frozen for this campaign; this operational status note is separate from that
fingerprint. Run roots from the lifecycle and recovery checks remain separate.

## Recovery and completion

Only after confirming that the previous runner has exited, resume from the
repository root with GPU access:

```bash
env LD_LIBRARY_PATH= CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/structure-net-matplotlib \
  OPENBLAS_NUM_THREADS=1 .pixi/envs/default/bin/python -u \
  -m experiments.structure_net.tinyllm_neural_gas_weather \
  --device cuda:0 --resume \
  --output data/experiments/tinyllm_neural_gas_weather/20261005_full_from_scratch_v1
```

Resume requires matching source/configuration/corpus fingerprints and checks
hashes before reusing completed runs. Do not launch a second producer into an
active root. A partial run recovers from its latest saved optimizer checkpoint;
updates after that checkpoint must be recomputed.

After all six training runs finish, the runner automatically writes fitted
forecasters, held-out forecast arrays and scores, the neural-gas map, PNG/SVG
figures, and `campaign_results.json`. Completion is reported in `status.json`.
Forecast failures remain counted and invalidate the affected aggregate rather
than being omitted. Forecast conclusions await these full-budget results.

## A₂ scope

Every telemetry checkpoint receives derivative measurements in the final
block's two-gain loss slice. Bounded multistart fold searches run for held-out
seed 101 at updates 0, 6,000, and 12,000. Only candidates passing the derivative
conditions receive slice track IDs. The diagnostic unfolding parameter is not
part of the training objective.

**This does not track every A₂ fold in the approximately 51M-parameter network.**
Full-space stationarity and completeness remain unverified; a slice fold need
not be a full-network minimum–saddle merger. An empty search is not evidence
that the network has no folds.

See the [frozen study design](../../01%20-%20Design/training-weather/tinyllm-neural-gas-weather.md)
for the methods, sampling limits, and source references.
