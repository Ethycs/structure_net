# WORD_GAMES: toward 70 tok/s and a loadable Triton runtime

The selected research Triton configuration is loaded and measures **66.965
full-request tok/s**, about 20.5% above the matched 55.556 native control. The
raw speed maximum is **72.446 tok/s**, but that speculative configuration fails
existing bridge checks and was not deployed. **70 tok/s without regression has
not been established.**

Study: `word-games-70tps-v1`. Exploratory systems tuning on the frozen
36-request WORD_GAMES replay. Original Empero Qwen3.8-35B-A3B Q4_K_M weights,
Q8 KV, Flash Attention, context 8192, greedy JSON generation and seed 73.
No additional weight quantization, training, expert pruning or cross-request
cache reuse. Native runtime includes the sampling-output acquisition patch;
the slower grammar-preview patch is excluded.

## Completed tuning stages

Each full result pools two complete 36-request passes. Three-request pilots
select up to two candidates; full ordering is reversed on the second pass.
Selection is adaptive and these are measurements on one machine, not a
confirmation set or a claim of population semantic accuracy.

| Stage and memory budget | Configuration | Full-request tok/s | Matched control |
| --- | --- | ---: | ---: |
| Initial, OCR resident | All-GPU experts, one CPU compute thread | 55.649 | 55.649 |
| Follow-up, OCR resident | Whole layers, native pipeline, microbatch 256, no unused checkpoints | 62.968 | 55.945 |
| Logical batches, OCR resident | Same, logical batch 1024 | 62.734 | 55.548 |
| Logical batches, OCR resident | Same, logical batch 2048 | 62.751 | 55.548 |
| Both resident models unloaded | Whole layers 8/14/18, microbatch 512, no checkpoints | 66.185 | 55.556 |
| Both resident models unloaded | Whole layers 8/14/18, microbatch 1024, no checkpoints | 66.926 | 55.556 |

The meaningful gain was grouping complete layers on each GPU, enabling the
native pipeline, and removing recurrent-state checkpoints unused by this
non-speculative, non-reuse workload. Native graph partitions fell from 42 to
4. The follow-up winner improves token throughput 12.55%, byte throughput
10.98%, and CPU time per output token from 18.02 to 16.03 ms. Its decode-only
rate is 85.39 tok/s; that is **not** the full-request rate.

Increasing the logical batch from 512 to 1024 or 2048 while retaining physical
microbatch 256 does not improve the result. All 216 parsed outputs in the
logical-batch stage equal the corresponding previously reviewed control or
whole-layer outputs. Larger physical microbatches with OCR resident failed
the 384 MiB free-memory guard. The original CPU-expert MTP probe also failed
that guard before any timed request; it is not a measured MTP speed failure.

The user's subsequent authorization allows gracefully unloading both Qwen3-8B
and GLM-OCR, expanding the tested memory budget. The resulting split puts trunk
layers 0–7 on the regular 2060, 8–21 on the SUPER and 22–39 plus output on the
3060. The larger physical batch reaches 66.937 and 66.916 tok/s in its two
passes: pooled **66.926**, +20.47% against its matched control. Byte throughput
improves 16.00%, CPU time/token falls from 18.15 to 15.09 ms, and total timed
request duration falls from 174.35 to 158.32 seconds despite 9.4% more generated
tokens. Total prompt time falls from 49.10 to 34.10 seconds. The minimum free
GPU memory is 1957 / 1027 / 531 MiB on 3060 / regular 2060 / SUPER.

All-GPU native MTP now fits and actually drafts: the three-request pilots accept
105/118 one-token proposals and 146/190 two-token proposals. Decode-only speed
reaches 94.55 and 95.77 tok/s, but full-request rates of 53.47 and 56.47 lose to
the 62.57–63.31 non-speculative pilots. They were not advanced to full replay.
The subsequent checkpoint follow-up distinguishes optional prompt snapshots
from native speculative rollback, which the source implements separately.

### Speculative decoding does not pass the current quality gate

| Final checkpoint stage | Full-request tok/s | Receiver states | Valid selectors |
| --- | ---: | ---: | ---: |
| Non-speculative expanded control | 66.715 | 24/24 | 2/2 |
| One draft token, microbatch 512 | 71.294 | 24/24 | 0/2 |
| Two draft tokens, microbatch 1024 | 72.446 | 24/24 | 0/2 |

Removing only the unused prompt context checkpoints allows MTP to exceed
70 tok/s. However, both advanced MTP configurations emit a selector with
`location: "dock"` and `deadline: "dock"`; the existing `Metaphor` validator
rejects it with `ValueError: distinct role labels required`. This is a measured
schema regression from the non-speculative expanded placement, which passes
the selector check. Receiver-state accuracy alone would have concealed it.

The two-token configuration additionally replaces `N1` with `shelf-position`
in the normal-reserve second-turn proposal. The one-token configuration makes
that substitution in the prompt-injection second-turn proposal. Frozen
receiver packets still yield correct receiver states, so those checks cannot
certify the newly generated sender path. Annotation summaries also become
less specific about the recorded condition.

Consequently the speculative profiles are retained as experiments, not selected
for deployment. The selected Triton profile is `free_whole1024`, which passes
the existing fixed receiver and selector checks. This is not a declaration of
full semantic equivalence; the known baseline summary defects remain. The
analyzer and typed registry now explicitly distinguish reaching the raw speed
threshold from reaching it while passing existing checks.

The one-token candidate accepts 4740/5278 drafts (89.8%) and the two-token
candidate 6414/8224 (78.0%). Their decode-only rates are 96.12 and 96.21 tok/s.
All 36 parsed outputs repeat exactly within each condition. This supports
reproducibility of both the speed gain and the observed schema failure, not
quality equivalence. All five native campaigns completed and restored their
resident services before deployment preparation.

## Quality and measurement boundaries

The completed stages score 24/24 fixed receiver-state checks per condition
(12 cases repeated twice). These checks compare authorized decisions and
occupied slots. They do not establish 95% semantic accuracy or end-to-end
firewall correctness. Receiver inputs are frozen authorized packets; they are
not regenerated from each new sender response.

Manual, non-blind review finds five existing unsupported approval/check
assertions in receiver summaries and ambiguous wording in the annotation
case. Sender choices can change with batch numerics and layer placement even
when weights and sampling settings are unchanged. Unchanged weights therefore
do not imply identical outputs or proven quality equivalence. No independent
semantic judge has been run in this speed study.

Native timing covers the completion HTTP call, including prompt evaluation and
generation, and excludes the separate template call, loading, warmup, artifact
writes and hashing. Triton timing includes the entire inference HTTP call,
template application, adapter and completion. Neither metric is concurrent
multi-user throughput. Visible output tokens exclude EOS; bytes and paired
request-rate diagnostics are retained because wording and output lengths vary.

## Evidence

Append-only raw roots under `data/experiments/word_games/`:

- `20261031_70tps_replay_v1`
- `20261031_70tps_followup_v1`
- `20261031_70tps_batch_mtp_v1`
- `20261031_70tps_free_gpus_v1`
- `20261031_70tps_mtp_nocp_v1`

Each stage retains source snapshots, source/input/runtime hashes, exact
commands, successful and failed arms, response hashes, GPU telemetry, CPU/RSS
observations, service unload/load receipts and original configurations.
Corresponding `*_review_v1` and `*_analysis_v1` roots bind review to raw evidence.
The analyzer checks source snapshots and hashes, recomputes receiver grading,
and requires two complete passes before reporting a candidate.

## Triton implementation

Project-owned source is in `experiments/word_games_triton/`. Its Python backend
owns one persistent native server and uses one request-level template call and
one completion call; Python is not involved per generated token. Explicit
Triton model unload sends SIGTERM and waits for native exit, with a bounded
force-stop fallback. A single KIND_CPU adapter instance owns all three GPUs;
Triton does not create a full model copy per GPU.

The dedicated Triton container exposes loopback ports 8020/8021/8022 and model
`word-games-qwen38-native`. Existing Triton containers and model configurations
are preserved. The start command leases their GPU memory by gracefully
unloading Qwen3-8B and OCR and stopping their containers, releasing the old
servers' CUDA contexts too. Stop restarts both containers and restores both
original model configurations.
Weights and native libraries are mounted read-only. Source and runtime remain
under the repository and `.pixi`, never a Python environment under `data`.

The patched native library needs a newer glibc than the existing Triton image.
An isolated loader and libraries extracted from the pinned native image apply
only to the native child. Triton's own loader and Python environment remain
unchanged. Model repository copies, runtime hashes and the exact selected
profile are recorded in `.pixi/word-games-triton/manifest.json`.

Input `text_input` is one BYTES string. Optional `sampling_parameters` accepts
`{"max_tokens": 128}` (integer 1–1024), and may repeat the fixed temperature 0,
seed 73 and `response_format: {"type":"json_object"}`. Generation settings
remain fixed to the measured configuration. Outputs are `text_output` and
`generation_info`, both BYTES strings. This is a loadable research inference
endpoint, not certification of the full WORD_GAMES firewall.

### Verified serving result

`20261031_triton_native_validation_v2` completes two 36-request passes at
67.082 and 66.848 tok/s, pooled 66.965. This includes template application,
adapter handling and the entire Triton HTTP inference round trip. It is within
0.1% of the selected native completion-only result; the difference is too small
to interpret as an improvement. All 72 parsed objects equal the corresponding
native outputs. Receiver states are 24/24 and valid selectors 2/2. These are
interface/regression checks, not an independent semantic evaluation.

Actual explicit load, inference, unload, GPU release and reload passed. GPU
usage fell from 9951 / 4712 / 7256 MiB to 311 / 244 / 260 MiB on 3060 / regular
2060 / SUPER after unload, releasing 21104 MiB. The second complete pass ran
after reloading. Minimum free memory under load remained 1957 / 1027 / 531 MiB.
The model is left READY at `http://127.0.0.1:8020`; old Qwen3-8B and OCR
containers are stopped while this configuration owns the cards.

Startup and reliability evidence is retained, including two initial failed
loads (missing Python activation script; backend plugin lookup through the
isolated loader), both of which restored the old services. The corrected backend
uses the existing Triton-compatible Pixi environment and `/llama` as its child
working directory. A reproduced idle `Broken pipe` was fixed by closing each
native request connection, without generation retries. The failed validation
and successful full rerun are separate roots. Unit checks cover shutdown,
mutable deployment state, fixed sampling parameters and connection handling.
All **26 targeted tests pass**. The post-run inference also passes after
72.9 seconds idle, exercising the connection failure that was reproduced earlier.

Deployment receipts: `20261031_triton_native_start_v1`, `_v2`, `_v3`;
final validation: `20261031_triton_native_validation_v2`; post-run idle check:
`20261031_triton_idle_validation_v1`. All are beneath
`data/experiments/word_games/`. The final validation preserves source snapshots,
72 response hashes, load/unload receipts, model-buffer verification and telemetry.

Registry ID `word-games-70tps-v1` records the raw speed winner as
`exploratory_70tps_reached_existing_checks_failed`, with qualified speed-target
success false and production deployment false. Separate fields bind the
passing research Triton profile, its measured throughput and lifecycle receipt.
The registry receipt is under `20261031_70tps_registry_v1`, with exact and
semantic read-back; it does not promote the failed speculative profile.

### Operating the prepared model

Run commands from the repository root with Pixi. On a fresh setup, preparation
verifies the full original weight hash, pinned native runtime and isolated
libraries, and creates the model repository without overwriting an existing
deployment:

```bash
pixi run python -m experiments.word_games_triton.deploy prepare --profile free_whole1024
```

Start the prepared configuration with a new evidence directory. This gracefully
unloads the old GPU models and stops their two containers before loading the
new Triton model. A failed start attempts to restore the prior services.

```bash
pixi run python -m experiments.word_games_triton.deploy start --evidence data/experiments/word_games/MY_FRESH_START
```

Readiness: `http://127.0.0.1:8020/v2/models/word-games-qwen38-native/ready`.
Inference uses the standard Triton V2 endpoint ending in `/infer`. Model
load/unload uses POST `{}` to
`/v2/repository/models/word-games-qwen38-native/load` or `/unload`. Explicit
unload releases the native model's GPU allocations while keeping the dedicated
Triton server running. To stop it and restore Qwen3-8B and OCR instead:

```bash
pixi run python -m experiments.word_games_triton.deploy stop
```

The prepared repository is `.pixi/word-games-triton/models/`; no existing
`/data/models/_triton` configuration is overwritten. Startup reuses the verified
full-file weight hash only when device, inode, size, mtime and ctime all match
the receipt; otherwise it hashes the checkpoint again. Runtime and prepared
repository hashes are checked each start.
