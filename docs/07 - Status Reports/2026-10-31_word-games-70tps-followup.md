# WORD_GAMES 70 tok/s: exploratory follow-up amendment

Written after inspecting the first stage's six pilots, while its two scheduled
control passes finish. Original plan and raw results remain unchanged. This is
an adaptive exploratory follow-up, not confirmation of the first selection gate.

Known results: whole-layer placement reduced backend partitions from 42 to 4.
With native pipelining, pilot decode improved from 77.48 to 86.33 tok/s, but
full-request rate was 52.04 versus 53.29 because prefill was slower and outputs
were shorter. It missed the 98% selection gate. Whole512 crossed the 384 MiB
guard (381 MiB observed on regular 2060). Quiet-control verbosity 3 suppressed
native INFO buffer evidence and was rejected before timing; this was a
measurement implementation failure, not a demonstrated placement failure.

Source/log inspection found context checkpoints enabled despite no prefix reuse
or speculation. Each snapshot is 62.813 MiB and several are created during
prefill. Disabling unused checkpoints is the new primary intervention.
Verbosity 4 retains native INFO placement evidence while suppressing DEBUG.
Native `GGML_CUDA_GRAPH_OPT=1` is an additional bounded probe of the runtime's
existing Q/K/V stream-concurrency optimization, not a new implementation.

Keep the original model, context, Q8 KV, Flash Attention, requests, grammar and
seed, one CPU compute thread, resident OCR and the same 384 MiB guard. No
quantization change, reduced context or cache reuse. Whole-layer mapping remains
0–9 regular 2060, 10–18 SUPER, 19–39 plus output 3060. All whole-layer arms below
permit native pipeline scheduling and retain verbosity 5 for exact assignments.

Run these fixed pilots in order:

1. `control`: unchanged parent expert placement, batch/microbatch 512, verbosity 5.
2. `quiet4`: identical, verbosity 4; repair of the failed quiet measurement.
3. `no_checkpoints`: expert placement, batch/microbatch 512, verbosity 4,
   `--ctx-checkpoints 0`.
4. `whole_no_cp256`: whole layers, batch 512/microbatch 256, checkpoints zero.
5. `whole_no_cp384`: whole layers, batch 512/microbatch 384, checkpoints zero.
6. `whole_no_cp512`: whole layers, batch/microbatch 512, checkpoints zero.
7. `whole_no_cp384_graph`: whole_no_cp384 plus `GGML_CUDA_GRAPH_OPT=1`.

Validate checkpoint suppression from the actual server log after timed runs.
Retain the first stage's 98% pilot-rate selection gate, at most two candidates,
and two full passes in forward/reverse order. Failed pilots stop their own arm;
service failure stops the campaign. No automatic retries or changed thresholds.
The microbatch-384 arm is a new declared memory/performance compromise; it does
not lower the guard to rescue 512. Interpret changes in generated length and
quality separately. No semantic-accuracy or deployment claim.

Use a fresh `20261031_70tps_followup_v1` root. Freeze both plans and the producing
source; preserve first-stage source snapshots. Report both stages and failed
attempts. Registry may index the best repeated measured arm across stages,
clearly recording adaptive selection and quality limits.
