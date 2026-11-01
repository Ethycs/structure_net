# WORD_GAMES: target 70 full-request tokens/s

Exploratory study `word-games-70tps-v1`, following the completed CPU expert
elimination study. The user requested improvement from 56.26 to 70 tok/s.
Primary target is pooled visible output tokens / completion-request wall time,
including prefill, with unchanged original weights and frozen requests.
No claim of semantic accuracy follows from speed or exact wording agreement.

The candidate mechanism is keeping attention, shared computation and routed
experts together in contiguous GPU layer groups. The parent has 42 backend
graph splits and remotely placed expert banks in 20 of 40 blocks. Whole layers
may reduce transfers and permit more fusion, although the older cards also
inherit attention/recurrent work. No native kernel changes in this campaign.

Use the verified sampling-acquisition-patch runtime, original Q4_K_M checkpoint,
all 256/top-8 experts, Flash Attention, Q8 KV, context 8192, seed 73, temperature
zero, JSON grammar, no prompt-cache reuse or speculation. CPU compute threads
are one throughout. Keep OCR resident. Temporarily unload Qwen3-8B on the
regular 2060 under existing authorization, wait for two release observations,
and restore readiness and exact config in finally.

First run six fixed three-request pilots:

1. `control`: parent `gpu40_cpu1`, batch/microbatch 512, verbosity 5.
2. `whole256`: whole layers 0–9 on regular 2060, 10–18 on SUPER, 19–39 on
   3060, including output on 3060; batch 512, microbatch 256, verbosity 5.
3. `whole512`: same placement, batch/microbatch 512.
4. `whole_pipe256`: same whole placement and batch 512/microbatch 256, with
   native pipeline scheduling permitted.
5. `expert256`: parent expert placement, batch 512/microbatch 256.
6. `quiet_control`: parent expert placement, batch/microbatch 512, verbosity 3.

The non-pipeline whole-layer arms explicitly override block 0's attention norm
to its already assigned CUDA device. This preserves placement while disabling
the runtime's automatic pipeline activation when tensor overrides exist.
Verify actual native pipeline status, layer assignments and model buffer sizes.
Quiet-control verification uses model buffer sizes and loaded-layer count,
because DEBUG assignment lines are intentionally absent. No CPU expert fallback.

Selected whole-layer weight estimates in physical order regular / SUPER / 3060
are approximately 4,946.5 / 4,289.2 / 10,667.2 MiB, excluding host embeddings.
Memory includes additional state and workspace. Stop an arm on less than
384 MiB free on any used GPU, OOM, invalid/truncated JSON, placement mismatch
or service failure. Keep failures as evidence and continue independent arms;
stop the entire campaign on resident service-health failures. Do not lower the
guard or alter precision/context to rescue a candidate.

Advance at most two candidates with valid pilots and >= 98% of control pilot
token throughput, ranked by that rate. Run two full passes per selected arm
and control, order control/candidates and then reversed. Selection is
exploratory and biased toward pilot winners. The repeated full suite is a
measurement check, not an independent unseen quality population. Record every
pilot, failed arm and selected full pass. No automatic retries.

After reviewing this stage, additional tuning requires a dated amendment and
fresh artifact root. Reaching 70 is a measured speed endpoint, not deployment
approval. Report request/decode/prefill time, output token and byte counts,
CPU time, observed memory, graph partitions and native pipeline status.
Check receiver state against fixed authorized packets, selector validity,
sender structure differences and manually inspect changed receiver summaries.
Do not conflate frozen receiver replay with a live two-agent bridge.

Use Pixi from root; freeze/hash source and inputs, preserve original data,
pass focused configuration/lifecycle tests plus CUDA check before launch.
Register conservatively with exact/search read-back and commit durable records.
