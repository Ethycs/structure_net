# WORD_GAMES: expanded GPU budget and loadable Triton configuration

User amendment: other GPU processes may be shut down gracefully to make space,
and the selected configuration should be loadable through Triton. This replaces
the unexecuted draft-microbatch-only memory proposal. Previous results and
failures remain evidence; their OCR-resident memory budget does not apply here.

Known repeated result before this amendment: whole layers, native pipelining,
physical microbatch 256 and no unused context checkpoints reached 62.968 tok/s
against 55.945 control. Larger logical batches did not show an obvious gain.
The CPU-expert MTP probe failed before timed output: 309 MiB free on 3060, with
259.54 MiB draft workspace and an additional recurrent rollback state.

Gracefully unload Qwen3-8B from Triton on port 8000 and GLM-OCR from the separate
Triton server on 8010. Record original readiness/configuration; wait for two
memory-release samples. Restore both models and configurations after this native
measurement stage, including failures. Additional deployment may gracefully
stop their containers under the new user authorization, with restoration tooling.

Study remains `word-games-70tps-v1`, exploratory. New root:
`data/experiments/word_games/20261031_70tps_free_gpus_v1`. Preserve original
weights, all experts/top-8 routing, context 8192, Q8 KV, Flash Attention, frozen
36 requests, JSON grammar, temperature zero and seed 73. No training, weight
quantization or cross-request reuse. Keep the 384 MiB memory guard.

Fixed five pilots, three requests each:

1. `control`: original all-GPU expert placement, batch/microbatch 512, one CPU
   compute thread, original checkpoints, sampling acquisition patch runtime.
2. `free_whole512`: contiguous whole layers 0–7 on regular 2060, 8–21 on SUPER,
   22–39 plus output on 3060; batch 1024/microbatch 512, native pipeline on,
   checkpoints disabled. Expected trunk GPU weights are 4,038.68 / 6,621.65 /
   9,242.57 MiB respectively, excluding host token embeddings.
3. `free_whole1024`: same placement, batch 2048/microbatch 1024.
4. `free_mtp1`: same whole-layer placement, batch 1024/microbatch 512, native
   MTP proposing one token (min/max 1, probability cutoff zero); entire MTP head
   on 3060, all CPU compute threads one. Context checkpoints remain enabled and
   native one-token rollback state is retained. Target verifies every proposal.
5. `free_mtp2`: same with two draft tokens and native two-token rollback state.

Use verbosity 5 while verifying native assignments and state. The model's MTP
head adds 521.38 MiB on 3060 without changing any weights. Its compatibility
with this distillation is unknown until measured. Verify actual draft activity.

Advance the fastest two valid candidates with >=98% of control pilot token
throughput (MTP also requires nonzero actual drafting). Two full passes each,
control/candidates then reversed. Stop an arm on memory/runtime/placement/JSON
failure, and the whole campaign on service-health failure. Keep all failed
arms. Selection is adaptive; report per-stage controls and resource budgets.
Report visible token/byte throughput, request/prefill/decode times, drafting and
acceptance, CPU time, memory and fixed receiver state checks. Manually inspect
changed summaries; no population semantic-accuracy claim.

After selecting a repeated configuration, create a project-owned Triton Python
backend that owns the native process and supports explicit model load/unload.
Use a dedicated three-GPU container because the existing two servers expose
subsets of the cards. Preserve their configurations. The native child needs a
newer glibc than the existing Triton image; use an isolated, pinned child runtime
or a compatible build rather than replacing Triton's system libraries.

Validate actual Triton load, inference, unload, GPU release and reload. Measure
through the Triton endpoint using the frozen requests, and distinguish its
whole-adapter timing from the native completion-only timer. Keep source/runtime
under repository/Pixi paths; deployed model-repository copies are artifacts.
Record service transitions and provide a reversible start/stop path. Publish no
unmeasured 70 tok/s or quality claim; register conservatively and commit.
