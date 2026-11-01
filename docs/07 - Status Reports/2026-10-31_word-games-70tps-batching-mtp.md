# WORD_GAMES 70 tok/s: larger prefill batches and native MTP

Adaptive exploratory amendment, written after the checkpoint-removal pilots
and first full candidate replay (62.918 tok/s versus 55.973 control). Repeated
passes are still running; the gain is not yet a repeated result. Preserve both
earlier stages, including failed memory/measurement arms and source snapshots.

Two remaining concrete mechanisms merit a bounded test before closing this
request. Larger logical prefill batches can expose more microbatches to native
pipeline scheduling without increasing the physical microbatch. The checkpoint
also contains one unused MTP/NextN head; the pinned runtime supports
`draft-mtp` against the same model. Its usefulness on this distillation is unknown.

Run four fixed pilots in a new `20261031_70tps_batch_mtp_v1` root:

1. `control`: unchanged all-GPU expert placement, batch/microbatch 512, one CPU
   compute thread, verbosity 5, no speculation; original checkpoints enabled.
2. `whole_b1024`: whole-layer mapping from the prior amendment, pipeline on,
   logical batch 1024, physical microbatch 256, checkpoints disabled.
3. `whole_b2048`: same, logical batch 2048.
4. `mtp1_cpu12`: original expert placement, batch/microbatch 512, native
   `draft-mtp`, exactly one draft token (min/max 1, probability cutoff zero).
   Keep trunk banks 0–39 on their original GPUs; place only the unused MTP
   head's routed projections (`blk.40.ffn_*_exps.weight`) on CPU. Keep its other
   weights on 3060. Use twelve target and draft CPU threads for these draft-only
   CPU projections. This intentionally reintroduces CPU work for drafting;
   every proposed token must be checked by the unchanged target. Enable the
   runtime's one-token recurrent rollback state. Keep normal context checkpoints
   enabled in this arm; do not remove speculation's state-management requirements.

MTP adds 498 MiB of CPU expert payload and about 23.38 MiB of GPU nonexpert
weights, plus native context/workspace/state. Do not quantize the head or target.
Verify loaded buffers and CPU routing of exactly those three MTP projections,
and require actual draft activity before advancing MTP. No second checkpoint,
tokenizer map or additional download is needed.

Keep context 8192, Q8 KV, Flash Attention, original requests, grammar, seed,
temperature and single-request timing. No concurrent-request throughput metric.
Same 384 MiB guard, OCR residency and Qwen3-8B restoration rules. CPU/GPU
arithmetic and changed prefill/verification batches can change generated text;
report that and check task state instead of promising bitwise identity.

Use the same >=98% pilot-rate gate and advance at most two candidates to two
full passes each against control in forward/reverse order. Preserve failures;
no retries or memory-guard relaxation. Record acceptance, drafting overhead,
token/byte throughput, latency, memory and quality diagnostics. This is the final
bounded tuning stage of the current request. If 70 is still unmet, report the
best repeated rate and remaining gap rather than implying the target was reached.
