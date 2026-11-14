# Parallel single-block distillation is running

Status on 2026-11-14: three independent GPU workers launched for
`JSPACE-QWEN-PARALLEL-SINGLE-LAYER-RECOVERY-02`.

Campaign: `data/experiments/jspace_parallel_single_layer/20261114_v1`.
Fingerprint: `65c755686637a5757e08942002439b1ccb9d7a07573869782e54417596a77cca`.
Trainer and coordinator were frozen in commit `d3c5200`.

| Worker | First block, zero-based | Reserved memory | First training lease |
|---|---:|---:|---|
| RTX 3060 | 2 | 7168 MiB | `35e706915676` |
| RTX 2060 | 3 | 5120 MiB | `e641b29d28e2` |
| RTX 2060 SUPER | 4 | 7168 MiB | `590e0f1a286f` |

Blocks 2 through 35 are queued. Each worker fits one compact block against the
original teacher's cached input/output pairs, using the proven 12-epoch FP32
recipe. Other blocks remain frozen. After selecting and independently verifying
its checkpoint, the worker takes another block. Artifacts and leases are separate
for each job. A quality miss is recorded; an execution failure stops the affected
worker without an automatic retry.

The shared FP32 native development cache reproduced all 64 native predictions,
including the original 50 correct answers. All 19 focused CPU tests passed.
Three simultaneous real GPU pilots passed native substitution, gradient and
memory checks. Each peaked at 1686 MiB reserved. Their elapsed times were
103.1, 104.7 and 106.3 seconds. These are systems-pilot timings, not generation
throughput benchmarks. The pilot weights do not initialize scientific training.

The completed first-block study recovered 48/64 versus native 50/64. The
separate two-block study selected 49/64 and passed fresh verification with
identical task logits. Its second block's initialization won over the trained
epochs; further distillation did not improve that initialization.

All current accuracy numbers use the repeatedly inspected development set.
Independent block results do not establish assembled-model quality. Once the
selected blocks are assembled, apply the user's cumulative five-percentage-point
accuracy allowance against the original native model and repair interactions
where necessary. The untouched 512-case confirmation set remains closed.
No full-model compression or speed result is claimed yet.

The coordinator PID at launch was `4040618`. Live evidence is in `events.jsonl`,
`logs/block_XX_train.log`, and each block's `train/epoch_XX_development.json`.
The final coordinator `result.json` will distinguish verified blocks, isolated
quality misses, execution failures and unattempted blocks. Do not launch a second
coordinator while this one is active; its exclusive run marker also rejects
duplicate invocation.

The prospective rules are in
[the parallel experiment protocol](2026-11-14_qwen-parallel-single-layer-recovery.md).
