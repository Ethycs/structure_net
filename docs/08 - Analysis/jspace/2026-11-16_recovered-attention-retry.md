# Matched quadratic-attention retry after response recovery

## Question and scope

Can the recovered three-block Qwen2.5-3B suffix accommodate quadratic attention in absolute layer 35, with matched recovery? This is a one-block intervention, not a full-model quadratic replacement or an inference-speed demonstration.

The current adapter preserves the same native score scale and BF16 score arithmetic, 16 query heads / 2 KV heads, 64-dimensional heads, 1024-dimensional residual coordinates, original RoPE, affine corrections, bridges and polynomial MLPs. Only layer 35's attention normalization changes between softmax and normalized `1+s+s*s/2`. Both arms use explicit Torch, cache-free execution. Prior CPU adapter verification passed 18 tests, including gradients, updates and mode-preserving export/reload; it did not establish GPU qualification or scientific quality.

## Why a matched retry

The earlier all-36 quadratic experiment scored 220/512 versus its softmax control's 211/512 and native 396/512. The paired quadratic-minus-softmax difference was +1.76 percentage points, with a family-stratified bootstrap 95% interval of [-3.125, 6.641]. That experiment did not establish that quadratic attention caused the main recovery failure, nor that it is harmless. One training seed per arm cannot estimate the probability that a future training run will succeed.

Current recovery changes the depth, objective, corpus, precision, anchors and MLP structure. Cross-experiment differences therefore cannot isolate the effect of improved recovery. The retry instead starts both arms from one authenticated checkpoint and holds their update budget and inputs fixed.

## Parent and resource boundary

Parent: `/data/structure_net_experiments/jspace_3b_usable_recovery/20261116_v1/stage_a`, fingerprint `2f7a29fa28d29b5af106a4d5c9e32ec403908679881f5d5c71e95d62733d0574`.

Wait for all five parent generation phases and available judgments, then use the parent's existing generation-first selection. A best-but-unqualified seed may initialize a diagnostic experiment; it must not be described as recovered or promoted. Missing evidence cannot be replaced by minimum NLL. Parent evaluations retain priority between their GPU leases.

Use `/data/structure_net_experiments/jspace_3b_attention_retry/20261116_v1` for new artifacts. The data volume had 680 GiB free when checked; the root volume had only 94 GiB. Reuse authenticated parent caches rather than copy them. Acquire every GPU phase through `inference-ctl lease run`, with 3584 MiB on the plain RTX 2060 and a 3328 MiB allocator cap. CPU waiting and judging hold no GPU reservation. `inference-ctl --wait` waits for an accepted lease's readiness; it does not queue a rejected conflicting reservation.

Do not modify or interrupt the progressive all-36 campaign or the ABLATION32 search. At the continuation check, progressive had completed eight compressed blocks, scoring 195/256 (the native MC score), and was training block nine. Its eight-block generation screen remained unqualified: MC retention alone is not usable recovery.

## Paired pilot contract

1. Require authenticated parent completion and frozen source/checkpoint/data identities.
2. Qualify each arm with discarded GPU systems updates and replay checks.
3. Start both scientific arms afresh from the same selected checkpoint, not pilot weights.
4. Train the same three compact blocks plus final normalization using exact full-vocabulary forward KL, temperature 1, CE weight 0, learning rate `2e-5`, BF16 live parameters and FP32 CPU Adam state.
5. Use the same first 71 existing TRAIN windows: 25,245 response/EOS tokens per arm. Do not optimize on DEV or TEST.
6. Evaluate each arm before and after recovery: response DEV loss/KL, MC256, six canaries, 16 generated cases, executable task checks and source-grounded semantic judgment.
7. Report paired MC wins/losses, generated-task usefulness, reliable stopping, repetition and numerical fidelity. Missing judgment remains unavailable, not a pass.

These are exposed development screens. There is no automatic full-pass expansion or multi-seed claim. Fresh source-group-separated confirmation and repeated matched seeds would follow a promising exploratory result.

## Implementation state

Implemented `jspace_3b_attention_retry.py` and `jspace_3b_attention_retry_controller.py`, with additive tests and no changes to active experiment producers. All 60 focused tests passed across the initial 55 passes and the five corrected test-fixture cases. Corrections concerned exclusive-write fixtures, the readout's selected-token shape, and complete receipt metadata; the checks were not weakened.

A CPU-only check against the real authenticated parent verified 71 windows, 1,136 TRAIN examples and 25,245 response/EOS tokens. The actual controller CLI with a zero-second wait exited `pending / parent_wait_expired` without launching GPU work. Its receipt is under `/data/structure_net_experiments/jspace_3b_attention_retry/20261116_pending_probe`.

At 23:09:11 UTC the deferred controller started with PID 3465672, reporting `waiting_parent` and `gpu_reservation_held=false`. Its state is under `/data/structure_net_experiments/jspace_3b_attention_retry/20261116_v1/controller_v1`. It waits at most 24 hours for terminal parent evidence, then authenticates the selected seed, runs both disposable pilots, and performs the two matched scientific arms. Accepted lease readiness has a separate 300-second limit. No extra GPU reservation is held while waiting, and no GPU retry result exists yet.

Failures, unavailable judges and partially written producers are preserved rather than silently retried. A CPU verification pass is not GPU qualification, generation recovery or evidence of a speedup.

## Planning pause — 2026-11-17

On the user's instruction to stop and make a consolidated plan, the deferred controller was stopped while still waiting for the parent. It had not acquired a retry GPU reservation or run either arm. The implementation and verification receipts remain preserved; no retry result is claimed. The [recovery reset plan](../../02%20-%20Implementation/jspace-qwen-compression-recovery-reset-plan.md) now places recovery-budget sufficiency and broader paired evaluation ahead of a new attention intervention. Existing progressive, three-block evaluation and ABLATION32 jobs were left running.
