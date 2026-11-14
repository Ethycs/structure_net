# Parallel single-block distillation and separate assembly validation

Study: `JSPACE-QWEN-PARALLEL-SINGLE-LAYER-RECOVERY-02`, exploratory.

On 2026-11-14 the user explicitly requested parallel execution now that the
single-block recipe works. This amendment changes the scheduling and training
input distribution for subsequent independent fits. It preserves the completed
single-layer study and the currently running block1 experiment. It does not
retroactively alter their protocols or results.

## Evidence available before this amendment

Qwen2.5-3B has 36 blocks. The native FP32 development control scores 50/64.
The enriched rank1024 block0 scored 46/64 initially and 48/64 after isolated
distillation. All 12 epochs and 384 updates completed. Accuracy-first selection
chose epoch7; fresh reconstruction and packing reproduced its task logits
exactly. Only block0 changed; the other 35 blocks remained native.

A separate experiment is fitting block1 behind that frozen recovered block0.
Its initialization scored 49/64. That experiment is still running when this
amendment is written. The sequential continuation coordinator has not launched.

## Parallel fitting

Fit blocks2..35 independently, one compact block per training process. Each
worker uses the ORIGINAL cached teacher input h[l] and target h[l+1]. It does
not consume another worker's outputs or update another block. Keep the fixed
training-derived enriched basis for that layer, rank1024 residual update,
native2048-wide residual carry, native128-dimensional attention heads, softmax,
and native11008 feed-forward activation width. Geometry stays fixed. This is
the verified recovery architecture; head reduction and polynomial attention
are not introduced in this campaign.

Preserve the single-block training configuration: original512 training cases,
12 epochs, FP32 parameters/forward/gradients/AdamW, learning rate1.2e-4,
microbatch4, accumulation4, clipping1, no weight decay, seed27864. Each process
trains only its own seven compact parameter tensors. The original teacher,
embedding/readout, norms in the native suffix, and all other blocks stay frozen.

The loss is the existing equal-case, equal-valid-token squared output error
normalized by detached ORIGINAL teacher-update energy. Fit its floor only on
training data:1e-4 times median per-token update energy. Padding is excluded.
Report ordinary, relative, retained-subspace and outside-subspace errors.
These feature metrics are not substitutes for task accuracy.

## Matched task checks

Generate one authenticated FP32 cache of all native development boundaries
using the original64 development prompts and the actual native model. This
cache is separate from the original BF16 teacher training cache. Preserve
identities, token lengths, causal masks and rotary positions. Its final
predictions must reproduce the declared native50/64 control.

For isolated block l, task evaluation starts from its cached native FP32
development boundary, applies the compact block, then original layers l+1..35
and the original tied readout. This evaluates one replacement in a native
context. A native-block substitution control must reproduce the native
development predictions on each worker's execution path before scientific
training. Use the actual full vocabulary readout weights, selecting the
benchmark's answer-token logits only for scoring.

Evaluate initialization and every epoch. Select highest development task
accuracy, breaking ties with lowest relative feature error. Preserve every
epoch's metrics, predictions and hashes, and the currently best checkpoint.
After training, strictly reload and independently verify the selected state.
An initialization may win; report it as such rather than attributing its score
to distillation.

The isolated diagnostic gate remains at least47/64 versus native50/64. A miss
does not cancel other independent fits. Record it and withhold an assembled
quality claim. The selected checkpoint must still be independently verified.
Fresh-process prediction parity is exact; numerical feature/logit tolerances
remain those of the existing single-block verifier.

## Resources and lifecycle

Use locked offline Pixi, the ncm environment, and inference-ctl reservations
for every GPU process. Use one worker per physical GPU: RTX3060 at7168MiB,
RTX2060 at5120MiB, and RTX2060 SUPER at7168MiB. Limit the Torch allocator to
the lease minus256MiB and retain the live reservation/peak checks. Reservation
sizes are runtime settings, not changes to the scientific objective.

First run representative longest-context pilots concurrently on the three
cards, then launch independent fits. Each newly prepared block retains its
pilot gate. Reuse only complete fingerprint-matched evidence. Do not retry
failed fits automatically, mutate completed artifacts, or overwrite another
worker's root. Infrastructure failures stop the affected worker; independent
workers can finish. Record commands, leases, producer snapshots, inputs,
configuration and terminal status. Keep the existing block1 job intact.

Planned roots:

- `data/experiments/jspace_native_dev_cache/20261114_v1`
- `data/experiments/jspace_parallel_single_layer/20261114_v1`

## Assembly remains a separate experiment

Independent fits do not establish that their errors compose harmlessly.
After selecting and verifying the independently fitted blocks, assemble them
with the accepted block0/block1 states and test actual end-to-end task
accuracy. The user's five-percentage-point allowance applies to the complete
model relative to the original native baseline, never once per block.

If assembly fails, localize the interaction and distill affected blocks using
the actual accepted student prefix, preserving the original teacher targets.
Freeze a new repair experiment before running it. Do not claim assembled
accuracy, size/speed success or deployment readiness from isolated passes.

The untouched512-case confirmation set remains closed until a final assembled
checkpoint is selected. Report all current evidence as development-only and
confirmed:false. Actual tensor storage and matched whole-model speed are
measured separately after the assembled model clears the accuracy gate.
