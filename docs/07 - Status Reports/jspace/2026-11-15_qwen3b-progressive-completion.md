# Progressive completion of Qwen2.5-3B compression

Study **JSPACE-QWEN3B-PROGRESSIVE-01**, exploratory, specified before new GPU
results. The user selected finishing the 3B model before the Qwen3-8B port.
All results remain unconfirmed until an independent held-out assessment.
Initial artifact root:
`data/experiments/jspace_qwen3b_progressive/20261115_v1`.

## Starting point and intervention

Continue the verified seven-block checkpoint from
`data/experiments/jspace_incremental_suffix/20261115_v2/train/selected.pt`,
SHA256 `061e1087857b114e0523da159949f817c5178df5f0444fb7bf22174ef4242b78`.
It scores 51/64 development questions; matched native BF16 scores 48/64.
The starting compact suffix covers blocks 29–35, with 7.315% whole-model
tensor-storage reduction. These are reused development observations.

Add exactly one preceding block per accepted stage: 28, 27, …, 0. Recompute
the new block's functional basis and polynomial fit through the current
accepted parent, then jointly distill the entire growing compact suffix.
Every inherited learned block, norm and interior bridge must be exactly
preserved at initialization. Only the entrance block and its bridge are new.
Never import recovered weights from the failed historical larger suffixes.

Retain the successful architecture and training method: residual rank 1024,
head dimension 64, original attention scale, softmax attention, 24 quadratic
polynomial anchors replacing 1024 MLP channels, fixed geometry, BF16 weights
and forward computation, and the existing FP32 polynomial calculations.
This protocol does not introduce quadratic attention or change the optimizer.

## Data and calibration

Reuse the authenticated 512 training and 64 development examples from the
original capture, with identical tokenization, offsets, four loss positions,
labels and balanced task families. Original native prefix boundaries exist
for all 37 boundaries. Reusing them is exact with respect to the frozen
native prefix; every compact block is recomputed during student training.

Use the original GroupedTeacherCapture estimator: 128 training calibration
examples, two independent groups of eight full-vocabulary random probes,
16 activation positions and position capacity 416. Calibration seed for new
cut c is `20861 + 100000*(29-c)`. Thus the first new cut uses 120861.
Keep the original full Gram eigenspace basis estimator and head projections.
The corrected rank fitter uses actual singular values, 24 anchors, 1024
removed channels, 128-column packing, seed 861, ridge 0.01 and the existing
development diagnostics. Development examples never fit basis/coefficients.

Teacher logits are freshly captured from the frozen accepted parent with the
one new block still native. This teacher is functionally the preceding
accepted model with an unchanged native prefix. Its differentiable Torch
execution and packed deployment may differ numerically in BF16; record the
choice-level difference separately rather than silently treating them as
bitwise identical. Teacher-cache accuracy is diagnostic, not a substitute
for the packed parent accuracy used in admission.

## Recovery and acceptance

Recover all compact blocks and the final norm together for exactly 12 epochs,
384 updates, microbatch four, accumulation four, learning rate 0.00012 with
cosine decay, zero weight decay, global gradient clip one, seed 20861.
Loss is `0.7*T²*KL(parent || student) + 0.3*CE`, temperature two, over the
original four full-vocabulary positions per case. CPU-streamed FP32 Adam
masters and moments retain the prior 32-GiB optimizer-state limit.

Select the initializer or one of the twelve epochs by highest development
correct count, then lowest full-vocabulary cross-entropy, then earlier epoch.
Require accuracy within five percentage points of BOTH the fixed native
48/64 baseline and the immediate accepted parent. The first stage therefore
requires at least 48/64; the global native floor is 45/64. Report also change
from the starting seven-block result, so cumulative parent drift is visible.

Do not expand through a stage that fails either gate. Preserve the last
accepted checkpoint and classify this campaign as an incomplete expansion;
diagnose the failure and specify any follow-up prospectively. A completed
training process does not imply successful full-model compression.

## Validity and systems checks

Before recovery at every stage, replay the packed parent from its matching
native boundary and replay the new native-block/parent seam. Require identical
64 predictions and choice logits within `0.03125 + abs(reference)/128`.
At the first stage also replay the native suffix with singleton batches,
matching the original reference's execution shape and expected 48/64.
Parent and compact scoring retain batches of four.

After calibration/fitting, run a discarded one-update pilot on the longest
16 training examples. Verify finite gradients and changes in FP32 optimizer
masters for every compact block and the final norm; BF16 stored changes are
separate rounding diagnostics. Check fixed geometry/readout and strict state
reload. Scientific training starts again from the untouched initializer and
must reproduce pilot initialization. Fresh-process verification reloads the
selected checkpoint, replays metrics and inventories actual packed tensors.

Each GPU phase runs under its own inference-controller reservation on the
RTX 3060: 11264 MiB reserved and 11008 MiB allocator cap. Every phase checks
the live lease. CPU preparation hides CUDA and uses the locked Pixi ncm
environment. The campaign runner releases each lease on process exit and
stops on failed execution, provenance, resource or quality checks. A phase
may reserve up to 120 minutes; no open-ended GPU reservation is needed.

Store artifacts on `/data/structure_net_experiments/jspace_qwen3b_progressive`
with a repository link. Authenticate original input hashes, checkpoint
provenance, each stage's consumed artifacts and the frozen source closure.
Archive sources before GPU work. Preserve every failed attempt. Capture and
training use disk-backed arrays; never retain the full teacher or prefix in
RAM during recovery.

## Reporting and completion

Report compressed block count, initializer/selected accuracy, both quality
gates, per-family results, paired changed decisions, checkpoint hashes, actual
packed whole-model storage, resource peaks and wall time at every stage.
Training/capture throughput is not inference speed.

Only an accepted cut-zero model is fully compressed. At that point verify
raw-token embedding input agrees with boundary zero, evaluate an untouched
held-out task set against the same native BF16 model, and benchmark warm
end-to-end prefill and decode under identical batch/length/dtype conditions.
Report the user's targets (38% smaller, 1.4× faster, at most five percentage
points lower accuracy) separately; partial suffix savings do not meet them.
Log typed exploratory results and an append-only research journal entry.

## Reproduction

Run CPU preparation and the CPU supervisor from the repository, in the locked
Pixi `ncm` environment with `CUDA_VISIBLE_DEVICES=''`. Preparation uses
`python -m experiments.structure_net.jspace_qwen3b_progressive prepare
--root data/experiments/jspace_qwen3b_progressive/20261115_v1
--seed-root data/experiments/jspace_incremental_suffix/20261115_v2`.
Then invoke the same module with `campaign --root <root>`. The supervisor
requests each worker lease using `inference-ctl lease run`; direct GPU phase
commands are rejected without the matching lease.
