# Qwen2.5-3B continuation under the original-native accuracy budget

Study **JSPACE-QWEN3B-NATIVE-BUDGET-01** is an exploratory continuation,
specified before its first GPU phase. The initial root is
`data/experiments/jspace_qwen3b_native_budget/20261115_v1`.
The user authorizes completing the 3B compression with a five-percentage-point
accuracy allowance relative to the original model. Immediate-parent accuracy
is retained as a diagnostic, not an additional acceptance requirement.

## Preserved historical outcome and prospective admission

The prior `jspace_qwen3b_progressive/20261115_v3` campaign stopped at cut26.
Its ten-block initializer was selected at 47/64 after all 12 epochs and 384
updates, and independently reloaded with exact predictions. It failed the
prior immediate-parent gate: the nine-block parent scored 52/64, requiring
49/64. Its original-native comparison passed: native BF16 scored 48/64,
requiring 45/64. That completed prior quality stop remains unchanged.

This new study admits only the verified cut26 checkpoint with SHA256
`d6954494f3fd74a5765c3f623b39891cae383339d1888bbef067dd6879bb4cb5`.
Preparation authenticates the old root, archived/current sources, all three
completed stages (28, 27, 26), their selected states and fresh verifications,
and the terminal quality-stop receipt. A separate admission receipt records
both the failed old dual gate and the passing prospective native-only gate.
This is an outcome-informed research continuation on reused development
examples; it does not retroactively convert the old study into a success.

## Unchanged experiment and staged continuation

Reuse the selected ten-block suffix 26–35 and its calibrated fits; add only
layers 25, 24, …, 0, one per stage. Recompute each new entrance block's
functional basis and polynomial relations through the accepted parent.
Initialize inherited weights, norm and interior bridges exactly from that
parent. Reuse original frozen native prefix boundaries; recompute every
trainable compact block on each student forward pass. No old GPU phase is
repeated and no previous failed larger-suffix state is imported.

Keep the v3 architecture: residual rank1024, head dimension64, original
attention scale, softmax attention, 24 quadratic anchors replacing1024 MLP
channels, and128-column packing. Keep BF16 execution and stored geometry,
FP32 polynomial calculations, CPU FP32 Adam masters/moments, frozen embedding and geometry, and
joint training of the entire compact suffix plus final norm.

Use the same512 training and64 development cases, original tokenization,
four full-vocabulary loss positions, and family-balanced splits. Fit bases
on128 training calibration cases using two independent groups of eight
full-vocabulary probes. Preserve the original estimator and rank fitter;
the calibration seed remains `20861 + 100000*(29-cut)`.
Keep differentiable calibration logits separately from packed-parent
distillation targets. The latter must reproduce parent predictions and
satisfy the existing BF16 logit tolerance.

Each new stage trains exactly12 epochs/384 updates, microbatch4,
accumulation4, learning rate0.00012 with cosine decay, zero weight decay,
gradient clip1, seed20861, temperature2, and loss
`0.7*T²*KL(parent || student) + 0.3*CE`.
The initializer remains eligible; select highest development correct count,
then lowest full-vocabulary CE, then earliest exact tie. Architecture,
losses, selected-checkpoint criterion, and training budget are unchanged.

## Acceptance and systems validity

The primary per-stage gate is fixed for the whole continuation:
`correct >= ceil(48 - 0.05*64) = 45`.
Always also report the immediate-parent five-point diagnostic, its minimum
count and pass/fail. A parent diagnostic miss does not stop this study;
a native-budget miss is independently verified and then stops expansion.
No confirmation-set data, timing gate, or family-specific gate enters this
development selection. All results remain `confirmed=false`.

Before training, require authenticated packed-parent and seam controls,
a discarded real update with finite effective gradients/changed FP32
masters for every compact block and norm, unchanged fixed state, strict
reload, and initializer replay. Fresh selected verification must reproduce
all64 development predictions with the prior BF16 logit tolerance
`atol=0.03125, rtol=1/128`; it must also preserve fixed state and metrics.
The first new stage repeats the matched native singleton control at its new
cut, requiring48/64 and matching original cached outputs.

All GPU phases use the reservation system: RTX3060 allocation11264MiB,
allocator cap11008MiB, separate120-minute leases. CPU preparation and the
campaign supervisor hide CUDA. Preserve phase logs, manifests, checkpoints,
both teacher banks and failure receipts. Existing complete matching phases
may be skipped; incomplete phases are never overwritten or automatically
retried. Source/input fingerprints and archived producers are immutable.

If all36 blocks pass, a separate predeclared full-model gate may evaluate
untouched confirmation accuracy, actual tensor storage, and speed against
an optimized native baseline. Neither development continuation nor a
single-stage pass establishes those full-model outcomes.
