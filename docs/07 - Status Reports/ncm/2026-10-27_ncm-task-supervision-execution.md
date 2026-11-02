# Execute the fixed task-supervision comparison

This is an implementation note for `ncm-task-supervision-expansion-v1`, not a
change to its fixed training or quality gates. At writing, application DEV is
still 0/128, public epoch4 is running, and the new teacher collection is partial.
No full new arm is admitted yet.

The existing-arm lifecycle can run on all 1,376 already admitted pairs while
teacher collection proceeds. The expanded lifecycle waits for the complete
collection, independent catalog audit and complete frozen-codec target audit.
Both lifecycle results and their independent CPU replays must pass before
either full arm starts. Full arms run sequentially; existing must finish and
pass its audit before expanded starts. Neither model is selected using DEV
while training; both use the previously fixed epoch2 model initializer.

`ncm_task_supervision_training` runs `pilot` or `train` with `--arm existing`
or `--arm expanded`. The pilot executes original-reference FP32 gradient checks
on four controls including the entire population's longest source, then two
real FP16 updates at effective batch32. It compares all serial and prefetched
FP32 inputs, saves both complete checkpoints, restores update1 and replays
update2 exactly, including optimizer/scaler/RNG state and source-only outputs.
The two pilot updates never initialize the full arm. CPU AdamW master weights
are freshly constructed only after strict loading of epoch2 model weights.

The pilot and full run use physical GPU0 RTX3060, locked Pixi ncm, deterministic
algorithms, TF32 off, and at most4.9GB whole-lifecycle GPU allocation. The same
256MiB cache and one worker with two prefetched microbatches stream full FP32
source features. The existing public run on the RTX2060 remains independent.
Immutable checkpoints contain all optimizer/scaler/RNG state and completed
positions. This version starts fresh only; it does not automatically recover
an interrupted full arm. Recovery requires an explicit new execution record
from the last verified complete checkpoint, without repeating a judged arm.

`audit_ncm_task_supervision` independently reconstructs every scheduled sample,
noise flag and code replacement count from labels. It verifies loss sums,
learning rates, gradient scaling, complete checkpoint cadence and bindings,
actual gradient-parity records, restart equality and recorded memory bounds.
It does not perform new training or semantic grading. Feature-position exposure,
native target positions and code positions are reported separately; FLOPs are
not measured, and no equal-compute or speed claim follows from equal updates.

`ncm_task_supervision_service` provides CPU-only `launch` and `observe` actions.
Workers use fixed commands, durable logs, no automatic restart and preserved
systemd invocation identities. Launch success is never training completion.
Record the actual launch-tool receipt, observed worker terminal state and
observer-tool receipt separately. Independent audits require a successful
terminal receipt bound to the exact stage artifacts. Preserve failures before
any changed execution; never overwrite this version's evidence.

The execution modules, tests, source dependencies, plans, data identities and
locked environment are hashed before a worker's first model operation. Run the
focused CPU tests first, and record their real exit status with source hashes.
The final semantic evaluation implementation remains a subsequent required
stage. Passing this lifecycle cannot change the latest measured DEV accuracy.
