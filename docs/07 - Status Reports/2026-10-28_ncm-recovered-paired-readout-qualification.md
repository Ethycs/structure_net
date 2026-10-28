# Qualification of the recovered model's paired readouts

The recovered-paraphrase training plan already fixes two readouts: the step6212
decoder alone, then the original decoder with source attention. Qualify their
shared prediction and decoder replacement before evaluating the future trained
model. This is a systems qualification; it adds no quality hypothesis or new
candidate, and cannot authorize a semantic or speed claim.

Use the exact public epoch2 initializer and the four existing TRAIN controls
selected by the task-training lifecycle (including source and target-length
extremes). Target lengths select controls only; generation receives source
features and a source mask, with no references or target codes. Use original
FP32 source features and FP32 inference on GPU0 RTX3060, TF32 disabled, batch1,
256MiB feature cache, and a 4.9GB allocation ceiling. No training or judge calls.

Require five focused CPU tests first: one shared prediction, independent
readouts, exact state restoration on normal and exceptional exits, stale context
and malformed input rejection, and complete same-case/code pairing. The first
test invocation inherited the repository-wide conftest and failed to collect
because scipy is absent from the locked ncm environment. Preserve that exit;
run the focused suite with `--confcutdir=tests/structure_net` using locked Pixi.
No dependency or environment modification is needed.

For each actual control, compare the new shared-source-prediction helper with
the unchanged original sequential predictor and source word readout. These
extra reference predictions qualify numerical parity only. In application
evaluation, predict once and share that exact saved code sequence between both
readouts. After all parent outputs are saved, install the completed decoder
checkpoint `7bc8beea1357c15f344903720f1207086591f0223b2e9ec81690c1af468f789b`
once, decode each predicted code sequence without word-source attention, and
require repeated word IDs to match. Verify unchanged nondecoder parameters,
unchanged codes, complete case pairing, and exact original-model restoration.

Use append-only root `20261028_recovered_paired_readout_pilot_v1`, freeze the
producer and dependencies, and preserve any failure. The future full predictor
must still finish its own training and independent audit; its source-only DEV
outputs must be evaluated with the established semantic rubric and canonical
first-judgment history. No DEV or Final320 examples enter this qualification.
