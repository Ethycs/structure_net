# Direct task evaluation of the new progressive checkpoint

Study `JSPACE-QWEN3B-PROGRESSIVE-TASK-PROBE-01`, exploratory follow-up.
The user asked to pursue progressive compression again after its earlier good
results. Continue the broader-corpus branch unless the user selects the older
recipe in the optional clarification. This evaluation is useful as a comparison
under either choice and does not train or alter any model.

## Known outcomes and fixed candidate

The completed v2 progressive rebuild compressed native layer35 and stopped
before adding layer34. Its packed MC DEV was198/256 at every checkpoint, versus
native195/256. The saved models' logits changed during recovery; identical
accuracy is not evidence that checkpoints were ignored. Full-response DEV NLL
fell3.45472to2.95058, versus native2.17769. Every checkpoint failed the declared
1.10times-native NLL gate, so the original producer skipped generation and
semantic grading. Preserve that failed classification and all evidence.

Direct task behavior remains unknown. This follow-up evaluates the saved
`update_0046` candidate because it has the lowest observed response DEV NLL and
ties the other candidates on MC accuracy. No generation outcome was inspected
to choose it. Its source checkpoint SHA256 is
`d2d1b8613474d4b5891d004e11c87c5610b9215e67d34b95465e8bb70f15a486`.
The source stage is
`jspace_3b_progressive_response_rebuild/20261116_v2/blocks_01/recovery`;
consume its existing `run/evaluations/update_0046/export` structurally, without
repacking, reconstructing fitted weights or training. Authenticate the completed
stage, candidate, checkpoint and export identities before execution.

## Measurements and execution

Use the original native controls, six canaries and the same16 selected response
DEV generation cases(two per eight families), same prompt tokens, greedy
decoding, EOS and128-token output cap. Recompute the actual native35-layer prefix
on generated histories, with the packed suffix and original tied readout. No
teacher-forced cache may substitute for generation. Retain raw outputs, token
counts, timing, peak memory and prefix-execution mode. Exact packed-state/reload
checks remain required. Do not read any sealed/final task contents.

Run generation under a separate3060 reservation(11264MiB,11008MiB allocator
cap), locked/offline Pixi. Release its lease before CPU/network grading. Grade
MBPP using the existing bounded sandbox, GSM8K by final numerical answer, and
semantic task success/generation/source support using the same blinded paired
rubric and expected `codex_sdk/gpt-5.6-sol` route. Use one judge worker and the
existing request limit/timeout. Preserve failed requests as unavailable; do not
retry selectively based on a semantic verdict or call missing evidence zero.

## Interpretation

Report native and student absolute rates plus paired differences. The direct
task retention screen allows at most5pp native loss on each declared screen;
use the new rebuild's native-relative canary rule, since the native teacher
itself passed only2/6 strict canaries. Missing grading or executable evidence
prevents a task-retention pass. These small exposed DEV samples cannot certify
population accuracy,95% quality or deployment readiness.

NLL remains reported as a distributional warning and is not a reason to skip
this study's task measurements. A task-retention pass here does not retroactively
pass the original NLL gate, establish semantic equivalence, or automatically
authorize stage2 promotion inside the old study. A subsequent progression needs
its own explicit prospective decision using these results. No extra training
epochs are part of this probe. Preserve the completed all-at-once run and its
failed quality results without restarting it.

Freeze new source/tests and write append-only artifacts under
`data/experiments/jspace_3b_progressive_task_probe/20261116_v1`, with separate
GPU and CPU-grading receipts. Keep `confirmed:false` throughout.
