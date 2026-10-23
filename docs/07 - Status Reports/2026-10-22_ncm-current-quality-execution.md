# Current NCM and matched parent quality execution

Date: 2026-10-22. Execution details for the already declared final-step5,120
endpoint and matched original-parent control. This changes no model, source,
rubric, checkpoint selection, admission threshold or test policy. The NCM
training worker has exited with `trained_pending_development_grading`.

Seal the final checkpoint, output, result and producer before judgment. Verify
the original producer dependencies and all64 source IDs/hashes against the
fixed development cohort. Grade every case separately with both qualified
original judges, four workers per role. Invalid outcomes retain their bounds;
valid negative outcomes are never retried. No intermediate-checkpoint selection.

Primary Sonnet is also producing the broader public training cohort with four
workers. Before concurrent primary grading, run a systems-only four-request
pilot using the first four immutable expanded-teacher pilot labels, separate
from both training and development. Its sole criterion is four valid judgments
while the teacher job is live and makes progress; semantic pass/fail does not
gate availability. These judgments never enter model task accuracy. Freeze the
teacher producer and progress snapshots in the receipt. A failed combined pilot
means primary grading waits until teacher completion; no higher concurrency.
Secondary Terra retains the already passed recovered-judge four-worker pilot.

Grade the final NCM and original parent in separate64-case campaigns under
each role. Admission still requires61/64 under each judge for the NCM, followed
by a separately declared sealed256-case test. Parent output is a diagnostic
reference, with its96-Llama-token cap versus48-GPT2-token NCM canvas disclosed.
Neither training reconstruction nor parent performance substitutes for student
accuracy. No speed measurement, original-body training or final-test opening.

New artifacts: `20261022_current_*`, `20261022_combined_primary_*` under
`data/experiments/ncm_paper_regime`. Preserve all earlier outcomes unchanged.
