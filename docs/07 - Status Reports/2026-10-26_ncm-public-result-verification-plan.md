# Public curriculum: verify training records and semantic milestone results

Prospective reporting work for `ncm-public-curriculum-v1`, written while epoch1
is training and before any public-milestone DEV output or judgment exists.
Depth12 and depth24 previously failed all128 complete-summary judgments.
Those failures and all4,435 canonical first judgments remain preserved.

At each scheduled epoch1,2,4,8 milestone, require actual successful completion
of the selected training segment and its frozen checkpoint. Independently
replay the committed training trajectory's ordered example IDs, populations,
positions, noise flags, deterministic recorded corruption counts, microbatch
geometry, actual tail denominator, learning-rate schedule and finite recorded
loss/gradient statistics. Bind full raw traces and checkpoint sidecars. Never
treat a prefix audit as proof of a complete epoch. Recovery must account for
the complete committed segment chain without counting discarded updates twice.

The CPU replay reads admitted native code labels and indexes; it does not load
source feature matrices or execute model training. Qualify it against the two
completed real batch128 pipeline-pilot updates and their exact restarted second
update, after focused corruption, ordering, boundary and failure tests. Use the
fresh `20261026_public_training_auditor_pilot_v1` root. This qualification is
systems evidence only and must retain an actual successful tool receipt.

For semantic reporting, replay all128 source-only outputs, tokenizer rendering,
exact source/candidate judge packets and canonical first responses. Preserve
the revised evaluation lineage and all previous milestone judgments. Recompute
both64-case cohorts, Terra admission, same-case Terra/Sol joint gates, and the
full128 denominator. An invalid judgment is unresolved, not a pass, a failure,
or grounds for dropping a case. Keep TRAIN loss and code agreement separate
from complete-summary task accuracy.

Report each completed milestone's semantic passes, valid judgments, cohort
counts, joint qualification, training exposure, selected checkpoint and source
guard. Register its conservative result and verify independent exact, semantic
and JSON readback. The earliest qualifying milestone is the only candidate for
the separately governed sealed final protocol. A qualified DEV result alone
does not establish the95% final task goal. No speed claim precedes measured
task quality; final320 stays sealed throughout these milestone reports.

The frozen training implementation and schedule are unchanged. The first
auditor entry point is `python -m experiments.neural_architecture_lab.audit_ncm_public_training pilot`,
using the locked Pixi `ncm` environment with CUDA hidden and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Actual full-trajectory replay and semantic
finalization must finish before any milestone is reported as verified.
