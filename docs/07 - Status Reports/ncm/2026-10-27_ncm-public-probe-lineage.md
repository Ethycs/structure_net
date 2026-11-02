# Preserve the XSum semantic branch in subsequent public milestones

Prospective provenance amendment for `ncm-public-curriculum-v1`. Epochs 1
and 2 remain completed failures at 0/128 application DEV passes. Epoch 4 is
running under its unchanged producer. The separate four-arm XSum semantic
probe is still running; no full-arm outcome has been used to choose this
amendment. Its purpose is to retain every first judgment, regardless of score.

After that probe completes, is independently replayed, reported, registered
and read back, insert its entire cache immediately after the completed epoch2
cache in the public evaluation ancestry. Require exact preservation of all
4,691 epoch2 firsts, exact packet/role membership for additions, all completed
probe receipts, and the matching audited count. Preserve valid negative and
exhausted invalid decisions. No request is made by this extension.

Use `experiments.structure_net.ncm_public_probe_lineage` for epoch4 and epoch8
inference and grading. It reuses the existing evaluation implementation and
adds the completed probe branch after epoch2, before any epoch4 judgments.
The existing evaluation schema and reserved, still-unused epoch4/8 paths stay
compatible with the frozen trainer's continuation gate. Producer inputs bind
this amendment, its implementation/tests, the old qualified preflight and
the new qualified preflight plus probe evidence. Completed epoch1/2 artifacts
are never rewritten. The old entry point remains historical provenance and
must not be used for new epoch4/8 evaluations.

The new CPU-only preflight uses
`20261027_public_curriculum_probe_lineage_preflight_v1`. It verifies the exact
merged cache and existing source guards on all128 DEV sources, with zero
judge requests, model updates or final-test access. Require focused failure
tests and an actual successful preflight receipt before evaluation. Inference
still uses its original GPU pilot; judging keeps its original four-case pilot.

For subsequent semantic replay and reports use
`experiments.neural_architecture_lab.ncm_public_probe_milestones`. Training
audits continue to use the unchanged interruption-recovery auditor. The new
report explicitly accounts for the additional branch between epoch2 and4;
it never pools XSum probe outcomes with application DEV outcomes. Reporting
and registration retain all existing quality gates and final-test exclusions.

No training, source selection, rubric, judge identity, target, 128-token limit,
61/64-per-cohort DEV threshold, model selection or final320 requirement changes.
This amendment is bookkeeping needed for faithful evaluation, not evidence
that accuracy improved or that any speed advantage exists.

Commands use repository-root `pixi run --locked -e ncm python -m ...`.
Set `CUDA_VISIBLE_DEVICES=''` for preflight, judging and audit/report; retain
the established deterministic GPU settings for inference.
