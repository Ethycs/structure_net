# Retain both completed probe branches before the next public milestone

Prospective provenance amendment to `ncm-public-curriculum-v1`. The original
XSum semantic probe and its author-rendering correction are now completed,
independently replayed, reported and registered. Both failed their improvement
gates. Epoch 4 is still training; no epoch 4 application outputs or judgments
exist. Epochs 1 and 2 remain 0/128 application DEV passes.

The next evaluation must inherit 5,263 first judgments: the 4,691 epoch 2
ancestor, 499 additions from the original probe, and 73 author-rendering
additions. Preserve every positive, negative and invalid first judgment.
Include both probe branches in provenance, never in application accuracy.
Require exact packet and role membership, all successful execution receipts,
producer hashes, audits, reports and independent registry readbacks.

Use `experiments.structure_net.ncm_public_author_lineage` for epoch 4/8
inference and grading, and
`experiments.neural_architecture_lab.ncm_public_author_milestones` for semantic
replay and reports. Earlier producing sources remain unchanged. The new wrapper
inserts the author branch only after the original probe branch at epoch 2;
it does not replace the original branch function (which is also called by the
author branch's own ancestry verifier).

The new CPU-only preflight root is
`20261027_public_curriculum_author_lineage_preflight_v1`. It supersedes the
unexecuted original-probe-only preflight, retaining the completed depth-lineage
preflight as its ancestor. Run focused failure tests, then verify all 5,263
firsts and the unchanged 128 DEV source guards. Record the actual successful
process receipt before future inference or grading. No judge requests,
model updates, final-test payload access or speed measurements occur.

All commands run from the repository root with locked Pixi `ncm`. CPU stages
use `CUDA_VISIBLE_DEVICES=''`; inference retains its existing deterministic
GPU pilot. The frozen live trainer, checkpoint gates, sources, labels, judges,
rubric, 128-token summary cap and 61/64-per-cohort DEV thresholds are unchanged.
The separate 320-source final test stays sealed until qualification.
