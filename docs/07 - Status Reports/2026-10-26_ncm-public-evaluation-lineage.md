# Public curriculum evaluation: preserve the completed depth comparison

Prospective evaluation amendment for `ncm-public-curriculum-v1`. The first
public-training epoch is running under its frozen original implementation.
No public milestone has generated or graded DEV summaries. The completed
12/24-block depth comparison returned zero complete-summary passes in both
arms and extended the canonical first-judgment cache from 4,179 to 4,435.

Use `experiments.structure_net.ncm_public_development_lineage` for public
milestone evaluation. Start with the unchanged 4,179-first ancestor, replay
the two completed depth-cache extensions, and retain all 4,435 first judgments
before incorporating any earlier public milestones. Require the recorded
depth campaign and cache hashes, actual successful receipts, unchanged prior
judgments, and exact packet/role membership for every added judgment. No new
judge call may replace an inherited first response or retry an uncertain one.

This changes evaluation provenance only. Preserve the source-only prediction
path, model selection, all-public/all-reviewed source guards, semantic rubric,
Terra/Sol identities, invalid-response handling, both >=61/64 DEV cohort gates,
128-token stored limit, and sealed final-test protocol. Keep training, optimizer,
corruption, eight-epoch schedule, checkpoints and training fingerprints unchanged.

The new producer has schema `nal.ncm-public-curriculum-dev.v2`. Retain the
previously reserved `20261026_public_curriculum_epoch{epoch}_DEV_{stage}_v1`
directory names: all sixteen are unused at this amendment, and the frozen
trainer's continuation gate already addresses those paths. Every invocation
still requires a fresh directory. No existing produced evidence is replaced,
and the unchanged original evaluation module remains available as provenance.

Before any milestone evaluation, run focused lineage/receipt tests and a
CPU-only, zero-request preflight into the new root
`20261026_public_curriculum_lineage_preflight_v2`. Verify all 128 DEV sources
against all 162,353 admitted public training identities and the reviewed corpus,
retain the existing exhaustive text-overlap-screen binding, and prove the full
4,435-first cache is preserved. The preflight is systems evidence, not quality.
Record its actual tool completion. Each real evaluation stage retains its
existing inference and judge pilots and its actual completion receipt.

Reproduction (repository root; locked Pixi `ncm` environment):

```text
python -m experiments.structure_net.ncm_public_development_lineage preflight
python -m experiments.structure_net.ncm_public_development_lineage inference_pilot --epoch 1 --attempt 1
python -m experiments.structure_net.ncm_public_development_lineage outputs --epoch 1 --attempt 1
python -m experiments.structure_net.ncm_public_development_lineage judge_pilot --epoch 1 --attempt 1
python -m experiments.structure_net.ncm_public_development_lineage grade --epoch 1 --attempt 1
```

Use CPU-only execution for preflight and judging; GPU1 RTX2060 for generation.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and retain strict deterministic execution.
This amendment does not establish task accuracy or authorize opening final320.
