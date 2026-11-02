# RTX2060 memory amendment for full732 training

Exploratory hypothesis `ncm-sequential-full732-v2`. This follows the unchanged
[full732 plan](2026-10-25_ncm-sequential-full732-plan.md) and the user's choice of
the physical GPU1 RTX2060. It does not change the dataset, model, loss, precision,
optimizer, clipping, seeds, update budget, quality gate or final-test rule.

Known before this amendment: the v1 lifecycle pilot passed the real RTX2060
packed-versus-sequential logit and gradient checks for first/longest sources.
Its first optimizer update completed without a CUDA out-of-memory error, then
the program stopped at its declared 2,200,000,000-byte allocation limit. The
exact peak was not written. No checkpoint or quality result exists. Keep that
failed root and its source snapshots unchanged; do not use its updated weights.

The next pilot restarts from the same qualified decoder and sequential generator
parent. Raise only the allocation budget to 2,400,000,000 bytes. The card currently
has 2,489 MiB free before our process. Existing Triton services remain untouched.
The original preflight still requires at least 2,400,000,000 bytes free after CUDA
initialization. This budget change is a systems amendment, not a quality rescue.

Record allocated, reserved, peak allocated, peak reserved and global free memory
after loading, both parity controls, initial diagnostics, optimizer creation,
each first-two-step microbatch, unscale, clipping, optimizer updates, final
diagnostics and strict checkpoint reload. Persist readings before checking the
limit, and preserve exceptions in `failure.json`. The peak covers the whole
lifecycle rather than resetting after diagnostics. The full-training loop uses
the same measurements after each update. No reduction in source length or model
precision is allowed to satisfy this check.

Use new roots `20261025_sequential_full732_{pilot,train}_v2`. Only a successful
two-update pilot with parity, finite updates, whole-lifecycle allocation within
the new budget, unchanged frozen tensors and strict checkpoint/optimizer/RNG/
output reload permits the fixed 2,048-update run. The pilot is systems evidence
only. If it fails, preserve the failure and do not start full training.

The remaining source-disjoint DEV evaluation and separate final-quality and
speed gates follow the original plan without changes. A fit claim covers the
NCM stage with cached Llama source features and CPU FP32 optimizer state; it does
not establish end-to-end residency of the Llama body on this card.
