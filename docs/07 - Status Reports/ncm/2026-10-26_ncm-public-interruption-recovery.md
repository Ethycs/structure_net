# Resume the interrupted public epoch from its committed checkpoint

The intentional conversation interruption removed training tool session68593,
Python process3056752 and completion-controller cell1426. Read-only checks found
no other public trainer, no result or failure record, and a trace ending at
update335. The original process exit code is unknown. Its last committed
checkpoint is update256, SHA256
`1d8708a72ab9f304edf22bbfa8b3cf3bfbb26e3692f55e7afae817426be97a97`.
No public DEV prediction or judgment has been inspected.

Preserve the entire first attempt and its79 uncommitted tail updates. Verify
the unchanged producing sources/inputs, checkpoint bytes, saved curriculum,
optimizer/scaler/RNG fields, and replay exactly updates1–256 with the qualified
CPU auditor. Retain the whole335-row trace in the artifact bindings. This is
operational recovery evidence, not an accuracy result or a new training recipe.

Use fresh root `20261026_public_epoch1_interruption_recovery_v1` for this
verification. Its actual successful tool receipt belongs to the verifier only;
never manufacture an exit code or success/failure receipt for the lost trainer.
Then run the unchanged `ncm_public_training --end-epoch 1 --attempt 2 --resume`
against the original `epoch1_attempt1/step_000256.pt`, on physical GPU1 RTX2060,
with locked Pixi ncm and the existing deterministic environment. Resume all
model/optimizer/scaler/RNG state; the next committed optimizer update is257.
Discard the uncommitted tail from the resumed trajectory but keep its evidence.

The original finalizer requires a known nonzero exit for interrupted ancestors.
The new `ncm_public_interruption_recovery` wrapper adds a narrowly bound branch
for this exact ancestor with unknown exit, observed absent process/tool handle,
and independently verified committed prefix. All other terminal-receipt checks
remain unchanged. Training-chain audits use fresh epoch-specific
`training_audit_v2` roots. Semantic replay/report roots and quality gates stay
unchanged; use this wrapper for `audit`, `replay`, and `report` after recovery.
Do not use the old full-chain auditor for this unknown-exit ancestry.

Recovery does not change the model, data, sampling, corruption, learning-rate
schedule, milestones, judges, or final95% endpoint. Final320 stays sealed.
