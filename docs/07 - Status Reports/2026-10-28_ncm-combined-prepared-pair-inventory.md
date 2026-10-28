# Combined prepared training-pair inventory

The user asked which dataset is largest and whether the datasets can be
combined after inspecting the latest recovered-model examples. Build an
exploratory, CPU-only union index of the existing 162,353 public XSum pairs,
43,894 recovered conversational pairs, and 2,384 reviewed anchors.

Use the completed original catalogs and admission records. Read every original
row, check source/target hashes where the text is present and complete native
target frames, and bind the complete recovered source/context catalog. Deduplicate
only exact source-text/target-text pairs. Retain every origin, original split,
source group and role/date-conditioned source-record pointer. Different targets
for a shared source remain distinct. Report distinct source counts and exact
cross-domain source overlap separately from pair counts.

The index references the original immutable data; it creates no features,
targets, summaries, model updates, or new split assignments. It is preparation
for a future mixed-domain experiment, not training admission or a new accuracy
result. In particular, it does not complete a union-wide near-duplicate split
audit or certify historical labels as factual gold. Keep Final320 sealed.

The larger archive inventory is a different population: 1,069,574 ordinary
candidate pairs, of which 259,636 fit the application's content-length limits.
Those are not added to this union because they include the recovered subset and
have unresolved provenance, split, codec-capacity or label-quality requirements.
The 99,367-frame decoder corpus supplies target-only reconstruction training;
it is not an extra set of source-summary pairs.

Require the three union/split contract tests before execution. Use locked Pixi
ncm from the repository root, and a fresh append-only artifact root
`20261028_combined_prepared_pair_inventory_v1`. Preserve failures and bind the
actual terminal result. This inventory does not alter any running experiment,
checkpoint, first judgment, or preregistered evaluation order.
