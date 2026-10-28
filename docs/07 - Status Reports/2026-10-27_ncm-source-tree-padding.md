# Per-source hierarchy depth repair

The activation-checkpointed attempt v2 fit the FP32 parity operation in
3,907,890,688 allocated bytes, but failed clean serial/batch gradient parity:
relative error 0.01830263, maximum absolute error 0.000188005. Logits were within
the fixed tolerance, with zero argmax disagreements. The dominant gradient
differences were all twelve newly added tree output projections; original leaf
and generator gradients differed only by ordinary numerical error. No optimizer
update occurred. Retain both v1 OOM and v2 parity failures and their receipts.

Diagnosis: tree depth was determined by the longest padded batch member. A
197-position source requires three eight-way ancestor levels, while the
2,273-position control requires four. The shorter member gained a duplicate
root when batched, changing its level mixture and gradients. Zero initial tree
outputs concealed this in forward-only initialization checks.

Give each example a level mask from its own right-padded valid source length.
An ancestor level is active only while its current child length exceeds one;
update that length by ceiling division by eight. Exclude inactive duplicate
roots from the level-router softmax. A one-token source has an all-zero ancestor
mixture with finite values. Retain every original token and pooled ancestor,
parameter, initializer, loss, sampler and checkpointing setting.

Add CPU regressions for lengths 1/3/17/65 in a single padded batch, including
nonzero tree outputs and level weights. Require individual/batched output and
input/parameter-gradient agreement. Confirm the old unmasked implementation
fails this comparison. Require exact checkpoint/recomputation equivalence with
and without dropout. Then rerun the unchanged real checkpoint lifecycle and
its existing strict GPU tolerances, mixed precision, 32-example accumulation,
two updates and exact restarted second update on GPU1 RTX2060.

Use root `20261027_source_tree_training_lifecycle_v3` and storage
`/data2/structure_net/ncm_source_tree_20261027_v1/lifecycle_per_source`.
The new producer binds the original plan, both failure roots, both amendments
and successful new CPU checks. Stop and preserve another failed contract.
This is still systems qualification; no summary-quality result is inferred.
