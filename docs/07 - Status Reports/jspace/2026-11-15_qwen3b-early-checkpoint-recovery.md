# Early high-accuracy checkpoint recovery

This is a separate, prospective warm-start experiment following the user's instruction to prioritize early high-accuracy checkpoints. Existing experiments, checkpoints, selection histories and producers remain unchanged.

## Starting point and scope

Use the saved fourteen-block Qwen2.5-3B checkpoint at cut 22, original joint epoch 1, with development accuracy 50/64 and SHA256 `814563a4de04d91c17bb02d670ed2d5a7609d3bb3460ed92073d721deac7e401`. Preparation requires its completed, exact fresh verification in the feature-resident campaign. A saved metric file without its corresponding weights is not an available checkpoint. Older compressed variants remain separately identified comparison points; differing depths and coordinate systems must not be merged implicitly.

The experiment stays at fourteen compressed blocks. It uses the existing 512 training cases and 64 reused development cases. Do not open the untouched confirmation population or claim final-model quality or inference speed from this recovery.

## Short recovery

Start from the verified checkpoint's BF16 stack and final-norm weights. Use three joint epochs, 32 effective-batch updates per epoch, learning rate 5e-6 and the existing cosine schedule applied to 96 new updates. Keep the remaining configuration, geometry, all-fourteen-block trainability, fixed embedding/readout, microbatching, clipping and arithmetic unchanged.

Retain the same accepted thirteen-block parent's packed output targets and actual final-answer compact-coordinate targets. Preserve the objective and the already calibrated feature coefficient `0.4192096950932534`; do not recapture targets or recalibrate the coefficient in this branch. Preserve the existing normalization energy and training-only target identities.

Only model weights were saved. Reconstruct FP32 master weights from the rounded BF16 checkpoint and reset Adam moments, counters, minibatch RNG and the schedule. This is a new warm start, not an exact optimizer resume. Use the qualified resident optimizer with the existing 6144 MiB state/work budget on the 2060 SUPER and CPU spill policy. Each GPU phase must use the reservation system and the existing allocator caps. The systems pilot performs and discards an update; scientific training starts again from the untouched source checkpoint and fresh optimizer state.

## Checkpoint retention and selection

Retain actual full stack/norm checkpoints for each new epochs 1, 2 and 3, together with its development metrics, records and checkpoint identity. The protected source is a separately referenced candidate, not a fabricated new epoch. Keep metadata for the best three candidates and preserve the available weight files.

Actual recovery selection ranks by development correct count first, then earlier cumulative joint updates at this compression depth. The source has 32 such updates; the new epochs have 64, 96 and 128. These are continuation-step counts, not claims to reproduce the original run's later epochs. Equal accuracy favors the earlier checkpoint regardless of a later candidate's lower CE. Keep CE, KL and inherited scientific selection as separate diagnostics.

The protected 50/64 source must win any 50/64 tie and cannot be displaced by a lower-accuracy candidate. Report the original-native 45/64 minimum separately from this stricter same-depth non-regression rule. This short branch does not establish a universal three-epoch optimum for other compression stages.

## Verification and reporting

Use explicit prepare, discarded-pilot, train and fresh-verify phases with source/input archives and immutable receipts. Validate the complete 96-update history and all three saved epoch states using this branch's own validator; the old twelve-epoch validator does not apply. Freshly replay the shortlisted checkpoints against their own saved development records, with strict state loading and unchanged fixed geometry. Promote only the highest-ranked candidate after its exact fresh verification; preserve the source if recovery does not improve task accuracy.

Development scores are repeatedly used for selection. Neither a new maximum on 64 cases nor lower training loss demonstrates better unseen-task quality. Broader confirmation, ordinary-generation checks, deployment storage and end-to-end runtime remain later requirements for the fully compressed model.
