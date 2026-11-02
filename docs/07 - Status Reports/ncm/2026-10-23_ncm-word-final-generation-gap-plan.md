# Diagnose the trained word-final concept-generation gap

Date:2026-10-23. Exploratory component diagnostic
`ncm-word-final-generation-gap-v1`. The512-update training worker exited0with
exact reload and unchanged codec. Its first two held-out outputs repeat or
corrupt entities. Full128-case source-only generation is running; no final
complete-summary accuracy has been measured. Preserve those outputs and
complete their fixed quality evaluation regardless of this diagnostic.

The training trace has low teacher-conditioned word loss and higher concept
loss. Test whether this reflects poor word reconstruction, difficulty predicting
concepts from sources, or disagreement between parallel training and inference.
This is a mechanism measurement on TRAIN labels, never a task-accuracy result.

Select16of252TRAINcases before prediction:four in each source-cl100k-token
band [0,512),[512,1024),[1024,1536),[1536,2049), ordered within each band by
SHA256(`ncm-word-final-generation-gap-v1:`+caseID). All four bands have at least
four cases. No development cases, critiques or final labels enter selection.
Require completed frozen128development outputs before running this local
diagnostic. It must not alter any output, checkpoint, judge request or fixed
quality gate, and cannot authorize training before that evaluation is archived.

Load only the final512checkpoint with the same frozen codec, FP32weights,
BF16autocast, full frozen source features and device-safe complete geometry.
Use physicalGPU0after source-only generation exits; the original parent is
not loaded. No gradients, parameter updates, generation fallback or repair.
The existing full-size inference and gradient pilots cover these operations.

For every selected TRAINsource, retain eight source-only predicted code levels
and the192raw generated word IDs. Separately run teacher-conditioned prediction
using correct target-derived earlier codes, then using the generated earlier
codes with a dummy unused finest-VQ level. Retain all code/word argmax IDs.
Compare non-PAD target word recovery for correct versus generated conditioning,
and word-ID agreement between actual generation and the full parallel forward
at that generation's own concepts. These literal comparisons are descriptive;
different wording can be valid, and none is a complete-summary quality score.

Report per-level concept recovery against the training labels, both over all
codes and over blocks containing at least one non-PAD target token. The latter
avoids hiding semantic errors behind entirely padded48-position blocks.
For wrong source-only predictions in active blocks, count how often the predicted
code belongs to the eight nearest non-self neighbors of the correct code.
This measures how well the current5%nearest-neighbor corruption covers actual
errors. Keep raw numerators and denominators, source identities and hashes.
Require model parameter and frozen codec hashes unchanged before and after.

The paper replaces conditioning codes uniformly within each codebook with
level-dependent rates, and discusses errors accumulating across levels.
Our5%nearest-neighbor recipe is an adaptation. Its AppendixF.3notes that using
actual model errors could improve the match to inference; it does not establish
that approach for our task. [NCM paper](https://arxiv.org/html/2610.14162v1#A6.SS3)

Use a fresh root `20261023_word_final_generation_gap_v1` under the paper-regime
experiment directory. Bind this plan, observer source, final checkpoint,
complete TRAINmetadata, frozen source caches and fixed development outputs.
Record actual terminal status. Interpret this result alongside the independent
128-case complete-summary evaluation. No speed, final accuracy or95%claim.
