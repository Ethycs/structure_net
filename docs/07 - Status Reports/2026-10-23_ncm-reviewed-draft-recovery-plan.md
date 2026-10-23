# NCM word training from reviewed teacher drafts

Date: 2026-10-23. This is a prospective recovery after a transport failure,
not a successful completion of the interrupted revision experiment. The goal
remains at least 95% complete-summary semantic accuracy on source-disjoint
evaluation, with the original Llama body and codec frozen.

The first four additional-label revision batches exhausted all three allowed
180-second attempts without a response. Their unchanged all-512-valid gate
became impossible. The remaining queue was stopped, with actual parent-observed
exit -15 recorded in `20261023_additional_revision_transport_abort_v1`. The old
label, data, word, and conditional final workflows retain their failures. No
revision candidate or judgment is regenerated.

## Training data

Replay the existing pilot and main teacher drafts and their first critiques.
Admit every additional first draft whose first Terra judgment passes and whose
unchanged format checks pass: at most 128 stored cl100k tokens, at most 192 GPT2
tokens including EOS, exact tokenizer round trips, nonempty evidence-linked
teacher claims, and no embedded special tokens. Selection uses teacher labels
only; no student evaluation is used. Retain all 512 scores and rejected labels.
Retain the complete latest critique judgment cache, including the original 930
entries, pilot revision judgments, and negative decisions. No new teacher or
judge calls occur during data preparation.

Keep all original 252 TRAIN rows and their word and concept IDs exactly. Match
new labels to the existing complete-source metadata and frozen parent feature
manifests. Recheck disjointness from both development cohorts and sealed source
metadata without opening either sealed source file. Encode accepted new targets
with the original FP32 CPU codec. A four-case pilot uses the first admitted label
in each length band, repeats target encoding exactly, and preserves those codes
in full preparation. First and longest target encodings also repeat in the full
preparation. Neither input nor target is clipped.

## Fixed model and numerical regime

Reuse the registered final 16,384-update concept checkpoint
`27f835d1391d7e14439716a0514a86d5b301bee79341199ab9dcfe88748e454d`.
The model, original codec and parent body remain frozen. Use the same independent
word branch and fixed configuration as the generated-history study: 32,768
updates, batch one, seed 20142, sampler seed 21142, AdamW with betas 0.9 and 0.95,
weight decay 0.01, gradient clipping 1, 256-step warmup to 0.0001 followed by cosine
decay to 0.00001. FP32 master parameters, optimizer, and loss; BF16 forward;
TF32 disabled. Preserve the architecture and dropout. Train primary word loss
on the frozen forecaster's actual source-only histories, plus 0.25 loss on correct
TRAIN histories. The word branch never forecasts its own coarse codes.

Keep the existing cache, maximum-frame memory, gradient, checkpoint, optimizer,
RNG, frozen-state, and composed-inference pilots. Preserve intermediate recovery
checkpoints but select only the fixed final update. All full source features and
192 raw output IDs remain available. No oracle concepts, repair, lookup,
autoregressive fallback, semantic rerolls, or best-of selection enter evaluation.

## Evaluation and completion

Use the unchanged 128-token semantic rubric and qualified models: Terra first,
then Sonnet on identical texts after each cohort's Terra admission gate. Both
64-case development cohorts require 64 valid judgments and at least 61 passes
per judge. Archive and register the complete development outcome before any
sealed source access. A failure leaves both sealed source files unopened.

Only a qualifying development result admits the original fixed 256 short and
64 longer sealed sources. Preserve all denominator cases and require every
judgment to be valid. Final success requires the same cases to pass both judges:
at least 244/256 short and 61/64 longer. Preserve raw first judgments, failures,
source and checkpoint identities, confidence intervals, and exact registry
read-back. A point estimate is not a population lower-bound or human-rating
claim. No speed or compression claim precedes measured task quality.

## Artifacts and execution

New data roots are `20261023_reviewed_draft_{pilot,data}_v1`. Model stage roots
use `20261023_reviewed_draft_history_*_v1`; conditional final stage roots use
`20261023_reviewed_draft_final_*_v1`. Word hypothesis is
`ncm-reviewed-draft-history-word-v1`; final hypothesis is
`ncm-reviewed-draft-history-final-quality-v1`. Both controllers capture this
plan, the recovery adapter, data builder, controller, and reused producing code.
The adapter changes destinations and provenance only; core training and quality
rules are reused unchanged. Actual child waits supply durable terminal receipts.

Before launch, test admission and rejection preservation, input identity guards,
adapter idempotence, distinct stage roots, and final child dispatch. Check CLI
entry points and the diff, then commit source, tests, this plan, and the completed
concept-training report. Run the four-case CPU target pilot before full target
preparation, then the existing CUDA pilots before expensive training. Keep all
unrelated work unchanged.
