# Train complete summaries from generated concepts

Date: 2026-10-23. Exploratory hypothesis `ncm-generated-history-word-v1`,
one model seed, underpowered. The latest word-final model failed 0/64 short
and 0/64 longer development summaries. Its 16-TRAIN-case diagnostic recovered
95.8% of words with correct concepts and 1.3% with generated concepts. The
fixed 16,384-update concept bootstrap is still running. No new task result has
been inspected. This study addresses the word-conditioning failure directly.

Use only the registered final 16,384-step source-hierarchy bootstrap after
its observed zero exit, complete trace audit and conservative registry readback.
Keep that concept forecaster entirely frozen and in eval mode. Copy all of
its new-head parameters into a separate word branch, sharing only the exact
frozen original codec. Original Llama weights and MLPs remain frozen and are
absent from optimization; complete cached source features are reused. This
is an NCM adaptation with two new-head parameter sets, not a paper replication.

Use all original 252 vetted TRAIN summaries plus every individually accepted
additional whole-summary revision from the fixed 512-source label population.
Require the expanded-data worker's actual zero exit, raw first-response and
judgment replay, unchanged old targets, complete feature manifest and all
declared source-disjointness checks. Report the accepted denominator without
removing negative teacher decisions. No development source or judge feedback
enters optimization. The original 16 root-capacity cases retain their earlier
training history as part of the original 252. No additional concept adaptation
is included in this candidate; its forecaster is fixed before word training.

First cache the forecaster's own eight coarse levels on source-only views of
every TRAIN input. The generation function receives no target text, target
words or target concepts. Store every predicted code and its hash. Use a
two-source cache pilot, first and longest source; compare all coarse codes
exactly with the existing full generation path, repeat deterministic generation
and separately exercise a 3,328-position padded source frame. Retain the two
native-frame pilot outputs exactly in the full cache. Verify unchanged
forecaster parameters and original codec. No word output is selected here.

Train only the independent word branch's source normalization/projection,
shared transformer and copied word projection/decoder. Freeze its unused
coarse heads and root queries. The fixed forecaster never receives gradients.
One final-level parallel forward receives complete source features and the
reconstruction of earlier predicted codes, with no previous-word inputs or
finest VQ labels. The main loss is non-PAD word cross entropy at the actual
cached generated history. Add 0.25 times the same loss at correct coarse
TRAIN concepts, using a separate forward/backward. Both histories are detached.
Do not train against approximate random neighbors in this phase.

Fix 32,768 AdamW updates, batch one and no accumulation. Uniformly sample all
expanded TRAIN cases. At the maximum 764 cases this is about 43 exposures per
case. Model/dropout seed 20142; independent sampler seed 21142. Warm 256
updates to 1e-4, then cosine to 1e-5 at the fixed final update. AdamW betas
(0.9,0.95), weight decay 0.01, gradient clip 1. Keep original module dropout.
FP32 parameters/optimizer/losses/codec reconstruction, BF16 forward autocast;
disable matmul and cuDNN TF32. Save 4,096-step recovery snapshots, select only
the final 32,768 checkpoint, and do not extend or select using losses or dev.

Before full training, require a real two-update GPU0 pilot on first and longest
TRAIN cases, padding the second source to 3,328. Exercise both loss paths in
each update, positive word/source gradients, no frozen gradients, unchanged
codec/forecaster/unused heads, peak allocation below 8GB, exact model/optimizer/
CPU-CUDA/sampler RNG reload and exact source-only composed inference after
reload. Qualify the two-model inference residency as part of that pilot. Full
training starts fresh from the registered forecaster, never pilot weights.
Stop on nonfinite values, changed frozen state, invalid routing, input/source
mutation or memory failure, preserving every artifact and actual child exit.

Evaluate the frozen forecaster followed by the trained word branch. Never use
the word branch's own coarse generator. Require a two-case deterministic
checkpoint-reload inference pilot, then retain those outputs in all 64 short
and 64 longer development cases. Retain all 192 raw word IDs and eight coarse
levels. Decode once at the first global EOS; do not truncate text to pass the
128 stored-token cap, repair output, supply oracle concepts/gold history, use
lookup or autoregressive fallback, reroll, or select among candidates.

Use the unchanged qualified 128-token rubric and four-worker Terra grading.
Each cohort needs 64 valid judgments and at least 61 complete-summary passes
before the same texts receive Sonnet grading with the same gate. Preserve
the latest full canonical first-judgment cache from expanded data, including
all 930 previous judgments and all new label negatives. A summary passes only
with supported material claims, useful salient content, readability, nonempty
text and at most 128 stored cl100k tokens. Record invalids and all failures in
the full denominator; no semantic rerolls. Use only proven public source spans
and locally derived outputs at the already authorized central-dev gateway.

Create separate roots `20261023_generated_history_{cache_pilot,cache,train_pilot,
train,inference_pilot,outputs,packets,grade,workflow}_v1`. Hash and copy every
producing file, plan, model, data, feature manifest and code history. A durable
supervisor records actual child exits before admitting each stage; observation
timeouts never cause a restart. Existing destinations are never duplicated.
Both final cohorts remain closed until both development gates pass and a
separate final-evaluation protocol is fixed. Neither development fit nor a
component score completes the 95% final task goal. Establish quality before
any end-to-end speed or compression claim.
