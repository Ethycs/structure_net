# Test broad adaptation of the released learned NCM codec

Date: 2026-10-23. Exploratory hypothesis
`ncm-broad-learned-codec-reconstruction-v1`. The isolated clean native teacher
is being evaluated separately; no outcome from it selects this codec recipe.

Question: can broader unpaired reconstruction improve the released learned
codec on the 252 complete summaries excluded from codec fitting? Earlier
small-data adaptation reduced novel token recovery from 96.93% to 92.61% on
20 short texts. Preserve that failure. This study tests the codec component,
with correct target-derived codes; it cannot establish source-to-summary
accuracy or a working NCM attachment.

Use the immutable `20261023_learned_codec_corpus_v1` corpus: 31,109 windows of
48 GPT-2 positions from 7,088 unpaired texts in 1,024 training groups, with
1,303,913 valid positions including genuine end-of-text markers. Whole-summary
targets and their source groups are excluded. Saved atoms supply unpaired text
only. The 252 complete diagnostic targets use four 48-position blocks and
retain all text, including the longest 136-token target. Concatenate predicted
IDs before the first global EOS and text decoding; do not insert EOS at block
boundaries, decode blocks into separate strings, trim targets or repair output.

Load the original released XSum tokenizer checkpoint and its unchanged
48-position architecture in FP32 on physical GPU2. Use all nine released code
levels and their 4,096-entry dictionaries. Original parent/MLP weights are not
loaded or updated. Only the new in-memory codec copy trains; preserve the
released file and all previous adapted checkpoints.

After focused CPU tests, run a real CUDA pilot with the same codec and
microbatch four: two training updates, accumulation one, using fixed first and
last corpus windows. Probe the first complete-summary diagnostic and the
longest target, with stable ID tie-breaking. Require finite reconstruction/VQ
losses and gradients, EMA codebooks excluded from AdamW, nonzero encoder and
decoder gradients, unchanged original checkpoint file, and exact model,
optimizer, CPU/CUDA/sampler RNG and output reload. Preserve nonpersistent
fine-dropout step state as well. Stop on OOM, nonfiniteness, drift or peak
allocation above 4.5 GB. The pilot is systems evidence only.

Full budget: 2,048 updates, microbatch four, accumulation eight, uniform
sampling of corpus windows with replacement. Seed8142 for model/dropout and
seed9142 for the independent sampler. AdamW beta=(.9,.95), weight decay .01,
clip norm one. Warm up128updates to2e-5, then cosine decay to zero at2048.
Use the released forward reconstruction CE (PAD ignored) plus its returned
VQ loss, keeping its .5 fine-dropout configuration and EMA updates. This
reconstructs a trainer around released modules; no official trainer was
released, and the module's all-level VQ loss is not claimed to reproduce every
detail of the paper's sampled-cutoff formulation. No Gaussian noise or source
prediction in this stage. Start fresh from the released checkpoint, never from
the pilot. Select only the final2048checkpoint.

Save recovery checkpoints every512updates. After terminal training, measure
both the original and final codecs on all252 complete targets, using each
codec's own target-derived codes and raw argmax. Report non-PAD token recovery,
whole-text matches, first-EOS agreement, active invalid vocabulary IDs and
changed-number cases for each arm, retaining every reconstruction and code.
Compare paired cases; do not choose between arms per case. Require token
recovery at least99.5%, first-EOS agreement at least99%, no active invalid IDs
and no increase in changed-number cases before a separate semantic codec audit.
Exact whole-text matching is descriptive, not the application's accuracy goal.
Passing this representation gate still does not admit a source-to-code quality
claim; conditional-generator training needs its own protocol and unseen task
evaluation. Both final task cohorts stay closed; no speed claim.

Fresh roots `20261023_broad_learned_codec_{pilot,train,reconstruction}_v1`.
Freeze producer/config/input/vendor hashes and runtime, retain failures and
actual terminal receipts, then aggregate, report and register the measured
result. GPU0 native development generation may run concurrently: no shared
weights or tensors and no throughput benchmark is taken during contention.
