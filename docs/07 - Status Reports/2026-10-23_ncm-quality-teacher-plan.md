# Establish faithful native quality before acceleration distillation

Date: 2026-10-23. Prospective exploratory hypothesis:
`ncm-isolated-native-quality-teacher-v1`. Planned; not executed.

The converged shared-adapter model scored 41/64 complete summaries, with
42/64 faithful versus the unchanged 3B baseline's 52/64 whole-summary passes.
Its errors include list counts, attribute binding, modality and broadened
scope. Original nine-pass and paired-loss failures remain fixed. The separate
word-bank pilot establishes exact cached parent identity and gradient routing,
not a successful summarizer. Do not resume either failed mixed/noisy recipe.

Restore the original frozen 28-block Llama3B body and original trained concept
bank. Freeze concept adapters/heads, books, hint and level biases. Add a fresh
independent rank-eight word bank, initialized with A seed 5142 and zero B;
do not initialize from the two-update pilot or the four-case capacity model.
All original MLP, attention, embedding and word-head weights stay frozen and
shared. Word-quality queries use native previous words without the old concept
hint/bias. This is a quality-teacher control preparing the NCM pipeline, not
a replacement standalone AR model claimed as NCM success.

Use all 143 existing accepted short-summary TRAIN labels plus all 109 vetted
longer-source TRAIN labels, 252 total. The latter cover 512–2048 cl100k source
tokens with accepted counts 27/28/26/28 by band. No diagnostic or sealed source
labels, new teacher calls, saved atomic snippets or development judgments enter
training. Verify source IDs, sessions, groups and exact hashes are disjoint
from all held-out cohorts using metadata, without opening sealed texts.
Verify canonical public provenance and all original label-audit artifacts.
Do not trim a target to fit: require its complete native text plus EOS to fit
128 tokens, storage at most128cl100k, with exact native round trip.
The provided native-training store includes single-atom targets, such as one
item from a multi-item passage. Those are not treated as complete-passage
summary labels in this control; the vetted whole-passage labels supply that
task supervision.

Prepare immutable source-only KV caches for the 109 longer training cases
using the unchanged native chat instructions. Retain existing 143 cache
entries, verifying their hashes. New native input frames round up to multiples
of256, at most2304tokens for the selected prompts; use complete sources and
reject truncation. No target enters source cache computation. A separate
cache pilot uses the first old short case and the absolute longest selected
new native prompt (stable ID tie break), not merely a convenient short example.

After terminal cache and adapter-identity controls, run a separate real CUDA
training pilot on those short/long cases, batch one, two updates, with exact
parameter/optimizer/RNG reload and frozen body/forecaster/books/cache checks.
The long case must exercise the real maximum frame. Require finite gradients
only in word A/B banks. Physical GPU0, BF16 native/FP32 new weights, TF32 off.
Do not launch full training if memory or identity fails.

Full fixed budget: 512 optimizer updates, batch one, accumulation eight,
uniform sampling over all252cases using seed7142. AdamW, weight decay .01,
clip norm one, hard-target native CE over valid positions only. No concept
classification loss, noisy word states, Gaussian noise, raw-language replay,
new positional weights, extra hint or preference loss in this study. Learning
rate warms linearly for32updates to5e-5, then cosine decays to zero at512.
Save optimizer/RNG checkpoints every128updates for recovery, but select only
the final512-update checkpoint for evaluation. Preserve every sampled ID and
loss/gradient trace. A failure or slow loss decrease does not authorize an
extended budget or best-checkpoint selection.

Generate all64 original complete development sources after training using the
isolated word path at a fixed128 whole-frame causal updates plus one full
verification pass, initialized with native EOS words. Require verified fixed
points. This deliberately unoptimized teacher endpoint measures quality; it
has no speed claim. Validate inference/reload on the two-case pilot first and
retain its first output exactly. No output repair or selection. Preserve all
256 prior first judgments (128 baselines,64failed-nine-pass,64converged) and
their negatives under exact packet/profile/role/model identity.

Grade all64 frozen summaries with qualified128-token Terra; require64valid
and61whole-summary passes before independent Sonnet on identical outputs.
Report complete-summary and component counts, invalids, cap failures and
Wilson intervals. Passing only admits a separately planned longer-source and
final evaluation and subsequent NCM acceleration training; it is not goal
completion. Failure requires grounded claim analysis before new changes.

For later acceleration distillation, generate the qualified teacher's own
native fixed points and intermediate histories on TRAIN sources. External
teacher wording is suitable for quality SFT but is not assumed to be the
native model's fixed point. Any learned fast/noisy bank must preserve the
qualified quality path, with actual unseen complete-summary checks before
latency or throughput claims. Neither final cohort is opened in this study.

Use fresh `20261023_isolated_quality_{data,cache_pilot,cache,train_pilot,train,
outputs,grade,results}_v1` roots. Freeze producers, plans and complete source/
label identities. Preserve failures, terminal receipts and registry read-back.
