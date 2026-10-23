# Train a word-final NCM attachment to an unchanged Llama parent

Date: 2026-10-23. Exploratory hypothesis `ncm-word-final-frozen-parent-v1`.
The original and EMA-adapted codec oracle semantic controls scored8/64 and6/64;
uncompressed labels scored63/64 Terra and64/64 Sonnet. Decoder-only transfer
completed2048updates but reached95.120%token recovery versus95.179%released,
failing its representation gate. Preserve every result and invalid audit.

This is a new NCM adaptation, not an exact paper replication. Retain eight
released coarse code levels and predict vocabulary tokens directly at the final
level. The old codec gates remain failed. The final word representation no
longer requires decoding a quantized word label. This does not qualify a model:
the objective remains at least95% complete-summary source-only task accuracy
on disjoint sources. Oracle words, training fit and interface tests do not count.

Use all252 vetted complete TRAIN summaries, the existing64short development
sources, and the64source-disjoint longer diagnostic sources. The64codec-audit
cases belong to TRAIN for this conditional model. Do not evaluate training fit
as task accuracy. Verify sessions, groups, IDs and source hashes are disjoint
between all three cohorts and from sealed cohorts using metadata only. Preserve
the original256 and longer64 final cohorts unopened. No failed development
summaries or judge critiques enter TRAIN. All sources must match the public
LongMemEval release exactly; no new teacher generation in this experiment.

CPU preparation retains the exact native chat prompt and every source token.
Use the original summarization SYSTEM prompt and original frozen28-layer
Llama3.2-3B-Instruct body. Encode full source prompts with no output head or
KV retention. The parent runs BF16 on physicalGPU0; require original body hash
75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c before and
after caching. All parent parameters, including MLPs, stay frozen and in eval.

NCM retains its pretrained GPT-2 source embedding path. Align every GPT-2 source
token to overlapping native source-token character offsets in the complete
native prompt, average the frozen hidden states in FP32, and store FP32 aligned
per-token features. Record all overlap indices/weights and verify positive
coverage and source identities. New LayerNorm/projection produces a parent
feature residual initialized to zero, preserving the raw source embedding path
at initialization. Right-pad sources into multiples of256, with explicit masks;
allow complete frames up to8192, reject overflow, and never clip or select shorter
cases. Rebuild only RoPE/attention geometry per frame, retaining learned weights.

Full targets use the existing four48-position GPT-2 blocks,192word positions.
Keep all text, one true global EOS50256, then PAD50257. Reuse SHA-bound original
target-derived code arrays, whose representation matches the released codec.
Only the first eight levels condition prediction. Target word IDs and unused
finest VQ labels must not enter the source or current-level inputs. Tokenizer
round trip and the128cl100k storage cap are required. During loss and generation,
use vocabulary IDs0..50256; PAD is ignored in loss and never generated. No output
repair, per-block EOS, fallback, reroll or best-of selection is allowed.

Initialize every generator weight from the released XSum checkpoint, repeating
root queries for four blocks and rebuilding geometry only. Retain the frozen
codec unchanged. Copy the released final projection and decoder into new
trainable word modules. The parent-feature projection is zero initialized;
there are no randomly initialized replacement generator blocks. Keep released
generator/decoder training dropout(.3/.1). Parent features are detached.

After focused CPU tests, use a real GPU0 parent-feature pilot on the first short
training case and the longest native prompt among all nonfinal sources. Require
exact repeated hidden/aligned features, correct complete maximum shape, original
parent hash unchanged, no gradients and allocation below8.6GB. Cache all380
nonfinal sources only after terminal success; retain the two pilot cache entries
exactly. Parent caching is preparation, never a quality measurement.

Then run a separate real full-size head training pilot on first/longest TRAIN
sources, padding the latter to the largest observed nonfinal source frame for
memory coverage. Use two updates, microbatch one, accumulation one, actual
coarse/word targets and real frozen features. Require finite loss and gradients
only in new modules, unchanged codec and caches, exact model/optimizer/CPU-CUDA/
sampler/noise RNG reload, exact deterministic inference after reload, and peak
allocation below8GB. Original parent is not loaded during head optimization.
Mixed precision: FP32 master parameters/AdamW, BF16 CUDA autocast, FP32 losses
and codebook distances, TF32off. Stop on systems failure; preserve it.

Full fixed budget:512updates, microbatch one, accumulation four, uniform sampling
over252TRAINcases with independent sampler seed14142. Model/dropout seed13142;
noise seed15142. AdamW beta=(.9,.95),wd.01,clip1; warm32updates to5e-5 then cosine
to zero at512. Only final512 is selected; save128/256/384 for recovery. Start
fresh from released/new-copy initialization, not pilot weights. No budget
extension or best-checkpoint selection based on losses.

For each microbatch, use clean and neighborhood-corrupted coarse histories.
Compute each codebook's eight nearest non-self centroids in FP32, bind the saved
neighbor table and reuse it. Independently replace5%of positions in each of the
first eight gold levels with a sampled listed neighbor. Never corrupt word
labels or source inputs. Optimize clean wordCE + .1*noisy wordCE + .1*mean clean
coarse-level CE. Each level CE is averaged over its own positions; PAD words are
ignored. This is supervised neighborhood corruption, not a claim to replicate
CLLM fixed-point consistency or released NCM training.

After terminal training, qualify deterministic source-only inference/reload on
the first two short development cases, retain outputs, then generate all64short
and64long diagnostic cases using eight predicted coarse levels and one word
level. Decode concatenated192IDs once, stopping only at the first global EOS;
retain all raw IDs and codes. Both cohorts must separately have64valid and at
least61complete-summary passes under qualified Terra and then independent
Sonnet, with length/readability/salience/faithfulness included. Preserve all
earlier first judgments by exact packet/rubric/model/role identity, including
the complete canonical label caches and negatives. Prove public payload identity
before external judging. Final cohorts require a separate frozen evaluation
protocol after development passes. No speed claim in this experiment.

Fresh roots `20261023_word_final_{data,feature_pilot,features,train_pilot,train,
inference_pilot,outputs,grade,results}_v1` under
`data/experiments/ncm_paper_regime`. Freeze source/config/input identities, record
actual terminal receipts, replay aggregates, report, register conservatively
and verify exact/semantic read-back. Training and caching use CUDA. CPU-only
metadata/offset preparation and contract tests are explicit preparation stages.
