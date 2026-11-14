# Compressed-layer combinations and targeted joint repair

Date: 2026-11-14. Exploratory studies
`JSPACE-QWEN-LAYER-COMBINATIONS-01` and
`JSPACE-QWEN-LAYER-GROUP-REPAIR-01`.

## Question and starting evidence

Which recovered compressed layers are compatible in the assembled Qwen2.5-3B,
and can joint distillation of a small harmful group repair its interactions?
The original FP32 model scores 50/64 on the existing development population.
The independently recovered 36-block assembly scores 27/64, while compressed
blocks 0 and 1 with a native suffix score 49/64. Restoring independently weak
layers 12 and 15 to native form scored 24/64: isolated quality is not a reliable
guide to contextual repair. A separate whole-model continuation scored 28/64
after 96 updates; it does not supply weights to this study.

Use the authenticated independent assembly from
`data/experiments/jspace_recovered_assembly/20261114_v1`, fingerprint
`987f470548540a645a2b0f1d3bc8fedc1fe043892591f7630e7b10a5fc0928a6`.
Each layer selects its exact recovered compressed checkpoint or the original
native block. Keep the original embedding, final norm and readout fixed.
Native BF16-stored weights are promoted to FP32; recovered weights and all
arithmetic remain FP32. Rank 1024, full 2048-wide residual carry, native head
dimensions, softmax and MLP width remain unchanged. No basis is refitted.

## Fixed exploratory search

Every mask executes the actual ordered model from original raw prompt tokens.
Use all 64 original development cases, 16 per family. Cached prefixes may be
shared only when the entire preceding native/compact decision sequence and
input batch are identical. Never substitute a native cached boundary after an
earlier compressed intervention. The untouched 512-case confirmation set is
excluded from every selection and training decision in this study.

Before inspecting search outcomes, fix these stages:

1. Controls: native 50/64, all compact 27/64, accepted two-block 49/64, and
   the known native-restoration pair {12,15}, 24/64. Require exact predictions
   and absolute 5e-4 plus relative 1e-5 logit tolerance against archived results.
2. In the all-compact background, restore each of 36 individual layers and
   every one of the 630 pairs to native. Also restore 12 groups of six: the six
   contiguous groups and six groups defined by layer index modulo six.
3. Choose six candidates deterministically. If the best six-layer group
   exceeds the best pair by at least two correct answers, use that group.
   Otherwise start with the two best singleton restorations, then add endpoints
   of beneficial pairs ordered by excess rescue, filling to six. Excess rescue
   is `correct(restore i,j) - correct(restore i) - correct(restore j) + 27`.
   A beneficial pair must improve on 27. Fill remaining slots from singleton
   ranking if necessary. Ties use correct count, native decision agreement,
   lower conditional choice CE, then lexical layer indices. This interaction
   quantity is descriptive and background-dependent, not a causal theorem.
4. Exhaust all 64 subsets of those six candidates in BOTH backgrounds:
   candidate subset compressed with every other layer native; candidate subset
   restored to native with every other layer compressed. Deduplicate exact
   masks across stages without discarding any result.
5. Report best and worst at each identical compressed-layer count, paired
   gains/losses, per-family accuracy, margins, choice CE and native agreement.
   Report the most compressed tested mask reaching 47/64 separately. This is
   not a search of all 2^36 models or a global optimality claim.
6. Select exactly three repair layers: the candidate triple whose restoration
   to native gives the best score in the full-compact background, with the same
   fixed tie-breakers. Record whether restoration actually improves on 27.
   Freeze this selection before training. Native rescue does not prove that the
   compressed group can learn the rescue.

The screen contains 681 distinct masks; refinement requests 128 masks with
cross-stage duplicates removed. The implementation determines the exact final
unique count from the fixed rules. All successful and failed executions remain
recorded; no outcome-based truncation or candidate changes are permitted.

## Runtime validity and resources

Use locked offline Pixi `ncm` and the experiment reservation controller for
every GPU process. Search holds native and compressed banks resident on three
GPUs: layers 0..17 on RTX 3060 (11264 MiB reservation), 18..28 on RTX 2060 SUPER
(7168 MiB), and 29..35 on RTX 2060 (5120 MiB). Resolve physical UUIDs from the
lease, cap each allocator at its lease minus 256 MiB, and check live reservation
identity and peak memory. Original embedding/readout stays on 3060.
Execution invokes its guard at every block; controller readiness reads may be
cached for at most one second, with forced checks at stage boundaries. Allocator
caps remain active continuously. Record guard calls and actual controller reads
separately so reservation signalling does not dominate the combinatorial scan.

Before the search, test the actual longest-context microbatch, compare shared
prefix execution with independent execution on overlapping masks, and check
unchanged parameter/buffer state. The pilot is systems-only; discarded pilot
outcomes do not select candidates. Pass all four native/compact controls on the
full population before the search. Microbatch is four, no gradients, no TF32.
The campaign reports decoder-call savings and execution wall time as search
telemetry, never as deployable token throughput. On memory/control failure stop
and preserve evidence; any revised runtime gets a new root and source identity.

## Targeted joint distillation after selection

Starting again from the independent all-compact assembly, train only the three
selected compressed blocks together. Freeze the other 33 blocks, original norm,
embedding/readout, all bases, anchors and other buffers. Frozen blocks between
and after trainable blocks must propagate input gradients. Every student batch
starts from raw tokens; no teacher state replaces the student stream.

Use the original 512 training cases and 64 development cases. Cache the
ORIGINAL frozen teacher's FP32 next-boundary states at the three selected
layers only; control its development decisions against native 50/64 and its
selected hidden states against the existing matched native FP32 development
banks. Retain existing matched FP32 full-vocabulary teacher logits at midpoint
and answer. Cache identities, offsets, precision, selected layers and hashes.
This stores about 1.92 GB of hidden targets, rather than all 37 boundaries.

For each selected layer, actual student output y, native next-state t and fixed
basis U, use the valid-token normalized loss

`L_layer = mean_tokens ||(y-t) U||² / max(mean_tokens ||t U||², 1e-8)`.

Average equally over cases and the three layers. Teacher targets and denominator
are detached. Preserve the FULL derivative through y, including earlier selected
layers. Numerically this matches reachable corrections from actual student
input s: predicted `(y-stopgrad(s)) U` and target `(t-stopgrad(s)) U`. Do not
subtract differentiable s from the prediction, which would cancel a residual
gradient path. No moving student-dependent normalization is used.

Add this auxiliary with coefficient 1.0 to the prior explicit decision objective:

`L = L_feature + .2 T² KL_full + .5 T² KL_choice + .3 CE_choice`, `T=2`.

Full-vocabulary KL uses midpoint and answer; choice KL and gold-label CE use
the valid answer choices. This is an exploratory targeted repair with a new
auxiliary and longer schedule, not a matched attribution of benefit solely to
group selection versus the earlier all-layer trial.

Fix six epochs, 192 updates, microbatch one, accumulation 16, AdamW LR 1e-5,
weight decay zero, global gradient clip one, seed 27866. Use FP32, the tested
two-GPU student partition (0..23 on 3060, 24..35 on SUPER), activation
checkpointing and bounded CPU optimizer state. The initializer remains eligible.
Select highest actual development correct count, then lower choice CE; retain
epoch predictions and metrics, and an atomic selected delta checkpoint for only
the three blocks. Do not change the chosen group mid-run.

Before scientific training, a separate one-update pilot must reproduce the
untouched 27/64, show finite gradients and changes in all and only the selected
blocks, and prove fixed state unchanged. Discard pilot updates. After training,
strictly reload the selected delta into the original assembly in a fresh
process and require matching predictions/logits. Compare subgroup feature loss,
task accuracy and teacher agreement separately. A lower feature loss is not
successful task recovery. The same model-level gate remains at least 47/64;
apply the five-percentage-point tolerance once, not per layer. Below that gate,
do not open confirmation or claim deployment speed success.

## Artifacts and outcome meanings

New append-only roots:

- `data/experiments/jspace_layer_combinations/20261114_v1`
- `data/experiments/jspace_layer_group_repair/20261114_v1`

Freeze outcome-relevant producers, tests and this protocol before execution.
Record every mask, candidate rule, exact selected group, predictions, controls,
source/data/checkpoint hashes, reservations, timing, memory and failures. Keep
search and training results distinct. Search findings remain exploratory after
many development comparisons; neither repeated reuse nor a single seed supplies
independent confirmation. Stabilized reports and typed registry records must
include exact read-back and `confirmed:false` unless a separate confirmation
study is actually completed.
