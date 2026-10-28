# Recovered conversational paraphrase training

Exploratory hypothesis `ncm-recovered-paraphrase-training-v1`: training the
existing frozen-body NCM code predictor on the larger, context-preserving
conversational corpus can improve complete-summary semantic accuracy over the
previously failed application models. This is a new model-development experiment,
not a retrospective rescue of any failed architecture or public-data hypothesis.
The final objective remains at least 95% complete-summary task accuracy on a
source-disjoint evaluation; reconstruction and training loss cannot establish it.

Known before this run: the latest completed application DEV result is 0/128.
Public epoch4 grading is incomplete with transport failures. Both task-supervision
training arms are complete but await their semantic evaluations; the source-tree
training comparison remains in progress. Improved decoder reconstruction with
correct codes is 56/64 reserved native summaries alone, versus 46/64 when combined
with the old source readout. Original decoder results are 45/64 and 44/64.
Those numbers are component diagnostics, not application accuracy.

## Data and exposure

Use all 43,894 pairs in `20261028_recovered_paraphrase_catalog_v1`, with complete
singleton target codes from `20261028_recovered_paraphrase_codes_full_v1` and
complete role/date-conditioned source features from
`20261028_recovered_paraphrase_features_full_v1`. The latter must finish, acquire
an actual successful terminal receipt, and pass its independent disk replay
before the full-population lifecycle or fitting starts. The existing 15-source
feature pilot is an I/O qualification only. No partial feature population may
stand in for the full corpus.

The 38,484 distinct source texts have one to four historical target variants.
Use the exact four-epoch sampler already qualified in
`20261028_recovered_source_balancing_v1`: one recovered presentation per source
per epoch, deterministic target-variant rotation, and three recovered examples
per reviewed anchor. Retain all 2,384 anchors from the isolated expanded v3
catalog, with their original prompts and features. The 881 overlapping source
texts are intentional additional anchor exposure, not independent coverage.
Historical targets remain noisy labels, not semantic gold. Do not add the
excluded substring targets, reserved diagnostic components, or held native
frames. Keep Final320 sealed.

Train all four epochs without choosing checkpoints using DEV: 205,248 examples,
6,416 optimizer updates, and 102,624 noisy-code histories. Use microbatch4,
effective batch32, retaining the last16-example update of each epoch with its
actual loss denominator. Seeds remain 271142 for source/anchor permutations,
272142 for target-variant ordering and 273142 for history noise. Dropout uses
274142. Save complete recovery states every256 updates and at epoch boundaries.

## Model and optimization

Strictly load the previously fixed public epoch2 model checkpoint
`c59051faea7d9005c2d91b1c88113d627d7545c9ae1a9c3dd7ac0a2e8f96f620`;
require model hash
`090ec8c105e47bd2412d4ab9b83411035fd73e2d8f52e575dfb7edea60043f22`.
Keep the existing 105,323,136 trainable generator/concept-source parameters.
This run uses the original three source-attention locations; do not import the
ongoing tree/dense architecture changes or claim to isolate their effect.
The Llama body is frozen and represented by audited cached source features.
The codec and word-source path stay frozen during fitting.

Use the existing sequential teacher-logit implementation, full 641-code objective,
level weighting and deterministic clean/noisy history contract. Reuse CPU AdamW
with FP32 master weights created only after strict model initialization, betas
(0.9,0.95), epsilon1e-8, weight decay0.01, gradient clipping1.0. Use FP16 autocast,
GradScaler initial128 and growth interval2000, deterministic algorithms, TF32 off.
Warm up for128 updates to3e-5, then cosine-decay over the fixed6,416-update schedule
to3e-6. No optimizer-state reuse from the initializer, skipped steps, source
clipping, target truncation, or on-the-fly teacher calls.

Run on physical GPU2 RTX2060 SUPER, allowing at most4.9GB allocated by this
process and requiring that much free memory before model construction. Keep
the existing GPU0 feature extraction and GPU1 tree/dense work unchanged. Stream
FP32 feature shards with one I/O worker, at most two prefetched microbatches and
a256MiB anchor cache. Host RSS limit is12GiB. This is a memory-constrained training
choice, not a measured inference-efficiency result.

## Qualification, artifacts and failure handling

Use locked Pixi ncm. Test configuration, source/context/target joins, source epoch
boundaries, complete/resumed presentation order, noise determinism, trace replay
and strict fingerprints. Before full training, run a separate complete-population
pilot: four controls covering first/shortest/longest source and longest target,
FP32 clean/noisy gradient parity, two real FP16 accumulated updates, complete
checkpoint save/reload, then restore update1 and reproduce update2 including
optimizer/scaler/RNG state and source-only outputs. Confirm frozen hashes and
actual concurrency memory. Independently audit the pilot before full fitting;
pilot updates never initialize the full model.

Use new append-only roots
`20261028_recovered_paraphrase_training_{prepare,pilot,pilot_audit,train,audit}_v1`
and `/data2/structure_net/ncm_recovered_paraphrase_20261028_v1/training`.
Freeze configuration, source, tests, this plan, data identities and environment
before execution. Failures are preserved. A stopped worker is never restarted
on the strength of an observation timeout; a changed or recovered execution
needs a new root and an exact committed-state binding. Full-run continuation
must not repeat previously completed optimizer updates. Successful launch is
not successful completion; audits consume actual terminal receipts.

The training audit must replay every saved sample identity, source/target
rotation, noise flag and actual corruption count; verify learning rates, partial
update denominators, loss sums, frozen states and all complete checkpoint
bindings. Lower CE or a passed lifecycle cannot satisfy the semantic gate.

## Evaluation and decision

After full training/audit and closure of preceding canonical judgment branches,
generate source-only predictions for the unchanged64 short and64 long DEV cases.
Reuse the original evaluation prompts and cached body features; no evaluation
conditioning change is bundled into this experiment. The reviewed anchors retain
coverage of those original prompt styles and multi-turn inputs. Report any
single-message versus multi-turn limitation explicitly.

Fix two readout candidates in this order: (1) improved standalone decoder using
the completed step6212 decoder checkpoint
`7bc8beea1357c15f344903720f1207086591f0223b2e9ec81690c1af468f789b`,
with frozen encoder/VQ unchanged and no word-source adapters; (2) the original
decoder plus original source-attention readout. Both use the same single set of
source-only predicted codes. These are paired output paths of one trained
predictor, not two independent training replicates. They do not use reference
codes or references during generation.

Grade complete summaries semantically using the qualified Terra judge and the
existing conditional Sol confirmation: faithful, salient, readable, at most128
stored tokens. Paraphrases are allowed; exact target matches are not accuracy.
Keep first valid judgments, original request identities, and the existing
transport/schema retry budget. Exhausted invalid judgments remain unresolved;
never count them as failures or select only valid successes. Report counts,
denominators, uncertainty and validity separately for each cohort and readout.

A DEV candidate requires at least61/64 joint passes in each cohort under the
existing fully valid dual-judge protocol. If both qualify, prefer the improved
standalone decoder, then the original source-bound readout. A qualified candidate
must still pass the existing sealed Final320 protocol before the global goal
is achieved; do not open or tune on Final320 prematurely. If neither qualifies,
record failure or unresolved validity and preserve all outputs before proposing
another experiment. One training seed gives no multi-seed robustness claim.
Speed measurement follows measured task quality, never training-loss improvement.
