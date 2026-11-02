# Vocabulary-balanced codec reconstruction

Date: 2026-10-22. `ncm-vocabulary-balanced-codec-v1` is complete, classified
**codec_transfer_gate_failed**. The95% full-summary goal remains unmet. No new summary
accuracy or speed result is measured; the existing full attachment remains
at its earlier0/64 Terra development outcome.

| Checkpoint | Cohort | Correct non-PAD target tokens | Exact target texts |
| --- | --- | --- | --- |
|baseline|raw_validation|6240/6477=96.3409%|121/256|
|baseline|novel|694/717=96.7922%|8/20|
|baseline|seen|647/672=96.2798%|4/20|
|final|raw_validation|6334/6477=97.7922%|161/256|
|final|novel|701/717=97.7685%|9/20|
|final|seen|669/672=99.5536%|17/20|

The codec misses its reconstruction endpoint despite complete content-token supervision. Per the prospective plan, further frozen-XSum-decoder data sweeps stop. The next study must change the representation or final text readout rather than extend this training recipe.

The unchanged released encoder and codebooks supply correct target codes.
Only the decoder trains, starting from the original release, for2,048 fixed
updates. The augmented raw bank contains57,258 public natural text windows and
114,516 synthetic native-token sequences, interleaved1:2. Each update samples48
entries from that bank plus16 old summary anchors: expected32 synthetic,
16 natural and16 summary targets. Synthetic sequences are random token-space
exercises, not natural summaries; semantic evaluation never uses them.

Their content follows balanced shuffled permutations of all50,256 nonreserved
GPT-2 IDs, length8..47, followed by EOS and PAD to48. The exact main sampler
schedule is replayed before training: all50,256 content IDs receive positive
target exposure, with minimum13 occurrences.
Seeds5042/6042 separate lengths and content; sampling2042 and dropout52 remain
the same as the natural-only study. No seed/LR/outcome retry or alternate
checkpoint selection occurred. Final-step2,048 is the fixed endpoint.

Raw256 validation and novel20 targets/source groups stay excluded from natural
training and are byte-identical to the earlier broad study. The gate requires
99.5% non-PAD tokens and90% exact texts separately on both cohorts. These are
known development diagnostics and target-aware reconstruction, not a new blind
task test. Complete vocabulary exposure is a validity measurement, not proof
of faithful text recovery or semantic transfer. The20 seen cases are training
diagnostics; high seen reconstruction cannot rescue a failed novel endpoint.

The wrapper changes only dataset/root/plan configuration in its process and
preserves/restores the shared training engine. Both source versions are frozen;
the original engine file is unchanged. FP32 quantization, BF16 decoder, batch64,
GPU0; real-batch lifecycle, nonzero decoder gradients, all frozen state, novel
code identities and exact checkpoint/prediction reload passed. Own peak GPU
allocation was2,770,218,496bytes. Original Llama and all
previous generators/codecs remain untouched. No new student training is admitted
by this report. The final task test remains sealed.

Aggregate: `data/experiments/ncm_paper_regime/20261022_balanced_codec_results_v1/campaign_results.json`; SHA256 `53b40956367e52db119b4f76f7ee339ffa632295f8896abd2f9cd9931a71dbfe`. Every fixed checkpoint,
full target/output, synthetic ID bank, actual exposure counts, cached codes,
producer/source seal and lifecycle pilot remain in append-only roots. Registry
confirmed=false and task_accuracy_measured=false. Reproduction follows the
immutable vocabulary-balanced plan and `ncm_vocabulary_balanced_codec` commands.
