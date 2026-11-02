# Broader decoder-only training fails to restore unseen reconstruction

Date: 2026-10-22. `ncm-broad-decoder-transfer-v1` is complete, classified
**codec_transfer_gate_failed**. Original body, generator and previously failed codec
remain untouched. The current summary attachment still has its earlier0/64
Terra result; no new summary accuracy or speedup is established here.

| Checkpoint | Cohort | Correct non-PAD tokens | Exact target texts |
| --- | --- | --- | --- |
|baseline|raw_validation|6240/6477=96.3409%|121/256|
|baseline|novel|694/717=96.7922%|8/20|
|baseline|seen|647/672=96.2798%|4/20|
|final|raw_validation|6250/6477=96.4953%|136/256|
|final|novel|680/717=94.8396%|8/20|
|final|seen|671/672=99.8512%|19/20|

This study starts from the original released codec, freezes its encoder,
codebooks and all nondecoder state, and trains only its decoder for2,048 fixed
updates on57,258 public text windows plus1,999 old summary anchors. Each batch
contains48 raw windows and16 summary anchors. The256 raw validation sources are
group/session-disjoint from codec training, and all20 novel summary texts and
their source groups are excluded from training. Data are exact untruncated
8/16/24/32-word public substrings, selected and split before training.

The joint diagnostic requires99.5% token reconstruction and90% exact texts
separately on raw256 and novel20. Both fail. Training-target reconstruction
improves while novel-summary reconstruction degrades. The fixed final checkpoint
is reported even though an earlier checkpoint had better raw validation. That
earlier checkpoint also missed the declared gates. More decoder training and
broader text alone are not a demonstrated cure; no generator run is admitted.

The matched baseline in this study is BF16 decoding at batch16, with FP32
quantization. It differs by one novel token from the earlier FP32/batch4 control;
the table uses this study's own baseline for the before/after comparison.
Encoder/codebook hashes and novel code assignments remain exact, no frozen
parameter receives gradients, cache and checkpoint/prediction reload are exact,
all2,048 optimizer steps are present and own peak allocation is
2,770,218,496bytes. This is one exploratory training seed,
not a statistical guarantee or a fresh blind task test. The20 novel targets were
already known development diagnostics motivating the remedy.

## Descriptive lexical coverage of the earlier codec errors

In the original adapted-codec novel control,44/53 wrong content-token positions
involve token IDs absent from the1,999 old training summaries. Among69 such
positions44 are wrong, versus7/134 positions with old token frequency1–5 and
2/494 positions with frequency above5. The original released codec makes
11/69,4/134 and7/494 errors in the same strata. These are post-hoc associations,
not proof that vocabulary coverage alone causes the failure. EOS/PAD are
excluded from this descriptive breakdown. The broader training bank expands
distinct content-token coverage from13,151 to33,635, but8 of the adapted model's
original error positions still involve IDs absent from that broader bank.
Natural windows alone do not cover the full GPT-2 vocabulary.

The next independent information control checks whether correct quantized
latents can be inverted using the encoder's original token dictionary. It
separates target-blind estimated normalization from target-aware oracle
normalization, with exact unquantized inversion as an implementation control.
No result from that new study is included here. Neither codec oracle metrics
nor known-example residual repair can rescue the failed summary task.

Aggregate: `data/experiments/ncm_paper_regime/20261022_broad_decoder_results_v1/campaign_results.json`; SHA256 `a0f460c2dd72bf53982da6424bf54d9cc0be06b9f9a2c0bc11a883532dca4b85`. Raw outputs, all fixed
checkpoints, cached codes, source splits, exclusions, lifecycle pilot and source
snapshots remain in append-only roots. Registry confirmed=false,
task_accuracy_measured=false, final task test sealed. Reproduction follows the
immutable broad-decoder protocol and `ncm_broad_decoder_transfer` commands.
