# Broad learned codec reconstruction on excluded summary texts

Date: 2026-10-23. Hypothesis `ncm-broad-learned-codec-reconstruction-v1`.
Classification **codec_reconstruction_fidelity_failed**. Aggregate `data/experiments/ncm_paper_regime/20261023_broad_learned_codec_results_v1/campaign_results.json`,
SHA256 `4a584c83892fe9e5161a523431c27424854dd75e718010f036e35c339c5d13c7`.

| Codec | Correct non-PAD tokens | Exact texts | Correct first EOS | Changed number sequences | Active invalid IDs |
| --- | --- | --- | --- | --- | --- |
| original | 20849/21905 (95.179%) | 23/252 | 251/252 | 53 | 0 |
| trained | 14713/21905 (67.167%) | 0/252 | 238/252 | 105 | 0 |

Both codecs receive their own full correct target-derived codes. These are
oracle reconstruction results, not source-to-summary task accuracy. The
representation gate **failed**: at least 99.5% token
recovery, at least 99% first-EOS agreement, zero active invalid IDs, and no
increase in changed-number cases. Exact text matching is descriptive. Numeric
sequence differences are a diagnostic and are not independently judged factual
errors. Passing would require a separate semantic audit before any generator
quality claim. The original parent is absent from this component study.

The complete 252 targets were excluded from fitting by source session, group,
exact input hash and exact target text. They remain training labels for a later
conditional summarizer, not a final application test. Every target fits four
48-position GPT-2 blocks without clipping; the longest needs 136 tokens with
EOS. Raw predicted IDs are concatenated before the first global EOS and text
decoding. There is no per-block EOS insertion, output repair or choice between
arms. All raw codes, word IDs, stopping decisions and counts replayed exactly.

The new codec copy started from the released 60.63M-parameter XSum codec. It
completed exactly 2,048 FP32 updates on GPU2, microbatch four and accumulation
eight, using 31,109 windows from 7,088 unpaired texts in 1,024 training groups.
The corpus contains 1,303,913 valid token positions including genuine EOS.
Saved atomic summaries supplied unpaired text, not complete-passage labels.
Encoder, decoder, quantization modules and EMA books used the released forward
with reconstruction CE plus its returned VQ loss and .5 fine-level dropout.
AdamW excludes EMA books. All original parent weights, including its MLPs,
remain outside this training process. The released checkpoint file is intact.

The actual two-update gradient/memory pilot passed before the fixed run. The
full run restored fresh released weights, not pilot weights, and evaluated
only its final checkpoint. Model, optimizer, CPU/CUDA/sampler RNG, nonpersistent
dropout step state and probe outputs reloaded exactly. Peak allocated memory
was 1,060,282,880 bytes. Original and final evaluation
states remained unchanged. Frozen code/config/input identities and observed
terminal-success receipts are retained in the aggregate's artifact map.

This is a transfer experiment around released NCM modules. The official release
does not include its trainer; the returned all-level VQ loss is not asserted to
reproduce the paper's sampled-cutoff objective. Four-block text representation
also differs from the original single 48-token target. No conditional concept
generator was trained here. The clean native teacher's separate 56/64 (87.5%)
development result remains below the goal. The original and longer-source final
task cohorts remain sealed, and no latency, compression or tokens/s benefit is
established by this component experiment.
