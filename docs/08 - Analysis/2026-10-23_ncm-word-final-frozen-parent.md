# Word-final NCM attached to a frozen Llama body

Date:2026-10-23. Hypothesis `ncm-word-final-frozen-parent-v1`.
Classification **word_final_NCM_development_quality_failed**. Aggregate `data/experiments/ncm_paper_regime/20261023_word_final_results_v1/campaign_results.json`,
SHA256 `6e946e9fb41da2c6591c766cbe1a9f79a5da2dbfecfc81ff4368c002dec9ce24`.

| Development cohort | Judge role | Complete-summary passes | Valid | Faithful | Salient | Readable | Over128tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short_development | secondary | 0/64 | 64/64 | 4 | 1 | 0 | 0 |
| long_diagnostic | secondary | 0/64 | 64/64 | 2 | 0 | 1 | 0 |

These are source-only complete-summary development results, requiring supported
material claims, useful salient content, readability, nonempty text and at most
128stored cl100k tokens. Each cohort must independently score at least61/64
with64valid Terra judgments before the same texts reach Sonnet; Sonnet requires
the same gate. The two development gates failed.
Per-cohort Wilson intervals remain in the aggregate. The reused short cohort
is a development diagnostic, not a fresh final test. Original256and longer64
final cohorts remain closed. No final95%quality, compression or speedup claim.

The model keeps eight learned concept levels and predicts final GPT-2 words
directly through a trainable parallel projection/decoder. It is an NCM
adaptation, not a replication of the published final VQ mechanism. Full frozen
28-layer Llama3.2-3B source features form a zero-initialized residual over the
released raw GPT-2 source embedding path. Every original parent weight and
MLP remains frozen; its source pass still computes all original blocks.

Training used252complete vetted summaries and exactly512updates, microbatch1
and accumulation4. Released generator weights were copied, root queries
repeated for four48-position blocks, and geometry rebuilt. New word projection
and decoder copy released weights;110,298,376new-module parameters may train.
The original codec remains exact. FP32master/AdamW with BF16autocast and FP32
losses/distances used clean wordCE+.1noisywordCE+.1meancoarseCE. Noise replaces
5%of earlier concepts using eight nearest non-self codebook neighbors. This is
supervised corruption, not a replication of CLLM fixed-point training. Only
the final512checkpoint was selected. All2048sample identities, frames, noise
hashes and learning rates replayed. No adaptive budget extension occurred.

The initial CUDA pilot stopped before updates because released RoPE frequencies
were onCPU while positions were onCUDA. Its exit1and traceback remain intact.
A separate device-only recovery calculates reference RoPE onCPU and transfers
it, passing exact CPU/CUDA geometry checks and the actual longest-frame
two-update gradient/memory/checkpoint/optimizer/RNG/inference reload pilot.
Full training started fresh after its observed exit0. Peak training allocation
was5,142,067,200bytes; this is not a latency comparison.

All64short and64longer sources are disjoint from252training cases by
case/session/group/hash. Source metadata for inference contains no target
summaries or concepts. Every source token and all192raw word positions were
retained, with eight predicted coarse levels and one word level. Text is
decoded once through the first global EOS; no oracle concepts, previous-word
labels, autoregressive fallback, per-block EOS repair, reroll or best-of
selection. The first two outputs were retained exactly after deterministic
checkpoint-reload qualification. Raw IDs, full texts and all first valid
judgments replayed. All802prior judgments, including failures, stayed exact.
Public full-source payload proofs were fixed before external judging. Detailed
failures are retained in `data/experiments/ncm_paper_regime/20261023_word_final_results_v1/case_failures.json`. All earlier codec and
native-hybrid failures remain failures. Quality must precede speed claims.

The separately planned16-TRAIN-case mechanism diagnostic recovered
1567/1635
non-PAD target words with correct earlier concepts, versus
21/1635 with
source-only generated concepts. Generation agreed with the full forward at its
own concepts on3067/3072raw word IDs.
The first concept level recovered only6/
176labels in blocks containing target text.
These are TRAIN-only literal component measurements, not task accuracy or a
new codec qualification. They support investigating source-conditioned concept
learning and the mismatch between training and generated histories. Alternate
wording can be valid; this diagnostic alone cannot assign semantic failure.
