# The repaired codec narrowly misses the public summary reconstruction gate

The repaired codec reconstructed **243/256 candidate public training summaries
exactly (94.92%)**, one below the fixed 244/256 requirement. Content-token
agreement was **6,620/6,633 (99.80%)**, and digit sequences were preserved in
**73/76** digit-bearing references. The released codec reconstructed 248/256
exactly and passed this component gate. Neither result measures source-only
summarization accuracy. The latest completed application experiment remains
at **0/128 semantic passes** for each of its two models.

Exploratory study `ncm-public-xsum-codec-v1` is complete and classified
`repaired_public_TRAIN_codec_reconstruction_gate_failed`. Its
[prospective plan](../07%20-%20Status%20Reports/2026-10-26_ncm-public-xsum-codec-plan.md)
fixed the sample, comparison and gate before reconstruction outputs. The
replayed aggregate is
`data/experiments/ncm_paper_regime/20261026_public_xsum_codec_results_v2/campaign_results.json`.
The gate required at least 244/256 exact native-rendered summaries, at least
99% content-token agreement, and at least 95% digit-sequence preservation.
The repaired codec passed the latter two requirements and failed the first.

| Fixed cohort | Released exact | Repaired exact | Released content agreement | Repaired content agreement |
| --- | ---: | ---: | ---: | ---: |
| Representative public TRAIN, 256 | 248/256 (96.88%) | 243/256 (94.92%) | 6,625/6,633 (99.88%) | 6,620/6,633 (99.80%) |
| Long-summary stress, 32 | 16/32 | 24/32 | 2,913/2,934 (99.28%) | 2,924/2,934 (99.66%) |
| Numerical stress, 32 | 30/32 | 30/32 | 896/898 (99.78%) | 896/898 (99.78%) |
| Existing TRAIN controls, 16 | 7/16 | 15/16 | 625/640 (97.66%) | 638/640 (99.69%) |

There were no empty or invalid outputs. Representative digit-sequence
preservation was 75/76 for the released codec and 73/76 for the repaired codec;
both recovered all 32 numerical-stress digit sequences. On the longest-summary
stress group, those figures were 9/12 and 10/12. Stress results and TRAIN
controls cannot rescue the representative gate.

Both arms received the same correct codes encoded from each reference. No source
features, source-attention adapters, generator or Llama body entered prediction.
The encoder and quantizer produced identical codes in both arms. The repaired
codec was extracted from the completed BROAD checkpoint, whose codec is also
unchanged in the latest matched training arms. This test isolates the codec's
reconstruction behavior; it does not diagnose how to predict the right codes
from a source.

The public population is the 163,051 complete XSum TRAIN pairs passing the
current token, exact-duplicate and protected-source screens. The full pinned
204,045-row shard was exported without string changes, independently replayed,
and screened in locked Pixi environments. Twenty-eight empty articles were
preserved and excluded. These references have not been verified as factual gold,
and comprehensive source disjointness has not been established. No public
validation/test payload or sealed application final text was opened.

The representative sample contains the 256 lowest prospectively defined article
ID hashes. The separate stress groups contain the 32 longest remaining targets
and 32 digit-bearing targets selected by the same hash rule after exclusions.
No reconstruction outcome determined selection. Public targets are mostly short:
the eligible population's native median is 26 tokens. The long stress group
averages 91.69 native content tokens. Its improved reconstruction under the
repaired codec therefore does not establish a population-wide improvement.

In a post-hoc paired comparison of the representative sample, 239 summaries
were exact under both codecs, nine only under the released codec, four only
under the repaired codec, and four under neither. On long summaries, ten were
exact only under the repaired codec and two only under the released codec.
This is evidence of a tradeoff across these tested inputs, not proof of a
universal preference for either checkpoint.

The repaired codec's 13 representative errors include quantity and name changes,
as well as a quotation-mark substitution. One reference's “dozens” became
“thousands”; that error does not affect a digit-only metric. Exact reconstruction
also penalizes harmless punctuation changes. Neither token agreement nor exact
matching substitutes for the unchanged complete-summary semantic endpoint.
All public mismatches and paired counts are retained in
[the descriptive error review](../../data/experiments/ncm_paper_regime/20261026_public_xsum_codec_error_review_v1/errors.json).
No new semantic judgments were assigned in that review.

The first pilot launch failed before encoder/decoder inference because the
standalone extractor used `codec.` instead of the checkpoint's
`concepts.codec.` prefix. A CPU diagnostic verified all 136 keys and the original
repaired-codec tensor hash. The
[execution amendment](../07%20-%20Status%20Reports/2026-10-26_ncm-public-xsum-codec-recovery.md)
preserves that failure and uses fresh execution roots with the unchanged sample,
checkpoints and metrics. The corrected pilot repeated its outputs exactly and
matched every saved raw code and word ID on all 16 prior TRAIN controls per arm.
Matching a control means reproducing its prior output, not perfect reconstruction
of that control.

All 672 full-run outputs passed the separate CPU replay of selection, native
frames, rendering, metric arithmetic, paired identities, pilot parity, saved
controls and checkpoint tensor hashes. That replay did not independently repeat
all GPU forwards. Twenty-four focused tests cover the measurement, recovery and
registry boundaries. Model execution used physical GPU1 RTX2060, FP32, TF32 off,
one case per forward and locked Pixi `ncm`; peak allocation was 758,468,096 bytes.
All codec tensors stayed unchanged. No training, new judge call or speed test ran.

The failed reconstruction gate remains failed. The small readout errors and the
much larger source-to-code failure need separate treatment. Broader conditional
training, mixed-domain reconstruction anchors and targeted factual contrasts
remain proposed interventions; this result does not demonstrate that a GAN is
needed or that a larger dataset will solve the task. A subsequent training study
must specify its data admission, codec handling, budget and application evaluation.

Reproduction entry points, from the repository root:
`experiments.structure_net.ncm_public_xsum_codec prepare`, followed by
`experiments.structure_net.ncm_public_xsum_codec_recovery {pilot,eval}` and
`experiments.neural_architecture_lab.finalize_ncm_public_xsum_codec`.
Completed roots are append-only. Source/data/model fingerprints and actual
terminal receipts are retained with each stage. The original 95% whole-summary
task goal remains unmet, and the sealed final320 remains closed.
