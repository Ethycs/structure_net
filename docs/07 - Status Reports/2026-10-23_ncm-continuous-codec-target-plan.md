# Continuous latent target fidelity with the original neural codec decoder

Date:2026-10-23. Exploratory CPU-only control
`ncm-continuous-codec-target-fidelity-v1`. The full frozen-body NCM goal remains
at least95% complete-summary quality on source-disjoint evaluation. This oracle
TRAIN reconstruction control cannot satisfy it or admit sealed sources.

The registered clean-word repair reaches2709/3197=84.74% correct-history TRAIN
word-token agreement. The previous word model's complete oracle TRAIN semantic
condition passes1/32 under32valid Terra judgments; the new model's semantic
conditions are still running. Frozen original codec reconstruction has also
been imperfect on conversation text. Previous unquantized dictionary inversion
with target-derived normalization statistics was exact, but that does not
measure the original neural decoder on unquantized encoder latents.

Before proposing continuous latent distillation, ask whether the original
frozen neural decoder preserves these targets better when quantization is
removed. Use the same original16/additional16 TRAIN cases from the completed
32-case diagnostic, unchanged whole192-position GPT2 word frames and saved800
level-major target codes. No new source, target, training, source-only generation, judge
request or output repair is introduced. Both arms know target-derived latents;
neither is a source-only model or a test of generalization.

Load the exact original released codec, state hash
50fc5680ce0fb3c82764d865243a2e509d8f16dd8c051113b16080b5d763d680.
Freeze all parameters and use eval mode. Fold each target into four independent
48-position blocks, preserving EOS50256 and right-PAD50257. Compare:

1. saved_quantized: reconstruct from the existing saved full target code arrays
   using the original learned quantizer/resampling/normalization operators.
2. continuous_encoder: call the original encode_z on the complete target words,
   retaining its trained positional embeddings and group normalization, with
   its eval-mode dropout disabled, then pass those latents to the SAME decoder.

Both arms use the original neural decoder, FP32 CPU computation, no autocast,
two intra-op threads, identical four-block folding, and the existing word-level
readout policy: slice vocabulary to IDs0..50256 and take argmax. PAD/unused rows
are excluded from both readouts, so this is not an exact reproduction of earlier
full-vocabulary codec numbers. Keep all192 raw output positions and decode up
to the first global EOS with the original GPT2 tokenizer. Include target EOS
and exclude target PAD from fitting counts. Verify each complete target decodes
to its stored text. Never use target normalization statistics to invert tokens.
Constructor seed42142 is fixed; every learned codec weight is restored from
the same released checkpoint before any measurement.

First run a two-case CPU lifecycle pilot on the first and longest-source TRAIN
probes, then repeat after destroying and freshly loading the original codec.
Require finite logits/latents, original codec state byte identity, no gradients,
exact latent digest/raw IDs/text/count reload parity and strict JSON. Select
neither model nor decoder by outcomes. A separate full32-case worker must retain
the pilot outputs exactly and leave the original codec unchanged. CPU-only is
intentional; do not initialize CUDA or compete with ongoing GPU training.

The component gate for continuous_encoder is unchanged199/200 active-word
agreement in EACH original16, additional16 and all32 group, inclusive integer
counts. Report both arms, exact target texts, first-EOS behavior, complete raw
outputs and CE; also report the paired difference. Classify
continuous_codec_target_fidelity_passed or continuous_codec_target_fidelity_failed.
This gate assesses a candidate dense supervision target and never admits final
evaluation, semantic quality, a new training stage or a speed claim. In
particular, a valid failure does not establish an information-theoretic limit.

Use fresh roots20261023_continuous_codec_target_{workflow,pilot,measure,results}_v1
under data/experiments/ncm_paper_regime. Implement and test block folding,
target-free readout, EOS/PAD counts, fixed group composition, inclusive gate,
state immutability and raw output replay; commit before the CPU pilot. Preserve
all earlier failures and all running source/configuration bytes.

The canonical audit replays the exact32 source/target/code identities, complete
raw-ID decoding/counts, group aggregation and gate; verifies pilot retention,
actual zero child exits, source/data/codec hashes and original frozen-state
identity. Hash immutable numerical evidence and first two workflow receipts,
excluding mutable parent progress and future finalization/registration receipts.
Write a measured control report and conservative exact registry readback.
Task_accuracy stays null, oracle_target_information is true, and
heldout_accuracy_measured/NCM_success_established/final_test_opened/speed_tested
remain false. Original320 sealed final sources stay unopened.
