# Targeted factual contrasts are better motivated than a generic GAN objective

The current evidence motivates a decoder/source-conditioning repair. It does
not show that a missing adversarial discriminator caused the failure. In our
[completed codec diagnostic](2026-10-25_ncm-codec-roundtrip.md), changing 2019 to
2021 changed 57 discrete codes and the reconstructed latent, yet every tested
codec-only decoder emitted 2019 for both variants. The repaired codec preserved
99.29% of reserved content tokens, while the latest decoder with source attention
preserved 92.65%. These are component preservation measurements, not task accuracy.

NCM's tokenizer objective combines token reconstruction cross-entropy with
commitment and quantization terms. Nested dropout trains reconstruction from
partial hierarchies. Its generator then learns with a frozen tokenizer and
corrupted conditioning codes. The published objective does not specify a GAN
loss. Appendix G documents numeral and named-entity substitutions among the
OWT tokenizer's residual errors, even at 99.99% token reconstruction accuracy.
That is a different tokenizer/domain from our LM1B-derived hybrid.
[NCM, sections 3.3–3.4 and Appendix G](https://arxiv.org/html/2610.14162v1).

VQGAN adds a patch discriminator and perceptual objective to improve visual
quality under image compression. This establishes an image-synthesis use for
adversarial learning; it does not establish preservation of source-specific
dates or quantities in text.
[VQGAN, section 3.1](https://arxiv.org/html/2012.09841v3).

A closer text precedent is CLIFF: reference summaries are positive examples,
automatically generated erroneous summaries are negatives, and contrastive
training improves factuality on XSum and CNN/Daily Mail in its experiments.
FactCC separately uses transformed source sentences to train a factual
consistency verifier. These support testing factual contrasts rather than
assuming that a generic fluent-versus-unfluent or real-versus-fake criterion
will preserve the right fact.
[CLIFF](https://aclanthology.org/2021.emnlp-main.532/),
[FactCC](https://aclanthology.org/2020.emnlp-main.750/).

The following is an inference and proposed experiment, not a proven NCM fix:

1. Retain clean-code reconstruction loss while exposing the generator to noisy
   histories and the decoder to actual generated-code errors. Do not replace
   every clean decoder example with generated codes again.
2. Build controlled date/quantity/name contrasts from TRAIN material only.
   For a codec-only round trip, encode each altered text and require decoding
   of that same altered text. This teaches discrimination without asserting
   that the altered statement is true of the original source.
3. For source-conditioned factual training, keep the original supported
   summary positive and use only verified contradictory alternatives as
   negatives. A changed number alone does not prove contradiction: the source
   can mention multiple values, and different phrasings can remain valid.
   Never pair an altered factual summary with an unchanged source as a positive.
4. Compare the same initialization, examples, update budget and clean/noisy
   mixture with and without a targeted contrastive term. Measure clean
   reconstruction retention, reserved numerical fidelity, and the unchanged
   complete-summary semantic endpoint. Reconstruction is diagnostic; the
   95% goal remains source-only task accuracy.

The 64 reserved codec/teacher summaries, all 128 DEV sources and the sealed
320 final sources remain excluded from TRAIN. The observed 2019/2021 example
must not be added to training. Broader reviewed teacher firsts have been
consolidated under the existing protected-source guards. No GAN, contrastive
training run, or new judge call was executed for this literature check.
