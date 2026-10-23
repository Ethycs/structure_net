# Isolate codec transfer from summary prediction

Date: 2026-10-22. EXPLORATORY `ncm-unseen-codec-control-v1`.
The current final NCM attachment produces0/64 acceptable unseen development
summaries and visibly broken text. Its strong training reconstruction does not
identify the failing component. First ask whether known correct codes can
reconstruct genuinely unseen target text without generating summary codes.

Use the20 completed public teacher-pilot labels at
`20261022_expanded_teacher_pilot_v2/train.json` as novel texts. These labels are
format-qualified, not quality-audited; their adequacy as summaries is irrelevant
to testing reconstruction of their exact text. They are not final-test examples.
Require all20 to be disjoint by ID, source/session, group and source hash from
all1,999 current training examples. Verify public source spans, target hashes,
and full untruncated GPT-2+EOS+PAD target IDs. Compare with20 old training labels
chosen by SHA256 ordering of `ncm-unseen-codec-control-v1:` plus ID. Do not select
examples by observed model behavior or remove difficult texts.

Evaluate both the original released XSum codec and the frozen polished current
codec. Each codec encodes each target into its own full correct code hierarchy,
then decodes its cumulative full-resolution latent with raw argmax. No
generator, body, source prediction, training, penalties or noise. These whole
codecs may differ in more than their decoder, so a difference cannot by itself
be attributed to decoder weights. Quantization and decoding are both FP32 on
physicalGPU1. Use batch4 and a real two-example-per-cohort lifecycle pilot before
the20-per-cohort diagnostic. Record actual peak allocation with the existing
resident services; stop on OOM, nonfinite logits, hash drift or reload failure.

Primary diagnostic: non-PAD target-token reconstruction, including EOS,
separately for each codec and cohort. Also report non-PAD correct/total, exact
decoded texts, every target/reconstruction, codes, frozen model hashes and
exact save/reload. Predeclare that the adapted codec's novel-text reconstruction
must be at least98% before attributing the attachment's novel-text failure to
the source-to-code predictor alone. Passing that diagnostic is not task accuracy
and cannot admit the95% summary goal. If it fails, inspect codec transfer before
scaled student training. Preserve both codecs' results regardless of thresholds.

New roots: `20261022_unseen_codec_{pilot,measure}_v1`. Full run requires passed
matching source/input-hash pilot. Freeze source, plan, vendor, tokenizer, label
and checkpoint hashes before execution. Save strict results and a completed
aggregate/report, then conservatively register/read back. Do not open the
sealed final test or modify any original weights or live producer sources.

Reproduction: `python -m experiments.structure_net.ncm_unseen_codec_control
{pilot|measure}` in the existing moba environment, offline HF, empty
LD_LIBRARY_PATH, OMP_NUM_THREADS=2, GPU1 selected before Python startup.
