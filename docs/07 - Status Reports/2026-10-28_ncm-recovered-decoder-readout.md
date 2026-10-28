# Compatibility of the improved decoder with source attention

Exploratory hypothesis `ncm-recovered-decoder-readout-v1`: the new decoder trained
on 99,367 recovered text frames preserves or improves oracle reconstruction when
combined with the application's existing frozen word-source attention. Known
before this test: standalone reconstruction improved from 101/128 to 117/128 on
the recovered diagnostic sample. A different earlier decoder adaptation harmed
the word-source path. Neither result establishes source-only summary accuracy.

Use the unchanged 64 reserved component summaries and 16 TRAIN controls from
`20261025_codec_roundtrip_prepare_v1`. Verify that no reserved native frame occurs
in the new decoder's fitting corpus. Compare four configurations: original
repaired decoder alone, original with source attention, new decoder alone, and
new with the same source attention. The concept generator is bypassed: all arms
receive each reference's complete singleton-encoded 641-code hierarchy. Sources
are streamed from the same existing FP32 cache with a 256 MiB bound.

Strictly load the original application checkpoint; require its codec,
word-memory, and word-adapter hashes to equal those protected in the completed
task-supervision run. Change only decoder tensors. Require the new codec hash
to match the completed decoder training/evaluation, the nondecoder model state
to remain identical, and every reference code vector to remain unchanged.
Parent output agreement with the earlier GPU1 diagnostic is reported separately:
this matched comparison uses physical GPU2, so historical byte identity is a
control observation rather than a fabricated same-device replication.

Use FP32, TF32 off, deterministic inference, batch one, and a 3.5 GB own GPU
allocation cap. Run the same two first/longest reserved pilot cases twice per arm;
then run all 80 cases, reproducing the pilot outputs exactly. No target truncation,
new judgments, model fitting, changed evaluation reservation, or speed test.
Freeze all producer/input identities and preserve failures in new roots
`20261028_recovered_decoder_readout_{pilot,eval}_v1`.

Report whole native-string reconstruction, content-token agreement, digit-sequence
preservation, and all paired fixes/regressions separately for reserved and TRAIN
cases. A descriptive integration check requires the new source-bound path to
be no worse than the old source-bound path on all three reserved metrics. Also
report the effect of source attention within each decoder; this can reveal
harmful interaction even if the old baseline is exceeded. The check does not
establish semantic equivalence, calibrated accuracy, or suitability for replacing
the decoder in the already-running matched architecture experiment. Independently
replay raw IDs, native renderings and all aggregate counts before using the result
to choose a future model. Final320 stays sealed.
