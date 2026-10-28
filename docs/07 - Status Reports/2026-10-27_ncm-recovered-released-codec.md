# Matched released-codec check on recovered labels

The repaired frozen codec reconstructed 101/128 fixed recovered labels exactly,
with 99.21% content-token agreement. Raw replay found material replacements,
including a year changing from 1896 to 1912 and a unit disappearing despite
unchanged digits. This motivates a matched comparison with the original released
codec; it does not establish a source-only model accuracy ceiling.

Use the same 128 cases, original native frames, original ordering, FP32 raw argmax
and renderer, without changing targets or selecting a subset of failures. Load
the original released codec, expected tensor hash
`9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32`.
Require every encoded hierarchy to equal the stored repaired-codec hierarchy;
that tests whether this contrast isolates reconstruction rather than changing
the target representation. Require repeat encodes and first/longest repeat
decodes to match. Report both full-denominator metrics and paired fixes/regressions.

Use a fresh root `20261027_recovered_released_codec_v1`, locked Pixi on physical
GPU2 RTX2060SUPER, deterministic FP32 with TF32 off, batch one and a 3.5 GB own
allocation cap. Bind the completed repaired-codec result and its actual receipt,
all cases and outputs, this plan and the released weights. No training worker,
checkpoint selection, semantic judging or source partition is changed. This
is an exploratory target-aware diagnostic, not a trained quality or speed claim.
