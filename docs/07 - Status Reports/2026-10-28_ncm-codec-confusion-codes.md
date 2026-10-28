# Do the remaining reconstruction confusions share identical codes?

The fixed broad-decoder reconstruction test improved from101/128 to117/128
exact native strings, with sixteen fixes and no regressions. Eleven errors
remain, including substantive word and name substitutions. Test every one of
these confusions, retaining all117 exact cases as controls.

Use the unchanged encoder/codebooks. Re-encode the reference native frame and
the trained decoder's output on the same active positions. Replace only padding
positions with their reference PAD value; require both native CLS boundaries
and all active vocabulary IDs to remain valid. Compare all641 code IDs and
report changes per native level. Require reference codes to equal the saved
ones, and the117 identical active frames to re-encode identically.

An identical full code vector for two distinct active frames establishes a
collision for that particular pair. Different code vectors establish that this
pair remains distinguishable at the code level; they do not prove global
injectivity or that the decoder can already reconstruct it. This is an encoding
diagnostic, not training, model-summary accuracy, or evidence of a speedup.
Use GPU2, locked Pixi and a new append-only artifact root. No final-case access.
