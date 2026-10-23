# Tokenizer lifecycle parity correction

Date: 2026-10-22. No tokenizer optimizer update has run yet. The first genuine
codec pilot stopped at its pre-update full-cutoff parity check; preserve
`20261022_codec_pilot_v2` and its original source copies. The conventional
straight-through expression `x + (quantized - x).detach()` can lose low bits of
the quantized value through cancellation. In a residual hierarchy those low-bit
changes can affect subsequent nearest-code assignments.

The new `ncm_codec_training_v2.py` instead preserves the exact forward lookup
using `quantized + (x - x.detach())`, with the same intended input gradient.
Apply the analogous value-preserving expression to the final straight-through
latent. No objective, data, learning rate or EMA setting changes. Require actual
released-model full-cutoff parity before training, plus the existing cutoff,
gradient and EMA controls. New entry point `train_ncm_paper_codec_v2.py`, fresh
pilot root `20261022_codec_pilot_v3`, full root `20261022_codec_training_v2`.
This is a systems correction; it cannot turn an old quality failure into a pass.
