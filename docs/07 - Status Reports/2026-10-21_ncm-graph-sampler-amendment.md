# Capture-safe top1 categorical sampling

The version2 systems pilot passed the scalar-allocation point, then stopped
inside PyTorch2.4's `torch.multinomial` during CUDA graph capture. No graph
quality or speed outcome was measured. Preserve both failed pilot roots,
their source snapshots, errors and logs. The parent model study remains intact.

Version3 preserves the fixed temperature1/top_k1/top_p0 prediction distribution
with a GPU-only inverse-CDF implementation. After top_k1 filtering, the original
sampler has uniform probability over every tied maximum and zero elsewhere.
Compute that maximum mask, its cumulative count, one uniform draw per row and
the first cumulative count exceeding the selected ordinal. With a unique
maximum this is exactly the same point-mass prediction. With ties it has the
same categorical law, but does not promise an identical RNG stream. Do not
replace ties with a deterministic first-index choice. Carry a finite-logit flag
through all sampling levels. CPU tests must cover unique maxima, exclusion of
nonmaximal logits and uniform tie behavior.

The pilot restores the original method AND native sampler for eager controls.
The original all48-native-ID parity gate on the actual cases is unchanged,
followed by all300 saved eager/graph output comparisons. No weights, masks,
level order, decoding, data, quality threshold or speed threshold changes.
Use new systems root `20261021_pilot_v3`. This is correction of unsupported
capture operators before numerical/timing outcomes, not a threshold retry.
