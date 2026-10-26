# Four-example training lifecycle on the actual public input extremes

This is a systems gate for subsequent larger-data training, not a quality
experiment. Keep the BROAD initializer, frozen codec and word-source path,
FP32 weights/features, FP16 training, TF32 off, CPU FP32 AdamW and physical
GPU1 RTX2060. Do not launch while the full codec-target producer occupies
GPU1; complete that preparation first. GPU0 feature extraction may continue.

Use the passed 16-case feature-store pilot. Select four distinct cases: the
shortest source, the unused source nearest 768 positions, the longest native
prompt and the longest source, with ID tie breaks and unused-case selection.
These include the measured 4,124-source-position and 2,275-native-prompt
extremes. Load their original admitted reference frames from the catalog,
and encode labels once per example using the unchanged FP32 codec.

First compare a four-example FP32 forward/backward against four separately
accumulated examples, for both clean and corrupted histories. Use the measured
per-example frozen VQ conditioning preparation, identical history RNG draws,
equal example/level weighting and full source masks. Require all trainable
gradient presences equal, relative gradient L2 error <=5e-5, maximum absolute
gradient error <=1e-4, logit allclose rtol/atol1e-4, and every argmax equal.
Retain the original batch2 threshold; no tolerances are widened for batch4.
Stop if either comparison fails. Preserve failures and select no fallback
batch size within this run.

Then perform two FP16 optimizer updates, each containing eight examples in
two microbatches of four. Each microbatch samples all four cases in a seeded
permutation; alternating cases use clean and corrupted conditioning. AdamW:
lr1e-4, betas0.9/0.95, eps1e-8, weight decay0.01, clip norm1. GradScaler starts
at128 with growth interval2000. All 105,323,136 previously selected generator
and concept-source parameters train. Codec, word-memory and word-adapters
stay frozen. No skipped optimizer update or nonfinite value is allowed.

Save full model/CPU optimizer/scaler/sampler/noise/CPU and CUDA RNG state after
the first update. Save the second-update state and four source-only outputs.
Reload the first checkpoint exactly, restore every RNG, and repeat update2.
Require exact second-update model/optimizer/scaler/RNG equality and identical
four source-only outputs. Check all protected tensors throughout, and retain
both checkpoints and per-update diagnostics. Peak allocated GPU memory must
stay <=4.9 GB during parity, training, evaluation and reload. An OOM or bound
failure leaves batch4 unqualified; it is not a quality result.

Evidence root: `20261026_public_batch4_lifecycle_v1`. No held-out sources,
new judge requests, full-data update budget or application speed claim occur.
The shakedown checkpoints are not candidates for the 95% task endpoint.
