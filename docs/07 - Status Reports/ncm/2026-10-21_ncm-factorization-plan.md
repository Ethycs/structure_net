# Factorize the distilled NCM generator

Exploratory, UNDERPOWERED hypothesis `llama32-3b-ncm-factorization-v1`.
Requested after the completed Llama-to-NCM study. The separate replica study
is running; do not overlap its GPU timing workers with this study. Parent
NCM checkpoints and 100/100 synthetic source review remain unchanged.

Adapt the existing Llama activation-scaled randomized SVD implementation to
NCM: all24 generator MLP c_fc/c_proj matrices become directly executed rank256
factors (one-third of hidden width768, analogous to1024/3072 in Llama).
Preserve original biases, all attention, embedding, norms, heads, learned
queries, and the entire codec. No dense residual or inference-time noise.
Use FP32 decomposition/master factors, FP16 deployment factors, BF16 runtime
autocast. Approximate factorization is explicitly not lossless.

Reuse original generator SHA ff7133b2de50bc1f66ef1e490fec4c723e0fc5a5c1957107209e923b198c3e3c
and codec SHA b87f7288edfc0eb1f5b9122d1139cc2e61aa2331be7d6fe53f0582993086ede2.
Calibration/local training: first64 parent training sources, balanced across
four families; local checkpoint selection: first8 development sources.
Encode actual stored Llama teacher summaries with the unchanged codec and
teacher-force all200 hierarchy positions. Cache dense original NCM inputs and
outputs of six independent two-block neighborhoods and input-channel RMS at
every MLP. Never include held-out or development sources in calibration.
This is a teacher-forced local objective; free-running generation must test
accumulated error independently.

Compare pure SVD initialization and local noise-repaired factors. Repair:
seed51021, independent decomposition/data/noise streams, one1% factor-RMS
Gaussian perturbation,5% channel-RMS Gaussian input neighborhoods with targets
recomputed by the original dense neighborhood on those same perturbed inputs.
Only factors train; biases and teacher stay frozen. AdamW lr2e-5, betas(.9,.999),
weight decay0, clip1,96 updates per two-layer group, batch one source. BF16
autocast, FP32 objective, no gradient scaler required. Select minimum
development boundary NMSE at0/32/64/96 using FP16 deployment factors; no
selection on final sources. Keep all histories and initial/noisy/selected
states. This is the same factorization/local-distillation family as the Llama
pass, not an exact rerun of its later full global-recovery campaign.

Before full execution: CPU factor/bias/gradient and checkpoint-contract tests,
separate CUDA pilot with two real updates, unchanged teacher and codec,
finite nonzero factor gradients, dense cache replay NMSE<=1e-6, exact save/reload
outputs, and single-copy graph/eager agreement. Stop on failed systems controls;
preserve failed roots and amendments. No threshold-miss retries.

Then build/train the fixed candidate. Evaluate both initialized and repaired
models on all64 development sources using the existing graph runtime; compare
source fields, supersession, length, and full text against the source. Promotion
requires64/64 faithful summaries. If both pass, prefer repaired factors; if
only one passes, select it. If neither passes, stop promotion, retain failed
outputs, and run only an8-development-source bounded timing diagnostic.
Do not add ranks/training beyond this plan after a threshold miss.

If promoted, evaluate all100 parent runtime-validation sources, three timed
repeats plus one excluded per-case warmup. Include CPU source IDs through
decoded text, common Llama output-token counts, graph setup, allocated/reserved
memory and actual unique model bytes. Require>=95/100 faithful sources and no
fewer passes than original NCM (therefore100 here). Fresh single-copy dense
NCM control runs on identical sources, precision and runtime. Original Llama
77.21 tokens/s is an explicitly reused secondary control. Primary engineering
success requires retained quality, >=10% model-byte saving and >=1.05x both
complete-request latency and token rate versus fresh dense NCM. Report the
broader2x-Llama goal separately. Model seeds are not replicated; confirmed:false.

Use new append-only roots `data/experiments/ncm_factorization/20261021_*`,
fingerprint/snapshot producers and inputs, retain all failed/partial/control
outcomes, audit/aggregate, write measured report, register/read back and commit
only study source/tests/docs. Limits remain synthetic bounded facts,512GPT-2
input/48output positions; no general Llama, production95%, fullapplication,
mobile or exact-compression claim. Concurrent factorized replicas are a later
conditional experiment, not assumed from this single-copy comparison.
