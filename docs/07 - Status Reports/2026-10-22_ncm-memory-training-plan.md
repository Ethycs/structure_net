# NCM training on shared memory summaries

Exploratory follow-up, `ncm-memory-summaries-v1`, 2026-10-22. The user requested
using the shared summaries directly as training targets, with a train/test split.
This is a new run; the completed synthetic-data and independent-quality studies
are unchanged. One training seed is underpowered for an architecture comparison.

Restore the 500 stores at memory-condense Git commit
`fc25ea0d3cccc589346ddc13747218b8dc7a1221` and lakeFS commit
`82d974caec273e113dd864f9da352c003540e9225fbcce3766a81c103e005dab`.
Verify archive/member hashes and hydrate atomic summary spans against original
turn hashes. Targets are the saved summaries, unchanged. Deduplicate identical
source/session/target triples. Make a deterministic 90/10 source-session split
(seed 20261022), keeping repeated non-bypass inputs in one partition. This uses
LongMemEval as training data; the resulting test split is not the untouched
benchmark. Hierarchy/parent sections remain in the restored source stores.

Export all atomic pairs and a native-compatible subset: at least 192 cl100k_base
input tokens (the real application's summarization branch), at most 512 GPT-2
input tokens, at most 47 GPT-2 target tokens plus EOS, and at most 128 cl100k_base
target tokens. No truncation. Record all counts and exclusions. Native model
capacity is unchanged in this initial adaptation.

Continue the existing NCM generator and codec, not random weights:

- Generator SHA256 `ff7133b2de50bc1f66ef1e490fec4c723e0fc5a5c1957107209e923b198c3e3c`.
- Codec SHA256 `b87f7288edfc0eb1f5b9122d1139cc2e61aa2331be7d6fe53f0582993086ede2`.
- Frozen encoder and multiscale codebooks; adapt the text decoder for 512 AdamW
  updates (batch 16, LR 2e-4) and generator for 1,024 updates (batch 8, LR 1e-4).
- Seed 42, independent sampling streams per stage; betas .9/.95, weight decay
  .01, gradient clip 1. Generator LR warms up for 50 steps then cosine decays to
  .1 of the peak. BF16 compute, FP32 master weights and code assignment.
- Codec token cross entropy ignores padding and retains EOS; generator loss is
  cross entropy over all multiscale target codes. Fixed final checkpoints,
  without test-set selection or tuning. Sample uniformly from training pairs.

Before training, run focused data/metric tests and one real CUDA update per
stage with save/reload parity in a separate systems-only root. Use one worker
on physical GPU0, RTX3060. Stop on a data/hash mismatch, nonfinite objective or
gradient, frozen-weight drift, or execution failure; preserve partial evidence.
Do not silently fall back to CPU.

After the fixed updates, evaluate the original and trained checkpoints on the
same hash-selected maximum 256 native-compatible test pairs. Report code CE,
codec text CE/token accuracy, free-generation ROUGE-1 F1, exact reference match,
empty outputs and output examples. These are reference-fit diagnostics, not a
human-calibrated 95% task accuracy measurement. No speed claim is made.

Artifacts: `data/experiments/ncm_memory/20261022_shared_v1/` (restored input),
`20261022_data_v1/` (splits), `20261022_pilot_v1/` (systems check), and
`20261022_training_v1/` (training and held-out comparison), each append-only.
Preserve producing source, complete configuration, data/checkpoint hashes,
optimizer/RNG continuation state, traces and final report.
