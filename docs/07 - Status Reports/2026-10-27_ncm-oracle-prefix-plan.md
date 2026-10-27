# Correct early codes, then generate the remaining hierarchy

Exploratory hypothesis `ncm-reserved-oracle-prefix-v1`: correcting at most the
first three code levels substantially improves the remaining rollout and text
reconstruction on the fixed64 reserved component cases. The preceding public
epoch2 diagnostic measured61.77% teacher code agreement,11.89% free rollout
agreement and3/64 correct roots; shuffled sources gave nearly identical code
agreement. Application semantic DEV remains0/128. These outcomes are already
known and motivate this new intervention; they will not be replaced.

Keep the same epoch2 checkpoint,64 cases, native target frames,641 oracle codes,
full frozen Llama source features, source order and pilot IDs. Reuse and verify
the completed independently audited/registered reserved-code diagnostic and
renew its TRAIN/source guards. Keep final320 sealed.

Five conditions use the correct source: no injected codes; the first1 code;
the first9 codes; the first25 codes; and all641 codes as a reconstruction
control. These are complete boundaries in the native12-level hierarchy.
The generation function receives only the selected prefix, never later target
codes. Advance the frozen quantizer with that prefix, then generate every
remaining level using the unchanged sequential greedy mechanism. With an empty
prefix, outputs must reproduce all archived source-only codes, words and text
exactly. With all641 codes, outputs must reproduce the repaired-parent
`parent_with_source` codec roundtrip outputs exactly. This second control checks
the reconstruction path remained unchanged during public generator training.

Primary measurements exclude injected positions: remaining-code agreement and
its difference from the no-injection baseline on those same remaining
positions. Also report content-token reconstruction agreement (excluding PAD
and CLS), exact native-rendered text count, per-level code counts, and digit
sequence preservation. These are target-aware component diagnostics, not
semantic task accuracy. The descriptive early-prefix rescue gate requires at
least one of1/9/25 injected codes to improve remaining-code agreement by at
least10 percentage points AND content-token agreement by at least20 percentage
points over baseline. Report every arm, including failure of that joint gate.
The all641 control cannot qualify the gate because no generation remains.

Use locked Pixi ncm, physicalGPU0 RTX3060, FP32, TF32 off, deterministic
algorithms, seed42, CUBLAS :4096:8, a256MiB feature cache and at most4.9GB peak
GPU allocation. The live epoch4 trainer continues unchanged on physicalGPU1.
No optimizer, backward pass, new semantic judgments, final-test access or
speed measurement is permitted. Preserve all prior and new failures.

Before GPU execution, test prefix boundaries, no later-target input, unchanged
empty-prefix rollout, target immutability, exclusion of injected codes from
metrics, strict source/output binding, and gate boundaries. Check every
fingerprint input exists before creating a run root. The first/longest-source
two-case pilot repeats all five conditions exactly and checks both archived
controls, full model hashes, absent gradients and memory. After its actual
successful receipt, run all320 case/condition combinations; pilot cases must
repeat exactly. Save raw codes, words and text, target/source bindings, counts,
producer/input hashes and actual process receipts. Independently replay all
metrics and controls before reporting, registering and reading back the result.

Fresh roots: `20261027_oracle_prefix_<pilot,run,audit,results>_v1` under
`data/experiments/ncm_paper_regime`. Producer:
`experiments.structure_net.ncm_oracle_prefix`. This measures the effect of
supplying answers; it does not establish that a learned correction can produce
them, or that exact-code improvements imply semantic improvements.
