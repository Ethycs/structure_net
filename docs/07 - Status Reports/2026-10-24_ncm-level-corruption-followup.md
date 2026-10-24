# Prospective NCM conditioning-corruption correction

2026-10-24. Known outcome: the released LM1B native generator, adapted decoder
and new source attention trained on655pairs for1024updates, failed0/128DEV.
This is a new mechanism for a future training protocol, not a rerun or a
retroactive change to the failed producer. No existing checkpoint or objective
module is changed. No quality or speed result is claimed for this helper.

The failed run used one uniform per-example rate in[0,.25] and protected the
first generated code from corruption. NCM Section3.4 and AppendixF.3 instead
describe replacements from each level's own codebook and a rate decreasing
linearly across levels. The root generated code and the fixed BOS/null prefix
are different: corrupting the root's use as conditioning can teach later levels
to recover from an incorrect coarse prediction without changing its clean
supervised target. Excluding that code was a deviation from the described rule.

Source: https://arxiv.org/html/2610.14162v1#A6.SS3

The new helper corrupts conditioning history at every generated level, using a
fixed linear schedule from.75at the coarsest level to.25at the finest (unweighted
level mean.5), same-level uniform replacement, and a separate explicitCPUtorch
generator. These endpoints are OUR choices inspired by the TinyStories ablation;
the public paper and inference repository do not specify an exact LM1B
fine-tuning corruption recipe. Do not label this an exact author replication.
Clean targets are never mutated. The BOS/null prefix is not in the code tensor
and is not corrupted. Record all per-level rates, selected/changed counts and
clean/history hashes for complete deterministic replay.

Before using this in training, define a fresh dataset/initializer/budget and
quality evaluation protocol, run its actual CUDA gradient/reload/memory pilot,
then freeze that producer. Do not silently replace the objective of any live or
completed run. Changing both data scale and corruption does not establish a
causal benefit for either; a matched control is needed for that claim. The
already failed655model, all canonical judgments and sealed final320 are retained.
