# Train the native coarse-to-fine mechanism on accepted examples

Date: 2026-10-23. EXPLORATORY `ncm-native-parallel-capacity-v1`.
The AR readout fits32/32 training texts and its independent raw replay matches;
its64-case development evaluation is pending and no transfer outcome was
inspected for this design. Native residual books passed their controls. This
study tests an actual coarse-to-fine native-word NCM adaptation, not AR-head
success or unchanged-paper replication. It cannot establish95%task accuracy.

Use the same32 previously accepted training summaries and original per-token
frozen-body features as the AR control. Initialize the frozen five-block native
readout and its trainable adapters from that final checkpoint. Freeze native
base/FC weights, tied word head, word embeddings and the fitted five residual
codebooks. New trainable parameters are256-way concept heads and level queries;
fusion/LoRA corrections continue training. Coarse queries start from native BOS;
the final word block starts from BOS followed by the shifted cumulative concept
reconstruction, with a zero-initialized level offset. This preserves native
predecessor alignment without supplying current/future word labels.

Generate parallel concept blocks of4,8,16,32,64 symbols, then128 parallel native
word positions. Current queries attend within their level and to source memory;
source never reads the current block. No other-level query blocks enter attention.
Cache source KV once in inference without modifying the saved source cache.
There are six readout evaluations after source prefill, not128 AR evaluations.
No parent generation or verifier is used after source feature extraction.

Train1,024 optimizer updates, batch2, accumulation4, AdamW LR1e-4, weight decay
0.01, gradient clipping1. Seed42; independent sampler2042, corruption3042,
level-selection4042. First256 updates sample uniformly among five coarse levels.
Thereafter choose the final native-word level with probability0.5, otherwise
uniformly choose a coarse level. Each microbatch uses a clean-history prefix
with input-code corruption, or after warmup with probability0.5 the model's
actual greedy earlier-level predictions. Corruption probabilities by level:
0.5,0.05,1/30,1/60,0. Recompute the current coarse expert against the actual
corrupted/predicted cumulative reconstruction. Final native word labels remain
clean, including EOS; positions after the first target EOS are loss-masked.
Do not condition on current/future labels or differentiate through sampled codes.

Before full training, a four-case BF16/GPU0 pilot does two actual updates, one
root and one final-word update. It exercises actual predicted histories and
the full128-position word block, checks finite values, both interface gradients,
frozen base/books, prefix-cache immutability, full/cached relative-L2<=0.02 and
top1 agreement>=0.95, repeated greedy outputs and exact trainable reload. All
component CPU tests and terminal passed AR/representation controls are required.
Only run after the native AR generation worker has terminated, with at least
6GB free GPU0. Stop on invalid numerics, OOM, identity/cache/reload or gradient
failure; preserve failed roots and do not retry seeds/configurations silently.

Measure only the final checkpoint, freely generating all five concept levels
and final words. Save every output/code tensor, not just scalar ablations.
Decision order: (1)target-token recovery>=98% and at least31/32 exact terminated
texts; (2)rolling source features/masks among cases drops token recovery>=5pp;
(3)rolling the model's own predicted concept plans among cases while keeping
source fixed drops recovery>=5pp. Otherwise classify respectively text-capacity
failure, source-dependence failure or concepts unused. Also report zero-concept
ablation as an out-of-distribution diagnostic, not a matched trained baseline.
These are training-only mechanism controls. No independent quality judgment or
held-out success follows from exact training targets. Broad student training
still needs the newly qualified teacher data and a separate task protocol.

Roots `20261023_native_parallel_{pilot,capacity}_v1`. Preserve complete producing
source/runtime/input identities, optimizer/RNG states, actual books, all raw
predictions, gradient checks and terminal receipts; finalize/register completed
results conservatively. Final task256 remains sealed. Full application input
range and128 stored tokens remain requirements before any95%or speed claim.
