# Recovered decoder exposure audit and fixed reconstruction readback

This implements the already specified decoder-adaptation evaluation. It does
not change training, data, budget or endpoints. Before the full run finishes,
audit its committed first epoch: all 99,367 domain frames exactly once including
the seven-frame tail, and every retention selection/augmentation. Capture and
hash that exact trace prefix; bind the committed step3106 checkpoint. Replay
two deliberately modified copies (domain index and retention augmentation) and
require rejection. Do not treat a first-epoch audit as full-run completion.

After actual successful full terminal execution, independently replay all 6,212
updates, both seeded domain permutations, retention RNG/augmentation, learning
rates, finite losses/gradient norms, and checkpoint metadata. Require 198,734
domain presentations with each unique frame visited twice, plus 49,696 author
TRAIN retention presentations. Require the final checkpoint's exact bytes and
the unchanged recorded frozen representation identity. The training worker's
actual final tensor-hash check supplies the representation-preservation check;
decoder-only checkpoints do not contain those frozen tensors themselves.

Then reconstruct the same complete held-out 128 targets used for the parent
codec comparison, using the fixed final checkpoint only. Load the original
repaired codec, replace only its decoder from the final checkpoint, and require
the full codec tensor hash to equal the completed training result. GPU2,
deterministic FP32, TF32 off, native BERT framing/rendering, no sampling or
checkpoint selection. Require all re-encoded target hierarchies to match the
parent's saved codes exactly, finite full logits, and unchanged inference weights.

Report exact text reconstruction, lexical agreement, digit preservation and all
paired fixes/regressions, using the original training plan's directional and
stronger reconstruction gates. Replay every saved raw output and metric on CPU
with the existing qualified codec auditor, including its full edit diagnostics.
No semantic judge requests, source-only accuracy or inference speedup is claimed.
The 128 source components and duplicate native frames remain excluded from the
training corpus. The application final cohort stays sealed.

Roots: `20261027_recovered_decoder_{first_epoch_audit,audit,evaluate,replay}_v1`.
All roots are append-only. Failure stops the affected attempt and is retained.
