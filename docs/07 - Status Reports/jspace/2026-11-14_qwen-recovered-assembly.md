# Assembly of independently recovered Qwen blocks

Study: `JSPACE-QWEN-RECOVERED-ASSEMBLY-01`, exploratory.

## Question and evidence fixed before execution

Do independently recovered rank1024 residual-update blocks preserve task
accuracy when composed through all36 layers of Qwen2.5-3B-Instruct? The
native FP32 development model scores50/64. Accepted blocks0/1 together
score49/64; block1's initializer won its subsequent distillation selection.
Independent blocks2..35 completed12 epochs each and fresh verification.
32 pass the isolated47/64 gate; blocks12 and15 score44/64 and46/64.
These isolated results are not assembled-model accuracy.

## Frozen arms and scoring

Evaluate these four arms, with no further checkpoint selection or training:

| Arm | Blocks |
| --- | --- |
| native | All36 original native blocks; control |
| accepted_two | Accepted compact blocks0/1, original blocks2..35; control |
| all_compact | All36 selected compact blocks; diagnostic |
| hybrid | Selected compact blocks except original native12/15; primary candidate |

Use exactly the original64 development prompts, original token IDs, full
36-layer execution from embeddings, and original tied vocabulary readout.
Batching remains4, with the established causal masks and rotary positions.
Weights and computation are FP32, including original BF16-rounded native
weights promoted to FP32. Preserve the full2048-wide residual carry and
native128-dimensional attention heads, softmax, and11008 MLP hidden widths.
Block0 must use its original first-block enrichment fit; blocks1..35 use
their respective all-layer enrichment fits. Strictly load the selected
checkpoint, then pack its projections; geometry and packed state are fixed.

The native control must reproduce50/64 and all64 reference predictions;
the accepted_two control must reproduce49/64 and its reference predictions.
Compare score logits with the existing absolute5e-4 plus relative1e-5
tolerance against their corresponding authenticated references. Stop
scientific interpretation if either control fails; preserve failed evidence
and investigate under a new root instead of modifying completed artifacts.

The primary hybrid passes the cumulative development quality gate at47/64
or better, relative to the original50/64. Report all four counts, paired
prediction changes, and per-task-family accuracy. A miss is a composition
failure, not a reason to silently discard the arm. The all_compact arm is
reported separately and cannot replace the declared primary outcome.

## Provenance, resource use, and limits

Pin the prior plans, selected-checkpoint hashes, selection history, fresh
verification receipts, code, and this protocol in a new root:
`data/experiments/jspace_recovered_assembly/20261114_v1`.
Read the nativeFP32 development cache only to authenticate references;
intervention evaluation starts from the original prompt tokens. Use one
block at a time to bound memory, locked offline Pixi's ncm environment,
and inference-ctl lease run for every GPU process. Keep live reservation
checks and an allocator cap below the lease. Run both controls before
intervention interpretation. GPU type is a recorded runtime factor, not
a change to the fixed scientific comparison.

This streamed evaluator measures accuracy, not deployment throughput:
its elapsed time includes loading and disk work and must not be reported
as inference Tok/s. Actual parameter storage must distinguish matched
FP32 comparisons from the native BF16 model. Any assembly passing here
still requires a separately frozen deployment benchmark and confirmation.
The fresh512-case confirmation set remains unopened until a final
assembled candidate is selected. All outcomes here remain development-only,
confirmed:false. An assembly failure leads to a new, explicitly scoped
repair study using actual student-prefix activations, not another claim
that isolated fits already establish end-to-end success.
