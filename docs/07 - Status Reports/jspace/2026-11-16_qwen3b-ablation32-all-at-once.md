# ABLATION32: joint finite-removal search for all36 Qwen blocks

Exploratory study `JSPACE-QWEN3B-ABLATION32-01`. The user supplied the recovered
Fashion-MNIST recipe and requested a second all-at-once experiment. The linked
Library report and ZIP could not be retrieved in this environment; implementation
uses the pasted description, without claiming inspection of the original bundle.
Keep the active progressive campaign and its3060 reservation untouched.

## Question and ordering

At fixed deployed dimensions, does learning which residual directions to remove
outperform removing the identical initial random directions? Freeze the original
Qwen2.5-3B-Instruct revision`aa8e72537993ba99e69dfaafa59ed015b17504d1`.
Jointly learn all36 discarded subspaces, select them on a separate TRAIN-derived
selection split, compile all36 retained complements, then distill sequentially
with exact native look-ahead. Do not substitute Jacobian Gramians, PCA energy
ranking, the prior all-at-once weights or progressive refitting of later bases.

## Necessary Qwen adaptations

Residual width2048 becomes1024. Qwen uses RMSNorm, so the channel-mean direction
is not a normalization null direction and must remain represented. Let E span
the2047-dimensional complement of the normalized all-ones direction c. Learn
Z_l of shape2047×1024, D_l=E qf(Z_l), and P_l=I-D_l D_l^T. Its rank is1024,
including c. Export an orthonormal retained basis B_l with B_l B_l^T=P_l.
Use differentiable thin QR during search; complete QR is allowed only for the
offline no-gradient complement export. Native normalization retains denominator
2048, with anchor norm and cross terms. Ordinary1024-channel RMSNorm is incorrect.

Use36 distinct bases, one per block, shared at that block's entrance,
post-attention residual and post-MLP residual. At each site apply
mu_l + P_l(h-mu_l). These three placements match the residual computation of the
existing affine compact compiler. Searching entrances alone while silently
adding intermediate projections during compilation would be a different method.
Mu_l is the absolute-token-position mean of native block-entrance activations
over basis-fitting TRAIN examples, up to416 positions; unseen positions use
the same fitting set's token-weighted global mean. Anchors remain fixed.

Keep native attention softmax/scale, all native head dimensions128 and native
MLP hidden widths11008 in this first paired experiment. This isolates residual
subspace choice. Head compression, polynomial attention and additional MLP
compression are separate follow-ups, not unreported changes to this comparison.
Measure the projected full model against the actual compiled model, including
BF16 folding differences; a projection-only model is not a compressed artifact.

## Data and search

Reuse the authenticated broad complete-response corpus (4441TRAIN/128DEV across
eight families; prompt≤288, response≤128, total≤416) and original native teacher
hidden cache. Choose500 selection examples first, then2500 fitting examples,
using deterministic family round-robin selection among distinct source groups.
Exclude every selection group from fitting, anchors and subsequent distillation.
Small families may exhaust before larger families; record actual counts.
Use no DEV or sealed final benchmark examples to optimize or select bases.

Freeze native weights and optimize only Z. For each example, average full-
vocabulary forward KL over all response-prediction positions including EOS,
multiply byT², then average examples. T=2. Use8 epochs, AdamLR0.01, effective
batch32, cosine schedule with floor0.15 and global gradient clipping1. Seed
layerl with220101+l and use shuffle seed220301. Evaluate selection KL at epoch0
and every epoch; choose the minimum, with earlier epoch breaking ties. Preserve
the epoch0 subspaces as the matched random control regardless of later results.
Also record KL, native/student greedy disagreement and actual token counts.

Keep FP64 QR parameters and optimizer state on CPU; perform FP64 QR on the
spare plain2060 with saved QR graphs offloaded to CPU, and apply FP32
projectors inside the frozen BF16 teacher on the2060SUPER. A reserved standalone
systems probe measured transfer-inclusive QR+VJP at0.159s/interface, versus
0.258s for eight-thread CPU, with raw-gradient relative difference below3.3e-15.
This selects the exact auxiliary-GPU arithmetic backend before scientific
search, not an accuracy approximation. QR is computed once per optimizer
window. Accumulate gradients with respect to detached FP32 D leaves over the
window, then propagate their summed gradient through the FP64 QR graph exactly
once per layer. This chain-rule implementation must match direct autograd on
tiny tests. CPU optimizer state is explicit memory offload, not CPU teacher
execution. Checkpoint frozen block computation for activation memory. No
quantization or hidden loss approximation is permitted to make the pilot fit.

## Compilation and student fitting

Compile every layer of both learned and initial-random controls before their
respective fitting campaigns. Use the same affine/normalization implementation,
data, architecture, initialization seed1001 and training budgets in both arms.
Freeze the original teacher and previously fitted student prefix. Fit each
current block against teacher coordinates, plus full-vocabulary teacher KL atT2
after its result passes through the exact frozen native suffix. Later stages
consume actual outputs from the selected student prefix. A frozen prefix may
be cached on disk for reuse without substituting native-prefix activations.

Use12 epochs per sequential fitting stage, AdamWLR4e-4, weight decay1e-4,
effective batch256, cosine floor0.15, gradient clipping1 and look-ahead coefficient1.
The distillation training inventory is every original TRAIN example whose group
is outside the500 selection groups. Coordinate MSE uses valid token positions;
output KL uses response/EOS positions. Select minimum selection distillation
loss including epoch0; retain early winners. Original tied vocabulary weights
remain frozen, and final norm may be fitted with the last block. No joint
recovery phase belongs to this recipe. Measure runtime before quoting an ETA.

## Execution and reporting

Use the idle2060SUPER via `inference-ctl lease run`, with7680MiB reservation
and7424MiB allocator cap, paired with2048MiB reservation/1792MiB allocator cap
on the plain2060 for QR. Authenticate both logical-device mappings and their
single shared lease. Perform a real discarded all36 forward/backward,
actual basis update and checkpoint/reload pilot first. Zero anchors may be used
only for a clearly discarded systems pilot while the full native bank is being
captured elsewhere; scientific fitting must use the authenticated fixed anchors.
Record host memory, GPU peaks, QR wall time, actual parameter changes and frozen
teacher identity. If8GB is insufficient, diagnose and implement exact layer
offload or schedule an explicit free3060 slot; never silently fall back to a
different objective or disturb the progressive reservation.

The full run comprises basis search, compilation, both sequential-distillation
arms and packed evaluation. Quality failures are reported and guide recovery;
they do not silently cancel this requested experiment. Invalid execution or
provenance requires repair. Keep every completed study immutable and use fresh
attempt roots for changed producers. Existing exposed MC256 and responseDEV128
plus actual generated responses are evaluation material, not independent final
confirmation. Report native, initial-random and learned results, actual bytes,
and measured warm timings separately. No96% accuracy,40% size reduction or
speedup is assumed from the Fashion-MNIST results.

New artifacts live under`data/experiments/jspace_3b_ablation32/20261116_v1`.
Source/tests/protocol are frozen before scientific execution. Maintain
`confirmed:false`; explicitly distinguish a passed systems pilot, trained
bases, an exported compressed model and completed task-quality evaluation.
