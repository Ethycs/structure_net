# J-space compression and recovery implementation

Maintainer guide, 2026-11-15. This describes the implemented Qwen2.5-3B paths
and their artifact contracts. Read the [design](../01%20-%20Design/jspace/residual-compression-and-recovery.md)
for the method and the [evidence synthesis](../08%20-%20Analysis/jspace/2026-11-15_compression-evidence-synthesis.md)
for measured findings. The [experimental master guide](experimental-system-master-guide.md)
governs source freezing, new artifact roots, reservations and interpretation.

The current progressive and all-36 campaigns are exploratory sparse
multiple-choice recovery. The separately implemented [complete-response recovery](distillation-recovery.md)
follows [NAL-STD-DISTILLATION v1](../05%20-%20Standards/NAL-STD-DISTILLATION-v1.md);
that standard has not been retroactively applied to the running campaigns.

## Compression order and tensor flow

The Qwen compiler acts on residual channels and attention-head coordinates.
Attention retains causal softmax and the original attention scaling. The
quadratic relations described below replace selected SwiGLU channels; they do
not replace softmax with quadratic attention.

1. **Capture the function to preserve.** Freeze the native model or admitted
   recovered parent, tokenizer/template and case order. Record residual
   activations, position means, head statistics and full-vocabulary VJP probes.
   Current progressive capture uses 128 TRAIN calibration cases, two independent
   groups of eight probes, 16 activation positions and capacity 416. Its means
   use all 4,608 TRAIN cases. DEV remains diagnostic rather than fitting data.
2. **Fit residual and head geometry.** The grouped fitter constructs the
   functional basis from its Gram/eigenspace estimator. The current recipe uses
   rank 1,024 in native width 2,048 and head width 64. Q/K retain complete RoPE
   frequency pairs; V uses an uncentered orthonormal basis folded into V and O.
3. **Compile the affine block before fitting its MLP relation.** For basis U,
   the position anchor is `a[p] = mean[p] - U Uᵀ mean[p]`, and the represented
   residual is `h ≈ a[p] + U z`, with `z = Uᵀ h`. This preserves a fixed
   background, not each input's omitted component outside U. Fold incoming RMS gains and projections into
   reduced matrices and position-dependent numerator shifts; fold outgoing
   projections into reduced coordinates. RMS normalization retains the native
   width and anchor energy. Capture this compiled non-polynomial interface.
4. **Fit and compile the polynomial substitution.** TRAIN activations select
   anchor channels and degree-two coefficients under the numerical-rank gate.
   Normally 24 anchors yield 325 constant/linear/quadratic features. Remove
   1,024 selected channels' gate/up rows physically, retain the other SwiGLU
   channels, and fold their predicted contribution into the reduced down map.
   The prediction defect is omitted; retained channels do not make this lossless.
   The all-36 initializer has an explicit 15-anchor exception at layer 2.
5. **Compose adjacent blocks.** Encode into the first basis once. A fixed bridge
   between blocks maps coordinates with `U_nextᵀ U_prev` and the projected
   anchor difference. Reconstruct native-width hidden states once at the exit,
   then apply the final norm and original tied vocabulary readout.
6. **Recover learned weights, verify, then pack.** Training uses the unpacked
   differentiable stack. Keep the selected weights together with their exact
   geometry and provenance. Packing is a later explicit transformation with
   its own parity and storage checks, not a replacement for recovery.

| Responsibility | Main implementation and boundary |
| --- | --- |
| Capture and fitting | `jspace_3b_progressive_capture.py`, `jspace_qwen3b_progressive_calibrate.py`, `jspace_qwen_parallel_fit.py`, `jspace_allblocks_rankfit.py`: teacher seam, grouped probes, basis, then compiled-interface relation fitting |
| Compiled representation | `jspace_qwen_affine_block.py`, `jspace_qwen_compact_stack.py`: reduced residuals, head projection, polynomial substitution and fixed bridges |
| Construction and reload | `jspace_qwen_parallel_model.py`, `jspace_qwen3b_progressive_model.py`: selective native loading, explicit per-layer fits, strict inheritance and packed clones |
| One-block recovery | `jspace_single_block_distill.py`, `jspace_single_block_reserved.py`: native input/output feature matching with the remaining native model fixed |
| Progressive continuation | `jspace_3b_progressive_ladder.py`, `jspace_3b_progressive_stage.py`: CPU controller, eight authenticated phases, DEV gate and parent fallback |
| Independent all-36 recovery | `jspace_3b_joint_relation.py`, `jspace_3b_joint_init.py`, `jspace_3b_joint.py`, `jspace_3b_joint_train.py`: historical-fit admission, fresh native initializer, joint training and exact replay |
| New response recovery | `src/neural_architecture_lab/distillation_{data,training,recovery}.py`, `jspace_3b_dense_adapter.py`: complete-response targets, bounded loss and NAL lifecycle |

Paths in the table without a prefix are under `experiments/structure_net/`.
This grouping identifies execution boundaries; study plans and archived source
closures identify the exact producer of each result.

## Three recovery workflows

**Single-layer recovery** changes one block and evaluates it through the native
suffix. The enriched block-0 implementation uses immutable native BF16 boundary
inputs/outputs, promoted to FP32 for training. Only compressed block parameters
train, using FP32 AdamW. Its equal-case/equal-valid-token output error is divided
by native update energy with a TRAIN-fitted floor. The embedding, layers 1–35,
final norm and readout remain fixed. It is a local interface experiment; its
selection rule and result do not certify a simultaneously compressed stack.

**Progressive recovery** starts from an authenticated, freshly verified suffix.
Adding predecessor `c` preserves every inherited learned block, final norm and
interior bridge exactly; only the new block and entrance bridge are initialized.
Its functional capture sees the new native block followed by the actual frozen
recovered parent. The output KD teacher remains the original native model.
The current ladder repeats these phases:

| Phase | Data flow and completion requirement |
| --- | --- |
| `boundary` | Capture native pre-c states; replay pre-(c+1) bit-exactly against the parent's authenticated bank |
| `capture`, `fit` | Probe the recovered-parent function; fit the new basis, then its compiled non-polynomial MLP interface and relation |
| `parentfeatures` | Cache the parent's actual post-block-35 answer coordinates for TRAIN and fit their normalization energy |
| `pilot` | Discard representative entrance and joint updates; verify gradients, FP32 master changes, exact restoration and fixed state |
| `prealign` | Train only the new block for one epoch; select initializer/end by DEV normalized interface MSE, earlier tie |
| `train` | Fresh optimizer; jointly train all compact blocks and final norm for one pass, retaining actual candidate checkpoints |
| `verify` | Fresh strict reload, unchanged complete fixed state, exact selected and distinct diagnostic DEV records |

Joint recovery uses 288 updates, microbatch 4 × accumulation 4, constant LR
`5e-6`, clip 1 and zero weight decay. Its loss is
`0.7 * T² * KL(native || student) + 0.3 * CE`, T=2, plus the fixed normalized
parent-feature term with coefficient `0.4192096950932534`. Prealignment has its
own one-epoch cosine schedule with peak LR `2e-5` and fresh optimizer state.
The original-native DEV floor is 183/256 versus native 195/256, with at most
six net lost cases per 64-case family. Select the highest eligible correct
count, then earliest candidate; the raw initializer precedes an equal warm
initializer. Failed gates retain the parent and stop expansion.

**All-36 joint recovery** is a separate experiment, not a deeper continuation
checkpoint or a controlled one-variable comparison with the ladder. Its learned
parameters are freshly projected from pinned native Qwen2.5-3B-Instruct
revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Historical task-Jacobian
bases and polynomial fits remain fixed and retain their original calibration
teacher provenance. Two missing relations are separately completed: layer 2
with 15 anchors, layer 5 with 24; both remove 1,024 channels.

Its boundary-zero cache is the original embedding lookup. All 36 compact blocks
and the final norm train jointly on original-native content-four KL+CE only,
with no parent-feature auxiliary. One pass uses 288 updates, microbatch 1 ×
accumulation 16 and the same LR, clipping and DEV eligibility limits. Retained
checkpoints are updates 0/72/144/216/288. A diagnostic best remains explicitly
unqualified if no checkpoint passes. Each experiment has its own discarded
pilot, initializer, optimizer state, source archive and verification chain.

## Data, precision and checkpoint contracts

The native loader reads local `model.safetensors` or indexed safetensors shards
and a pinned Qwen2/Qwen2.5 config. It selectively loads blocks and one shared
tied embedding/readout. These are floating-point source tensors, not a generic
GGUF or low-bit quantized-model conversion path. Native BF16 rounding is part
of the reference and must not be silently replaced by FP32 native inference.

Current broad data contain 4,608 TRAIN and 256 DEV cases, four task families,
fixed option permutations and input lengths at most 384. The immutable native
logit bank has eight slots: legacy four followed by content four. **Slots 4:8**
are three meaningful user-content next-token positions plus the answer position.
They are full-vocabulary rows, but only four positions per case. They are not
complete responses or EOS supervision. Repeated positions across the two banks
are intentional. Boundary/logit `.npy` arrays store lossless BF16 bits as int16
and are read by mmap with authenticated offsets and case identities.

Native boundary caches replace an unchanged native prefix; every trainable
compact block is still recomputed. Parent answer-coordinate caches supply the
declared feature target, not reusable student activations. No audit/final-test
row may enter these banks or selection. New NAL response records instead carry
verified prompt/response boundaries, all response targets and observed EOS.
Its `loss_mask[b,t]` selects the logit predicting token `t+1`; targets are already
shifted. The separate response runbook documents exact chunked forward KL,
teacher hidden-state/readout targets, rollout freshness and generation gates.

Weights, computation, geometry and optimizer precision are separate choices:

| Path | Model and fixed state | Optimizer location/precision |
| --- | --- | --- |
| Single-block feature recovery | FP32 trainable block from native BF16-rounded source | FP32 AdamW on the model device |
| Current progressive stack | BF16 learned parameters; BF16 stored bases/bridge matrices; required anchors, energies and polynomial statistics retain FP32 | FP32 masters/moments on the 2060 SUPER with bounded CPU spill |
| Current all-36 stack | Same explicit BF16 compiler convention; unfused Torch forward with block checkpointing | CPU BF16 gradient bank and FP32 masters/moments; bounded 32 GiB state, host admission with 8 GiB reserve |

Do not call `.half()` on the entire stack. BF16 stored changes can round away
even when FP32 masters changed; master-change receipts do not prove every stored
BF16 tensor changed. Native RoPE constructed on
CPU then moved to GPU and compact RoPE constructed on GPU have distinct recorded
fingerprints. A correct weight reload with different rotary bits is insufficient.

An unpacked checkpoint contains `stack` and `norm` state, explicit contiguous
layer indices and geometry metadata. Reload requires the matching native source,
per-layer `fit.pt`/`relation.pt`, packing mode and buffer conventions. The builder
strictly checks same-cut reconstruction or exactly one preceding-block expansion.
Keep input hashes, source archive, initializer/selected checkpoint SHA, immutable
file identities, fixed-state fingerprints and phase-manifest dependencies together.
Weights alone do not identify the model. The verified 15-block checkpoint has
DEV evidence; its whole-model size and speed still need matched measurement.

## Inspect current work without launching it

Snapshot on 2026-11-15: the verified 15-block seed is
`data/experiments/jspace_3b_broad_recovery/20261115_v2/progressive15`.
The active continuation root is
`/data/structure_net_experiments/jspace_3b_progressive36/20261115_v1`;
the independent all-36 root is
`data/experiments/jspace_3b_broad_recovery/20261115_v2/all36_joint`.
Phi work is paused after a native 58/64 baseline; that is not a compressed Phi
result. Older frozen prospective documents retain their original launch status;
use current controller/phase receipts to establish live progress.

From the repository root, these inspect status without starting a worker:

```bash
git status --short
/usr/bin/python3 /usr/local/bin/inference-ctl status
/usr/bin/python3 /usr/local/bin/inference-ctl lease list
ps -eo pid,args | rg 'jspace_3b_(progressive_ladder|progressive_stage|joint)'
env CUDA_VISIBLE_DEVICES='' PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache \
  UV_CACHE_DIR=/tmp/structure-net-uv-cache \
  /home/rabbit/.pixi/bin/pixi run --locked --offline -e ncm python \
  -m experiments.structure_net.jspace_3b_progressive_ladder status \
  --root /data/structure_net_experiments/jspace_3b_progressive36/20261115_v1
tail -n 1 data/experiments/jspace_3b_broad_recovery/20261115_v2/all36_joint/progress.jsonl
```

`status` authenticates archived ladder evidence, reads events and probes its
existing controller lock; it does not advance phases. Inspect `plan.json`,
phase `manifest.json`/`result.json`, launch receipts and logs before interpreting
a partial root. Missing terminal output alone never authorizes a second launch.

For a **separately authorized new experiment**, prepare a new dated root only
after reviewing its plan and fingerprints. This example is a standalone 16-block
study from the verified 15-block parent, not a command to resume the active root:

```bash
export CUDA_VISIBLE_DEVICES=''
export PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache UV_CACHE_DIR=/tmp/structure-net-uv-cache
JSPACE_NEW_ROOT=/data/structure_net_experiments/jspace_3b_progressive36/YYYYMMDD_newstudy
test ! -e "$JSPACE_NEW_ROOT" && \
  /home/rabbit/.pixi/bin/pixi run --locked --offline -e ncm python \
  -m experiments.structure_net.jspace_3b_progressive_ladder prepare \
  --root "$JSPACE_NEW_ROOT" --target-blocks 16 \
  --parent-root data/experiments/jspace_3b_broad_recovery/20261115_v2/progressive15 && \
  /home/rabbit/.pixi/bin/pixi run --locked --offline -e ncm python \
  -m experiments.structure_net.jspace_3b_progressive_ladder run --root "$JSPACE_NEW_ROOT"
```

The CPU supervisor reserves before starting Python workers. Single-card phases
use RTX3060 `GPU-0370fdb0=11264` MiB; joint `pilot`/`train` also reserve RTX2060
SUPER `GPU-7b1db3cb=7168` MiB. Allocator caps are 11,008/6,912 MiB, with 6,144 MiB
SUPER optimizer residency and CPU spill. The independent 6 GB RTX2060
`GPU-8cf440df` all-36 worker reserves 5,632 MiB, cap 5,376 MiB, using CPU Adam.
Native experiment reservation conflicts receive HTTP 409: no automatic queue
or preemption. The ladder retries only declared transient failures before a
worker creates artifacts; a conflict or failed worker is preserved for diagnosis.

## Packing, generation and deployment boundaries

`packed_clone` copies through CPU, folds offsets and polynomial output columns,
aligns packed widths to 128 columns and strips unused interior bases/nonexit
anchors. This reduces deployed state, but stripped constituent blocks cannot be
recomposed without their external fit artifacts. Packed CUDA inference can take
the BF16 Triton path only when its dtype/device, eval, no-gradient and packed-state
conditions hold. FP16 worker execution and unfused Torch training are distinct.
Packing or a fused kernel being present establishes neither parity nor speed.

The current geometry supports **416 total positions**, including template,
prompt and requested output budget. Oversize requests must fail rather than
clamp anchors or silently truncate. Native long-context config does not expand
that capacity. See [bundle inference](jspace-3b-bundle-inference.md) for the custom
loader, source/version checks, reservation guard and greedy-generation CLI.

Final qualification requires the actual full compact checkpoint, fresh reload,
complete-response generation assessment, independent confirmation, export parity,
actual bundle/storage accounting and isolated matched prefill/decode measurements.
Isolated final-block speedups are positive local evidence, described in the
evidence synthesis. Training throughput, partial-suffix savings and another
model family's results cannot establish Qwen whole-model speed or quality.
