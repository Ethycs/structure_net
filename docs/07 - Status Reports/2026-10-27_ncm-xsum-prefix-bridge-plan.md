# Preserve released XSum conditioning through a frozen-Llama bridge

Exploratory hypothesis `ncm-xsum-prefix-bridge-v1`: a small supervised bridge
can preserve the released XSum NCM's source-prefix representations and outputs
when its only content input is the existing frozen Llama source features.
The public-curriculum epoch4 experiment continues unchanged on GPU1. Its first
two application DEV milestones remain 0/128. This separate mechanism study
does not change its model-selection schedule or qualify the 95% task goal.

The XSum release already conditions on a raw GPT2 token prefix. Keep its codec,
generator, hierarchy, decoding policy and all parameters fixed. Learn only a
3072 -> 1024 -> 384 GELU MLP. Its output replaces the codec token embeddings
used by the generator's raw-prefix path. All active prefix token IDs supplied
to the bridged generator are dummy zero IDs; only the source length determines
padding. No real content-token IDs may reach generation through that path.
The bridge consumes the original complete FP32 aligned frozen-body features.

Select from the already admitted 162,353 public TRAIN sources. Require complete
source length at most 512 GPT2 positions; never truncate. Sort eligible IDs by
SHA256 of `ncm-xsum-prefix-bridge-v1:` plus ID. The first 512 distinct source
hashes are bridge-fit cases; the next 128 are bridge-probe cases. Reject any
duplicate ID/source hash in the selected population. Fix selection before
training. These are public TRAIN sources and may have been seen by the released
teacher; the probe tests bridge transfer, not unseen summarization accuracy.
Application DEV, reserved and final cases are not used. No summary targets,
semantic judges or new LLM labels are needed.

Normalize each input feature dimension using fit-only token-weighted mean and
standard deviation (floor 1e-5). Add the fit-only mean codec embedding to the
MLP output; initialize its last linear layer to zero. Minimize MSE after the
released generator's frozen input projection, divided by fit-only projected
target variance. Each complete source contributes one mean loss and one AdamW
update. Train ten epochs over all 512 fit cases, reshuffling with seed 5827.
Use seed 5827 for initialization, FP32 throughout, TF32 off, deterministic
algorithms, AdamW betas 0.9/0.999, eps 1e-8, weight decay 0.01, clip norm 1,
and cosine learning rate 1e-3 to 1e-4 over 5,120 updates. Evaluate only the fixed
final checkpoint; no probe-based checkpoint or hyperparameter selection.

Use locked Pixi ncm on physical GPU0 RTX3060, with CUBLAS `:4096:8`, OMP/MKL
threads 2 and peak allocated GPU memory <=4.9GB. Stream one complete feature
matrix at a time from the existing immutable disk store. Do not load the Llama
body, quantize features, modify the active GPU1 job or use GPU2's service.

Before training, CPU tests cover deterministic disjoint selection, the bridge
shape/normalization and gradient route, dummy-prefix isolation, hook cleanup,
and exact raw-count replay. A GPU pilot uses the first and longest fit cases.
Compute normalization from fit cases only. Require finite updates, exact
optimizer save/reload replay of update2, and nonzero gradients after update2.
Replacing the native prefix with its identical frozen embeddings through the
new hook must reproduce native codes and decoded token IDs exactly. Repeat
native generation exactly. Verify frozen codec/generator hashes and intended
concurrent GPU1 service identity before and after the pilot and each full stage.

After the successful pilot's actual receipt, train once from the fixed initial
bridge, using the pilot's exact normalization tensors. Preserve per-update
losses, selected identities, final bridge, frozen model hashes, complete
source/input hashes and actual process results. Any failure is retained;
there is no silent retry, tolerance widening or quality-based rerun.

On all 128 bridge-probe cases compare native generation, learned-bridge
generation and constant-fit-mean prefix generation. Use the released greedy
code generation and released frequency/deficit/adjacency word policy.
Record complete raw codes and decoded IDs, projected prefix MSE, token counts,
and model identities. The mean control uses the same source lengths/padding.
Primary go/no-go gate: learned-bridge decoded-token agreement with native
generation >=90% over all 128 x 48 positions AND projected-prefix MSE <=50%
of the mean control. Report EOS/pad-inclusive and non-pad agreement separately;
the gate requires >=90% non-pad agreement too. Also require >=90% agreement
on native content positions before the first EOS (excluding padding), and
nonempty native content on every probe. This prevents EOS or unused positions
from carrying the gate. This is a mechanism-preservation
gate, never semantic accuracy. No speed is measured.

A CPU audit independently replays every code/token count, both MSE aggregates,
all 5,120 update identities and the decision. Save raw projected predictions
and targets for independent MSE replay; use float64 sums, rtol 1e-10 and atol
1e-8 for replay comparisons. Codes and token counts require exact equality.
Save and register the conservative
result with exact/semantic/JSON read-back. The native 512-input/48-output frame
does not cover the full application range. Passing would motivate a separate
range-extension and source-disjoint semantic evaluation; it cannot establish
the 95% objective. Failure rules out only this fixed small-bridge protocol.

Fresh roots: `data/experiments/ncm_paper_regime/20261027_xsum_prefix_bridge_`
plus `prepare_v1`, `pilot_v1`, `train_v1`, `evaluate_v1`, `audit_v1`, `results_v1`.
Entry point: `experiments.structure_net.ncm_xsum_prefix_bridge`.
Audit/report/register/readback entry point:
`experiments.neural_architecture_lab.ncm_xsum_prefix_bridge_results`.
