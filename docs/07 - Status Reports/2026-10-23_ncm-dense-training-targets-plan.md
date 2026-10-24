# Dense latent targets for all approved NCM training summaries

Date: 2026-10-23. Exploratory hypothesis `ncm-dense-full-TRAIN-targets-v1`.
This prepares labels for a future source-only student across the complete690
approved TRAIN summaries. The32-case dense student established strong fitting;
its128-case semantic evaluation failed, with0/64 short and1/64 long passes.
Inspection found that the single passing output exactly copied a TRAIN label
from a nearly identical source under a different session ID. Before any CUDA
launch, require the original semantic study and a separate source-overlap
amendment to finish, be audited and registered. The original quality failure
stays unchanged; its source-disjoint generalization claim is invalidated.

Use the ordered690 TRAIN rows from `20261023_reviewed_draft_data_v1/cases.json`,
SHA256 ac3c56f7d03cf9160de9e719ef042b44a0bf12dececd778949b46e7288aac983.
Preserve the original252 and additional438 populations, all targets and full
192-word frames. Retain the historical metadata exclusion checks against old
development, binding the separate audit showing why those checks were
insufficient. All690 cases remain TRAIN data. Future student evaluation must
use fresh development sources passing source-family, public-turn and normalized
text-overlap checks before generation. Never read either sealed final source file.

Reuse the exact32 registered256-step target latent, initial latent and AdamW
states, complete traces and raw outputs from the successful latent convergence
control, campaign SHA256
1a5587221af8048f3052296bb17d6c6ab40afdf7e62d689088b7b0061070a834.
Bind every reused target to the current approved complete label and reproduce
its original/refined raw readouts on the same physical GPU2. Optimize the
remaining658 targets once, without selecting, excluding or repairing cases.

For every fresh case start at the original continuous encoder latent. Optimize
only that complete192 by384 tensor for256 fixed AdamW updates: learning rate
0.02, betas0.9/0.99, epsilon1e-8, weight decay0, clip norm1. Loss is nonpadding
word CE plus0.001 mean squared distance to the original full latent. Include
EOS50256; ignore PAD50257 in CE. Bound word logits to IDs0 through50256. Keep
the four independent48-position decoder blocks unchanged. No model parameters
receive gradients. No source features, parent model, forecaster or student are
loaded. Target words are labels only, never source-only inference inputs.

Use physical GPU2, FP32 throughout, math SDPA, no autocast, TF32 off, and less
than2GB peak allocation. Constructor seed50142 identifies fully restored
weights. Before full preparation run focused CPU tests, actual-data bindings
and a two-case CUDA lifecycle pilot. Select the first previously unoptimized
TRAIN case and the longest target among the other fresh cases, breaking ties
by case ID. Give each two updates; check positive input gradients, absent model
gradients, exact latent/optimizer serialization and fresh decoder readouts.
Full preparation starts fresh and retains exact pilot baselines and update
prefixes. Pilot final latents do not initialize full optimization.

Save each complete latent/optimizer state and raw baseline/refined readout with
its full trace in immutable per-case files. Hash all files in an ordered
manifest; retain all failures. Independently replay target/EOS decoding, latent
digests, optimizer configurations and counters, complete traces, all32 reused
states, lifecycle prefixes, counts and actual subprocess exits.

Require at least199/200 active-token reconstruction separately on the original
252 cases, additional438 and union690. This token fidelity gate qualifies label
preparation only; it does not establish95% complete-summary task accuracy.
Classify `dense_full_TRAIN_targets_fit_passed` or `...fit_failed`, with a measured
report and exact conservative registry read-back. No student training follows
automatically. A separate plan must specify initialization, the complete TRAIN
population, fixed optimization budget and source-disjoint semantic evaluation.

Use fresh roots `20261023_dense_training_targets_{workflow,pilot,measure,results}_v1`
under `data/experiments/ncm_paper_regime`. Freeze source, tests, plan and CPU
auditor before launch. One supervisor owns pilot, target preparation, audit and
registration with observed child exits. The original frozen-body95% summary
goal remains active; neither final evaluation nor speed testing is admitted.
