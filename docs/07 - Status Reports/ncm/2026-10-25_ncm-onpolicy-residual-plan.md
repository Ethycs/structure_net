# Learn from generated histories, with and without residual retargeting

Prospective exploratory study `ncm-onpolicy-residual-v1`. The broader generator
scored 0/128 valid complete-summary DEV judgments. The completed 16-case oracle
probe showed that recomputing the last three residual levels after nine
generated levels raises active-token reconstruction from 62.84% to 81.20%.
That probe supplies a mechanistic motivation, not task-quality evidence.

Compare two fixed arms, both initialized from broad-generator checkpoint
`74cdce0eb14bb99c801b6ec0a025c092f95249663becc05561af8dfbad6040ff`:

1. `fixed_targets`: train against the original clean code labels even when the
   input history consists of this model's actual generated codes.
2. `residual_targets`: for those same types of generated histories, recompute
   each level's target by quantizing the remaining clean encoder latent after
   subtracting the contributions of the actual earlier generated codes.

The generated history is detached, greedy FP32 source-only inference by the
current arm before its update, in evaluation mode. Target frames must never
enter this inference call. Restore the exact pre-call module training flags.
For retargeting, subtract each historical code's contribution, not the newly
computed oracle target. This labels the states the policy actually visited.
Packed next-level prediction still consumes only strictly earlier history.
The full reference latent may supervise TRAIN labels; it is forbidden in DEV
or deployed inference.

In each arm, half of presentations use clean histories and clean targets as an
anchor; half use the current generated history. No random code replacement is
added. The sole between-arm objective difference is residual retargeting.
Arm histories may diverge as their weights diverge; each arm observes its own
policy. Neither arm adds a discriminator or modifies the codec.

Use all 1,376 unchanged admitted complete-summary TRAIN pairs and their full
FP32 source features. Train only the existing generator, concept source memory
and concept adapters; freeze the codec, word source memory/adapters and Llama
body. Each arm gets exactly 1,024 optimizer updates, accumulation4, half clean
presentations, CPU FP32 AdamW with betas0.9/0.95, epsilon1e-8, weight decay0.01,
gradient clip1. Warm64 updates to1e-4, then cosine to1e-5. Use shared
without-replacement sampler seed191142 and dropout seed193142. Run arms in the
fixed order above. This is one matched seed, underpowered for general claims.

Use locked Pixi ncm from repository root on physical GPU1 RTX2060, FP32 weights,
FP16 loss/backprop, FP32 generation/evaluation, TF32 off, and the existing256MiB
lossless disk-feature cache. Keep complete inputs and targets. Cap measured
allocated GPU memory at4.9GB. Preserve source/checkpoint/data/environment
fingerprints, actual process receipts, sampling traces, history/target-change
counts, losses, model state, optimizer and RNG states.

Before full training: focused tests must verify residual target semantics,
detachment, exact module-mode restoration, shape and bounds checks, and label
changes that correct a wrong prefix. A CUDA lifecycle pilot for each arm must
verify real updates, finite gradients, unchanged protected tensors, strict
reload, and packed/explicit loss and gradient parity for the selected histories
and labels. A clean history must yield the original native targets. Systems
pilots do not initialize the full arms. Stop on integrity, numerical, memory,
or lifecycle failure; preserve every failed artifact.

Only each arm's fixed final checkpoint is eligible. Evaluate all1,376 TRAIN
rollouts and the same16 shuffled-source controls diagnostically. After both
completed arms, generate all128 fixed source-disjoint DEV outputs per arm and
apply the existing whole-summary semantic rubric, regardless of TRAIN scores.
Preserve all3,923 prior first judgments and historical uncertain attempts;
never reroll first judgments to improve a score. Score `fixed_targets` before
`residual_targets`, sharing the growing first-judgment cache. Invalid transport
remains unresolved evidence, not model inaccuracy.

The primary endpoint is complete-summary semantic accuracy. In each arm, Terra
requires at least61/64 in both fixed cohorts to admit Sol, and at least61
same-case joint passes per cohort qualifies DEV. Positive but subthreshold
results are partial progress; zero passes fails that arm. Compare paired DEV
pass/fail decisions descriptively, without converting code/token agreement into
task accuracy. If both qualify, choose higher joint DEV accuracy, breaking an
exact tie in favor of `residual_targets`, before opening the sealed final320.
The final test and95% task requirement remain unchanged. No speed claim is
authorized by this training comparison.

Use new roots `20261025_onpolicy_residual_{arm}_{pilot,train}_v1` and separate
DEV/audit/results roots under `data/experiments/ncm_paper_regime`. Audit,
aggregate, report and register both arms and their failed predecessors. Do not
overwrite completed evidence or silently change these budgets after seeing
quality outcomes.
