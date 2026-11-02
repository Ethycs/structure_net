# Coarse-code competence and residual frozen-body conditioning

Date: 2026-10-22. EXPLORATORY, hypothesis `ncm-coarse-curriculum-v1`.
The completed noisy-pair campaign remains quality-unmeasured pending judge
recovery. Inspection shows invented entities and malformed text. Its tokenwise
objective assigned only 4/200 of the weight to the coarsest level, whose final
training CE remained above 5. This follow-up tests a mechanism before another
large conditional campaign; it cannot rescue any previous failed study.

Question: can a continued NCM generator learn an accurate free-running code
hierarchy on 32 training pairs with stronger coarse supervision and a stable
frozen-Llama interface? This is a memorization/conditioning diagnostic, never
held-out task accuracy. Use the first 32 teacher pairs in the immutable saved
training order. No test or development examples participate. Known label noise
and codec reconstruction errors remain disclosed.

Initialize from `20261022_noisy_native_v1/model.pt` with its exact trained codec.
Keep the entire codec and original Llama body frozen. Preserve the generator's
native GPT2 source embedding channel. Add LayerNorm(3072) and a zero-initialized
Linear(3072,768) of the final valid frozen Llama hidden state as a broadcast
residual on valid source positions. Zero residual must exactly preserve the
native context. Cached features are a training convenience only.

Use the released implementation's optional autoregression over its first four
codes (12 generator evaluations rather than 9), explicitly switching the first
target block to a causal mask after checkpoint loading. Teacher-force only past
coarse labels; check first-code invariance to changed labels. Finer levels use
the normal residual code hierarchy. These interface/curriculum changes form one
engineering intervention; this study cannot attribute a benefit to one factor.

Train 1,024 updates, batch4, on physical GPU0 with one worker. First256updates
supervise only the four coarse codes. Remaining768 supervise all levels with
equal total weight per level (alpha1), clean targets and uniform same-level
conditioning corruption .50 to .10. Coarse AR past inputs remain clean.
Use dropout.1, AdamW(.9,.95), LR3e-4 for generator and bridge, WD.01, clip1,
warmup32 then cosine floor.1, BF16 autocast/FP32 parameters and code assignment.
Separate model/dropout/sampler/augmentation seeds42/52/2042/3042. Save every256
updates and retain raw predictions, code targets, token diagnostics and states.

Before training, a separate two-update CUDA lifecycle pilot must establish
finite gradients in query and bridge, frozen-codec identity, causal coarse
inputs, exact zero-residual native equivalence, actual GPU allocation and exact
reload. No CPU fallback, service changes or changes to completed artifacts.
Stop on a lifecycle mismatch, OOM, nonfinite objective or changed dependency.

Mechanism gate at the fixed final checkpoint: >=90% free code accuracy across
all positions, >=95% non-PAD target token reconstruction, and >=20 percentage
points more correct coarse codes with matched versus cyclically shifted source
contexts. This controls for unconditional memorization. Failure stops expansion;
inspect the preserved evidence before defining another experiment. Passing
only authorizes a separately specified held-out adaptation study. It does not
establish 95% complete-summary quality, final test success or any speedup.

Artifacts: `data/experiments/ncm_paper_regime/20261022_coarse_{pilot,train}_v1`.
Source/hash copies, checkpoint/codec identities, seed streams, losses, reload,
memory and matched/shifted diagnostics must be recorded. Register the completed
result conservatively; one seed is underpowered. Keep the 256-case final test
sealed. Actual final-summary quality remains the user's primary endpoint.
