# Continued tensor training with Gaussian noise and global recovery

Date: 2026-10-14. EXPLORATORY, UNDERPOWERED (one parent and one noise seed).
Hypothesis: `llama32-3b-noise-tensor-continuation-v1`.

Authorization: the user requested continued training, "noise + tensors", and
research on related embedded/blockwise distillation solutions. The completed
parent study remains unchanged and failed its development quality gate. This
is a new follow-up, not a retry that can promote that parent.

Both arms start from the parent's 14 selected rank-1024 FP16 factor checkpoints
after 96 updates/group. These are original-weight SVD factors that were already
trained; they are not randomly initialized weights. Adam state was not saved,
so this is weight-only continuation with new optimizers, explicitly not exact
optimizer resume. Attention, embeddings, norms, normal residual connections,
factor rank and inference shape stay fixed. There is no dense correction.

Arms: `clean` and `noise`. The noise package adds seeded Gaussian noise with
standard deviation 1% of each factor's RMS once at continuation initialization.
On every second local update it also perturbs the cached teacher input with
Gaussian noise at 5% of each input channel's RMS. Both frozen teacher blocks
and student blocks receive that same noisy input; recompute the teacher target
instead of pairing a changed input with the old target. This is a near-data
Gaussian-input adaptation, not a reproduction of pure-noise vision training.
The combined package does not separately identify weight-noise and input-noise
effects. The clean arm receives the same input schedule with neither noise.

Local continuation: 256 additional updates per each of 14 two-layer groups,
AdamW LR 0.00002, zero weight decay, norm clip 1, batch one request, seed 41014.
Use FP32 factor master weights and FP16 forward operations. The local objective
is half per-token relative hidden-state MSE over the complete input and half
over reference-response positions, with an energy floor 1e-8. Validate the
actual FP16 representation at steps 0/64/128/192/256; select the lowest
development response-balanced loss. Local inputs still come from the original
teacher, so a further assembled-model recovery phase is required.

Global recovery: assemble each local candidate, then train against the original
teacher's output distributions on reference-response positions. Use KL at
temperature 1 plus equally weighted reference-token CE, LR 0.00001, AdamW,
clip 1, 32 updates for each active two-layer group, visiting groups 0–13 once.
Freeze all other factors and original weights. Gradient-checkpoint the full
assembled decoder with non-reentrant checkpoints. This is memory-bounded block
coordinate optimization of a global loss, not independent local training or
the full all-parameter global optimizer in Puzzle. Save the final 32-update
state per group. Reference-response teacher hidden states are extracted from
the existing original-teacher cache; the original frozen norm/output matrix
produces soft targets. No extra full teacher forward is necessary. Record
global teacher KL/CE on all development requests after each group.

Data: reuse the same 64 train and eight development requests, declaring that
their prior outcomes are known and development is adaptive. The 20 final-test
requests remain unopened for training/selection. Inputs, summaries and output
caps match the parent. No cross-request runtime KV reuse. Input caches are
training resources only. Noise/data-order RNG streams are separate and arms
share the example schedule. This remains a narrow synthetic final-decision
proxy, not production summarization quality.

Gates: CPU tests, a full-size local noise lifecycle pilot, then a longest-input
full-decoder global-gradient memory pilot on GPU 0; serial execution only.
Require finite losses/gradients, unchanged frozen weights, gradients only in
the active factor group, and exact FP16 checkpoint reload. Preserve failures;
any implementation fix needs a fresh producing root. No silent CPU fallback.

Train both complete fixed arms. Evaluate all eight development requests after
global recovery with the parent's source-grounded rubric and token/cardinality
rules. There is no independent judge model. Wrong owner, budget or deadline,
unsupported claims, missing fragments or >128-token summaries fail a request.
Noise benefit means more complete development requests than clean; ties do not
establish benefit. Lower global KL is secondary and cannot rescue quality.

If either arm passes all eight, select among passing arms using lower mean
development response KL (tie: clean), then evaluate the unopened 20 requests
alongside frozen original FP16 and Q4 controls. Require >=95% complete requests
and no fewer passes than either control. If neither passes, leave final test
unexecuted. Timing uses a fresh native Q4+lookup control and identical four
development prompts, 32 forced token steps, three repeats. This diagnostic
does not establish the user's 2x useful-summary throughput target. A passing
quality candidate would still need native kernels/quantization and a matched
five-repeat complete-summary campaign before a deployment speed claim.

Prospective classifications distinguish quality recovery, no recovery, and
noise benefit. The parent is never reclassified. Store all initial/local/global
checkpoints, histories, data identities, source snapshots, actual updates,
precision, optimizer resets, failures and conditional exclusions under new
roots `data/experiments/llama_neighbourhood_continuation/20261014_*`. Independently
audit, report, register conservatively with exact/semantic readback and commit
only this study's source/tests/docs. Preserve unrelated worktree changes.

Research motivation: [Neighbourhood Distillation](https://arxiv.org/pdf/2010.01189)
uses Gaussian neighbourhood inputs and final assembled-model fine-tuning.
[Puzzle](https://arxiv.org/pdf/2411.19146) applies local block distillation to
LLMs and adds global recovery for inter-block compatibility. Neither source
establishes that weight noise improves our particular Llama summarizer.
