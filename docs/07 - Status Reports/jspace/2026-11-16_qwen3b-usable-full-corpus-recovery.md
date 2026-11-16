# Qwen3B recovery selected for usable generation

Exploratory study `JSPACE-QWEN3B-USABLE-RECOVERY-01`, 2026-11-16. The user
explicitly prioritizes the best recovery of this compression technique and
usability. Follow NAL-STD-DISTILLATION-v1. This new companion study preserves
the active progressive and ABLATION32 producers and their immutable results.
It does not change their objectives or discard their checkpoints.

## Diagnosis and intended result

The current progressive search uses a broad eight-family corpus, but joint
recovery sees only the same 736 examples and 25,172 response/EOS tokens at each
depth. All 4,441 TRAIN examples are available, spanning 4,393 source groups.
The existing two-block checkpoint retains 198/256 MC answers against native
195/256, while generated outputs fail: no EOS on 22 probes, with both early
substantive mistakes and destructive repetition after some correct answers.
Lower reference-prefix loss and one additional MC answer do not establish
usable recovery. A generation inference error remains a competing explanation
until actual-current-checkpoint execution paths are compared.

The first fixed-depth experiment uses the three-block suffix (native layers
33–35 compressed). Before the new diagnostic outcomes, choose its retained
update46 checkpoint: response DEV NLL2.3195738, MC197/256. Update19 has one
additional MC answer but worse NLL2.4657182; that single answer will not dictate
this recovery initializer. Preserve and diagnose both versions.

Initializer:
`/data/structure_net_experiments/jspace_3b_full_progressive/20261116_v1/blocks_03/recovery/run/checkpoints/update_0046.pt`

SHA256: `44349fede75578ebe1e2dbda4713edc86f3dee7d36ffe89dad4d0eedbc1fd6c0`.
Original teacher: Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, frozen BF16 weights.

## Check execution before training

Use exact stored prompt and growing generated token histories. Compare the
trained unpacked student with a freshly packed Torch student using the same
native-prefix boundary and original vocabulary matrix. Include full-vocabulary
logits, original teacher logits, next-token agreement, teacher KL and EOS
probabilities/ranks. On supported hardware also compare packed Triton. The
plain RTX2060 does not support the deployed BF16 Triton kernel profile;
unavailability is explicit, never an implicit fallback or a passed parity test.
Stored Triton-generated next tokens can supply a diagnostic comparison on the
identical histories but cannot replace a full paired logits test.

Verify template, causal mask, token positions, fixed geometry, normalization,
readout and native-prefix boundary against the original model. A meaningful
execution mismatch requires diagnosis before spending the new training budget.
Save diagnostic inputs and results. No parameter update belongs to this phase.

## Stage A: full-corpus dense recovery

Use every admitted TRAIN example exactly once in a fresh deterministic order:
4,441 examples and 105,089 response/EOS tokens. Source groups remain disjoint
from the 128 exposed DEV examples. Preserve the actual repeated-within-group
TRAIN examples; do not incorrectly describe them as distinct groups.
Use existing authenticated disk-backed native prefix/final-hidden targets;
do not recapture or copy the entire activation bank.

Keep the fitted bases, relations, bridges, anchors, native prefix and tied
embedding/readout fixed. Jointly train the existing compact suffix parameters
and final normalization. The reference objective is exact full-vocabulary
forward KL at temperature1 on every response-prediction position including
the actual EOS, token weighted. No CE mixture or new geometry is silently added.
Use the tested CPU FP32 master/Adam optimizer machinery, BF16 live trainable
weights, LR2e-5, effective accumulation16, clipping1, no weight decay, and the
existing memory-bounded token-chunked loss. This LR is a declared new recovery
setting, not a continuation of the old optimizer state.

Record initializer and approximately25k/50k/75k/full-pass checkpoints. Evaluate
early candidates; they remain eligible even though the run must report whether
it actually completed the full training inventory. Retain real optimizer and
model state where resumability is claimed. A disposable systems pilot must
demonstrate finite gradients, actual BF16 changes in every intended block,
fixed teacher/geometry and a fresh selected-state reload. Scientific training
starts anew from the declared initializer, never pilot-mutated weights.

Do not automatically repeat three or twelve epochs. First complete this full
pass and measure useful generation. Allocate additional training based on that
evidence, distinguishing inadequate coverage, execution errors, failure to
terminate and inability to represent the task.

## Selection and evaluation

Use paired native/student free generation, with the same prompt, template and
128-output-token cap within the validated416-token total envelope. Preserve
the existing six deterministic canaries and16 balanced exposed DEV generation
cases. Report a broader exposed DEV panel when promoting a candidate; these
small screens cannot establish population noninferiority or general accuracy.

Task completion, instruction compliance, factual/numerical consistency,
coherence and natural termination determine usable generation. Report correct
partial content separately from clean completed answers. No repetition penalty,
stop-token trick, answer trimming or scoring-rule relaxation may masquerade as
recovered model behavior. Native truncation/failures remain visible.

Compare semantic answers by meaning, not wording. Use existing blinded judge
provenance and executable/structured checks where applicable. The exposed
MC256 screen is a retention constraint (at most5 percentage points overall
loss,10 points within a family), not the primary ranking objective. Generation
eligibility precedes ranking; among eligible candidates prioritize useful
completed tasks, then generation reliability, then response DEV loss, with
earlier checkpoints winning ties. If none qualify, preserve the best observed
candidate as exploratory and explicitly unqualified. Missing judging is
unavailable evidence, not an inferred quality failure or success.

## Stage B if reference-prefix recovery is insufficient

The next targeted comparison uses short student-generated TRAIN continuations,
with the original frozen teacher scoring those exact token histories. Start
with256 balanced TRAIN prompts and a32-token rollout cap, at most8,192 rollout
tokens. Bind each round to its generating checkpoint and refresh after updates;
stale trajectories are explicitly offline. Supervise all generated positions,
including real EOS when present. A capped rollout never receives an invented
EOS target. Preserve repetitions in the loss; budget adaptation applies to
future rollouts only. Freeze the precise round objective, refresh interval,
allocation and admission rules in a separate implementation receipt before
launch. Stage B is independently runnable and is not claimed implemented by
the Stage A worker.

This direction is motivated by distribution mismatch in
[GKD, ICLR2024](https://arxiv.org/abs/2306.13649), and generation collapse after
compression in [ShortOPD v2](https://arxiv.org/abs/2607.13124v2). Their findings
motivate our experiment; they do not validate these fitted Gröbner subspaces.

## Resources and continuation

Prefer the plain RTX2060 under a separate3840MiB reservation and3584MiB
allocator cap, alongside the existing2048MiB QR reservation if the real
concurrency pilot fits. Cached-boundary training requires no resident native
prefix. Generation may reuse the existing exact layer-streaming prefix mode:
CPU stores the original weights and CUDA performs the forward computation.
Declare packed Torch execution on this card and include transfer costs in
timing. This profile makes no inference-speed claim. Reject memory overrun;
do not stop another experiment or silently change precision to make it fit.

Use locked/offline Pixi and inference-ctl for every GPU launch. Root owns
preparation and launch. Source/data/initializer/diagnostic fingerprints and
full progress counters must be saved. Current progressive compression still
targets36 blocks, and ABLATION32 remains the separate paired recipe. A recovery
recipe that improves this fixed-depth usability test can then be applied to
the deeper saved models with explicit lineage, without erasing either run.
