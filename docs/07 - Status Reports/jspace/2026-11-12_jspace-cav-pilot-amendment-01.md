# J-space pilot amendment 01: finite computation and a competent parent

Date: 2026-11-12 (America/Los_Angeles). Hypothesis: `JSPACE-CAV-PILOT-01`.
Profile: exploratory follow-up. This amendment follows inspection of development
outcomes only. No evaluation outcomes, causal interventions, or rank sweeps have
been inspected. It supplements the [original plan](2026-11-12_jspace-cav-pilot-plan.md).

The first reserved campaign completed its stopping rule but exposed an invalid
numerical measurement. Its Qwen2.5-0.5B base parent scored 3/12. Its 1.5B parent
emitted token zero (`!`) for every question; a separate reserved diagnostic
confirmed nonfinite FP16 activations starting in block zero on all twelve cases.
The checkpoint weights themselves were finite. BF16 and FP32 forward passes
were finite on all twelve cases. The original 0/12 is therefore not evidence
about the 1.5B model's knowledge.

An FP32 development-only check gave 3/12 for the 0.5B base model and 5/12 for
the 1.5B base model. Removing trailing prompt whitespace changed neither score.
Neither parent passes the fixed 8/12 competence gate. These results, the original
campaign, and its inaccurate automatic classification are preserved at
`data/experiments/jspace_cav_pilot/20261113T0715Z_pilot01/`; interpretation must
include this correction rather than silently replacing its evidence.

The follow-up tests exactly one new parent:
[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Use its native BF16 precision on the
RTX 3060, with FP32 projection, accumulation, and SVD. The repository supports
this Qwen decoder without a dependency change. All hidden states, logits, and
captured observations must be finite; otherwise fail execution before scoring.
A real forward/backward/save/reload shakedown precedes the development screen.

Use the checkpoint's chat template with one user message whose content is this
fixed instruction followed by the original, unedited task prompt:

> Complete the fact below. Reply with only the missing word or number, without explanation.

Insert two newlines before the original prompt and request the assistant's
generation prefix. There are no examples, alternate prompts, answer aliases,
judge calls, or additional model candidates in this amendment. The output parser
and eight-token cap remain fixed.

Retain the exact original split manifest, 8/12 competence threshold, 48 evaluation
cases, 94-token inventory, 16 calibration paragraphs, intervention definitions,
rank sweep, control families, uncertainty calculations, and success thresholds.
The generic-text Jacobian/PCA calibration remains unchanged. Stop if this parent
fails competence; do not lower the gate. Freeze the new producer, model hashes,
prompt formatting, and dtype in a new artifact root before this parent's outcomes.

This is a new exploratory model/interface test, not a repaired replication of the
original model or a pooled result. A positive outcome still establishes only
bounded, context-conditioned intervention behavior, not autonomous compression.

Execution uses the deployed controller's automatic Infisical authentication:
`/usr/bin/python3 /usr/local/bin/inference-ctl lease run ... -- pixi run --locked ...`.
The system Python trusts the installed internal CA; the shell's separate global
Pixi Python does not. No TLS checks are disabled, caller keys are not written to
disk, and no identity permissions are changed. Lease release is verified after
each process exits.
