# Full source-only DEV of the FP16 level-conditioning candidate

2026-10-24. Hypothesis `ncm-level-conditioning-native655-source-only-DEV-v1`.
Exploratory follow-up. The training workflow is live; no candidate held-out
output has been inspected. The previous full-native655 candidate failed0/128
under Terra. The new fixed1024 candidate changes precision, level corruption,
microbatch structure and seeds, so this study cannot attribute a difference
solely to noise. It cannot establish final95% accuracy or speed.

Use the entire final checkpoint from
`20261024_level_conditioning_FP16_native655_training_train_v1`, only after
actual whole-tool exit0 and independent CPU trace/model/optimizer/scaler/RNG
audit exit0. Reject pilot/native16/old-native655 states. Enforce all655 original
unchanged TRAIN targets,1024 updates,127732736 trainable parameters, original
warm initialization, frozen encoder/VQ and parent Llama body/MLPs. Strict-load
the complete model and compare native generator, decoder, attention and
protected-state hashes before freezing it for inference.

Preserve the pinned date23Oct2026 full source metadata (SHA1395a35a...) and
FP32 full-source feature manifest (SHA637be63f...). All64 short and64 long
source families must remain disjoint from the655 TRAIN families. No target
words, target length, target code, output repair, restricted vocabulary,
candidate selection or retry based on quality enters inference. Predict128
raw word positions and all641 native codes. Render with the existing native
CLS/EOS policy and retain empty, overbudget, malformed or poor outputs.

GPU execution is physical2/logical0, FP32 with TF32 disabled. First run the
actual first/longest-source pilot, generate each twice, check equality and the
complete model hash, then generate every128 source. Pilot outputs must match
the corresponding full outputs exactly. Require5GB free before inference;
do not contend with the live training workflow. CPU stages have no GPU.

Inherit all3350 canonical judgments from the fully registered Sol-teacher64
campaign, including historical Sonnet identities and negative first outcomes.
Use the existing model-aware raw-request/first-valid replay. The already
prospectively qualified replacement primary is `codex_sdk/gpt-5.6-sol`;
secondary is `codex_sdk/gpt-5.6-terra`. Both must retain their actual50/50
qualification proofs. The reviewer-switch amendment predates candidate DEV
inspection. Do not relabel historical entries, branch from3222 or3094, or
reroll a valid negative.

Run two secondary validity-only pilot cases per cohort, retain those firsts,
then grade all128. Admit primary only if each cohort has64 valid and at least
61 passing secondary judgments. Qualify DEV only if at least61 of the SAME64
cases in EACH cohort pass both judges and all judgments are valid. These are
whole-summary faithfulness, salience, readability, nonempty and<=128 storage
token checks, accepting semantic paraphrases. Missing primary measurements
remain null, not fabricated zero scores. Final320 sources stay unopened and
unhashed; a passing DEV would require a separately frozen final protocol.

Fresh roots use `20261024_level_conditioning_native655_development_*_v1`.
Freeze new wrapper, imported inference code, full checkpoint, new training
audit, lineage, rubric and judge qualification inputs. CPU tests must reject
wrong training precision, scaler, pilot steps and target-bearing inputs, and
check strict full-state loading and current canonical lineage. Independent
CPU finalization must replay outputs, packet construction, every first score,
same-case admission and all artifacts, then conservatively register and verify
backend read-back. An actual whole-workflow exit receipt remains required.
