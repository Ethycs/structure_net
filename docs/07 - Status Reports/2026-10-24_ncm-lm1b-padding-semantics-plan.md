# Native NCM padding and semantic reconstruction qualification

Date: 2026-10-24. Hypothesis `ncm-lm1b-CLS-padding-semantic-control-v1`.
Conformance: EXPLORATORY target-code representation study, no model training.
This plan fixes selection and judgments before inspecting any CLS-tail outputs.

Known: the corrected native LM1B codec reconstructs63872/65063 tokens on all690
existing TRAIN targets,271 exact native sequences,684 correct window boundaries,
and21 outputs above the128-cl100k storage cap. The group-invariant dictionary
shortcut is worse and remains a closed negative result. The source-only frozen
student still passes0/128 date-matched development summaries. None of the native
codec outputs has yet received a source-grounded semantic judgment.

The authors'71825 released native frames all begin/end with CLS101 and contain
zero PAD0. Our variable-length application frames use PAD0 after their genuine
CLS end. Compare exactly one change: replace only these tail PAD0 positions with
CLS101. Keep every lexical token, genuine endpoint, target, source, learned
weight,128-position frame, native level, precision and raw argmax unchanged.
Use the same predicted-BOS/next-predicted-CLS parser; true lengths enter scoring
only. Preserve all690 targets, including all25 overlength targets and715 complete
windows. Multi-window reconstruction is not a single-frame generation claim.

Require repeated first/longest numerical pilot outputs and a fresh full690 load
retaining both. FP32, TF32 off, GPU2, one worker,4GB free/4.7GB peak. Cache the
known vocabulary size for validation. Keep all prior PAD0 and failed readout
artifacts unchanged. Numeric token fidelity is secondary to semantic quality.

Fix64 semantic TRAIN cases before generation: choose32 from the original252 and
32 from the reviewed438 by SHA256 seeds64142/65142, independent of outputs.
Compare four anonymous single-candidate arms against each complete original
public source: uncompressed approved target, unquantized BERT text roundtrip,
learned codec with PAD0, and learned codec with CLS tail. Their role is to
separate target quality, text normalization and neural reconstruction errors.
No exact reference-summary match determines semantic correctness.

Use the unchanged qualified Terra and Sonnet judges, each50/50 on its controls,
and the source-grounded faithful/salient/readable, nonempty,<=128-token rubric.
Run a four-case-per-arm schema-only pilot; valid negatives pass this systems
gate and remain first decisions. Grade all64 in every arm with Terra. Admit
Sonnet for an arm only if the uncompressed target and that arm each have at
least61/64 Terra passes and all64 decisions valid. Qualification requires at
least61 of the SAME64 cases to pass both judges, with all decisions valid, for
the uncompressed reference and the codec arm. Record BERT control independently.
An invalid result cannot silently shrink a denominator. Four concurrent judge
workers, at most three transport/schema attempts, no semantic rerolls.

Inherit all2051 canonical first decisions from the completed date control,
SHA256 `ea07f76887bd15c07ffaea8fc70339fee8ffa41731c6ce5572916dea51b2c4a4`.
Identical packets share one first decision across arms. Preserve every pilot,
valid failure and exhausted invalid. Use one serial cache lineage throughout;
never revert to an earlier1925/1797 cache.

Stop and preserve any systems/provenance failure; a valid quality miss is a
completed negative result. Independently replay all raw counts, packets and
first judgments, report and register conservatively with exact readback.
These are target-aware TRAIN controls, not source-only model accuracy. No
95% task success, speed gain or final-source access is admitted. Original Llama
and its MLPs remain frozen and absent. The next conditional training decision
must follow semantic findings rather than token agreement alone.
