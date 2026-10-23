# Broader teacher labels fail the independent quality gate

Date: 2026-10-23. `ncm-expanded-valid-label-recovery-v1` is complete, classified
**label_quality_gate_failed**. Recovery preserves7,923 existing valid labels
but does not qualify them for a larger student-training run. This is teacher
label quality, not student accuracy. The NCM attachment retains its earlier
0/64 development result under Terra; no new model score or speed is established.

The fixed independent sample passes **113/128=88.2812%** against
the unchanged complete-summary faithful/salient/readable rubric and128-token
cap. The predeclared point gate is122/128. Wilson95% interval is
81.5634–92.7683%. All128 judgments
are valid; none needed a retry. First-valid raw responses, parser replay, case
order, source/label identity, tokens and every pass decision are verified.
Failure flags: `{'faithful': 15}`. The15 failed labels are readable and salient
but introduce unsupported assertions or remove material restrictions. Examples
include dropping the qualifying-purchases restriction on reward points and
identifying a speaker as an AI when the supplied passage does not establish it.
These failures are not exact-reference disagreements or decoder corruption.

The original8,192-source teacher campaign finished2,048 batches with2,045 valid,
three invalid after the existing attempt cap and36 retried batches. It keeps
its **failed_data_gate** classification. Of269 excluded labels,257 exceed the
prototype's48-GPT2-token canvas without truncation and12 have no valid response.
No missing or long label was quietly replaced. The separate recovery replayed
all raw requests/responses, preserved each first valid answer, reconstructed
accepted/excluded files exactly and copied the7,923 labels byte-for-byte. No
additional teacher request or semantic quality filtering occurred.

Audit cases are the first128 in the prospective hash order, one source/session/
group per example. A separate four-case systems pilot returned four valid
judgments and three semantic passes; four valid responses alone satisfied its
declared lifecycle requirement. The audit uses the qualified Terra model
`codex_sdk/gpt-5.6-terra` with four workers after the teacher process terminated.
The128 sample is a development data-quality gate, not independently collected
human gold, a confidence guarantee, a student result or a final task test.

Because the gate fails, do not feed these labels to a scaled student run under
this protocol. A new declared data-quality intervention must qualify its final
labels before training. Source-copying feasibility may use weak summaries only
as diagnostic semantic queries: judge its actual resulting summaries directly,
and do not treat the references as gold or the selectors as trained NCM success.
The failed codec and failed free-text generator are separate defects; imperfect
labels do not by themselves explain or excuse the student's0/64 result.

Aggregate: `data/experiments/ncm_paper_regime/20261023_expanded_label_quality_results_v1/campaign_results.json`; SHA256 `93a95e7dd076d74c2a9b310b4f31deb4a2f2e428d17d323e2f03058817b37b0e`. Parent raw responses,
failed batches, exclusions, exact recovered file, all first-valid judgments,
source/protocol seals and the authoritative worker-terminal receipt remain
append-only. Registry confirmed=false, student_accuracy_measured=false. The
final task test remains sealed and original-body weights stay unchanged.
