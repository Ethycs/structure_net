# NCM paper-regime controls: adaptation stopped before generator training

Date: 2026-10-22. Hypothesis `ncm-paper-regime-recovery-v1`. Classification:
`stopped_before_generator_adaptation`. Profile EXPLORATORY, one model seed.
Aggregate: `data/experiments/ncm_paper_regime/20261022_stopped_v1/campaign_results.json`;
SHA256 `fce8287603c5bbfec9c3732f08af0501bf546203efd61b566699629b894e1f0e`.

The released NCM scored **0/64 complete summaries (0%)** on the new conversation
development cohort under the qualified secondary judge. All64 judgments were
valid, with no retries. Wilson95% interval0–5.66%, excluding judge error. This is
a development baseline, not the accuracy of a newly adapted generator. The new
full generator was not trained because both competence gates failed.

| Control | Measured result | Gate |
| --- | --- | --- |
| New teacher labels fitting native canvas |1,999/2,048;49excluded without truncation|Passed minimum1,792|
| Independent teacher-label audit |30/32 complete-summary passes,0invalid|Failed required31/32|
| Released-codec non-PAD reconstruction |67,112/69,682=96.3118%|Failed required98%|
| Released-codec exact decoded texts |676/1,999|Diagnostic only|
| Full FP32 codec diagnostic |67,121/69,682=96.3247%|Does not explain away codec failure|
| GPU generator lifecycle |Unpaired, bridge-only and joint updates; exact reload; codec frozen|Passed systems only|
| Frozen Llama feature extraction |64cases; unchanged body digest|Passed systems only|

The 2,048 training passages,64development passages and256sealed final-test
passages are disjoint by source group, session and source hash. Development and
test selection excluded earlier quality cohorts and calibration sessions; one
group per development/test example. Teacher requests used only source spans
reconstructed from the SHA-verified public LongMemEval release. All512teacher
batches yielded valid JSON;7needed transport/schema retries. The independent
audit retained both valid negative judgments: one unsupported stage-management
role and one overstatement of qualified festival descriptions. No failed label
was retried for a better semantic verdict.

The precision diagnostic fixed the target codes in FP32 and crossed latent and
decoder precision. It exactly reproduced the original BF16/BF16 integer counts.
FP32/FP32 recovered67,121tokens, FP32/BF16 67,111, BF16/FP32 67,122, and
BF16/BF16 67,112. Mixed precision changes a few tokens; it is not the main
reason reconstruction is below the gate. These are token reconstruction rates,
not semantic accuracy. No change of precision rescues the original failed gate.

The implementation adapts the paper's tokenizer-first, frozen-codec,
unpaired-then-conditional staging. It adds a frozen Llama source bridge and
corrects left-padded context placement for NCM. The numerical configuration and
boundaries are in the [plan](../07%20-%20Status%20Reports/2026-10-22_ncm-paper-regime-plan.md)
and its training addendum. Source/tests were frozen in commits ce5142e and
4791fc4; raw producers also retain source copies and hashes. The released
checkpoint's nine-level schedule is retained despite the paper table listing
eight. The authors' XSum result is not a promise of95% task accuracy here.
[Paper](https://arxiv.org/html/2610.14162v1).

The v1 study is closed at its failed controls. The user's95% objective remains
active in a separately declared [v2 study](../07%20-%20Status%20Reports/2026-10-22_ncm-paper-recovery-v2-plan.md):
train the actual codec encoder/EMA books/resampling/decoder with nested dropout,
and screen the paired labels before conditional training. No v2 quality result
is included in this aggregate. The final256sources remain sealed. There is no
new speed claim, human accuracy claim, or full application input-range result;
the native prefix/canvas are512/48GPT-2tokens.

Reproduce with the moba environment Python, offline HF cache and
`LD_LIBRARY_PATH='' OMP_NUM_THREADS=2 PYTHONPATH=src:.`; select physical GPU0
before CUDA. Add memory dependencies and tiktoken cache for grading. Entry
points: `ncm_paper_data`, `ncm_paper_controls`, `ncm_paper_features`,
`ncm_paper_train`, `ncm_paper_grade`, and `ncm_codec_precision` under
`experiments.structure_net`. Pinned roots are declared in source; changed
protocols require new roots. The registry retains confirmed=false.
