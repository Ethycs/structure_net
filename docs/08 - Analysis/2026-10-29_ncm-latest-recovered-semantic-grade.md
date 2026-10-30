# Latest NCM summary accuracy remains 0%; remaining grading stopped

The latest recovered-data checkpoint scored **0/128 complete-summary semantic
passes (0.0%)** with the improved standalone codec readout. All 128 judgments
were valid, and an independent audit reproduced the result from saved first
responses. This is source-disjoint development accuracy; the final 320 cases
remain sealed. The 95% quality goal is unmet, and no inference speedup has been
established.

The user explicitly stopped the remaining original-readout comparison. At stop,
that arm had **0/113 valid judged cases**, with 15 of 128 still ungraded. Its full
grade and independent audit did not complete. This partial observation must not
be reported as 0/128 or as an audited full comparison.

| Evaluation | Valid judged cases | Complete summaries passing | Status |
| --- | ---: | ---: | --- |
| Improved codec, short sources | 64/64 | 0/64 | Complete, independently audited |
| Improved codec, long sources | 64/64 | 0/64 | Complete, independently audited |
| Improved codec, combined | 128/128 | 0/128 | **0.0% DEV accuracy** |
| Original NCM readout with source attention | 113/128 | 0/113 judged | Intentionally stopped; partial |

The judge was `codex_sdk/gpt-5.6-terra`. A complete summary must be faithful,
salient and readable, nonempty, and within the 128 stored-token cap. Paraphrases
are allowed; exact reference wording is not required. For the completed arm,
faithfulness passed on 1/128, salience on 1/128, and readability on 0/128. No
outputs were empty or over the cap. Sol was not admitted because neither cohort
reached the fixed Terra gate. The valid responses support a model-quality
failure rather than an unavailable-judge explanation.

The selected checkpoint is `step6416.pt` from the recovered paraphrase run,
trained with 43,894 recovered pairs and 2,384 anchors. Both evaluated readouts
use this same source-conditioned predictor; the original-readout arm is not the
original full Llama model. Generation used source features only. All 128 code
sequences match between arms, while their final text matches on 12/128 cases.
Those 12 exact text matches reuse the preserved judgments. The partial arm also
contains four pilot judgments and 97 fresh grade rows.

The qualitative failures are substantial. A source about neurological symptoms
of celiac disease produced an invented person and age followed by repeated
therapy and anxiety fragments. A volleyball reminder became repetitive snack
and breakfast fragments. Small stylistic changes between readouts did not yield
coherent summaries.

The [October 25 completed evaluation](2026-10-25_ncm-broad-generator.md) also
scored 0/128 under the complete-summary rubric. Thus the four-day interval has
produced no measured improvement on the task endpoint. Codec reconstruction,
data preparation, and evaluation infrastructure have improved in separate
diagnostics, but these do not establish usable source-conditioned summaries.
The two training configurations differ, so this is a lack of demonstrated task
progress, not a controlled attribution to any single change.

The decision is to stop scaling this configuration. The evidence suggests an
unresolved source-to-code learning and generation problem, with error
accumulation a plausible contributor; it does not identify a single proven
cause or invalidate the published NCM architecture. Any future restart should
begin with a bounded test showing reliable source-only generation on a tiny,
clean training set, then a separate held-out set. No such restart is launched
by this report.

The grading supervisor, semantic worker, and pending independent auditor were
stopped. Actual terminal receipts record the RPC client exiting 1 after worker
termination and worker/auditor exiting 143. These are deliberate cancellation
exits, not new endpoint failures. The shared judge service remains running.
Fifteen request directories have no durable row. Whether any corresponding call
was in flight is retained as uncertain; these directories must not be assumed
unsent or silently retried with reset attempt budgets.

Evidence is preserved under `data/experiments/ncm_paper_regime/`:

- Completed grade and independent audit:
  `20261029_latest_recovered_new_codec_only_DEV_{grade,audit}_v1`.
- Partial original-readout grade:
  `20261029_latest_recovered_parent_with_source_DEV_grade_v1`.
- Stop intent, exact process identities, actual tool exits, per-case partial
  snapshot and SHA-256 preservation manifest:
  `20261029_latest_recovered_user_stop_v1`.
- Terminal handoff:
  `20261029_handoff_v238_latest_recovered_user_stop.json`.

The completed independent audit receipt has SHA-256
`a8acdafe7224e4a629e10c18173971f905e7fdc0fb11c0ec1ce60ffeed21e212`.
The trained checkpoint has SHA-256
`edc88690abaa22b29989be5042269d86f011483a76c8f7b473e4e21b41e30799`.
