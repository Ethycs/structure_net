# Matched prompt-date control before further NCM retraining

Date: 2026-10-24. Hypothesis `ncm-pinned-date-full690-semantic-development-v1`.
Conformance: EXPLORATORY, one existing trained model, no new optimizer updates.
This plan precedes inspection of any date-matched development predictions.

The original full690 student produced 0/64 short and 0/64 long complete-summary
passes from Terra; all judgments were valid. Its body features used the next
calendar date because the Llama chat template inserts the current date. Explicit
`date_string="23 Oct 2026"` exactly reproduces every one of the 690 TRAIN prompt
token sequences and hashes. Switching to 24 October changes one token before the
source. A two-TRAIN-case neural control found 1.7% and 2.7% relative feature
changes but identical output words and concept codes. It does not explain the
held-out failure or remove the need to control the input.

The sole intervention is the date string. Recompute frozen original 28-layer
Llama3.2-3B BF16 features for all the SAME 64 short and 64 long development
sources. Preserve all text, alignment, spans, ordering, source frames and
source-disjoint guards against all690 training cases. Require exactly one changed
native prefix token per case, with no source-field changes. Keep all320 new final
source contents sealed and unopened. Earlier failed outputs remain unchanged.

Use the same fixed final32768 word checkpoint
`bdc462897a81e49a3dd0f16e1d48b63ebbfe078859095f274a97efb3e3f546a6`,
the same actual-history forecaster and frozen released codec/decoder, same
constructor seed57142, BF16 transformer computation, FP32 masters/readout,
TF32 disabled, nine passes and one deterministic argmax. No repairs, target
codes, autoregressive fallback, clipping, output selection or retraining.

Before all128 inference: a two-case representative feature pilot must repeat
exactly; full caching must retain those features. A two-case inference pilot must
survive a fresh model reload and reproduce both registered TRAIN constructor
probes exactly. Full inference must retain its pilot outputs exactly. Body and
head processes run serially on physical GPU0, with 8.6GB/8GB peak limits.

Reuse the unchanged qualified source-grounded judges and 128-storage-token
rubric. A four-case schema-only pilot retains its first decisions even if all
fail. Grade every case with Terra; admit Sonnet within a cohort only after
Terra's unchanged61/64 and64/64-valid gate. Success requires at least61 of the
SAME64 cases to pass both judges in each cohort with no invalids. This is semantic
faithfulness, salience and readability, not exact reference-summary matching.
The full final95% target remains unproven even if this development control passes.

Inherit ALL1925 canonical first judgments from
`20261024_clean_semantic_dev_grade_v1/judgment_cache.json`, SHA256
`5c094eb72ca43bb87c459770b89bcb739d91efed625eeaa679e87b69ec636961`.
Identical packets reuse those decisions; valid negatives and exhausted invalids
are never rerolled. At most three transport/schema attempts per new packet,
four judge workers. No changes to prompts, cases or weights after outcomes.

Stop on any systems/provenance failure; preserve the full failure root. A quality
miss is a completed negative result. Record all receipts, raw predictions,
packets and judgments, audit the aggregate, register conservatively and read back
the exact hypothesis. No automatic final-source admission or speed claim.
Use fresh roots under `data/experiments/ncm_paper_regime/20261024_pinned_date_*_v1`.
Entry point: `python -m experiments.structure_net.ncm_pinned_date_development workflow`.
