# Audit source overlap in the dense NCM development evaluation

Date: 2026-10-23. Exploratory data audit
`ncm-dense-semantic-source-overlap-v1`. During inspection after all128 first
Terra judgments, the single passing long-source prediction exactly matched a
TRAIN summary. Its source was contained in a training passage two characters
longer, under session IDs sharegpt_uD7CBng_2 and sharegpt_uD7CBng_4. Original
metadata checks missed this overlap. This audit is specified after that finding
and cannot establish an untouched confirmatory estimate.

Let the fixed semantic supervisor finish its CPU audit and registration
unchanged. Preserve its original128 outputs, every first judgment, denominator,
quality-failure classification and artifacts. Then audit all690 approved TRAIN
sources against all64 short and64 long development sources, plus all pairs
within development. No new model outputs, judge calls, labels or exclusions.
Never open or hash either sealed final source file.

In addition to existing identity/group/source-hash checks, join observed
sharegpt session variants with trailing numeric suffixes; hexadecimal8-character
session roots and variants; and answer hexadecimal roots including abs/numeric
variants. Preserve full ultrachat numeric IDs. These are conservative identifier
families, not a proof that all arbitrary identifiers expose conversation roots.
Also flag shared public full-turn hashes, equality after NFKC/case folding and
Unicode word tokenization, and at least32 shared contiguous word5grams covering
at least80% of the shorter source's unique5grams. Save every conflicting pair
and reason. Derive all counts from bound source bytes and raw saved scores.

Report each original cohort's raw passes, overlapping source count and exact
TRAIN-target predictions, including passing overlaps. Keep original accuracy
counts unchanged and explicitly invalidate their use as clean generalization
evidence when overlap exists. Do not drop conflicts and recalculate a favorable
rate. Classify `development_split_contaminated` or `no_overlap_detected_under_fixed_checks`.

Run focused CPU tests, freeze source/tests/plan, then execute a CPU supervisor
with actual audit and registry child exits. Use fresh roots
`20261023_dense_semantic_source_overlap_v1` and
`20261023_dense_semantic_source_overlap_workflow_v1`; write a separate analysis
amendment and conservative registry record. Completed semantic files stay
unchanged. Future evaluation must use fresh sources passing these checks before
generation. Label preparation may retain all690 approved TRAIN cases, but
cannot treat the contaminated development population as independently held out.
