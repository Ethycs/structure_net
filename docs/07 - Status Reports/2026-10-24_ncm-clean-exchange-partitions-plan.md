# Reserve full evaluation cohorts from complete public exchanges

Date: 2026-10-24. Prospective data hypothesis
`ncm-clean-exchange-source-partitions-v2`. The registered whole-turn-only
construction failed: each of the two longest bands had16 clean components,
below the32 required across development and final. Preserve that failure and
all original artifacts. This revision changes input-fragment construction;
the full128 development and320 final sizes, input ranges and95% task target
remain unchanged. No model outputs or labels select sources.

Use the same complete500-sample public LongMemEval-S release with SHA256
d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442.
Keep the2828 known training/development sources and320 old final identities
excluded, using the existing metadata-only recovery. Do not open or hash the
two original sealed source files. Require the registered original allocation
failure archive733f237ccf38130e6fc3d735f96401ffd3781b76493f13d9c07444035d4f803f.

Candidates include complete public turns of192–2048 cl100k tokens and bounded
fragments made from contiguous complete user-led exchanges. A fragment starts
at the session start or a user turn, and ends before the next user turn or at
session end. Concatenate literal complete turn contents with one newline.
For each ordered public-session variant, keep the first window found in each
512–767,768–1023,1024–1535 and1536–2048 band, traversing starts and ends in
ascending order. Empty constituent turns are format-ineligible. Stop extending
a window once it exceeds2048 tokens; never cut a turn or a generated output.
All public roles remain eligible. Whole turns above2048 remain grouping
bridges but cannot be selected. Deduplicate exact source text while retaining
every public origin and family alias.

Build a complete public full-turn token inventory. Source-family and old-group
aliases remain unchanged. Shared full-turn-hash edges apply only to turns of
at least192 tokens: generic bypass greetings must not join unrelated families.
Apply this same rule to known-source and candidate graph views and direct
selected-source checks. Retain every literal word and original provenance span
in actual sources and exports. Full source text always receives NFKC/case/word
normalization and exact5gram checks (at least32 shared unique5grams and80% of
the shorter set). This changes only uninformative shared-turn-hash aliases,
not content or the complete-text overlap thresholds.

Join these relations into complete connected components before selection.
Use exact dictionary IDs for normalized words and5grams; postings compute
exact intersections while retaining only earlier gram-set sizes. Already
connected pairs need no additional edge. Tests must match the reference
component partition across varied ordering, overlap and transitive blocking.
This memory optimization introduces no probabilistic hash comparison.

Keep one case per unblocked component. Use the unchanged deterministic quota
matcher and namespace `ncm-clean-exchange-source-partitions-v2`. Reserve256
final and64 development short sources in192–511 tokens, plus64 final and64
development long sources,16 per long band in each cohort. Never shrink quotas.
Choose rows, final/development assignment and public origins by their fixed
SHA256 ordering. Before sealing, verify every complete literal public span,
all direct known-source comparisons and every pair across the448 selected
cases under the same guards. No quality filtering or outcome selection.

Use CPU only with CUDA hidden and a24GiB peak-RSS guard. Run focused tests,
actual-data/admission checks, a producer-capture shakedown and a two-public-sample
lifecycle pilot before full preparation. The pilot uses the complete public
inventory and all exclusion metadata, exports up to one fresh source per band,
and requires exact provenance and JSON readback. Full preparation starts fresh
with all500 samples. Freeze source, tests, plan and auditor before launch.

Record all public origins, complete session/turn metadata, component membership,
selection and final file creation hashes/stats. Seal the two new final exports
after creation. Independent CPU finalization replays metadata, literal span and
exchange boundaries, deterministic full quotas, development bytes and final
stat receipts; it never reopens or rehashes sealed content. Keep final paths
outside generic artifact-rehash lists. Exact full-text/provenance checks ran
before sealing and need independent repetition when final use becomes eligible.

One supervisor owns pilot, full preparation, audit and conservative registry
read-back with actual child exits. Use fresh20261024 clean_exchange_partitions
workflow/pilot/main/results v2 roots. Report
`clean_exchange_sources_reserved_quality_unmeasured` only after full completion;
preserve every systems or availability failure otherwise. No student training,
labels or model accuracy follows from data reservation.

Future development still requires61/64 SAME cases passing both qualified Terra
and Sonnet in each cohort, all valid. Final still requires244/256 short and61/64
long, all valid, after fixed-candidate development and original frozen-body
integration checks. Keep all1797 canonical first judgments and every previous
failure. The95% task goal remains active; no speed claim before verified quality.
