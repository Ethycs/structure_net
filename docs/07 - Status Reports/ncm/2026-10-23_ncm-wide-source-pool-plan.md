# Prepare public sources covering the application's longer fragments

Date: 2026-10-23. Exploratory CPU data preparation only. The current short-source
training remains fixed; no checkpoint quality has been inspected. The input
audit found a maximum of 444 cl100k tokens in summary training and 437 in model
development, versus the application's 2,048-token fragment ceiling.

First audit availability in the original full `train.jsonl`, not the previously
filtered native subset. Verify its original manifest hash and the canonical
public LongMemEval file hash. Admit only exact public spans. Exclude all IDs,
sessions, groups and source hashes from the original 2,048 training sources,
8,192 expanded candidates, their pilot, the older selected cohorts, and current
development. Never export or transmit old saved targets. Do not read original
sealed final sources or the original test JSONL.

Use four source-length bins: 512–767, 768–1023, 1024–1535 and 1536–2048 cl100k
tokens. Audit all eligible public rows and unique groups before choosing sample
counts. Counts for the later unlabeled pool are explicit command parameters,
fixed before selection and before any generation or grading on that pool.
Availability can determine a feasible dataset size; it cannot change a model
quality threshold. The pool is a new operational holdout inside the original
training partition, not an untouched original benchmark partition.

Select deterministically by SHA256 with a fixed namespace. Process bins from
longest to shortest, reserving sealed-final groups first, then diagnostic and
training groups. Use one source per group, with no repeated session/hash/ID,
across all splits. Stop and preserve an insufficient-availability result if any
requested quota cannot be met; never reduce quotas silently. Selected complete
sources must reproduce their stored cl100k lengths and public-span proofs.
Record native chat-prompt lengths without truncation or a short-source filter.
Export only source/provenance fields and a split designation, never old targets.

Roots `20261023_application_source_availability_v1` and
`20261023_application_source_pool_v1`. Preserve strict JSON, producer/config/input
hashes and source snapshots. New sealed texts receive no model outputs, teacher
labels or manual source inspection before the development admission gate.
Pool preparation does not admit a new training run or establish 95% accuracy,
raw-batch/hierarchy quality, compression or speed. Those need separate protocols
and the original final 256 cases remain sealed.
