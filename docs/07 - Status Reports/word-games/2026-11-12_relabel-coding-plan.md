# Qwen coding with consistent labels

Date: 2026-11-12. Prospective exploratory study:
`word-games-isomorphic-label-coding-v1`.

Test the user's proposed configuration: the same computation with different
domain labels. Stronger Terra controller interprets, Qwen translates labels,
Qwen executes in the existing bubblewrapped OpenCode harness. A matched direct
arm uses the same initial controller instruction and unrenamed data. Alternate
arm order; fresh workspaces/sessions per arm; at most two controller clarification
revisions per arm with same-session actor resumption. No grading feedback or
answer repair. Stop the campaign for transport/route/producer failures. Count
context overflow, length termination, bad model protocol, unresolved questions
and incorrect code as failures.

Freeze the eight existing task definitions, original four inputs each, plus 32
generated inputs per task (seed 731112 plus task number): 36 tests per task,
288 tests per arm. Extra inputs exercise empty collections, equal deadlines,
title ties/case, missing and mismatched permits, empty notes, mixed-case/empty
queries, category names appearing as literal text, rule minima and event selection.
Only the old example input is visible to actors. Terra sees the original goal
and schema/types, never concrete examples, reference code or grading answers.
Generate/hash all hidden data before the first campaign call. These are new inputs
for familiar tasks, not new held-out task families.

Use separate typed tables for fields, category values and identifier prefixes.
Rename category values only in category-bearing fields and category-count output
keys. Change ID prefixes consistently, preserving suffixes and within-namespace
lexical order. Preserve numbers, dates, nulls, booleans and free text exactly.
This intentionally does not hide text contents: it is not a secrecy test or the
old opaque privacy layer. No host-computed match flags or sorting ranks may supply
the answers. The target oracle is independently expressed using target field
names. Verify input round trips and decoded target/source oracle equality on all
288 cases before model inference.

Both arms use the same compact Python function contract, without the irrelevant
reservation-transaction manual. The Qwen translator receives the source controller
instruction plus separate typed tables and target schema; no untyped regex or
opaque literal replacement. Its grammar has disjoint instruction/request branches
and host validation rejects mixed responses. These changes are part of the new
configuration; comparison with older runs is not a single-factor causal estimate.

Keep Qwen's 8K context, 1,024 output tokens and 12 steps per invocation. Require
the final actor finish reason to be `stop`, no transport/errors, and actual passing
code for primary success. Report artifact-only correctness separately. Primary:
tasks passing all 36 tests, out of eight. Secondary: tests passed out of 288,
original cases out of 32, clarification/resource outcomes and task latency.
The existing network-isolated grader executes code and checks exact decoded
results and input preservation. Median latency includes failed tasks and charges
the common initial controller call to both arms; it is not model tokens/s.

Use one development-only grammar/route check if necessary. Preserve failed
attempts, producer/input hashes, raw calls/events and independently audited grades
in new artifact roots. No DataAnnotation material or production service changes.
Do not claim general coding accuracy, a privacy guarantee, or an isolated benefit
of label substitution from one eight-task configuration.
