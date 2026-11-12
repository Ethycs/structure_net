# WORD_GAMES: coding utility and information recovery

Status: completed exploratory pilot, independently audited. Study:
`word-games-structured-proof-coding-recovery-v1`.

**The bounded reservation adapter passed its structural check, but the coding
adapter did not preserve task performance: native Qwen passed 8/8 tasks, versus
3/8 through the translated view.** The translated packets hid the tested literal
titles and canaries from the recovery model. That is a narrower result than
information secrecy: most schema field names remain unchanged, and some renamed
fields were inferred correctly.

All tasks were newly authored synthetic exercises in this repository, based on
the supplied public-style reservation fixture. No DataAnnotation material was
used or transmitted. This is **not SWE-bench**, a real-world repository-repair
benchmark, or evidence that the reservation HTTP API now supports arbitrary code.

## What ran

Eight frozen Python-function tasks, four hidden test inputs each, with paired
native and translated arms. Both used the installed OpenCode 1.19.5 agent loop,
the existing Qwen3.8-35B-A3B LiteLLM/Triton route, fresh bubblewrapped workspaces,
8,192-token context, 1,024-token model output limit, 12 agent steps and disabled
automatic compaction. Arm order alternated by task. No model answer was repaired
using grading feedback or retried after failure.

The native actor received source-domain task/data. The translated actor received
a Qwen-translated goal, the actual frozen mapping's target manual and data encoded
with the deployed `Privacy` plus `Mapping` components. It received no source names,
inverse alias table, expected code or hidden tests. The actor wrote `solution.py`.
A separate **network-isolated** bubblewrap executed `solve(data)` against hidden
inputs, with CPU/memory/time bounds. The grader checked exact decoded JSON values
and input preservation. Expected answers stayed host-side.

The coding adapter is new evaluation glue around existing components. The source
Engine and deployed bridge service were not changed. The full double-pass bridge
integration remains unfinished.

## Coding results

| Task | Native hidden tests | Translated hidden tests | Translated outcome |
|---|---:|---:|---|
| Find free slots | 4/4 | 0/4 | Context overflow; artifact also failed |
| Earliest reservation deadline | 4/4 | 4/4 | Pass |
| Count items by category | 4/4 | 4/4 | Pass |
| Minimum inherited deadline | 4/4 | 0/4 | Context overflow; written code separately passed 4/4 |
| Case-insensitive note substring | 4/4 | 1/4 | Notes/query were opaque; only empty-data case passed |
| Alphabetical title ordering | 4/4 | 1/4 | Sorted opaque handles; only empty-data case passed |
| Detect missing/mismatched approvals | 4/4 | 0/4 | Context overflow; artifact also failed |
| Newest item in a specified event | 4/4 | 4/4 | Pass |
| **Total** | **32/32** | **14/32** | **8/8 versus 3/8 complete tasks** |

Complete-task rates in this pilot: **100% native versus 37.5% translated**.
Hidden-test rates: 100% versus 43.75%. These are observed fractions on eight
related synthetic tasks, not population accuracy estimates.

Three translated sessions overflowed the fixed context; no native session did.
If session completion is ignored and only written code is graded, translated
artifacts passed 4/8 tasks and 18/32 tests. That secondary result does not repair
the end-to-end failures. The larger translated packet and tool history exposed
a real context-budget limitation of this configuration.

## Information recovery

Fresh Qwen calls attacked each receiver packet. Blind recovery had no original
schema vocabulary and attempted to recover the original domain, exact original
titles and a synthetic irrelevant private canary. A separate, stronger attack
received the public source vocabulary and target key names, but not the mapping,
and tried to invert the eight genuinely renamed keys.

| Blind recovery input | Original domain recovered | Exact titles recovered | Exact canaries recovered |
|---|---:|---:|---:|
| No view | 0/8 | 0/48 | 0/8 |
| Raw source view, positive control | 8/8 | 48/48 | 8/8 |
| Translated receiver view | 0/8 | 0/48 | 0/8 |

The blind model labelled all eight translated packets as the displayed airline
domain, despite being asked for the original domain. This is observed
misclassification by one recovery model, not proof that the source domain is
unidentifiable.

The literal scan agreed: the raw Mapping codec retained all 48 title occurrences
and all eight canaries; the Privacy+Mapping receiver packets contained none.
Thus literal concealment came from the privacy layer, not the metaphorical
renaming. The inverse with the private map and aliases recovered the original
example exactly in **8/8** cases, as intended.

The source-aware attacker recovered **24/64 renamed-key guesses (37.5%)**:
`locations → galleries`, `record_id → id`, and
`target_location_ref → to_gallery_id` in every task. Separately, **44/52 schema
field names are unchanged by the actual mapping**. Do not interpret literal
name/canary non-recovery as hiding the task's operational structure or every
source property.

These controls repeat one source domain, six fixture titles and one model, with
fresh contexts and per-task canaries. They are not independent samples from a
general adversarial population. No source implementation was supplied, so this
is not a source-code reconstruction benchmark. Failure of this recovery model
does not prove that stronger attacks cannot infer more.

## A constructive limit, independent of model quality

A secondary deterministic witness constructs two valid source worlds. In one,
Bronze Heron's unique note contains "urgent"; in the other it contains "routine".
Their private alias dictionaries assign that note the same opaque handle, with
all other visible equality patterns unchanged. The complete encoded input is
identical, but the correct substring-search answer differs by `loan:bronze`.

No receiver given only that identical packet can always return both different
correct answers. A stronger model cannot recover information that the public
representation omits. Opaque equality handles preserve identity; they do not
preserve substring predicates or lexical order. This explains the observed
note-search and title-ordering failures and is separate from context overflow.

The next useful change is a task-specific projection that releases authorized
derived facts such as a query-match flag or ordering relation, while retaining
opaque names where the names themselves are irrelevant. That requires an explicit
task contract and new paired tests; it is not established by this run. Likewise,
reducing unrelated manual/data context or validating standard harness compaction
is preferable to treating unfinished sessions as successes.

## Structured proof and its boundary

The companion [bounded commutation report](2026-11-11_structured-commutation-proof.md)
records **85,176 exhaustive cases, zero mismatches**, 3,457 commits, 81,719
rejections and one replay per case. All three symbol tables are total and
injective. A corrupted decoder was detected. The proof covers the declared
states/contexts and atomic words of length at most two through the actual adapter
and authoritative Engine. It does not prove arbitrary JSON invertibility,
natural-language equivalence, general code equivalence or confidentiality.

The unrestricted codec already has counterexamples: it drops legacy `target_id`,
and `annotate` and `maintenance_note` both encode to `maintenance_note` outside
the typed domain. Those failures are preserved, not silently repaired.

## Attempts, evidence and reproduction

V1 stopped with 0 complete pairs because single-line JSON was truncated by
OpenCode's file tool, hiding the goal. V2 used multiline JSON, passed native c01,
then stopped on translated context overflow. Both attempts, source hashes and
partial producer snapshots are retained. An absolute-path recording defect meant
the v1/v2 snapshots omitted the runner itself; v3 corrected that defect and its
complete frozen producer snapshot passed the audit. The recorded v3 amendment
counts context overflow as an arm failure
and continues independent cases. No partial v1/v2 result is pooled into v3.

Final campaign: `data/experiments/word_games/20261111_coding_recovery_v3/result.json`.
SHA-256: `ddc1aac58cb3938fa38bbd2931ccd2d36bc7e379f4051d008cd28fbd6deb70e4`.
Elapsed campaign time: 464.41 seconds; this is not decoding tokens/s.

Proof: `data/experiments/word_games/20261111_structured_proof_v1/result.json`.
SHA-256: `ef0e49177c26f2a6b8d85ce0fca634b18696540e43a1953c5e3c085a065290d6`.

Independent artifact read-back:
`data/experiments/word_games/20261111_structured_coding_audit_v1/audit.json`.
It verifies producer hashes, full completion, per-case grades and aggregate counts.
The constructive witness is retained alongside it. Focused tests: 19 passed.

```sh
pixi run -e word-games python -m experiments.word_games.coding_recovery --output data/experiments/word_games/NEW_CODING_RECOVERY --key-env-file .env
pixi run -e word-games python -m experiments.word_games.audit_structured_coding --proof data/experiments/word_games/20261111_structured_proof_v1 --coding data/experiments/word_games/NEW_CODING_RECOVERY --output data/experiments/word_games/NEW_AUDIT
```

Use fresh directories. The registered record remains `confirmed: false` for any
general accuracy/secrecy claim. The actor sandbox is implemented and tested;
the separate bridge HTTP service remains unchanged and is not bubblewrapped.
