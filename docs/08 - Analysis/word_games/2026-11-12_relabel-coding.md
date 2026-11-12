# Qwen coding with different labels

Date: 2026-11-12. Status: completed, independently audited and re-executed.
Study: `word-games-isomorphic-label-coding-v1`.

**The relabeled configuration preserved coding performance on this suite.** Both
arms passed all eight tasks and all 288 test cases, including normal completion
of every model turn. There were zero paired case disagreements, clarification
requests, context overflows or truncated completions.

| Metric | Direct | Relabeled |
|---|---:|---:|
| Tasks passing every test | 8/8 (100%) | 8/8 (100%) |
| Test cases passed | 288/288 | 288/288 |
| Original regression cases | 32/32 | 32/32 |
| Median task latency | 34.60 s | 36.74 s |
| Clarification requests | 0 | 0 |

Median latency increased by 2.14 seconds (about 6.2%). It includes the shared
initial-controller latency charged to each arm, translation where applicable,
OpenCode startup and tool work. Offline grading and independent auditing are
excluded. It is not decoding tokens/s or
a statistically established throughput difference. The full campaign took
466.35 seconds, before independent re-execution.

## What actually ran

The automatic path was **Terra interprets → Qwen translates labels → Qwen
executes**. Terra used the existing `codex_sdk/gpt-5.6-terra` route; Qwen used
`qwen3.8-35b-a3b` through LiteLLM/Triton. The same Terra initial instruction was
shared between paired arms. The direct arm bypassed translation. Arm order
alternated. Eight controller API calls, eight translation API calls and sixteen
fresh OpenCode actor invocations were used; the actor invocations contain their
own native model/tool steps. No human or grading feedback repaired an answer.

The Qwen executor retained its 8,192-token context, 1,024-token output cap,
12-step invocation budget and bubblewrap sandbox. Every final actor finish
reason was `stop`. Both arms received the same compact Python function contract
instead of the old reservation-transaction manual. The translator used disjoint
instruction/request grammar branches, so a mixed verdict could not be accepted.

## The change in representation

The typed codebook changes labels while preserving the computation:

| Source field | Target field |
|---|---|
| loans | jobs |
| galleries | pools |
| medium | hardware_class |
| restricted | requires_permit |
| reservation | allocation |
| approval_id | permit_id |
| title | label |
| notes / text | messages / query |

Category values map separately: `metal → cpu`, `paper → gpu`, `stone → tpu`,
`textile → fpga`. Category values change only in category-bearing fields and
category-count output keys. There is no merged field/enum/prose regex. For example,
a note containing the word `metal` remains literal text rather than becoming `cpu`.

Identifier prefixes change consistently (`loan:atlas → job:atlas`), while suffixes
remain unchanged. Within each identifier namespace this preserves lexical order
and reference equality. Numbers, dates, booleans, nulls and all free text are
unchanged. The host does not compute match flags or sorting ranks for the model.

Before inference, all 288 inputs passed an exact round trip and an independent
source/target reference check: decoding the target reference output equalled the
source reference output. These are bounded checks on the declared test domain,
not a universal bijection claim for arbitrary JSON or natural language.

## Coverage and grading

Each task used its original four regression cases plus 32 newly generated cases,
frozen before inference with seed 731112 plus the task number. Generated cases
covered empty collections, date ties, title ties and case, missing/mismatched
permits, empty notes and queries, mixed-case substrings, category words occurring
as ordinary text, rule minima and event selection. These are new inputs for
familiar tasks, not eight newly held-out task families.

| Task | Direct | Relabeled |
|---|---:|---:|
| Free slots | 36/36 | 36/36 |
| Earliest reservation deadline | 36/36 | 36/36 |
| Count by category | 36/36 | 36/36 |
| Minimum inherited deadline | 36/36 | 36/36 |
| Case-insensitive substring search | 36/36 | 36/36 |
| Alphabetical title order | 36/36 | 36/36 |
| Missing/mismatched approvals | 36/36 | 36/36 |
| Newest matching creation event | 36/36 | 36/36 |

Only the original example input was visible to each executor. Terra saw the goal
and schema/types, not concrete examples or answers. Actual generated programs ran
in a separate network-isolated sandbox. The grader checked exact decoded values,
ordering and input preservation. The audit reconstructed every score from raw
responses and also checked exact target-language outputs before decoding.
It captured all sixteen programs, reran them in fresh execution workspaces and
reproduced every grade. Source, input, route and artifact hashes were checked.
Twenty-nine focused tests passed for the codec, protocol, finish handling, existing
grader/feedback behavior and conservative registry record.

## What this establishes, and what it does not

Consistent relabeling can preserve Qwen coding performance on these eight task
families. The previous opaque-bridge run passed 2/8 code tasks (1/8 with normal
completion required); the original 32 cases now all pass. However, typed mapping,
meaning-preserving data, a compact contract, the response grammar and completion
handling changed together. This is not a single-factor causal estimate.

This configuration deliberately leaves free-text contents, identifier suffixes
and the synthetic canary visible. **It is not an information-secrecy result or a
production information firewall.** Hiding those contents would require a separate
policy and recovery evaluation while retaining task-required comparisons.

Eight known task families, one codebook and one run do not establish 95% or 100%
general coding accuracy. The registry keeps `confirmed: false` for any broad
accuracy/privacy claim. No model weights, GPU placement or production service
configuration were changed. No DataAnnotation material was used.

## Evidence and reproduction

Run: `data/experiments/word_games/20261112_relabel_coding_v1/`.
Result SHA-256:
`ad51989a32e8a46fa0d654baed5dac519d10d8e70735d2a23cccab2690351080`.
Audit: `data/experiments/word_games/20261112_relabel_coding_audit_v1/audit.json`.
Saved programs are under that audit directory's `generated_solutions/`.
The complete codebook and frozen 288-case suite are in `plan.json` and `suite.json`.

```sh
pixi run -e word-games python -m experiments.word_games.run_relabel_coding --output data/experiments/word_games/NEW_RELABEL_CODING --key-env-file .env
pixi run -e word-games python -m experiments.word_games.audit_relabel_coding --run data/experiments/word_games/NEW_RELABEL_CODING --output data/experiments/word_games/NEW_RELABEL_AUDIT
```

Use new directories. Prior runs remain unchanged; no failed task was removed or
relabelled as a pass after seeing its score.

Report correction, 2026-11-12: the initial registered report incorrectly said
per-task latency included grading. Timing ends before offline grading. Numerical
results are unchanged; the original report and registration receipt are preserved
in the audit directory, alongside `registry_report_amendment.json`.
