# WORD_GAMES: broader coding with relabeled interfaces

Date: 2026-11-12. Status: completed, independently audited and re-executed.
Study: `word-games-humaneval-labels-v1`.
Classification: `sampled_public_coding_evaluated`; broad confirmation is false.

**Both arms passed 30/32 official task suites (93.75%).** The passing task sets
differed: 29 passed in both, one passed only direct, one passed only relabeled,
and one failed in both. An additional post-run edge case exposed a further error
in a relabeled program that the official tests had passed. This does not establish
95% general coding accuracy or production readiness.

| Measurement | Direct | Relabeled |
|---|---:|---:|
| Official task suites passed | 30/32 (93.75%) | 30/32 (93.75%) |
| Descriptive Wilson 95% interval | 79.85–98.27% | 79.85–98.27% |
| Normal actor completions | 32/32 | 32/32 |
| Clarification requests | 0 | 0 |
| Median task latency | 27.22 s | 29.32 s |

Independent sandbox execution reproduced all 64 saved-program results. Artifact
correctness and strict completed-delivery correctness were identical. All tasks
were retained; there were no retries, answer repairs, fallback models, context
overflows or output-limit terminations. Normal transport completion alone did not
guarantee a complete instruction or correct program.

The relabeled median was 2.10 seconds higher (7.7%). Timing includes the shared
initial controller latency charged to each arm, translation where applicable,
actor startup and tool work; it excludes offline grading and independent replay.
The full campaign took 1,516.41 seconds (25.27 minutes). These are task latencies,
not decoding tokens/s or a statistically established speed difference.

## The earlier “0/8” was not an identical accuracy test

The first coding attempt, `20261111_coding_recovery_v1`, stopped with **zero
completed pairs**. OpenCode truncated a long task-file line, hiding the goal,
and exhausted its context before delivering code. Its result has no coding
aggregate. It did not measure eight incorrect solutions.

The latest eight-task relabel run reused the same task definitions and original
32 inputs, adding 256 generated inputs. Its configuration changed: typed maps
replaced conflicting substitutions, free text remained visible, irrelevant
manual text was removed, protocol branches were separated and normal completion
was required. Therefore its 8/8 result was not an identical rerun of the old
opaque information-hiding setup. The completed opaque pilots scored 3/8 and then
2/8 working programs (1/8 when strict normal completion was applied to the latter).

This HumanEval run uses a different task set, sampled before inference. Historical
scores must not be pooled or treated as a single-factor improvement curve.

## What ran

Thirty-two problems were selected without replacement from the 164 official
[HumanEval problems](https://github.com/openai/human-eval), using seed 731113 and
repository revision `b9453e8f1865df0a0057a1383767caa400acfbcf`. The immutable
[prospective plan](../../07%20-%20Status%20Reports/word-games/2026-11-12_humaneval-labels-plan.md)
lists every selected ID. Coverage includes strings, lists, primes, polynomial
roots, recurrences, parsing, dates, filenames and expression evaluation.

The pipeline was **Terra interprets → Qwen translates interface names → Qwen
executes**. The direct control skipped translation and shared the same initial
Terra instruction. The configured routes were `codex_sdk/gpt-5.6-terra` and
`qwen3.8-35b-a3b`, with the existing pinned gateway deployment IDs. There were
32 controller calls, 32 translation calls and 64 fresh OpenCode actor invocations;
each invocation contains its own native model/tool steps. No weights changed.

Terra saw the public problem, including public examples and supplied helpers,
but no canonical solution or hidden tests. It could explain an algorithm in
plain language. Actors received their controller instruction and interface
scaffold; the relabeled actor did not receive the inverse map or original prompt.
The function became `bridge_operation`, with parameters `parcel_N`; literals,
operations and return values remained unchanged. This changes interface names,
not the entire narrative domain, and makes no secrecy claim.

Both arms used OpenCode 1.19.5 in bubblewrap, 8,192 context tokens, 1,024 actor
output tokens, 12 agent steps and disabled compaction. The actor had file tools;
it did not have a shell execution tool. Arm order alternated. Grading used the
unchanged official `check(candidate)` assertions in separate network-isolated
sandboxes, adapting keyword names only for the renamed callable. All 64 original
and renamed canonical solutions passed before inference. Twenty-four focused
tests passed, including wrong-answer, timeout and completion-gating controls.

This is one sample per arm from a controller-assisted pipeline, not a bare-model
leaderboard score. The original official tests were used, not HumanEval+ expanded
tests. Public-benchmark training contamination is unknown. Thirty-two functions
do not characterize general repository repair or long-running coding work.

## Failures and a test-suite blind spot

| Task | Direct | Relabeled |
|---|---|---|
| HumanEval/32, polynomial root | Search loop never expands bounds; execution limit | Translator retains original function name; executor leaves renamed entry point as a stub |
| HumanEval/113, odd-digit output formatting | Pass | Misses required replacement of every lowercase `i` in a literal string |
| HumanEval/160, expression evaluation | Operator/operand indexing error produces `TypeError` | Official tests pass; additional associativity check fails |

For /113 the translated instruction still states the key replacement rule, but
its example ends halfway through a string despite `finish_reason: stop`. The
executor ignores the stated rule. These traces do not isolate how much the
incomplete example contributed to the failure.

After primary grading, inspection of /160 found that the relabeled program
evaluates chained exponentiation left-to-right. A separately recorded diagnostic
checked `2 ** 3 ** 2`: the canonical solution passes the expected value 512, while
both generated programs fail. The relabeled program computes the equivalent of
`(2 ** 3) ** 2`, or 64. This check was selected after observing the code, so it is
**post hoc**, not part of the preregistered primary score. With this one extra case
included, the task counts would be 30/32 direct and 29/32 relabeled (90.625%). The
official 30/32 scores remain unchanged in their original artifacts.

The concrete next improvements are validating the required callable interface
before accepting delivery, carrying literals/examples as structured data through
translation, and testing saved programs with expanded independent cases. These
are proposals, not demonstrated fixes. The equal official totals alone do not
prove performance equivalence, and the extra case demonstrates why a passing
official suite is weaker than complete functional correctness.

## Evidence and reproduction

Run: `data/experiments/word_games/20261112_humaneval_labels_v1/result.json`.
SHA-256: `3860f37fefbb9ccee10413b8aa8d9a0b13592c79f54e351a0fc5df00c613b997`.
Audit: `data/experiments/word_games/20261112_humaneval_labels_audit_v1/audit.json`.
Audit SHA-256: `841897c64e07662a5407423d5d97769dd70f1e9cbc109baddaa569632eab8902`.
Post-hoc diagnostic: `data/experiments/word_games/20261112_humaneval_posthoc_associativity_v1/`.
Saved programs: each run task's `native/solution.py` and `relabeled/solution.py`.
Dataset and MIT license: `data/experiments/word_games/20261112_humaneval_data_v1/`.

```sh
# Fetch only if the pinned data directory is absent.
pixi run -e word-games python -m experiments.word_games.fetch_humaneval_labels
pixi run -e word-games python -m experiments.word_games.run_humaneval_labels --preflight --output data/experiments/word_games/NEW_HUMANEVAL_PREFLIGHT
pixi run -e word-games python -m experiments.word_games.run_humaneval_labels --output data/experiments/word_games/NEW_HUMANEVAL_RUN --key-env-file .env
pixi run -e word-games python -m experiments.word_games.audit_humaneval_labels --run data/experiments/word_games/NEW_HUMANEVAL_RUN --output data/experiments/word_games/NEW_HUMANEVAL_AUDIT
```

Use fresh directories. Earlier reports, campaign results and failed solutions
remain unchanged. No production deployment, GPU placement or DataAnnotation
material was involved.
