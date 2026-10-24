# Prospective replacement of the exhausted Sonnet reviewer

2026-10-24. The user explicitly requested a model switch after Sonnet's weekly
quota stopped the 10,000-source teacher workflow. The direct Claude API also
rejected the availability canary because its account lacks credits. GPT-5.6 Sol
returned HTTP 200, the exact requested model alias, and the requested readiness
JSON. That canary proves availability only. Preserve every old raw response,
first candidate, failure, denominator and semantic judgment.

This amendment replaces the blocked review model with
`codex_sdk/gpt-5.6-sol`, while retaining `codex_sdk/gpt-5.6-terra` for drafts.
No training or evaluation threshold changes. Before judging summaries, qualify
Sol on the original 50 budget128 controls using the identical source bytes,
candidate texts, expected decisions, rubric, parser, at most three transport or
schema attempts, and first schema-valid reply. Four network workers. Require
all 50 decisions correct and every packet valid. No retries for a semantic
failure. Independently replay the raw attempts and scores after actual tool
completion. A failed qualification stops Sol admission; its outcomes remain.
These agent-authored controls are not independent human annotation.

The new isolated qualification root is
`20261024_budget128_judge_sol_qualification_v1`; its read-only replay uses
`20261024_budget128_judge_sol_qualification_audit_v1`, both under
`data/experiments/ncm_paper_regime`. The module, plan, tests, inherited rubric,
control inputs and availability canary are frozen or content hashed before
requests. Original Sonnet qualification does not qualify Sol.

For subsequent teacher recovery, use the unchanged evidence-bound single-source
draft/review prompts and length contracts. Select fresh qualification sources
from the 8,883 unattempted reservations only, by source metadata and fixed hash
order, before generating outputs. Keep the original 32 short plus eight in each
of four long bins. Preserve all 1,117 observed original firsts, including the
63 invalid ones. No teacher bulk recovery is admitted until a real concurrency
pilot, raw first-response replay, and fresh 64-case semantic qualification pass.
Require at least 61/64 SAME cases to pass both judges, all requests valid and all
labels within the unchanged native and stored capacities. No capacity or quality
rerolls. The new teacher-label quality is separate from student accuracy.

For future student evaluation only, Sol may replace Sonnet as primary after its
new qualification and audit pass. Terra remains secondary. This is prospective:
no held-out output from the currently running FP16 level-conditioning candidate
has been inspected. Keep the full original 128 development sources and features,
the same complete-summary rubric, both 61/64 cohort gates on SAME cases for both
judges, all-valid requirement, no repair or oracle codes, and all prior 3,222
canonical first judgments. The old native655 failure remains reported as Terra
0/128, with Sonnet not admitted. Do not reinterpret old results as Sol results.

Two models from one provider are correlated automated judges; passing this
protocol does not establish human-reviewed 95% accuracy. Final 320 source files
remain closed and unhashed until a separately verified complete candidate passes
development and has its own final protocol. Quality precedes speed claims.
