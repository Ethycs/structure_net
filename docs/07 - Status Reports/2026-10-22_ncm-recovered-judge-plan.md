# Resume development grading after judge connectivity recovery

Date: 2026-10-22, before new development quality outcomes.
The connectivity-only probe `20261022_gateway_recovery_probe_v2` returned200 for
both original judges in under5seconds after earlier transport failures. This
does not certify their quality. Preserve all prior timeout/credit/missing-route
failures and the separate local-Qwen qualification study.

Use the original qualified models `claude_code/claude-sonnet-4-6` and
`codex_sdk/gpt-5.6-terra`, same source-grounded rubric/parser and48-control
qualification artifacts. Before full grading, each model must complete a fresh
four-concurrent-request systems pilot on public development sources with new
systems-only packet IDs. Require4/4valid packets; no systems scores are pooled
into the actual evaluation. No semantic failures are retried. Model identity,
raw payloads/replies, public exact-span verification and strict schema validation
remain enforced.

Then grade all64fixed development outputs from each completed noisy-pair arm at
step2048, independently under both judges. One candidate per request; workers4
per judge, two judge processes total. Reuse no prior candidate judgment; these
outputs have not been judged before. Up to3transport/schema attempts, first
valid semantic decision wins. Complete-summary pass is faithful AND salient AND
readable AND nonempty AND <=128cl100ktokens. Invalid replies are missing evidence,
never semantic failures or model passes; task accuracy is absent if any remain.
Report numerator/denominator, all case scores, Wilson intervals and all4arm/judge
results without selecting the most favorable judge. Admission to the separate
sealed final test remains >=61/64under both; final success requires >=244/256.

This recovers the diagnostic's originally intended judges rather than changing
its success threshold. New roots:
`20261022_noisy_{native,hybrid}_grade_{primary,secondary}_v1` under
`data/experiments/ncm_paper_regime`. Append an aggregate/report when grading
stabilizes. No changes to model checkpoints, train labels, prior judgments or
existing campaign results. No speed claim from quality evaluation.
