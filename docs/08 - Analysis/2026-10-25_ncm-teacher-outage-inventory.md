# Stopped teacher continuation: all first results preserved

Independent CPU replay completed with actual tool exit 0. All 144 saved first
pipelines and 303 raw request attempts match their original sources, requests,
model identities, parser results, token counts and native frames. There are
135 syntax-valid, capacity-fitting targets and nine invalid pipelines: one
exhausted support-syntax failure and eight exhausted service failures. These
135 targets have not been individually judged for semantic quality.

The failed collection remains failed (actual workflow exit 1). Its pending job
count is zero. All 8,487 untouched source IDs are enumerated; no failed first
received a new allowance. Together with the preceding immutable collections,
1,513 firsts are now observed, including 73 preserved invalid firsts. This audit
made no model requests, changed no semantic cache, admitted no training data,
and opened no sealed final source. Six focused tests passed, including forged
request/identity/termination/validity rejection and exhausted-allowance checks.

Evidence: `data/experiments/ncm_paper_regime/20261025_sol_teacher144_partial_inventory_v1/`.
Audit session 90192 and test session 17171 both actually exited 0. The audit uses
the frozen recipe directly without rebinding completed producer globals.

Two separate, synthetic-only inference readiness requests to the existing Terra
and Sol aliases both timed out after 40 seconds. Probe session 79382 exited 0
after recording those failures; that exit is not evidence of working inference
or judge qualification. Raw probe evidence is in
`data/experiments/ncm_paper_regime/20261025_service_probe_v1/`.

Complete-summary semantic task accuracy remains unmeasured for the latest
student. Its existing 10 empty outputs among 128 source-only DEV cases still
rule out 95% on that cohort. No speedup is established. The broader non-LME
archive remains the next data priority.
