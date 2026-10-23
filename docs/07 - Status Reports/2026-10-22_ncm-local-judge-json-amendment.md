# Local judge constrained JSON amendment

Date: 2026-10-22, before qualification or candidate task grading.
The GPU2 systems pilot returned a valid first packet, then malformed JSON for
the second packet. Raw failures remain preserved. A new systems/qualification
cell uses the same Qwen weights, server, GPU placement, rubric and strict parser
with llama.cpp's JSON-schema constrained decoding. No semantic judgment is
changed or retried for a passing outcome. Previous systems-only decisions are
not included in qualification. This is a formatting intervention, not evidence
that the judge is accurate.

Require an evaluations array of the packet's size; candidate IDs and evidence
IDs are restricted to the packet's supplied IDs. All three booleans, claim
verdicts, citations and nonempty reasons are required. The schema does not fix
any verdict or pass/fail decision. The existing parser still checks complete
unique candidate binding and faithful/claim consistency, preserving conservative
failures. Up to three syntax/transport attempts, first valid reply wins. Raw
payloads/replies and the exact schema are retained. The48-control qualification
gate remains24/24positive and24/24negative with no valid-failure retries.

New roots: `20261022_pilot_gpu2_json_v1` and
`20261022_qualification_gpu2_json_v1` under `data/experiments/ncm_local_judge`.
The earlier pilot stays a separate failed/invalid systems result. No resident
services or model weights change. Actual NCM task accuracy remains unmeasured.
