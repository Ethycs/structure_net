# Requalify unchanged judges after the endpoint repair

The user paused work during a LiteLLM outage and explicitly requested resumption
after repairing the endpoints. Both paused processes retained their identities.
The local guidance diagnostic resumed unchanged. The semantic pilot had not
created a producer or sent an application request when paused. Its supervised
tool session86715 was intentionally terminated (actual chunkb0256d, exit143)
because its loaded route could not incorporate the repaired provider source.
This is an infrastructure interruption, not a summary-quality failure.

The first post-resume neutral probe stopped before obtaining credentials or
sending requests: the original gateway fingerprint rejected changed provider
files. Preserve `20261028_user_resume_endpoint_probe_v1`, actual chunk5a949d,
exit1. The original service and all qualified judgments remain historical
evidence. Do not edit their hashes to match the new files.

The endpoint repository is clean at commitada4a43. Relative to the saved gateway
source, only `providers/codex_pool.py` and `providers/codex_sdk.py` changed. The
repair disables configured MCP servers at each thread creation and supplies a
finite default timeout when none is supplied. The evaluation already provides
an explicit timeout and a persistent-only launcher with MCP isolation. Preserve
the user's changes. Hard-bind the two observed old/new file digests; reject any
additional source change rather than extending this admission silently.

Retire only our idle isolated gateway after observing zero in-flight requests,
then start a fresh service invocation with current source on the same loopback
port8092. Preserve the persistent-only launcher, separate SQLite runtime,
single-worker dispatch, MemoryHigh3GiB/MemoryMax4GiB and no automatic restart.
Bind a new producer and copied source snapshot under
`20261028_repaired_SQLite_gateway_service_v1`. Do not restart or modify shared
endpoint services. The new unit is
`structure-net-ncm-repaired-sqlite-judge-v1.service`.

Before service launch, run four repair contracts and the four unchanged local
semantic-consumer contracts. A neutral availability probe must return exact OK
from both original model aliases. Then repeat the unchanged 50-control suite
for Terra, independently audit its raw first replies, repeat for Sol, and audit
Sol. Both must pass all50 controls before any new application judgment. Keep
all first valid failures and the original three-attempt ceiling. A partial or
failed qualification never admits application grading.

New neutral and qualification roots use `20261028_repaired_SQLite_*_v1`.
The wire endpoint, model aliases, rubric, packet keys and transport identity
remain unchanged, so original raw replies retain their original valid meaning.
The new service/qualification producers identify the changed implementation;
historical first judgments are neither regenerated nor relabeled.

After complete qualification, continue the same unstarted expanded judge pilot,
grade and audit, then the original hierarchy/recovered evaluation order. The
existing consumer still verifies all inherited firsts and complete admission.
Generation roots and outputs remain unchanged. Bind this amendment, recovery
source, tests, CPU receipt and consumer-retirement receipt into future producers.
The new entry point is `experiments.structure_net.ncm_repaired_judge_route`,
with `serve`, `probe`, `qualify|audit --role secondary|primary`, and
`consume --family task|hierarchy|recovered --stage judge_pilot|grade|audit --arm ...`.

This amendment changes infrastructure provenance only. It does not change any
model, input, decoding rule, first-judgment policy, 61/64 cohort gate or Final320
reservation. It establishes no accuracy or speed improvement.
