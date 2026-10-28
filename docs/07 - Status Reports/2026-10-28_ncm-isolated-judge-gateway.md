# Isolated completion transport availability experiment

The existing loopback8091 route passed Terra's50 controls, then exhausted
transport attempts on5 of13 Sol packets. Sol has32 correct controls and18
unresolved controls. Preserve that qualification failure; neither valid
decisions nor exhausted application attempts are reopened.

Test the unchanged local gateway implementation in its own supervised process
on loopback8092, with persistent Codex hosting disabled and one worker. This
is a transport diagnosis, not a model change or a claim that the previous
failure's cause is known. Keep existing services untouched. Use the gateway's
locked Pixi environment, preserve its source hashes and process invocation,
and generate an in-memory authentication key without printing or saving it.

First make one neutral request per unchanged model, Terra then Sol, asking for
exactly OK. Use backend timeout30s, client40s, max_tokens256 and temperature0.
Preserve raw responses and failures. Require exact requested model identity
with only the already documented optional codex_sdk prefix. No source summaries,
evaluation cases or calibration controls enter this availability experiment.
The service and probe use separate new append-only roots ending
`20261028_isolated_judge_gateway_{service,probe}_v1`.

Even two successful probes do not qualify semantic grading. A separate
prospective qualification must test all50 unchanged controls for both models,
replay raw first responses, and bind the full application judgment history.
The unchanged61/64-per-cohort dual-judge DEV gate and sealed Final320 remain.
No accuracy or speed claim follows from this transport experiment.
