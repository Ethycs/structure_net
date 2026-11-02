# Separate runtime database for the persistent judge

The MCP-disabled persistent HTTP service failed both neutral requests before
semantic qualification. A direct startup diagnostic recorded a SQLite runtime
initialization failure in the shared Codex directory. That service is retired;
its results remain failed, with no application judgments.

A new isolated SQLite directory removed that startup error, but its first
initialization exceeded the provider's 15-second initialization timeout. A raw
initialization-only RPC with an explicit `sqlite_home` configuration completed
in 24.78 seconds and closed successfully, without requesting model generation.
The [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
documents this separate runtime-state directory. Authentication, user
configuration and model names remain in their existing locations.

Use the initialized `/data2/structure_net/ncm_judge_runtime_20261028_v3/sqlite`
directory through a new persistent-only launcher. Preserve explicit MCP-disable
flags, rejection of exec fallback, one worker, 1,024 MB host recycling and
16-turn recycling. Use a new service with a 3 GiB soft limit, 4 GiB hard limit,
no automatic restart, loopback port 8092 and fresh
`20261028_persistent_SQLite_*_v1` artifact roots. Do not change previous producers.

Require four focused contract checks and the actual successful initialization
receipt before launch. Then repeat the two-model HTTP neutral probe. If both
models answer correctly, run the original 50-control qualification and raw
response audit for Terra, followed by Sol. Preserve the original packets,
expected answers, rubric, model aliases, first-valid-response rule and three
transport/schema attempts. A valid negative decision remains terminal. Both
judges must score 50 of 50 on this configuration before application grading.

The initialization result diagnoses startup only. It establishes neither judge
qualification, summarization accuracy nor NCM inference speed. All previous
failed qualifications remain failed; no scores from different routes are pooled.
