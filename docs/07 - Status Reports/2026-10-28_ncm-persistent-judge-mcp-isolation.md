# Persistent judge gateway with MCP servers disabled

This systems experiment tests an isolated persistent Codex gateway for the
unchanged NCM semantic evaluation. The original central route times out on
neutral generation requests. Its local shared gateway is heavily memory
throttled; an attempted soft-limit adjustment was rejected by the operating
system, and no shared setting changed.

The previous nonpersistent isolated route remains failed: Terra scored 49 of
50 controls and Sol scored 50. The valid Terra disagreement is preserved.
The earlier local route also remains failed. Neither route admits application
grading, and their marginal successes cannot be combined.

The new configuration addresses a measured startup problem. The provider passes
`mcp_servers={}`, which leaves configured servers enabled after configuration
merging. Explicit per-server disable flags must come after that argument.
The actual CLI listing verifies both configured servers disabled with this
ordering. A separate two-request persistent-host probe answered both neutral
prompts, observed no MCP descendants in 33 samples, and measured a peak process
tree RSS of 198,639,616 bytes. This small availability check does not establish
sustained capacity or semantic competence. The setting is documented in the
[official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

Run a new user-owned service on loopback port 8092 after verifying the previous
service is retired. Use the existing locked Pixi gateway packages, one worker,
the explicit MCP-disable launcher, a 1,024 MB host recycle threshold and a
16-turn recycle limit. Give the service a 3 GiB soft limit and 4 GiB hard limit,
with no automatic restart. Preserve user configuration, shared services, model
names, rubric, source packets and expected answers. Reject the exec fallback
before it can send a model request; the gateway must expose persistent-host
failures as failures. Record every service invocation and producer dependency.

Require the wrapper contracts, actual CLI configuration check, completed direct
host probe and new gateway contracts before launch. Then run a new HTTP neutral
probe for both exact aliases. Only after that probe completes successfully,
qualify Terra and independently audit its responses, then qualify and audit
Sol. Retain the existing 50 controls, three-attempt transport/schema budget,
first-valid-response rule, temperature zero, 4,096 output-token request limit,
160-second backend timeout and 180-second client timeout. A valid negative
decision is terminal. Both judges must score 50 of 50 on this configuration;
otherwise close the failed qualification without application calls.

Use fresh `20261028_persistent_no_MCP_gateway_*_v1` service/probe roots and
`20261028_persistent_no_MCP_judge_{role}_{qualification,audit}_v1` qualification
roots. Keep all original qualification and application records unchanged.
Successful qualification would only admit this transport. Application grading
must still preserve canonical first judgments and satisfy the existing complete
summary gates on source-disjoint DEV and sealed Final320. No accuracy or model
inference-speed claim follows from this systems experiment.
