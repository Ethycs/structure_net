# V3 autonomous mapping bundle review

The supplied V3 bundle adds the operational mapping pipeline missing from the
active WORD_GAMES orchestrator. Its saved checks pass, and its ordinary prompt
sizes fit the current 8,192-token context. It is a smaller first integration
target than V4's recursive capsules. This review made no fresh generation calls
and changed no active vendor or serving configuration.

Evidence: `data/experiments/word_games/20261101_v3_review_v1/result.json`, SHA256
`bcd72a849e59b83e476a9b7b537ba0a0e01ef9f8b78f46c1825a80229f657c2a`.
Archive SHA256:
`e0e651aaf2c00c60aeb13893d1297ea580aecd1a9df37f4ef9f2c3a74bf06e7a`.

V3 uses a fresh translator to choose a target domain and operational vocabulary,
freezes the validated map, encodes source requests, supplies translated scoped
state to fresh target-only solvers, then decodes their operations into the
unchanged source Engine. Actual committed state continues across events. The
generated example domain is accelerator workload placement control.

The source Engine is byte-identical to the active V2 vendor. The generic V3
runtime is byte-identical to V4's mapping runtime. V4 adds self-translation and
checkpoint capsules around that core. Neither bundle provides an automatic
provider client or a general model-executed REPL. The original bundle calls its
external symbolic state plus fresh interpreters “RLM-style”; that foundation
already exists. Autonomous operational mapping is the missing active component.

Checks reproduced from the extracted archive:

| Check | Result |
| --- | ---: |
| Runtime unit tests | 5 passed |
| Generated mapping probes | 91/91 |
| Sequential full states versus independent oracle | 16/16 |
| Exact decoded transaction events | 11/11 |
| Clarification events, with unchanged state | 5 |
| Broader-capability control events | 4/4 |
| Final execution freeze hashes | 5/5 matched |

These are saved-answer replays. The original model/provider version is not
established by the call ledger. Narrow operation/object capabilities can cue
answers in the primary trajectory; four broader controls are post-design checks,
not a second held-out sequential run. The decoder was hardened after E05 while
E06 was in flight. Final-code replay establishes consistency, not a uniformly
frozen original experiment or fresh Qwen accuracy. The new checks do not replace
the archived semantic review of clarification questions.

The deployed Qwen tokenizer measured the following saved artifacts, excluding
chat framing:

| Stage | Input | Output | Combined |
| --- | ---: | ---: | ---: |
| Generate mapping | 1,341 | 2,106 | 3,447 |
| Encode all 16 requests | 3,305 | 678 | 3,983 |
| Largest solver prompt/output pair | — | — | 4,413 |

Across 20 solver pairs, the maximum input is 4,329 and maximum output 129;
those maxima are from different pairs. Ordinary saved tasks fit the existing
context and 1,024-token output cap. Fresh mapping creation needs a larger output
cap. This says nothing yet about fresh-generation lengths, latency or accuracy.

Integration must preserve the stricter WORD_GAMES disclosure policy. A synthetic
private note survives V3's translated view but is absent from the current
authorized packet. V3 intentionally retains protected literal text; that behavior
is incompatible with blindly forwarding our private notes. Apply host policy to
both the translated state and encoded request before a receiving agent sees them.

All extraction and replay work used `.pixi/word-games-v3-review-v1/`. Three
independent audit scripts needed one hardcoded root literal relocated in copied
scripts; original extracted scripts and all model outputs remained unchanged.
The evidence records original/executed hashes and commands. No files were placed
in the archive's hardcoded `/tmp/autonomous_bridge_v3_20261030` location.
