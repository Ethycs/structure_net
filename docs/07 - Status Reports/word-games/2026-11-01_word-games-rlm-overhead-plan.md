# WORD_GAMES worker-environment overhead optimization

Exploratory study `word-games-rlm-overhead-v1`, November 1, 2026.
The active research thread has twenty completed turns and a preserved checkpoint.
This scope is its tool/host execution environment, not a new recursive reasoning
algorithm, V4 capsule integration, changed model, or broader disclosure policy.

The original sixteen-turn trace records 109.405 seconds of client time and
106.618 seconds inside the subprocess invocation. Claude's per-invocation
duration sums to 95.102 seconds: process lifecycle overhead is approximately
11.516 seconds. The CLI's API-duration field is cumulative across resumed turns;
do not sum it. Its final value is 93.226 seconds. These observations motivate a
persistent stream-JSON Claude process and persistent read-only tokenizer process.

Keep the same model alias, system prompt, structured output schema, fact menu,
source witness checks, metaphor and canonical projection. No answer cache,
relaxed evidence checks, speculative release, or changed output budget.
Preserve the original twenty-turn thread and resume it only after validation.

Before live comparison, test framing, timeouts, EOF/failure handling, sequential
requests, explicit close, and canonical authorization controls with fake child
processes. A two-turn live lifecycle pilot must preserve one session identity
and return valid proposals. Stop on transport failure; do not retry that cell.

Compare two fresh independent sessions over the same sixteen research questions:
one subprocess per turn versus one persistent stream process. Alternate which
arm runs first on successive questions. Both use the identical frozen mapping.
Record per-request wall time, worker time, tokenizer time, validation time,
native output usage, public reference tokens, accepted/rejected facts, actual
tool use, and native session identity. Check every public projection and canary.
Cloud latency and generated evidence may vary; report raw wall-time differences
separately from measured host overhead, without claiming a population speedup.

Benchmark tokenizer counts on captured public packets in alternating order;
all counts must match exactly, including Unicode and empty text. Include first
process start and report warm repeated requests separately. No GPU serving
change is planned. Persist sources and all outcomes in fresh artifact roots.

Promote the persistent transport only if it passes lifecycle/boundary controls
and its frozen-run catalog coverage and delayed recall do not fall below the
fresh baseline. If quality or timing is inconclusive, preserve the comparison
and keep the corresponding existing default. No repeated search for a lucky run.
