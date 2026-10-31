# WORD_GAMES: Phi-4 14B toward 40 output tokens/s

**Status:** completed exploratory optimization; target not met.
**Hypothesis:** `word-games-phi4-native-40tps-v1`.
**Classification:** `below_40tps`; observed quality gate also failed.
Aggregate: `data/experiments/word_games/20261031_native_40tps_analysis_v1/campaign_results.json`.
SHA256: `54489e4a9b6dbfbda47c9bfc971cb2b7ee19497264bed735b8c6befe601a20c8`.

The fastest complete frozen replay reached **38.30 visible output tokens/s
including prompt processing**, and **52.95 tokens/s during decoding**. This
is 1.70 tokens/s short of the requested end-to-end target. It also failed the
task-quality gate: 9/12 live workflows passed versus 10/12 for the native
baseline. The existing Triton serving path was not replaced.

## Full request replay

All rows use the same 36 frozen prompt/cap pairs, Phi-4 14B Q4_K_M weights,
8,192-token context, greedy decoding, one RTX 3060, original JSON constraints,
and no cross-request cache reuse. The final row changes KV precision and
prompt batch size explicitly. No training or second model is involved.

| Configuration | Request seconds | Visible output tokens | End-to-end tok/s | Decode-only tok/s |
|---|---:|---:|---:|---:|
| Native baseline; Q8 KV, batch 512 | 99.112 | 2,705 | 27.29 | 34.12 |
| Lookup n=4, draft=16; Q8 KV, batch 512 | 71.947 | 2,703 | 37.57 | 51.83 |
| Lookup n=3, draft=8; Q8 KV, batch 512 | 72.358 | 2,711 | 37.47 | 51.67 |
| Lookup n=4, draft=16; FP16 KV, batch 1,024 | 71.224 | 2,728 | **38.30** | **52.95** |

The fastest row reduced summed request latency by **28.1%** versus the native
baseline (1.392x latency speedup). It generated slightly more tokens; it did
not obtain that reduction by shortening responses. It matched baseline text
exactly on 26/36 requests. Exact text equality was diagnostic, not the quality
gate: the user accepts different wording when the task remains correct.

Its 71.224 seconds comprise 19.506 seconds of server prompt evaluation,
51.521 seconds of server decoding, and approximately 0.197 seconds of remaining
HTTP/client time. Further client-only cleanup cannot close the remaining gap
by itself. The observed FP16/batch-1,024 post-load GPU footprint was 10,684 MiB
including the existing 291 MiB allocation, versus 9,788 MiB in the initial
Q8/batch-512 pilot. These are snapshots, not peak-memory claims.

For historical context, the earlier Python pilot measured 14.63 tok/s. Native
baseline measured 29.76 tok/s on those same three inputs, approximately twice
the throughput. That comparison combines a newer llama.cpp build, a native
request loop, thread settings and grammar implementation; it does not isolate
Python overhead as the sole cause. The full replay table above is the current
matched-input comparison.

## Live task quality

The receiving agent still sees only the authorized translated packet. The
model freely selects the metaphor from public task structure. Live evaluation
checks sender operations, host boundary decisions, receiver structured state,
invariants and canary disclosure. It is not exact-summary matching.

| Live configuration | Sender operations | Boundary decisions | Receiver state | Joint workflows | Observed canary leaks |
|---|---:|---:|---:|---:|---:|
| Native baseline | 10/12 | 11/12 | 12/12 | **10/12 (83.3%)** | 0/12 |
| Q8 lookup n=4/draft=16 | 10/12 | 10/12 | 12/12 | **9/12 (75.0%)** | 0/12 |
| FP16 lookup n=4/draft=16 | 10/12 | 11/12 | 11/12 | **9/12 (75.0%)** | 0/12 |

Both lookup arms reuse the baseline's synthetic context IDs after asserting
identical authorization metadata, eliminating random handle-copying differences.
All twelve context records match. The FP16 arm selected `shelf-position`
instead of `shelf-number`; that was allowed and became part of its downstream
prompts. Subsequent model choices remain free to differ.

The Q8 lookup arm introduced failures in outside-entity operations and the
intentional stale-revision submission, while fixing the baseline's
outside-destination failure. The FP16 arm retained the baseline's two failures
and introduced a cancellation error: it reported an occupied slot called
`shelf-position` even though the authorized packet and its own prose said the
entity occupied no slot. These are task errors, not harmless wording changes.

Manual review covered the packets and responses of all twelve scenarios in
each arm, plus changed source proposals. The cancellation prose also claims
an approval absent from the packet in the baseline and both candidates.
Blocked-case descriptions sometimes overgeneralize the authorized view to the
whole library. There was no independent external semantic judge, and the
structured percentages above are not broad semantic-accuracy estimates.
Review and its 147 artifact hashes:
`data/experiments/word_games/20261031_native_semantic_review_v1/review.json`.

One small exploratory suite does not establish a statistically reliable
accuracy difference or general security guarantee. The observed new failures
are nevertheless sufficient to fail this study's no-regression gate.

## All pilot attempts

Each row below uses the same three pilot requests; speeds exclude EOS.

| Pilot | End-to-end tok/s | Finding |
|---|---:|---|
| Native baseline | 29.76 | Valid lifecycle baseline |
| n=3, draft=8 | 36.94 | Promoted to full replay |
| n=2, draft=8 | 35.24 | More rejected candidates; slower |
| n=4, draft=16 | 37.20 | Promoted to full replay and live evaluation |
| Above plus backend sampling | 37.12 | Runtime disabled backend sampling because grammar was active |
| Compact JSON, n=3/draft=8 | 36.83 | Valid JSON syntax but invalid selector/receiver fields; rejected |
| n=4, draft=32 | 37.25 | Effectively tied; no convincing pilot improvement |
| n=4, draft=7 | 34.42 | Eight-token MMVQ kernel boundary was slower here |
| n=4/draft=16, batch 1,024, Q8 KV | 37.08 | Larger prefill batch alone did not help |
| n=4/draft=16, batch 1,024, FP16 KV | 37.92 | Promoted; fastest complete replay, quality gate failed |
| FP16 plus minimal greedy sampler chain | 37.91 | No observed improvement |
| Minimal GPU sampler, no generation grammar | 34.50 | 0/3 responses parsed as JSON objects; Markdown/prose; rejected |

The first two raw native pilot aggregates counted terminating EOS tokens.
They remain unchanged. The analysis recomputes their visible-output rates
from raw responses, as prospectively clarified before later pilots. Decode
timing follows the native server's convention and is reported separately.

There were 288 measured requests: twelve three-request pilots, four full
36-request replays, and three live 36-request executions. These are repeated
measurements on one workload, not 288 independent semantic test examples.
Load/warmup and checkpoint hashing are excluded from inference throughput.

## Mechanism and reproducibility

Native prompt lookup searches the current request and generated continuation
for an n-token match, proposes up to the configured draft length, and asks the
full Phi model to verify the candidates together. It stores no cross-request
answer corpus. The fastest full run proposed 2,729 tokens and accepted 1,079;
only actual emitted output tokens enter throughput. Every frozen replay
response reports `cache_n=0`.

The backend is pinned to the CUDA 12.8 asset for
[llama.cpp b11960](https://github.com/ggml-org/llama.cpp/releases/tag/b11960),
the nightly referenced by v0.8.0. Binary hashes, model/input hashes, source
snapshots, commands, raw responses and server logs are retained per run.
The downloaded archive SHA256 is
`64d57beb9abb637c177c6afb313760a0c0782224cac71e74869eed5b8f9a1ea4`.
The host's older glibc could not load that binary, so it runs in a separate
Ubuntu 24.04 image pinned to
`sha256:24cbda07ad69c186ebaa9927292448d82e705d0d5d68c68e6c0f8eab420b64dd`.
Model loading after checkpoint verification/page-cache warming took about
4.3 seconds; this is not a cold-disk startup measurement.

Upstream [MMVQ selection](https://github.com/ggml-org/llama.cpp/blob/b11960/ggml/src/ggml-cuda/mmvq.cu)
motivated the seven-token draft pilot. Upstream
[sampler construction](https://github.com/ggml-org/llama.cpp/blob/b11960/common/sampling.cpp)
explains the grammar/backend-sampling incompatibility. Different numerical
batch paths are a plausible contributor to changed greedy answers, but this
study did not capture logits to establish that cause.

Use the commands in [the runner README](../../experiments/word_games/README.md)
and the [protocol](../07%20-%20Status%20Reports/2026-10-31_word-games-phi4-40tps-plan.md).
All code executes from the repository root through Pixi; native dependencies
live under `.pixi/` or in the pinned container, with evidence under data.
The `20261031_native_*` roots are append-only. Full replay roots are
`baseline_full_v1`, `lookup_n4d16_full_v1`, `lookup_n3d8_full_v1` and
`b1024_f16_full_v1` with that prefix; corresponding live roots end in
`baseline_live_v1`, `lookup_n4d16_live_v1` and `b1024_f16_live_v1`.

The focused suite passed 49 tests, including token-accounting, classification
and matched-fixture authorization checks. All temporary servers were stopped;
GPU memory returned to 291 / 5,646 / 2,702 MiB on the three cards. No unrelated
endpoint was unloaded. Registry status remains tested, unconfirmed, below
40 tok/s, with the failed quality gate explicitly recorded.

The useful result is a substantial native-runtime speed improvement. A
40-token/s, quality-preserving configuration has **not** been established.
