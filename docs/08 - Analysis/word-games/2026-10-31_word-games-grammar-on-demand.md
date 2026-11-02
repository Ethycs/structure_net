# Grammar logits on demand

Completed exploratory systems study. The final patch avoids the full-vocabulary
download on successful eligible grammar checks and preserves all replay outputs.
It is **not a demonstrated throughput improvement** and remains opt-in.

Both `word-games-grammar-on-demand-v1` and its implementation follow-up
`word-games-grammar-on-demand-v2` have the interpretation
`exact_outputs_less_transfer_no_consistent_speedup`, with `confirmed: false`.
The final candidate is `.pixi/word-games-grammar-v5/candidate`; no production
endpoint or original runtime was replaced. Qwen3-8B readiness and configuration
were restored, and OCR remained ready.

## Final measurements

The control includes the preceding sampling-output acquisition patch. Each full
pass contains the same 36 frozen requests and produces 4,778 visible tokens.
Order is control/candidate, then candidate/control. Throughput includes the
whole HTTP completion request, including prefill. Server decode timing is a
separate measurement.

| Final implementation | Control tok/s | Candidate tok/s | Control seconds | Candidate seconds |
| --- | ---: | ---: | ---: | ---: |
| Full pair 1 | 47.433 | 48.608 | 100.731 | 98.297 |
| Full pair 2, reversed order | 49.645 | 47.555 | 96.243 | 100.473 |
| Pooled tokens / pooled seconds | 48.514 | 48.076 | 196.973 | 198.770 |

The candidate is 0.90% slower in pooled throughput. Median paired per-request
throughput ratios are 0.9731 and 0.9662; the pooled median is 0.9697, or 3.03%
slower. Only 8 of 72 paired requests were faster. Slow requests affect the
aggregate: for example, pair 1's control cancellation turn took 5.622 seconds
versus 2.729 for the candidate. No samples were removed. Pair 2's candidate
also had a slow metaphor request. These are repeated measurements on a shared
workstation, not independent model or task-population replicates.

Decode-only rates were control/candidate 66.143/70.025 in pair 1 and
70.450/68.103 in pair 2. Candidate prefill totals were 29.861 and 30.089 seconds,
versus control 28.291 and 28.219. Neither the decode rates nor the prefill
measurements establish a speed gain.

| Measured readback, each full process | Control | Final candidate |
| --- | ---: | ---: |
| Raw vocabulary calls | 4,822 | 2 |
| Raw vocabulary bytes | 4,789,596,160 | 1,986,560 |
| Compact preview bytes | 0 | 96,400 |
| Other instrumented GPU output bytes | 44,574,400 | 44,574,400 |

Both final repetitions have these same counters: **99.9585% fewer raw vocabulary
bytes**. Counters cover the entire server process, including startup, warmup,
requests and shutdown. They instrument actual `ggml_backend_tensor_get_async`
calls by backend device and tensor identity; they are not estimates of all
PCIe traffic or measurements restricted to the timed request window.

All 72 paired comparisons have exactly equal token IDs, text, parsed JSON and
stop conditions. Every full pass has 36 valid JSON objects and zero truncations.
The three-request pilot also matched exactly. This establishes preservation on
this frozen suite, including its existing semantic mistakes; it is not a new
semantic accuracy evaluation or evidence for a 95% application accuracy claim.

## Implementation and scope

Apply `native_sampling/grammar-on-demand.patch`, then
`native_sampling/grammar-partitioned-preview.patch`, on top of patch 1. The
explicit switch is `LLAMA_WORD_GAMES_GREEDY_PREVIEW=1`.

- Eligible requests use temperature-zero greedy sampling, grammar, no requested
  token probabilities or reasoning budget, and filters that preserve a unique
  finite maximum. Other requests retain the existing sampler.
- The backend returns two scores, two token IDs and a scalar finite-value
  guard: 20 bytes. The CPU chooses the higher of the two scores, without
  assuming backend top-k ordering, and checks that token against grammar.
- For an accepted unique finite maximum in a single-output batch, the full
  vocabulary stays on the device. Grammar rejection, ties, non-finite values,
  grammar-first sampling and explicit raw-logit reads materialize the original
  vocabulary and use the original CPU chain. A finite-sum overflow also falls
  back conservatively.
- Multi-output batches copy raw logits eagerly. A later microbatch or graph
  reservation materializes pending raw output before its storage can be reused.
  `llama_get_logits()` returns the full output; the added
  `llama_get_raw_logits_ith()` explicitly bypasses backend compact logits.
  Existing `_ith` sampled-output behavior remains available.
- The final backend selector finds two winners per 128-token group and then
  two global winners. Unsupported sizes use the original top-k operation.
  The context owns its stateless preview sampler across request lifetimes,
  including disable/reactivate transitions.

Model weights, numerical precision and the model's forward arithmetic are
unchanged. Only `libllama` and `libllama-common` differ from the matched control;
the server executable and CPU/CUDA backend libraries are byte-identical.
Configuration remains the cached Empero Qwen3.8-35B-A3B Q4_K_M, Q8_0 KV,
Flash Attention, context 8192, batch/ubatch 512, 12 CPU threads, seed 73,
generic JSON grammar, no speculation and no cross-request prefix reuse.
Packed38 expert placement spans RTX 3060, RTX 2060 SUPER and RTX 2060, with
two CPU expert layers and attention on the 3060.

## Validation and earlier attempts

The cached Qwen2.5-3B CUDA integration checks force known logits after a real
model forward pass. Accepted single-output cases read zero raw bytes plus
20 preview bytes; fallback cases read 607,744 raw bytes plus the preview.
Checks cover rejection, ties, NaN/infinities, default filter chains, high token
IDs, ties across token groups, successive decodes, explicit reads, clone/copy,
request-sampler replacement, preview off/on transitions, two simultaneous
outputs, an output before a later microbatch, and eight eligibility exclusions.
Patch 1's raw/logit/probability/token accessor compatibility checks also pass,
as does upstream `test-sampling`. Eight result-classification tests pass.

The first implementation is preserved as replay v1. It reduced raw downloads
by approximately 76%, but was consistently slower:

| First implementation | Control tok/s | Candidate tok/s |
| --- | ---: | ---: |
| Full pair 1 | 50.035 | 47.446 |
| Full pair 2 | 50.134 | 47.458 |
| Pooled | 50.085 | 47.452 |

All its replay outputs also matched. The follow-up was declared in an
implementation amendment and is reported separately. Its development tests
caught unordered top-k outputs causing unnecessary fallback, and double
initialization when reactivating a cached preview. Those failed tests remain
under native v3/v4 and v5; the corrected build v5 passes native v6. The initial
const-correctness compile failure remains under build v1. No failing full
replays were omitted or overwritten.

One remaining overhead is visible in the server logs: slot reset calls
`llama_set_sampler(..., nullptr)`, and subsequent registration still causes
roughly 40 ms of graph reservation per request. Persistent sampler ownership
avoids backend reinitialization but does not cancel that reservation. The
partitioned selector also uses several backend operations. Avoiding redundant
reservation and fusing selection plus the finite guard are plausible next
experiments; their benefit has not been measured here.

## Evidence and reproduction

All paths below are relative to the repository. Completed artifacts are
append-only. Source snapshots, commands, model/request hashes, runtime hashes,
native logs, GPU telemetry, responses and transfer counters are preserved.

| Evidence | Path | SHA-256 |
| --- | --- | --- |
| First replay | `data/experiments/word_games/20261031_grammar_replay_v1/campaign_results.json` | `867f7eab47c27045392c6f719f94efdc42063577206922359d2d40dd91de9a3f` |
| Final replay | `data/experiments/word_games/20261031_grammar_replay_v2/campaign_results.json` | `e8a52ad41788430e5ba16a76a7523a91d9ad15f5d8a92ca4cc3083581eafadf8` |
| Final build | `data/experiments/word_games/20261031_grammar_build_v5/result.json` | `0ab39a595c3cf239ffb11c77bb12609091dd229193ee831786bce87f702a199c` |
| Final native checks | `data/experiments/word_games/20261031_grammar_native_v6/result.json` | `1daeb2cba9523fb0a150f3f30c84fd42fa01704a6ccbc97e238f94e75315142b` |

See the [original plan](../07%20-%20Status%20Reports/2026-10-31_word-games-grammar-on-demand-plan.md),
[implementation amendment](../07%20-%20Status%20Reports/2026-10-31_word-games-grammar-reduction-amendment.md),
and [patch 1 build prerequisites](2026-10-31_word-games-sampling-output-once.md).
From the repository root, with that verified patch 1 build available, use fresh
paths for every command:

```sh
pixi run python -m experiments.word_games.prepare_grammar_runtime --root .pixi/NEW_GRAMMAR_BASE --output data/experiments/word_games/NEW_GRAMMAR_BASE_BUILD
pixi run python -m experiments.word_games.prepare_grammar_reduction --parent-manifest data/experiments/word_games/NEW_GRAMMAR_BASE_BUILD/result.json --root .pixi/NEW_GRAMMAR_FINAL --output data/experiments/word_games/NEW_GRAMMAR_FINAL_BUILD
pixi run python -m experiments.word_games.check_grammar_reduction --root .pixi/NEW_GRAMMAR_FINAL --build-manifest data/experiments/word_games/NEW_GRAMMAR_FINAL_BUILD/result.json --output data/experiments/word_games/NEW_GRAMMAR_NATIVE
pixi run python -m experiments.word_games.run_grammar_reduction --root .pixi/NEW_GRAMMAR_FINAL --build-manifest data/experiments/word_games/NEW_GRAMMAR_FINAL_BUILD/result.json --native-tests data/experiments/word_games/NEW_GRAMMAR_NATIVE --output data/experiments/word_games/NEW_GRAMMAR_REPLAY
```

The replay temporarily unloads Qwen3-8B from the regular 2060, enforces the
384 MiB free-memory stop rule, keeps OCR resident, and restores Qwen3-8B in
`finally`. Registration uses
`experiments.neural_architecture_lab.register_word_games_grammar_output` with
both completed `--campaign` paths, this `--report`, and a fresh `--output`.
It checks saved producer/response hashes, recomputes output parity, and verifies
the exact and searchable registry read-back. Records retain `confirmed: false`.
