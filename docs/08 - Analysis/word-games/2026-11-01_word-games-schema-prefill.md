# WORD_GAMES: public schema prefill reaches 70.92 tok/s, fails serving adoption

Completed exploratory study `word-games-schema-prefill-v1`.
Classification: `exploratory_speedup_requires_quality_review`.
Public prefix caching improved whole-request throughput **67.43 → 70.92 tok/s
(+5.18%)**, but introduced an incorrect sender action field in both repetitions.
The original uncached Triton configuration is restored and READY. Caching is
implemented as an optional backend feature and is **not enabled in serving**.

Aggregate: `data/experiments/word_games/20261101_schema_prefill_analysis_v1/campaign_results.json`

SHA-256: `6d58c54e8e930807f173452894198ed984c4488abb5e0514fbdabacdf52c0890`

Raw campaign: `data/experiments/word_games/20261101_schema_prefill_full_v1/result.json`

SHA-256: `0441472d054c0c00a5ba6f9d204f237b163b817ac12eb79fe52eaaabd296d0cd`

## Matched results

Four 36-request passes ran control/cached/cached/control through the same Triton
adapter, with startup/warmup outside steady-state request timing. Whole-request
timing includes HTTP, formatting, tokenization, prefix restoration, remaining
prefill and generation. Weights, precision, context, sampling and GPU placement
were unchanged from `free_whole1024`; no speculative decoding was used.

| Measurement, two passes each | Uncached | Public prefix cached |
| --- | ---: | ---: |
| Whole-request output tok/s | 67.431 | 70.925 |
| Individual pass tok/s | 67.469 / 67.394 | 70.846 / 71.003 |
| Request wall time | 157.138 s | 146.578 s |
| Visible output tokens | 10,596 | 10,396 |
| Native prefill time | 34.070 s | 22.223 s |
| State restore time, included above | 0 | 2.105 s |
| Receiver decision/occupied-slot checks | 24/24 | 24/24 |
| Valid metaphor selectors | 2/2 | 2/2 |

Prefill time fell **34.77%**. Total request time fell **6.72%**. Output length
also changed, so the latency difference is not entirely a fixed-output kernel
comparison. Both controls exactly reproduced all prior baseline parsed outputs;
the two cached passes exactly reproduced each other.

All 70 eligible cached requests reused exactly their configured public prefix;
the two selector requests remained uncached. Saved lengths were 466 sender
tokens and 138 receiver tokens, including chat framing/static delimiters and
excluding the final boundary token. Total reuse was 24,748 tokens. There were
no unexpected misses or extra reused private/event tokens.

## Quality failure and mechanism

Only **10/72 complete parsed outputs** matched their paired controls. Most
differences were explanatory prose, but **2/72 comparisons changed an action
field**: the same `outside_destination/sender/turn_01` case in both repeats.
The cached proposal used `slot: "shelf-position"` instead of the requested
`slot: "A1"`. The metaphor's name for a slot was substituted for its identifier.
This is an additional incorrect field, not just a wording difference.

The baseline already mishandles this request by substituting `gal:north` for
the requested Archive Vault destination. Neither arm passes that full task.
The 24/24 receiver result uses fixed authorized input packets and does not grade
live packets produced from these sender outputs. It cannot establish end-to-end
accuracy or a 95% quality claim. Manual review also found retained unsupported
approval statements in two cached receiver summaries; three corresponding
baseline approval claims disappear. No independent semantic judge was used.

A separate post-outcome diagnostic evaluated the changed-slot request with
split prefill **without saving or restoring a snapshot**. Its complete parsed
output exactly matched the cached result; fresh whole-prompt prefill exactly
matched control. Thus the batch split alone reproduces this difference; the
snapshot restore operation is not necessary. Floating-point differences caused
by changed batch shapes are a plausible explanation, but exact logits and
numerical error were not measured.

The predeclared automatic adoption gate required a speed gain, all existing
checks, and all 72 paired outputs matching. That gate failed. The action-field
error provides an additional substantive reason to retain baseline serving.

## Cache cost and isolation

Native sequence snapshots contain both Q8 attention KV and recurrent state.
The sender file is 70,947,840 bytes and the receiver file is 67,370,016 bytes:
**131.91 MiB total host files**, mostly recurrent state. Measured construction
took 0.945 and 0.929 seconds after model readiness. Mean restore time was
**30.07 ms per hit**, already included in the throughput result.

The measured request-time gain amortizes this extra construction cost after
approximately 6.3–6.4 requests at the measured mix. This estimate includes
output-length changes and excludes ordinary model loading; it is not a general
startup or cold-storage guarantee. Files were read shortly after creation, so
OS page caching was available; no cold-disk restoration benchmark was run.

Snapshots are built anew on each model load from operator-configured public
contracts only. Each request verifies exact prefix tokens and restores from
the appropriate sender/receiver snapshot; unknown prefixes run uncached.
No event data, retrieval results, authorized packets or private workspace are
saved into reusable snapshots. A single serialized slot prevents concurrent
restore/generate interleaving. Files are removed when the model unloads.

Observed minimum GPU free memory remained 1,957 MiB on the 3060, 1,027 MiB on
the regular 2060 and 531 MiB on the 2060 SUPER, above the 384 MiB guard. The
original prepared repository and manifest were restored byte-for-byte; a real
Triton inference returned the expected ready object. The existing Qwen3-8B/OCR
containers retain their prior stopped state and unchanged configurations.

## Integrity and reproduction

The [plan](../../07%20-%20Status%20Reports/word-games/2026-11-01_word-games-schema-prefill-plan.md)
and [lifecycle amendment](../../07%20-%20Status%20Reports/word-games/2026-11-01_word-games-schema-prefill-lifecycle-amendment.md)
preceded the full replay. Source, inputs, weights identity, native binary hashes,
runtime manifests, responses and per-load cache receipts were preserved.
Repetitions measure one model/corpus; they are not independent unseen datasets.

Three lifecycle attempts stopped before timed replay: an absolute-path source
snapshot error, a too-strict interpretation of the runtime's zero-token request,
and a public directory permission issue for host evidence collection. The
successful v4 pilot verified role switches, unrelated intervening requests,
exact prefix lengths and stable repeated outputs. In this pinned runtime,
`n_predict=0` samples one unused token but does not feed it back into state;
both the cached length and saved length are checked against the public prefix.
All failures remain in their original evidence roots.

Ten focused backend tests passed, including prefix isolation, receiver/sender
separation, every-request restore, boundary mismatch fallback, rejecting extra
cached tokens, request connection handling and graceful model shutdown.

Run from repository root with Pixi and fresh output paths:

```bash
pixi run python -m experiments.word_games_triton.schema_prefill --pilot --output <fresh-pilot-root>
pixi run python -m experiments.word_games_triton.schema_prefill --output <fresh-full-root>
pixi run python -m experiments.neural_architecture_lab.analyze_word_games_schema_prefill --campaign <full-root>/result.json --output <fresh-analysis-root>
```

Supplementary evidence is in `20261101_schema_prefill_state_diagnostic_v1`,
`20261101_schema_prefill_restoration_check_v1`, and the analysis root's
`manual_review.json` and `analysis_producer.json`, all under
`data/experiments/word_games/`. Registry metadata remains `confirmed: false`.
