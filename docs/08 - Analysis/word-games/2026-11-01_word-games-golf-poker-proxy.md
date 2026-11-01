# Golf played through poker: proxy result

Completed exploratory diagnostic `word-games-golf-poker-proxy-v1`. The poker-only
player preserved optimal play in this small executable game: 15/15 holes finished,
56/56 optimal actions, and the same 58 realized strokes as direct golf. This is
a model-generated poker vocabulary over a host-defined golf simulator, not
ordinary poker rules or a physical golf simulation.

The requested approximately 10K run produced **9,291 visible output token
equivalents**, plus **40,021 input token equivalents**, through
`codex_sdk/gpt-5.6-sol` on the central LiteLLM proxy. The returned alias matched
every request; its underlying checkpoint identity was not independently exposed.

## Measured performance

| Measurement | Direct golf | Poker-only view |
| --- | ---: | ---: |
| Completed holes | 15/15 | 15/15 |
| Legal actions | 56/56 | 56/56 |
| Expected-cost optimal actions | 56/56 (100%) | 56/56 (100%) |
| Optimal actions with multiple legal choices | 37/37 | 37/37 |
| Correct cumulative ledger | 56/56 | 56/56 |
| Realized strokes | 58 | 58 |
| Expected excess strokes from chosen actions | 0 | 0 |
| Visible output token equivalents | 4,592 | 4,616 |
| HTTP request seconds | 292.05 | 357.19 |
| Output token equivalents/second | **15.72** | **12.92** |

Including the 83-token mapping response, the aggregate is **14.09 token
equivalents/s over 659.24 request-seconds**, about 11 minutes. There were 29
successful measured calls: one mapping call and 28 playing calls. Median request
latency was 19.52 seconds; maximum was 75.20 seconds. The poker arm's observed
throughput was 17.8% lower than direct golf. One sequential run with varying
response lengths and backend latency does not establish a stable overhead ratio.

These are **Qwen-token-equivalent visible-text counts**, not native provider
usage. The SDK proxy returned zero prompt, completion and total tokens even for
nonempty responses. We therefore used the deployed Qwen GGUF tokenizer's read-only
API, excluded special tokens and independently recomputed every count. Inputs
exclude chat framing and any SDK-injected context; hidden reasoning and native
billed tokens are unknown. The clock includes the HTTP proxy round trip, prompt
processing, model work and response delivery. It does not isolate decoding,
time to first token or proxy coordination overhead. It is not a measurement of
the local three-GPU model's generation speed.

## What actually ran

The model generated a bijective vocabulary, including driver → `all_in`, iron →
`flat_call`, putt → `river_bet`, tee → `preflop`, and holed → `showdown`. The host
preserved exact transition probabilities and stroke costs while encoding them
as poker states, moves and chips spent. Player requests supplied only the
appropriate arm's vocabulary and current synthetic game state. No mapping
dictionary or golf source view was included in poker player requests.

The model chose actions; the host sampled outcomes, applied penalties and kept
the authoritative ledger. The host never replaced a proposal with an oracle
action. A shared random stream by hole/move paired the arms, whose order alternated
each wave. The run finished complete paired blocks and stopped after reaching
the prespecified minimum of 9,000 output reference tokens. No answer was retried
or repaired, and no holes failed.

The operational explanation also survived translation. On hole 1, the poker
player valued `all_in` at about 4.16 remaining chips versus 4.37 for `flat_call`,
matching the golf oracle's driver-versus-iron choice. At the next position it
chose `min_raise`, correctly preferring its approximately 2.54 expected cost over
2.61 for `three_bet`. These are examples, not a separate exhaustive prose grade.

The judge was executable mathematics, not an LLM self-rating or exact summary
matching. Value iteration computed the optimal expected-stroke policy before
inference. An independent linear-system policy-iteration audit agreed within
8.49e-12, replayed all 112 decisions, checked every supplied transition table,
ledger and sampled outcome, and found no discrepancies. The expected optimum
across these 15 courses was 66.85 strokes; the realized 58 reflects favorable
sampled outcomes. Realized strokes versus par are not the accuracy metric.

This establishes task preservation for a small, explicitly specified state
machine. The host supplies the semantics and exact numeric transition model;
the model generates vocabulary and chooses actions. It does not establish
autonomous discovery of a full domain mapping, natural-language ambiguity
handling, adversarial information-firewall security or population accuracy.
The original V3 placement codec and V4 recursive capsules were not integrated.
The local serving configuration remained unchanged.

## Evidence and reproduction

The first `claude-sonnet-5` attempt failed before generation because the Anthropic
route reported insufficient credits. The Sol attempt produced the valid mapping
but stopped on its zero usage receipt. The final attempt imported that exact
mapping response and original elapsed time, then made fresh playing calls. All
attempts remain preserved; the mapping was counted once, not regenerated.

Final evidence root: `data/experiments/word_games/20261101_golf_poker_proxy_v3`.

- `result.json` SHA256:
  `d13ef6b55efaa848a87c7a8417d321e85d43e6ead950689d3ea6343e12f2868f`.
- `audit.json` SHA256:
  `7f9795fb7c6b6af0f4c8bdb78855099d57b47fd787f8d67492ffc0ffb0901ad3`.
- Every request, raw response, usage receipt, latency, course, oracle and state
  transition is retained. Producer source and tokenizer manifest are hash-bound.
- Three focused tests passed before generation. The final independent audit
  verified the complete trajectory and token counts.

The [plan](../../07%20-%20Status%20Reports/word-games/2026-11-01_word-games-golf-poker-proxy-plan.md),
[route amendment](../../07%20-%20Status%20Reports/word-games/2026-11-01_word-games-golf-poker-proxy-route-amendment.md)
and [token measurement amendment](../../07%20-%20Status%20Reports/word-games/2026-11-01_word-games-golf-poker-proxy-token-amendment.md)
record the design and transport changes before gameplay.

```sh
pixi run pytest tests/word_games/test_golf_poker_proxy.py -q
pixi run python -m experiments.word_games.golf_poker_proxy --model codex_sdk/gpt-5.6-sol --output data/experiments/word_games/NEW_GOLF_POKER
pixi run python -m experiments.word_games.audit_golf_poker_proxy --result data/experiments/word_games/NEW_GOLF_POKER/result.json --output data/experiments/word_games/NEW_GOLF_POKER/audit.json
```

Use fresh directories. Reproduction requires the configured proxy credentials
and the existing local tokenizer service. The shared seed fixes simulator
randomness, not remote model determinism.
