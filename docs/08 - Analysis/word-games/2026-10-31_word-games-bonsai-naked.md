# Standalone Bonsai 2 27B on WORD_GAMES

Completed exploratory run, 2026-10-31. Hypothesis:
`word-games-bonsai27-naked-v1`. Classification: `higher_joint_task_success`;
`confirmed: false`. Bonsai improves the structured bridge result from Qwen's
5/12 to 10/12 while using less GPU memory, but is slower on this workload and
adds unsupported approval claims to receiver prose. This is not a deployment
or 95% task-accuracy result.

Aggregate: `data/experiments/word_games/20261031_bonsai_analysis_v1/campaign_results.json`.
SHA256: `36e66862e643be4beed9c028b2b2472b88bb9c7938dd149fca8e81fcb30ea612`.
The [prospective protocol](../../07%20-%20Status%20Reports/word-games/2026-10-31_word-games-bonsai-naked-plan.md)
fixes the sequence, settings and interpretation.

## Configuration and integrity

Official [Bonsai 2 27B Q2_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
revision `94dd51cdbbec5309ed8b76a12f3268bc4fdceb93`, 7,626,008,928 bytes;
checkpoint SHA256 `4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e`.
No training, adapters, draft model or lookup. The existing native llama.cpp
b11960 runtime loaded 402 Hadamard-folded weights, including one inverse lookup,
and offloaded 65/65 layers to the RTX 3060. Its CPU-mapped embedding/model buffer
is 341 MiB; full layer offload does not mean zero host memory.

Thinking is disabled through the model's embedded template and verified on
each rendered prompt. Temperature 0, seed 73, generic JSON-object constraint,
selector cap 256, other output caps 1024, context 8192, Q8 KV, batch/ubatch 512,
four CPU threads and no cross-request prompt cache. This is the short-output
application condition, not the authors' long-thinking benchmark or recommended
non-thinking sampling settings. The official template trims surrounding user
whitespace and supplies an empty, closed thinking prefix; Qwen's native
template supplies its default system message. Those template differences are
part of the model comparison.

The first pilot passed model loading and JSON warm-up, then stopped before any
pilot outcome: the input-preservation check incorrectly rejected native
trailing-newline trimming. Its evidence remains in `20261031_bonsai_pilot_v1`.
The explicit protocol amendment permits only that whitespace normalization;
the corrected v2 pilot passed all three inputs. No prompt, checkpoint, sampling
or task gate was changed to improve an outcome. All subsequent calls completed
without retries. Producing sources, runtime files, model and data are hashed;
source snapshots and original failures remain append-only.

## Speed and memory

Both target-only models completed the same 36 frozen user requests on the same
RTX 3060. Each produced 36 valid JSON objects, with zero truncations. The Qwen
control is a fresh full replay, not an extrapolation from its earlier pilot.

| Measurement | Bonsai 2 27B Q2_0 | Qwen2.5-14B Q4_K_M |
| --- | ---: | ---: |
| Visible output tokens / full request second | 24.31 | 27.46 |
| Native decode-only tokens/s | 32.18 | 32.92 |
| UTF-8 output bytes/s | 79.42 | 88.89 |
| Total completion-request time, 36 requests | 170.39 s | 123.77 s |
| Output tokens | 4,143 | 3,399 |
| Output bytes | 13,533 | 11,002 |
| Native prefill time | 41.45 s | 20.35 s |
| Native decode time | 128.74 s | 103.24 s |
| Total GPU memory used immediately after load | 7,950 MiB | 9,526 MiB |

Completion-request timing includes prefill and HTTP completion overhead;
template application, model loading, hash checks, warm-up and artifact I/O are
excluded. GPU figures are point-in-time whole-device usage including a 291 MiB
baseline, not measured peaks. Native model-load times were 4.83 and 4.39 seconds,
after file verification. The two unrelated GPU services remained resident.

Bonsai took 37.7% longer for the frozen suite. It produced more text and spent
about twice as long in prefill; decode rate was close. Different tokenizers mean
tokens/s is not an equal-text comparison, so bytes and full paired workload
time are included. This is one sequential pass per arm, with no confidence
interval. Bonsai used diagnostic verbosity 4; Qwen retained its default 3.
The results establish no speedup and do not meet 40 tok/s.

## Live task result

The model selected the `museum storage` metaphor from the public task.
Twelve live cases used the prior Qwen baseline's synthetic context IDs after
authorization metadata checks. The receiver saw only the host's authorized
translated packet. Metaphors and generated tool-call paths can differ by model.

| Measurement | Bonsai | Prior target-only Qwen |
| --- | ---: | ---: |
| Correct sender operations | 10/12 | 5/12 |
| Expected host boundary outcome | 12/12 | 9/12 |
| Correct receiver decision and structured state | 12/12 | 12/12 |
| All three correct in the same case | **10/12 (83.3%)** | **5/12 (41.7%)** |
| Canary disclosures / forbidden source identifiers | 0 | 0 |
| Completion calls including selector | 27 | 35 |
| Total completion-request time | 118.79 s | 107.48 s |

Bonsai preserved all five prior structured successes and added normal reserve,
prompt injection, outside entity, stale revision and changed due date. There
were no component regressions. Both remaining failures substituted names for
canonical identifiers: `Archive Vault A1` instead of `gal:archive`, and `North`
instead of `gal:north`. The host blocked both transactions, as expected, but
their sender operations failed the fixed exact-action contract. These negative
cases deliberately ask the sender to propose invalid operations so the host
can reject them; their score measures protocol fidelity.

Four conditional private-variation checks also passed. This is evidence for
these synthetic checks, not proof that all possible covert channels are closed.

## Semantic review of receiver summaries

The structured receiver check verifies fields, state and nonempty prose; it
does not judge the prose's factual meaning. A separate manual assistant review
compared every generated summary with its authorized packet, without reference
summary exact matching. It found:

- Eleven summaries assert unprovided approval events: successful actions become
  “The curator signed off,” and blocked actions become “The curator sign-off
  was denied.” The packet's metaphor dictionary names approval but does not
  establish that such an event occurred or explain a rejection.
- The approved-annotation summary describes recording a sign-off instead of
  recording the approved condition check. This is a weaker, interpretive
  fidelity concern. Placement state is still correct.
- The placement dates, including the changed `2027-01-11`, are preserved in
  structured state and placement prose. No tested private canary was disclosed.

This review is not an independent judge-model evaluation, and repeated packets
and identical summaries are not independent semantic trials. Its 12 flagged
summaries must not be converted into a calibrated population accuracy estimate.
The 83.3% number above is explicitly **joint structured case success**, not
overall semantic accuracy. The original raw records retain
`receiver_semantic_review: not_performed`; the later review is separate,
hash-bound evidence rather than an edit to those records.

Review: `data/experiments/word_games/20261031_bonsai_manual_review_v1/review.json`.
SHA256: `991dc187a25f1ec4c612ced0bd22603b9f08cc1654d82b950de3db2ea8b20ba3`.

The next useful experiment is to distinguish metaphor vocabulary from factual
events in the receiver prompt, then evaluate the resulting summaries separately
from state accuracy. This run leaves the prompts and active Qwen default intact.

## Evidence and reproduction

All paths below are under `data/experiments/word_games/`:

- `20261031_bonsai_models_v1`: pinned download provenance.
- `20261031_bonsai_pilot_v1`, `20261031_bonsai_pilot_v2`: preserved startup failure
  and corrected lifecycle check.
- `20261031_bonsai_full_v1`: 36 frozen Bonsai requests and runtime memory/logs.
- `20261031_bonsai_live_v1`: 12 live cases, packets, proposals and raw answers.
- `20261031_qwen_control_full_v1`: fresh 36-request target-only Qwen control.
- `20261031_qwen_baseline_live_v1`: prior 12-case target-only comparison.
- `20261031_bonsai_analysis_v1`: hash-checked aggregate and registry readback.

Run from the repository root with fresh output paths:

```sh
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.prepare_bonsai --output data/experiments/word_games/NEW_BONSAI_DOWNLOAD
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.bonsai_runtime --stage pilot --output data/experiments/word_games/NEW_BONSAI_PILOT
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.bonsai_runtime --stage full --output data/experiments/word_games/NEW_BONSAI_FULL
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.bonsai_runtime --stage live --output data/experiments/word_games/NEW_BONSAI_LIVE
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.qwen_runtime --stage full --output data/experiments/word_games/NEW_QWEN_CONTROL
```

Controllers use the locked Pixi environment; weights reside in the model store,
and native executable code remains under `.pixi/`. Each run closes its own
server. The RTX 2060 Qwen3-8B service and RTX 2060 SUPER service were preserved.
Focused verification: 72 WORD_GAMES tests passed. The conservative registry
records the measured structural classification with `tested: true`,
`confirmed: false`; its exact and semantic readback receipt is stored beside
the aggregate.
