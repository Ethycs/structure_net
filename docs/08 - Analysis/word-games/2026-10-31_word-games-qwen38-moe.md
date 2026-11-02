# Qwen3.8-35B-A3B distillation with 26 CPU MoE layers

Completed exploratory attempt, 2026-10-31.
Hypothesis: `word-games-qwen38-moe-v1`. Classification:
`metaphor_gate_failed`; `confirmed: false`.

The requested Q4_K_M / Flash Attention / Q8 KV / 26 CPU MoE configuration runs
successfully. Its 36-request replay measured **24.37 output tokens/s including
prompt processing** and **37.68 decode-only tokens/s**. The live bridge stopped
at metaphor selection because the model reused one label for two distinct
roles. **No live workflow accuracy percentage was measured.**

Aggregate: `data/experiments/word_games/20261031_qwen38_analysis_v1/campaign_results.json`.
SHA256: `c2ce2af11ded623c274fa25059d8fbe1a5589c19e5ce805b43bf163f401a60c7`.
The [prospective protocol](../07%20-%20Status%20Reports/2026-10-31_word-games-qwen38-moe-plan.md)
records configuration, sequence and stop rules.

## Exact model and configuration

The available named release is
[Empero's Qwen3.8-35B-A3B distillation](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill),
which uses the Qwen3.6-35B-A3B backbone. The public Qwen organization search
did not return an official Qwen3.8-35B-A3B release. This result applies to the
identified community model and quantization.

The [publisher's Q4_K_M GGUF](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF)
is pinned to revision `b1f9d1dcc3de8aa867669b0ab919384aeeb9b8d5`, file
`Qwen3.8-35B-A3B-Q4_K_M.gguf`, 21,713,462,944 bytes, SHA256
`196103269085bc54c9b8f49ed21e9f53e1b56b465e8b796c6d8e31e06f63cfa5`.
Its GGUF identifies `qwen35moe`, 40 main layers plus one MTP layer, 256 experts
with 8 routed per token. MTP/speculation was disabled.

Exact serving settings:

- Native llama.cpp b11960 in the existing pinned Ubuntu runtime image.
- RTX 3060, GPU layers 99, first **26 routed-expert blocks (0–25) on CPU**,
  automatic fitting off, single GPU, no draft or lookup.
- **Q4_K_M weights**, **Flash Attention on**, **Q8_0 K and V cache**.
- Context 8192, batch/ubatch 512, 12 CPU threads on the 12-core Xeon E5-2697 v2,
  load mode `none`, diagnostic verbosity 5.
- Thinking disabled through the embedded template; temperature 0, seed 73,
  the same generic JSON constraint and caps as the preceding trials: selector
  256, other calls 1024. No cross-request prompt cache and no training.

The [publisher recommends long reasoning outputs with stochastic sampling](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill).
Our bounded, non-thinking condition is an application test, not validation or
refutation of that recommended mode. Task prompts and firewall policy were
unchanged. The receiver still gets only authorized translated packets.

The runtime confirms all 78 routed-expert tensors in blocks 0–25 use
`CUDA_Host`, which is pinned CPU memory. It reports 7,878.90 MiB of CUDA model
buffers and 12,296.81 MiB of host model buffers. Its generic “42/42 layers
offloaded” log must not be interpreted as all weights executing on GPU: the
explicit CPU expert overrides remain active. The native b11960 model
[implementation supports this hybrid architecture](https://github.com/ggml-org/llama.cpp/blob/b11960/src/models/qwen35moe.cpp).

## Full frozen replay

All configurations below use the existing 36 frozen user prompts. Prior
controls were reused from completed evidence, not rerun. Tokenizers and output
lengths differ, so token rates alone do not measure equal work.

| Measurement | Qwen3.8 distill, 26 CPU expert layers | Bonsai 2 27B | Qwen2.5-14B |
| --- | ---: | ---: | ---: |
| Output tokens / full completion-request second | **24.37** | 24.31 | 27.46 |
| Native decode-only tokens/s | **37.68** | 32.18 | 32.92 |
| Output UTF-8 bytes/s | 87.40 | 79.42 | 88.89 |
| Summed request time | 199.48 s | 170.39 s | 123.77 s |
| Visible output tokens | 4,861 | 4,143 | 3,399 |
| Output bytes | 17,435 | 13,533 | 11,002 |
| Native prefill time | **70.22 s** | 41.45 s | 20.35 s |
| Native decode time | 129.00 s | 128.74 s | 103.24 s |
| Valid JSON / requests | 36/36 | 36/36 | 36/36 |
| Truncations | 0 | 0 | 0 |
| Whole-device GPU usage after loading | 8,938 MiB | 7,950 MiB | 9,526 MiB |

Qwen3.8's post-load GPU usage is about **8.7 GiB**. Server RSS was
13,091,760 KiB (about **12.5 GiB**) after loading and 13,314,900 KiB after the
replay; observed RSS high-water mark was 13,379,316 KiB, with zero server swap.
These process/device measurements have different accounting scopes and should
not be added into a single exact model-size number. GPU figures include the
existing 291 MiB baseline and are snapshots, not measured GPU peaks.

The three-request pilot reached 25.08 full-request and 41.86 decode-only tok/s;
the full replay above supersedes that small pilot for throughput claims.
The model did not sustain 40 tok/s over the full replay, even on the decode-only
metric. Historical target-only Phi reached 27.29 full-request and 34.12
decode-only tok/s on these prompts.

Completion-request time includes prefill and HTTP completion overhead. It
excludes checkpoint verification, model loading, warm-up, template application
and artifact I/O. The full replay's native model-load time was 20.65 seconds
after verification. This one-pass comparison changes model, quantization,
tokenizer, placement, CPU threads and diagnostic verbosity; it does not isolate
MoE architecture as a causal speed improvement. Qwen3.8 generated substantially
more text and spent 35.2% of measured request time in prefill. It was not faster
at completing this suite.

## Live bridge gate

The model-selected metaphor was:

```json
{"name":"warehouse","entity":"cargo","location":"dock","slot":"bay",
 "place_verb":"park","return_verb":"ship","approval":"gate","deadline":"dock"}
```

The unchanged selector prompt explicitly requires distinct entity, location,
slot, approval and deadline labels. `location` and `deadline` are both `dock`.
The host correctly raised `distinct role labels required`. The same selection
appeared in the pilot and frozen replay; there was no model retry or repair.

The live attempt completed one selector request, then stopped with
`requested: 12`, `completed: 0`, `status: metaphor_selection_failed`.
Joint task accuracy is **null/unmeasured**, not 0/12 and not 12/12. Live canary,
boundary and receiver checks were not reached. Thus this model has not matched
the completed 10/12 structured results of Phi or Bonsai in the current bridge.
This is a specific instruction/format failure under the tested condition,
not a general estimate of model capability.

## Separate frozen-receiver diagnostic

After inspecting the selector problem, the assistant reviewed all twelve
receiver outputs already generated by the frozen replay. Those prompts supply
a preselected library metaphor and preconstructed authorized packets, bypassing
the failed selector and sender/host workflow. This is a **post-hoc diagnostic**,
not a substitute live arm or an independent semantic judge.

All twelve copied decision and structured occupied-slot state correctly.
Five summaries nevertheless invented an applied librarian approval absent from
their packets: normal reserve, private note, private title, changed due and
cancellation. The approved-annotation summary also blurred recording a condition
check with applying approval, a weaker interpretive concern. The six blocked
summaries were consistent with their packets when read as descriptions of the
requested change. Repeated packets and identical answers are not independent
semantic examples. Dates, including the changed `2027-01-11`, were preserved.

Diagnostic: `data/experiments/word_games/20261031_qwen38_replay_review_v1/review.json`.
SHA256: `cde24d9211cc522bcf14bcae27bbfe930961a6c90c0f794a855eebfd2a650556`.

The next useful study would test a bounded selector-repair policy and the
model's recommended reasoning mode as explicit new conditions. This run did
not change prompts or repair the selector to turn the failure into a pass.

## Integrity, recovery and reproduction

The v1 lifecycle worker was stopped during startup, before any model call,
because its placement guard would reject the native name `CUDA_Host`. The
guard was corrected to recognize pinned CPU memory while still rejecting
device buffers, missing projections and any expert blocks outside 0–25.
The explicit amendment, original source/logs and operator note are preserved.
The corrected v2 pilot passed. No requested runtime setting changed.

Raw roots under `data/experiments/word_games/` are
`20261031_qwen38_models_v1`, `20261031_qwen38_research_v1`,
`20261031_qwen38_pilot_v1`, `20261031_qwen38_pilot_v2`,
`20261031_qwen38_full_v1`, `20261031_qwen38_live_v1`,
`20261031_qwen38_replay_review_v1`, and `20261031_qwen38_analysis_v1`.
Sources, checkpoint, runtime and inputs are fingerprinted and frozen. Failed
and completed evidence is append-only. The searchable registry records
`tested: true`, `confirmed: false`, `metaphor_gate_failed` and zero completed
live cases; exact and semantic readback are saved beside the aggregate.

Use fresh output directories from the repository root:

```sh
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.prepare_qwen38_moe --output data/experiments/word_games/NEW_QWEN38_DOWNLOAD
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.qwen38_moe_runtime --stage pilot --output data/experiments/word_games/NEW_QWEN38_PILOT
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.qwen38_moe_runtime --stage full --output data/experiments/word_games/NEW_QWEN38_FULL
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.qwen38_moe_runtime --stage live --output data/experiments/word_games/NEW_QWEN38_LIVE
```

Executable code/environments remain outside `data/`; weights are in the model
store. Test workers are stopped after each run. The resident RTX 2060 Qwen3-8B
and RTX 2060 SUPER service were preserved. Focused verification: 76 WORD_GAMES
tests passed; no production model switch was made.
