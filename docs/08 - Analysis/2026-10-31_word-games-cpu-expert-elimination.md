# WORD_GAMES: removing the last CPU expert banks

Status: completed exploratory comparison, 2026-10-31. Hypothesis:
`word-games-cpu-expert-elimination-v1`. Classification:
`exploratory_speed_and_cpu_cost_measured_quality_unestablished`;
`confirmed: false`. No production deployment or model training.

Aggregate:
`data/experiments/word_games/20261031_cpu_experts_analysis_v1/campaign_results.json`.
SHA256: `80f774140ee2c8deeadb034527164aa670101c51a354385e0737774b19f0e615`.

**Moving the remaining two expert banks to GPUs, with unchanged weights,
increased full-request throughput from 49.60 to 56.26 tok/s (+13.4%).** Server
CPU time per output token fell 70.5%, and maximum observed server RSS fell
934 MiB. Further compression was unnecessary to eliminate CPU expert work.

Selectively requantizing those two banks to Q2_K saved 426 MiB and allowed both
to reside on the 3060. That variant reached 55.65 tok/s, slightly below the
unchanged-weight all-GPU placement. Reducing CPU compute threads from twelve
to one after moving the experts gave no measurable additional benefit.

## Design and completed attempts

The [prospective plan](../07%20-%20Status%20Reports/2026-10-31_word-games-cpu-expert-elimination-plan.md)
fixed four arms, a three-request pilot per arm, and two full 36-request replays.
All four pilots and eight full passes completed: **300 timed requests**, zero
failed or excluded cells, no retries. Warmups are separate and excluded.
The full-pass order was baseline, GPU40, GPU40/CPU1, Q2; then the reverse.
Every pilot passed the predefined validity and 90%-of-baseline throughput
gate. This gate admitted measurement; it did not establish a speed win.

Hardware was the local RTX 3060 12 GB, RTX 2060 SUPER 8 GB, regular RTX 2060
6 GB, and Xeon E5-2697 v2. Logical CUDA order was 3060, SUPER, regular 2060.
The regular 2060's Qwen3-8B service was temporarily unloaded; OCR remained
resident on the SUPER. Qwen3-8B readiness and identical service configuration
were restored after the campaign, with OCR ready.

The model was the existing Empero Qwen3.8-35B-A3B Q4_K_M GGUF. All 256 experts
and top-8 routing were retained. Attention, recurrent and shared computation
remained on the 3060. Context was 8,192; batch and microbatch were 512; KV was
Q8_0; Flash Attention was enabled. Temperature was zero, seed 73, JSON grammar
enabled, with no speculative decoding or prompt-cache reuse.

All arms used the verified **sampling-output acquisition patch only**, from
`.pixi/word-games-sampling-v2/candidate`, built from llama.cpp b11960. The later
grammar-preview patch was not included: its preceding study reduced transfers
but did not improve throughput. The present gains are against the acquisition
patch baseline, not an unpatched runtime.

| Arm | CPU routed banks | 3060 routed banks | SUPER routed banks | Regular 2060 routed banks | CPU compute threads |
| --- | --- | --- | --- | --- | ---: |
| `packed38` | 24–25 | 18–22, 26–39 | 0–7, 23 | 8–17 | 12 |
| `gpu40` | none | 18–22, 25–39 | 0–7, 23 | 8–17, 24 | 12 |
| `gpu40_cpu1` | none | 18–22, 25–39 | 0–7, 23 | 8–17, 24 | 1 |
| `q2_3060` | none | 18–22, 24–39 | 0–7, 23 | 8–17 | 1 |

Only banks 24 and 25 have changed precision in `q2_3060`. “On the 3060” means
those two formerly CPU banks; the whole model still spans all three GPUs.

## Throughput and CPU cost

Primary throughput is visible output tokens divided by full completion-request
wall time, including prefill. These are pooled rates, not averages of individual
request rates. CPU seconds are server-process user plus system time sampled
around each complete call; this also includes template handling. That scope is
slightly wider than the completion wall timer and is not a kernel profile.

| Arm | Pass 1 tok/s | Pass 2 tok/s | Pooled tok/s | Change | CPU ms/output token | Max observed RSS MiB | Max observed server threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `packed38` | 49.457 | 49.743 | 49.599 | — | 60.809 | 2,285.1 | 46 |
| `gpu40` | 56.277 | 56.235 | 56.256 | +13.42% | 17.924 | 1,351.4 | 35 |
| `gpu40_cpu1` | 56.258 | 56.260 | 56.259 | +13.43% | 17.922 | 1,351.6 | 35 |
| `q2_3060` | 55.563 | 55.739 | 55.651 | +12.20% | 18.114 | 1,366.9 | 35 |

| Arm | Output tokens, both passes | Request seconds | Visible bytes/s | Decode-only tok/s | Total prefill seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `packed38` | 9,556 | 192.664 | 181.622 | 70.212 | 56.152 |
| `gpu40` | 9,686 | 172.178 | 208.296 | 78.571 | 48.504 |
| `gpu40_cpu1` | 9,686 | 172.168 | 208.309 | 78.560 | 48.477 |
| `q2_3060` | 9,146 | 164.346 | 201.575 | 79.120 | 48.356 |

The unchanged-weight placement improved in both passes. Its median paired
request throughput ratio was 1.1233, consistent with the pooled gain. The
0.0034 tok/s separation between the two unchanged-weight GPU arms is negligible;
the registry's numerically fastest arm is `gpu40_cpu1`, but this is a practical
tie. Their generated outputs were identical on all requests.

Removing CPU experts reduced observed process threads from 46 to 35 even with
the twelve-thread setting. Setting that knob to one did not remove more threads
or improve throughput. The remaining CPU work includes native orchestration,
sampling and driver activity; this is not CPU-free inference or removal of
Python worker processes. Host model buffers, tokenizer data and staging still
use CPU RAM. RSS readings are observations after requests, not continuous
peak-memory instrumentation.

Q2's shorter total request time partly reflects fewer generated tokens: 4,573
per pass versus 4,843 for unchanged GPU placement. Its decode-only rate was
slightly higher, but full-request token and byte throughput were lower. A fixed
token replay or kernel trace would be needed to separate arithmetic, transfers
and changed generation paths precisely.

## Compression and memory

Original banks 24 and 25 occupy 432 and 498 MiB respectively, totaling 930 MiB.
Requantizing their six routed up/gate/down projections to Q2_K produced 504 MiB:
**426 MiB saved, or 45.81% of these banks**. The derived file is about 2.06%
smaller overall. All other **747 tensor payloads were verified byte-identical**;
model, tokenizer and template metadata were unchanged. Q2_K is lossy weight
compression. The unchanged-weight placement does not introduce this loss,
although CPU/GPU arithmetic can still change generated text.

The native quantization step took 138.68 seconds. Preparation and checkpoint
hashing were outside inference timing. Original weights were preserved.

| Arm | Minimum free 3060 MiB | Minimum free SUPER MiB | Minimum free regular 2060 MiB |
| --- | ---: | ---: | ---: |
| `packed38` | 961 | 597 | 875 |
| `gpu40` | 627 | 599 | 445 |
| `gpu40_cpu1` | 627 | 599 | 445 |
| `q2_3060` | 621 | 599 | 877 |

All observed samples exceeded the fixed 384 MiB guard. Unchanged GPU placement
left 61 MiB above that guard on the regular 2060. These margins apply to this
context, batch and resident-service configuration. They do not establish memory
capacity for longer contexts or concurrency.

The original weight-only estimate excluded Q3_K because both banks would need
660 MiB on the 3060. Actual all-GPU runs showed approximately 164 MiB more
headroom than the estimate, consistent with reduced workspace requirements.
Using the measured Q2 headroom, Q3 projects to roughly 465 MiB free
(621 minus 156). **This is a revised estimate, not a measured Q3 result.** The
prospective plan remains unchanged; Q3 was not run.

A reversible zlib level-1 probe sampled 16 one-MiB blocks from each selected
tensor, 96 MiB total. It saved only **1.62%**, with decompression round trips
verified. This is a sampled storage-codec result; no GPU decompression kernel
was implemented, and it does not establish a limit for every lossless codec.

## Quality evidence and its limits

Each arm passed **24/24 receiver state checks**, comprising the same twelve
authorized packets replayed twice. These checks compare decisions, occupied
slots, entity IDs and due dates against packet truth, not baseline wording.
All requests produced valid, untruncated JSON. Selector schema validity was
0/2 for the baseline and 2/2 for each candidate, again a repeated single prompt.

Receiver summaries were manually reviewed against their authorized packets;
the review is nonblind and has no independent semantic judge. Five of the
twelve cases retain unsupported approval/check statements in every arm:
`cancel`, `changed_due`, `normal_reserve`, `private_note` and `private_title`.
The annotation case is ambiguous in the baseline and unchanged-weight arms;
Q2 explicitly describes the authorized condition check. The six blocked cases
have supported summaries. Second-pass parsed outputs match their respective
first-pass outputs, so repetition adds timing evidence, not independent quality
examples.

The unchanged-weight GPU arms correct one inspected sender slot from
`shelf-position` to the requested `N1`; baseline and Q2 retain that error.
The outside-destination case still invents an unverified destination ID in all
arms. Sender structure agreement with baseline is 21/23 for unchanged GPU
placement and 22/23 for Q2; these are agreement diagnostics, not accuracy.
Exact text agrees on 23/36 and 16/36 requests respectively, also not accuracy.

Receiver packets are frozen and are not regenerated from each candidate's
sender output. Therefore this is not an end-to-end live bridge evaluation,
not a 95% semantic-accuracy result, and not proof of unchanged overall quality.
The defensible next runtime candidate is the unchanged-weight all-GPU placement:
it gains speed and removes CPU expert work without requiring extra quantization.
Live bridge quality remains a separate evaluation requirement.

## Evidence and reproduction

Source is in `experiments/word_games/{prepare_expert_compression,cpu_expert_elimination}.py`;
analysis and registration are in `experiments/neural_architecture_lab/`.
Focused tests cover placement, thread settings, hook restoration on failure,
GGUF payload hashing/truncation, and grading against packet truth.

All evidence roots below are under `data/experiments/word_games/`:

| Root / principal file | SHA256 |
| --- | --- |
| `20261031_cpu_experts_prepare_v1/result.json` | `50cfa1c627f8c76ab04271b76a983b12805f52e375d08e3b1f0c1743fe22f38c` |
| `20261031_cpu_experts_replay_v1/campaign_results.json` | `006e0c42a670a4e6a030ef5001f4ab356c5e469189a792d2b813729b6e271f4d` |
| `20261031_cpu_experts_review_v1/review.json` | `aaff2aca77a9eaa5294f065734d250d785282b577c9fdd6cbe89d1592128d741` |
| `20261031_cpu_experts_analysis_v1/campaign_results.json` | `80f774140ee2c8deeadb034527164aa670101c51a354385e0737774b19f0e615` |

The replay preserves frozen producer source, scientific fingerprint, runtime
hashes, all commands, raw requests/responses, placement verification, per-request
CPU observations, GPU telemetry and service restoration receipts. Analysis
verifies source snapshots, input/response hashes and manual-review evidence,
and recomputes receiver state grading before aggregation.

Original checkpoint SHA256:
`196103269085bc54c9b8f49ed21e9f53e1b56b465e8b796c6d8e31e06f63cfa5`.
Derived checkpoint:
`/data/models/word-games-expert-compression/qwen38-bank24-25-q2-v1.gguf`;
SHA256 `e61f3f3364b85714937c54ee22be9326a7034177fb5b552d1d2528b56e2fe6e7`.
Frozen requests SHA256:
`f9830c9480f8a4694630a06102cfb33e9e16517faec22e4f9f59888137fd53cb`.

Use Pixi from the repository root and fresh output paths. The replay temporarily
unloads Qwen3-8B and restores it; it needs all three cards in the measured layout.

```sh
pixi run pytest tests/word_games/test_cpu_expert_elimination.py tests/word_games/test_three_gpu_experts.py tests/word_games/test_packed_experts.py -q
pixi run python -m experiments.word_games.prepare_expert_compression --output data/experiments/word_games/NEW_CPU_PREP --weights /data/models/word-games-expert-compression/NEW_CPU_Q2.gguf
pixi run python -m experiments.word_games.cpu_expert_elimination --preparation data/experiments/word_games/NEW_CPU_PREP --output data/experiments/word_games/NEW_CPU_REPLAY
```

Recompute analysis of the existing immutable evidence into a fresh directory:

```sh
pixi run python -m experiments.neural_architecture_lab.analyze_word_games_cpu_experts --root data/experiments/word_games/20261031_cpu_experts_replay_v1 --preparation data/experiments/word_games/20261031_cpu_experts_prepare_v1 --review data/experiments/word_games/20261031_cpu_experts_review_v1/review.json --output data/experiments/word_games/NEW_CPU_ANALYSIS
```

The registration CLI is
`experiments.neural_architecture_lab.register_word_games_cpu_experts`, with
`--analysis`, `--report` and a fresh `--output` directory. It binds this report
and the aggregate by SHA256, writes the conservative typed record, and requires
both exact-ID and semantic-search read-back. The campaign receipt location is
`data/experiments/word_games/20261031_cpu_experts_registry_v1/registry_readback.json`.
