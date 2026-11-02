# Acquire sampling outputs once

The requested native patch passes correctness checks and removes redundant
scheduler synchronization calls. All 36 frozen WORD_GAMES requests produce
exactly the same token IDs, text, parsed JSON and stop reasons as the matched
control. The control also matches the earlier packed38 replay exactly.
The measured timing benefit is small and uncertain: the median paired request
throughput increase is 0.742%, while the pilot is effectively unchanged.

This is a completed systems regression and exploratory timing comparison,
`word-games-sampling-output-once-v1`, not a new model-quality evaluation.
The [prospective plan](../../07%20-%20Status%20Reports/word-games/2026-10-31_word-games-sampling-output-once-plan.md)
defines the gates. No weights, arithmetic precision, grammar or sampling policy
changed. The original serving runtime remains available; the candidate lives
under `.pixi/word-games-sampling-v2/candidate`.

## Implementation and correctness

[sampling-output-once.patch](../../../experiments/word_games/native_sampling/sampling-output-once.patch)
adds `llama_get_sampling_output_ith()`. It synchronizes once and returns a
read-only, context-owned view of logits, backend probabilities/logits, candidate
IDs, counts and any selected token. The common sampler acquires this view once
per sampling call. If grammar rejects the first token, it rebuilds the mutable
candidate array from the same view. It does not cache pointers across decode
calls. Existing individual getters retain their synchronization behavior, and
the server still synchronizes completed decoding before sampling.

The control and candidate libraries were built from the same pristine b11960
archive with identical compiler/options. Only `libllama` and `libllama-common`
were replaced in each isolated runtime; server executables and CPU/CUDA
backends match the original runtime byte for byte.

Both builds pass upstream sampler tests. Native GPU integration tests use the
cached Qwen2.5-3B Q4_K_M model and count real calls to
`ggml_backend_sched_synchronize` through ELF interposition:

| Sampling path | Control calls | Candidate calls | Selected token |
|---|---:|---:|---|
| Greedy, no grammar | 6 | 1 | Identical |
| Grammar rejects first choice | 10 | 1 | Identical, permitted by grammar |
| Grammar applied first | 6 | 1 | Identical |

Additional checks pass for raw logits, backend top-k logits, backend greedy
tokens, backend probabilities/tokens, two output rows, positive and negative
indices, counts, subsequent decode/reacquisition, and common-sampler consumption
of those views. The forced-rejection test also checks that sampling does not
mutate the original logits.

## Frozen WORD_GAMES replay

Both arms use the existing packed38 placement on the RTX 3060, RTX 2060 SUPER
and RTX 2060: 38 GPU expert banks, two CPU banks, Q4_K_M weights, Q8_0 KV,
Flash Attention, context 8192, batch/ubatch 512, 12 CPU threads, temperature zero,
seed 73, generic JSON grammar, and no speculative decoding or prompt reuse.
The three-request pilot passes exact-output parity before the full replay.

| Full 36-request replay | Control | Candidate |
|---|---:|---:|
| Visible output tokens | 4,778 | 4,778 |
| Valid, untruncated JSON objects | 36 | 36 |
| Completion request wall time | 99.4583 s | 94.9464 s |
| Full-request throughput | 48.0403 tok/s | 50.3231 tok/s |
| Decode-only throughput | 67.2153 tok/s | 71.3986 tok/s |
| Server prefill time | 28.1705 s | 27.8275 s |
| Server decode time | 71.0850 s | 66.9201 s |
| Process scheduler synchronization calls | 33,806 | 9,706 |

The raw full-request throughput difference is +4.752%. It is not a reliable
estimate of the patch's speedup: `cases/outside_entity/sender/turn_01` took
8.1183 seconds in control and 4.7401 seconds in candidate, accounting for 74.87%
of the total wall-time gap. Median per-request throughput improvement,
`median(control_seconds / candidate_seconds - 1)`, is 0.742%.
The pilot measured 49.6275 versus 49.5551 tok/s. There was one full run per arm,
control first, with no randomized crossover or uncertainty estimate. No
confirmatory speed claim follows from these results.

Synchronization counts include startup, warmup, requests and shutdown; both
arms use the same counter library. Removing 24,100 calls does not establish a
corresponding reduction in GPU waiting time: many calls find work already
completed. Reported sampling-timer changes are not used as speed evidence,
because the combined accessor now runs before that timer starts. Completion
wall time excludes template formatting and artifact writing, as in the
existing benchmark.

Qwen3-8B was temporarily unloaded only for the three-GPU replay and restored
with matching configuration and readiness afterward. OCR stayed resident and
was healthy afterward. Minimum free memory in both full runs was 961 MiB on
3060, 875 MiB on regular 2060 and 597 MiB on SUPER, above the 384 MiB stop rule.

## Evidence and reproduction

- [Matched build manifest](../../../data/experiments/word_games/20261031_sampling_output_build_v4/result.json)
- [Native integration result](../../../data/experiments/word_games/20261031_sampling_native_v2/result.json)
- [Replay aggregate](../../../data/experiments/word_games/20261031_sampling_output_once_v1/campaign_results.json)
- [Exact full-suite parity](../../../data/experiments/word_games/20261031_sampling_output_once_v1/full_parity.json)
- [Service restoration](../../../data/experiments/word_games/20261031_sampling_output_once_v1/service_restored.json)

Each evidence directory retains producing hashes, commands and logs. Earlier
build/harness attempts remain separate: manifest-path construction, the
unusable host `patch` executable, and an append-only recorder error were fixed
before the successful runs. Recovery verified every cached source file against
the pristine tar archive and applied the patch with the pinned build image's
Git. No failed attempt is included in the timing table.

Run from the repository root, with fresh runtime and artifact directory names:

```sh
pixi run python -m experiments.word_games.prepare_sampling_runtime --root .pixi/NEW_SAMPLING --output data/experiments/word_games/NEW_SAMPLING_BUILD
pixi run python -m experiments.word_games.check_sampling_output --root .pixi/NEW_SAMPLING --build-manifest data/experiments/word_games/NEW_SAMPLING_BUILD/result.json --output data/experiments/word_games/NEW_SAMPLING_NATIVE
pixi run python -m experiments.word_games.run_sampling_output --root .pixi/NEW_SAMPLING --build-manifest data/experiments/word_games/NEW_SAMPLING_BUILD/result.json --native-tests data/experiments/word_games/NEW_SAMPLING_NATIVE --output data/experiments/word_games/NEW_SAMPLING_REPLAY
```

The replay command temporarily unloads and restores the configured Qwen3-8B
Triton service. Its recorded pinned images, cached source archive and model
files are prerequisites. Exact agreement on this fixed suite establishes a
patch regression result; it does not regrade the model's semantic accuracy.
