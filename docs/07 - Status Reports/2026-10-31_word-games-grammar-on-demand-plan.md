# Grammar logits on demand

Prospective systems check and exploratory timing study:
`word-games-grammar-on-demand-v1`. Compare patch 1 against patch 1 plus this
patch, using the same b11960 source, compiler, backends and packed38 placement.
Preserve existing evidence and unrelated worktree changes.

Enable the candidate explicitly with `LLAMA_WORD_GAMES_GREEDY_PREVIEW=1`.
Eligible grammar requests must use temperature-zero deterministic greedy
selection, no requested probabilities or reasoning budget, and only filters
that preserve a unique finite maximum. Preserve the original CPU sampler for
fallback. Compute the two highest logits/IDs and a finite-value guard on the
backend. Check only the winning token against grammar on the CPU. A tie,
non-finite guard, grammar-first request or grammar rejection uses full raw
logits and the original CPU chain. Single-output batches retain the original
GPU logits until needed; multi-output batches initially use eager raw output
to avoid retaining tensors across microbatch graph reuse. Existing public
logits reads must return valid data. Do not change model weights/precision.

Gates, in order:

1. Build and upstream sampler checks. Real CUDA native tests for accepted and
   rejected grammar choices, ties/non-finite handling, eligibility exclusions,
   explicit full-logit reads, successive decode calls, clone/copy, multi-output
   and microbatch lifetime. Measure actual output transfer calls/bytes, not
   inferred bytes avoided. Stop on incorrect output or stale pointers.
2. The three frozen pilot requests must have exactly equal token IDs, content,
   parsed JSON and stop conditions. Confirm actual reduction in raw vocabulary
   transfer. Invalid/truncated JSON, OOM, service failure or memory below
   384 MiB stops the run.
3. Two full 36-request replays per arm, ordered control/candidate then
   candidate/control. Require exact output equality for every paired request.
   Report completion wall time, prefill/decode timing, all download counters
   and individual repetitions. No population accuracy or confirmatory speed
   claim from this fixed suite.

Keep existing model/request identities, temperature zero, seed 73, Q4_K_M,
Q8_0 KV, Flash Attention, context 8192, 512 batch/ubatch, 12 CPU threads,
generic JSON grammar, no speculation and no cross-request prefix reuse.
Reuse the previously authorized temporary Qwen3-8B unload on regular 2060,
restore its readiness/configuration in `finally`, and keep OCR resident.
Native checks use the cached 3B on 3060 with both services resident.

Use new roots under `data/experiments/word_games/20261031_grammar_*` for
build, native checks and replay. Source/build/runtime stay under `.pixi` and
tracked source directories, with Pixi orchestration from the repository root.
Changed producers get new evidence roots. Deliver a reviewable patch,
reproducer and measured report without replacing the production endpoint.
