# Acquire sampling outputs once

Status: prospective systems validation for the requested first native patch.
ID: `word-games-sampling-output-once-v1`. No training or model changes.

The change adds a combined read-only sampling-output accessor that synchronizes
the context once. The common sampler acquires that view once and reuses it
when grammar rejection requires rebuilding the mutable candidate array. Existing
public getters and the server's completed-decode synchronization remain intact.
No persistent pointer cache crosses decode calls. Grammar, sampler order, RNG
advancement and model arithmetic remain unchanged.

Build matched control and candidate `libllama` and `libllama-common` from the
pristine cached b11960 archive, SHA256
`2b37ffa871595f90b94164f6c510ad9a34bd7f94da21557cc9878587e163af37`.
Use the same compiler/image/options for both. Retain the original server and
CPU/CUDA backend libraries byte for byte in isolated runtime directories under
`.pixi/word-games-sampling-v2`. Do not replace the original benchmark runtime.

Validation order:

1. Native sampler behavior tests and combined-accessor checks: raw logits,
   backend logits/probabilities/tokens, multiple output rows and negative
   indexing. Count scheduler synchronization calls around acquisition and
   sampling. Force grammar rejection to verify candidate reconstruction reuses
   the same immutable view and still selects a permitted token.
2. Three frozen WORD_GAMES requests on each runtime using the established
   packed38 model placement. Require exact emitted token IDs/content and valid,
   untruncated JSON between matched control and candidate before advancing.
3. One matched full 36-request replay per runtime. Require all token IDs,
   content, stop reasons and parsed objects to agree. Record request wall time,
   server prefill/decode timing and synchronization counters. One pass per arm
   supplies exploratory timing only; correctness and speed remain separate.

Temperature zero, seed 73, Q4_K_M weights, Q8_0 KV, Flash Attention, 8192 context,
512 batch/ubatch, 12 CPU threads, existing generic JSON grammar, no speculation
or cross-request cache. Preserve the fixed request corpus and weight identity.
If rebuilt control differs from historical results, report that separately;
the causal comparison is the freshly rebuilt matched pair.

The native API checks can use the cached Qwen2.5-3B model on the 3060 to keep
the lifecycle check small. This is not model-quality evidence. Packed-model
checks retain the user's earlier authorization to temporarily unload only
Qwen3-8B from the regular 2060. Keep OCR resident, require 384 MiB free per GPU,
stop on OOM, invalid output, mismatched tokens or service failures, and restore
Qwen3-8B readiness and configuration in `finally`. Record failed attempts in
their original directories. No unchanged-accuracy population claim follows
from exact agreement on this fixed suite.

Build evidence: `data/experiments/word_games/20261031_sampling_output_build_v4`.
Earlier v1/v2 attempts stopped at manifest-path construction after compilation;
v3 passed control sampler tests but the host `patch` executable could not run.
The v4 recovery verifies every source file against the pristine archive, uses
the build container's Git to apply the patch, and preserves the earlier evidence.
Native integration checks: `data/experiments/word_games/20261031_sampling_native_v2`.
The native-v1 harness stopped before compiling the checks because its command
recorder attempted to rewrite an append-only manifest. V2 records each command
in its own immutable file; the model and native patch are unchanged.
Runtime validation: `data/experiments/word_games/20261031_sampling_output_once_v1`.
Both replay arms use identical scheduler-call counter instrumentation; process
counts include startup, warmup and shutdown as well as measured requests.
Changed test code or configurations use a new artifact root. The requested
deliverable is a reviewable patch, reproducible build and validation result;
no production deployment is implied.
