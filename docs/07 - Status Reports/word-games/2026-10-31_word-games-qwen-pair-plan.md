# WORD_GAMES: switch to Qwen2.5 14B / 3B

Date: 2026-10-31. Status: prospective exploratory systems validation.
Hypothesis: `word-games-qwen25-pair-v1`.

The user requested a Qwen 14B target and Qwen 3B draft in place of Phi.
Use the matching Qwen2.5-Instruct family, official Q4_K_M GGUF weights pinned
by revision, size and SHA256 in `experiments/word_games/qwen_pair.py`.
This changes the model, so Phi quality or speed cannot be inherited.

Preserve the completed Phi tokenizer-map and speed artifacts. Download the
new pair under `/home/rabbit/models`; code and environments remain outside
data/. All controllers run through the locked word-games Pixi environment.

Decision order:

1. Verify all checkpoint shards and tokenizer semantics on CPU, including
   vocabulary pieces, special tokens, merges and pre-tokenizer configuration.
   Inspect any padded vocabulary differences. Check the frozen 36 request
   corpus and Unicode/JSON controls; do not assume shared token IDs from names.
2. Run focused contract tests. Start a pinned native llama.cpp b11960 server
   with 14B on RTX 3060 and, for speculation, 3B on RTX 2060 SUPER. Keep other
   GPU services running. Use Qwen's embedded native chat template, JSON-object
   constraints, temperature zero, seed 73, 8,192 context and no cross-request
   prompt cache. Record model identity, hashes, runtime command and memory.
3. Run the existing three-request lifecycle/speed pilot with target alone and
   native draft-simple K=3. Count accepted visible output tokens per full
   request second, including prefill and draft work. Report native decode time
   separately. This pilot is not a quality or general speed claim. Preserve
   failures; stop an arm on OOM, incompatibility, invalid JSON or transport error.
4. Run the twelve-case live WORD_GAMES suite on the Qwen target. If the draft
   pilot is valid, run the same live suite with the draft and paired synthetic
   context IDs. Keep model-selected metaphors and authorized-only receiver
   packets; do not alter task prompts or firewall policy to favor Qwen.

Quality means the joint sender-operation, boundary-outcome and receiver-state
check, with all twelve cases in the denominator. Also report disclosure
canaries, each component and regressions. These are structural task checks,
not an independent semantic judge or a production security certification.
Speculation is promoted only with observed speed benefit and no per-component
regressions in the matched live suite; otherwise target-only stays the default.
Exact generated text and task quality are reported separately. Float kernel
batching may change greedy outputs despite target verification.

Outputs use fresh `data/experiments/word_games/20261031_qwen_*` directories.
Freeze producing sources before each run and verify after completion. Record
the switch, results and remaining limits in a dated report and conservative
registry entry. No training or production bridge deployment is implied.
