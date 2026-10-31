# Standalone Bonsai 2 27B on WORD_GAMES

Prospective exploratory protocol, 2026-10-31.
Hypothesis ID: `word-games-bonsai27-naked-v1`.

The user requested "naked Bonsai": test the released model by itself, without
a draft, lookup speculation, training or adapters. Use official Bonsai 2 27B
Q2_0, revision and file hash pinned in `prepare_bonsai.py`, with the existing
native llama.cpp b11960 runtime (supports the required Hadamard integration).
Keep the information firewall and task prompts unchanged.

The deployment condition is short structured output, not replication of the
authors' long-thinking benchmark. Disable thinking through the embedded chat
template, verify the closed empty thinking prefix before inference, and use
the same temperature zero, seed 73, generic JSON-object constraint and output
caps as the Qwen baseline. Report the departure from the author's non-thinking
sampling recommendation. Do not silently retry with different prompts or
sampling if task quality fails.

Use RTX 3060 only, context 8192, Q8 KV, batch/ubatch 512, four CPU threads,
full GPU layer offload, no cross-request prefix cache. Preserve the 2060 and
2060 SUPER resident services. Use Pixi controllers from the repository root;
weights and raw evidence go in their existing model/data stores, executable
code and environments remain outside data/.

Decision order:

1. Verify the complete downloaded checkpoint and metadata. Run focused CPU
   tests. Pass a real JSON warm-up and three frozen pilot requests, checking
   all layers fit on the GPU and the rendered native prompt disables thinking.
2. Replay all 36 frozen requests to measure visible output tokens per full
   request second, plus decode-only timing and per-request latency. Output
   token counts across tokenizer families are not equal amounts of text;
   also report bytes/sec and paired elapsed time. Model load, warm-up and
   file I/O are excluded. No 40 tok/s claim from a three-request pilot.
   Because only a three-request Qwen pilot exists, collect a fresh target-only
   Qwen36 replay as the paired latency/byte-throughput control if Bonsai's pilot
   passes. Use its existing pinned runtime entry point unchanged; record the
   Bonsai diagnostic verbosity difference and avoid broad speedup claims.
3. Run the same twelve live WORD_GAMES cases, using the completed Qwen baseline's
   synthetic context IDs after checking identical authorization metadata.
   The model still selects its metaphor. A selector-format failure stops the
   live suite and is reported as a pre-case failure, not fabricated accuracy.
4. Report sender operations, boundary decisions, receiver state and their
   per-case joint success with denominator 12. Check canary disclosures and
   review generated receiver summaries for obvious semantic errors. Distinguish
   manual review from an independent semantic judge. No production promotion
   follows automatically from this small synthetic suite.

Stop an arm on OOM, transport failure, unsupported model transforms, bad identity,
invalid/truncated pilot JSON or changed fingerprints. Preserve failures and
append explicit amendments for infrastructure recovery. Do not change weights
or runtime format to rescue a quality failure. Compare task success with the
Qwen target's 5/12 result; historical Phi results are context, not matched arms.

All producing sources/checkpoints/runtime and paired data are fingerprinted.
Use fresh `data/experiments/word_games/20261031_bonsai_*` roots; finalize a measured
report, conservative registry entry and focused source commit. This experiment
has one model configuration and one deterministic pass, not statistical proof
of 95% accuracy, lossless compression or a general hardware speedup.

## Lifecycle amendment before pilot outcomes

The first launch verified 65/65 GPU layers and passed the JSON warm-up, then
stopped before any pilot request because the native template uses `|trim` on
user content. The input-preservation guard incorrectly required the original
trailing newline. Permit only native leading/trailing whitespace trimming and
still require the exact user turn plus closed empty thinking suffix. Preserve
the original startup under `20261031_bonsai_pilot_v1`; use v2 for the corrected
lifecycle. No task prompt, sampling, weights or quality gate changes.
