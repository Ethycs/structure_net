# Llama 3B summarization throughput study

Date: 2026-10-13. Hypothesis `llama32-3b-summary-throughput-v1`.
Profile: EXPLORATORY. Status: prospective; no new outcomes inspected.

## Question and acceptance

Can current inference techniques double output tokens per second for fresh-input
summarization with Llama 3.2 3B Instruct, while retaining high task quality?
The user explicitly replaced bitwise equality with task quality. Completed exact
studies remain unchanged. No model substitution, truncated source, shorter output
cap, cross-request prefix reuse, or increased hardware counts as a speed gain.

The primary speed endpoint is total generated non-EOS tokens divided by summed
request wall time (prefill, token selection and generation included), on identical
requests with normal EOS stopping. Also report request latency, prefill time,
decode tokens/s, output lengths, fixed-length capacity tests and model residency.
A twofold speed claim requires >=2.0 versus a matched optimized FP16 baseline;
separately report versus the existing Q4_K_M deployment, never conflating them.
Paired latency must also improve, so verbosity cannot manufacture a success.

## Workload and quality

Stored summaries are limited to 128 tokens. Source text below 192 tokens bypasses
summarization and is stored verbatim. Raw fragments are at most 2,048 tokens;
raw batches at most eight fragments and 6,300 prompt tokens, with 768 generated
tokens maximum. Hierarchy merges have a 512-token generation cap; their production
input distribution is unknown. The extractor's 32-word atom target is not a token
cap. Context capacity for maximum raw generation must be at least 7,068 tokens.

Use new, labeled synthetic dialogue fixtures spanning short/medium/maximum prompt
sizes and raw/hierarchy modes. They are a proxy, not production application data.
Keep source facts, prompts, tokenizer, template, EOS rules, and caps matched.
Quality permits paraphrases: score required fact coverage, unsupported additions,
contradictions, attribution and updated decisions. Inspect raw outputs with source
evidence. Deterministic fact checks are diagnostic, not sufficient certification.
95% task-success remains the target; no production 95% claim without an adequate
independent application evaluation. Keep pilot/development and held-out examples
separate. Do not tune prompts or thresholds on held-out results.

## Interventions and controls

1. Pinned llama.cpp runtime with FlashAttention, full GPU residency, per-request
   KV state and greedy generation; existing Q4_K_M is a measured reference.
2. FP16 and locally quantized Q8/Q5/Q4 variants from the pinned original checkpoint.
   Quantization is an explicitly CPU-only preparation cell, not model training.
3. Prompt-lookup speculation at draft lengths 4, 8 and 16, matching n-grams of
   lengths 4 through 2 in the current prompt only. The target model verifies every
   candidate; trim rejected KV entries before continuing. No draft model, training,
   persistent document cache or cross-request lookup.
4. Cache precision and runtime batch sizing may be screened under a prospective
   amendment. More methods require new frozen configurations and artifact roots.

Prompt lookup references:
https://huggingface.co/docs/transformers/assisted_decoding
https://github.com/ggml-org/llama.cpp/tree/master/examples/speculative
Quantization reference:
https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md

## Execution and integrity

Use physical GPU 0 only, with visibility fixed before runtime initialization.
Existing worktree modifications, cached inputs and completed studies are preserved.
Producer revision starts at `ba94b96`; hash and snapshot all added sources, inputs,
compiled binaries and shared libraries. Record original checkpoint and GGUF hashes.
Use new roots under `data/experiments/llama_summary_throughput/20261013_*`.

Ladder: CPU contract tests, tiny real CUDA decode/rollback check, representative
full-context memory pilot, then serial full measurement. No concurrent GPU arms.
Each process retains one physical model; compare decoding strategies in alternating
order after warmup. Quantization comparisons use independently loaded processes
and counterbalanced run order; loading is reported separately from warm inference.
Use five repeats for primary timing. One frozen model is not five training seeds.

Preserve every failure. Stop a configuration on runtime error, nonfinite scores,
invalid tokenization/context, or incorrect speculative cache management. Numerical
token differences from batch arithmetic are reported and evaluated for task quality;
they must not excuse acceptance of an unverified draft token. Do not claim timing
is isolated when unrelated GPU utilization interferes. Do not mutate producers
while executing. Strict JSON includes configuration, raw timings, tokens, text,
draft acceptance, fingerprints, failures and declared limitations. Finalization
independently recomputes summaries and checks evidence hashes, followed by a report,
conservative registry readback and a commit.
