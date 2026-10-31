# WORD_GAMES: smaller-Phi speculative decoding

October 31, 2026. EXPLORATORY. Hypothesis:
`word-games-phi4-mini-speculation-v1`. No training.

Question: does a Phi-4-mini draft reduce latency for the existing Phi-4 14B
WORD_GAMES requests while preserving target-verified greedy output?

Keep the target checkpoint, Q4_K_M precision, Q8 KV, 8,192-token context,
JSON grammar, temperature zero and original request caps fixed. Use the
existing llama-cpp-python 0.3.23 serving image in a separate temporary
container. Target runs on the RTX 3060. The Q4_K_M Phi-4-mini 3.8B draft runs
on the RTX 2060 SUPER, with the same 8,192-token capacity if memory permits.
This uses an additional GPU; it is not a same-resource efficiency claim.
Do not change or unload unrelated serving endpoints.

Phi-4 has 100,352 vocabulary entries; Phi-4-mini has 200,064 and different
chat markers. Direct token-ID drafting is invalid. Re-encode the confirmed
continuation in the draft's native chat format, decode its candidate bytes,
then tokenize only those new bytes for the target. Never rewrite a confirmed
target prefix. Use the target runtime's greedy verification and JSON grammar
for every emitted token. Count proposed and accepted target tokens, draft
time, target evaluation time and target evaluation calls. A poor candidate
can waste time but cannot bypass target verification.

Freeze all 36 raw request inputs from `20261030_phi4_full_v2`, including the
metaphor selector, retrieval turns and receivers. This is a paired request
replay benchmark, not a fresh semantic accuracy evaluation. Do not change
prompts or repair historical errors. Both models start each request with
empty caches; allow prefix reuse within a draft's ongoing request only.
Exclude load and warmup from request throughput; report them separately.

First test candidate conversion and target-only rejection behavior. Run a
separate systems pilot on the selector, the final normal-reserve sender turn,
and the cancellation receiver. Compare no draft with 2, 4 and 8 draft tokens.
Only settings with complete output-token equality are eligible for the full
comparison. Select the eligible setting with lowest summed pilot latency,
even if it is slower than baseline; report that selection and all pilots.
If none preserve output, stop the speed claim and retain the failed controls.
Infrastructure or implementation fixes get a new artifact root and amendment.

For the full comparison, replay all 36 requests once per arm, alternating
baseline-first and speculative-first pairs. Record exact target token and text
equality, output length, request wall time, time to first sampled token,
candidate acceptance, memory and environment/source hashes. The primary
speed ratio is total baseline request time divided by total speculative
request time. Report output tokens per full request second, including prefill,
drafting, conversion, target verification and Python overhead. Never count
unaccepted draft tokens as generated output or compare against yesterday's
unmatched 14.5 tokens/s estimate.

Classify observed faster only if all 36 pairs complete, every target token
sequence matches and aggregate latency improves. Otherwise report slower,
no improvement, or output mismatch as appropriate. One pass on one machine
is exploratory timing evidence, not a generalized speedup or security claim.
Preserve failed pilots, fingerprints and raw outputs. Finish with a report,
conservative NAL registry/readback and focused tests. Release only the newly
created worker and its GPU memory when complete.

## October 31 implementation amendment after pilot v1

The first pilot stopped after the selector's four conditions. The longer
sender prompt exposed an adapter allocation defect: switching `_logits_all`
after constructing the target left only `n_batch` score rows. llama-cpp-python
then failed copying a later prompt batch into that array. This was an
implementation failure, not a model or out-of-memory failure. Allocate the
full-context score buffer at construction; keep score collection disabled for
baseline calls and enabled for speculative calls. Preserve the interrupted run.

The first speculative selector call also included substantial cold-path
startup overhead. Before new measured calls, warm the speculative path for
each draft length with an unrelated long formatting prompt and record warmup
separately. Retain a dedicated CUDA JIT cache between disposable workers.
Include draft-adapter setup in request wall time. These changes create pilot
v2; do not pool pilot v1 timings with it. No primary full comparison has run.
