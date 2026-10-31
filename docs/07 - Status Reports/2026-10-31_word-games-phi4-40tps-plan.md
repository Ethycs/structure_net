# WORD_GAMES: Phi-4 14B toward 40 output tokens/s

October 31, 2026. Exploratory systems optimization, no training.
Hypothesis: `word-games-phi4-native-40tps-v1`.

The requested target is 40 accepted output tokens per full request second
on WORD_GAMES. Preserve Phi-4 14B Q4_K_M weights, Q8 KV, 8,192-token context,
the native chat template, greedy decoding, JSON output constraints and the
original prompt/cap pairs. Use the RTX 3060 alone. Do not change other
resident services. Exclude model loading and explicitly recorded warmup;
include prompt processing, decoding and HTTP/client overhead. Report native
decode-only throughput separately. No inter-request prompt/KV reuse and no
counting proposed or rejected tokens as output.

First test pinned native llama.cpp b11960 (v0.8.0 release's pinned nightly),
CUDA 12.8, against the prior Python implementation. Downloaded release SHA256:
`64d57beb9abb637c177c6afb313760a0c0782224cac71e74869eed5b8f9a1ea4`.
All executable dependencies live under `.pixi/`; only evidence lives in data.
Bind commands, binary/checkpoint/input/source identities to each run.

Use the three previous pilot requests (metaphor, normal-reserve final sender,
cancel receiver) for initial native baseline and native prompt-lookup trials.
Prompt lookup proposes continuations from tokens already in this request;
the full target verifies candidates. Try short n-grams and bounded drafts,
with native backend sampling as a separate intervention if supported.
Record every completed and failed configuration. Additional tuning must be
described as exploratory rather than held-out evidence.

Evaluate promising settings on all 36 frozen request inputs. Retain every
raw response, token count, request wall time, server timing and configuration.
Choose by summed request latency with quality as a separate gate. Exact text
equality is diagnostic: the user accepts differing outputs when task quality
is preserved. Parse/schema checks alone do not establish task quality.

If outputs change, run the existing 12-case live firewall campaign for native
baseline and the selected optimization. Compare sender operations, boundary
decisions, receiver structured state, joint case success and disclosure checks;
manually inspect changed receiver summaries for factual errors. Do not call
this a broad semantic accuracy estimate. Preserve the receiver's authorized
translated view; do not expose source prompts or hidden data to improve speed.

Claim the 40-token/s target reached only for a complete 36-request replay at
or above 40 output tokens/full-request-second, with no observed task-quality
regression in the paired live suite. Repeat if close to the threshold. A
decode-only or pilot result does not meet that claim. Otherwise report the
best measured setting and remaining gap. Finish with reproducible commands,
focused checks, an evidence report and conservative registry readback.

## Accounting clarification after the first two pilots

The native API counts the terminating EOS token in `tokens_predicted`, while
the previous Python API excludes it from completion usage. Subtract one for
EOS-terminated requests when reporting visible output throughput. Retain
the first two raw pilot aggregates unchanged and recompute corrected values
in the final analysis. Native decode timing excludes prompt eval's first
token; use its reported decode rate and timing for that secondary metric.

## Exploratory compact-JSON follow-up

The full original-format trials reached 37.57 tokens/s (n=4, draft=16) and
37.47 tokens/s (n=3, draft=8). Receiver prose and indentation provide few
matches. Test a separate serialization constraint that removes whitespace
outside JSON strings while retaining the same fields, permitted values,
input prompts, target weights, temperature and token caps. Start with a
pilot, then full replay if promising. This changes the output format grammar;
report it separately and compare parsed task results in live campaigns. It
does not authorize shorter task content, input caching or answer reuse.

The compact pilot failed task schemas in the selector and cancellation
receiver despite syntactically valid JSON. Do not promote it to a full run.
Continue with the original JSON format and test a 32-token lookup draft
before the live quality comparison. Retain the failed format pilot.

The 32-token pilot was effectively tied (37.25 versus 37.20 tokens/s). A final
kernel-boundary pilot uses n=4 and draft=7: upstream b11960 selects MMVQ up to
eight verification tokens on this GPU, whereas eight draft tokens plus the
current token cross that boundary. This is a prospective performance test,
not an assumption that the smaller kernel is faster. Live quality checks use
the unchanged fixture generator, including fresh opaque context IDs; they are
matched scenarios rather than byte-identical prompts.

After observing that the baseline live fixture has new opaque context IDs,
pair the optimized live arm to those saved IDs. Rebind only the isolated
synthetic engine's context handles after asserting identical authorization
metadata; do not change vendored code or a production context. Hash all twelve
saved fixture contexts. This removes a copying/numerical confound. Model-selected
metaphors and subsequent retrieval decisions remain free to differ.

## Prefill and KV follow-up after live quality regression

The original-format n=4/draft=16 run reached 37.57 tokens/s in frozen replay
but scored 9/12 joint live workflows versus the native baseline's 10/12.
Outside-entity and stale-revision scenarios regressed, while the invalid
destination scenario improved. It is not eligible as a quality-preserving
optimization. No tested canary leaked. Keep these outcomes visible.

Test prompt batch/microbatch 1,024 instead of 512, first with Q8 KV, then with
FP16 KV as an explicitly different precision arm. Keep Q4_K_M model weights,
8,192 context, one RTX 3060, original JSON format and all prompts/caps fixed.
FP16 KV uses more memory; record allocation and do not unload other services.
Use pilots to choose a full replay setting, then a fresh matched live quality
suite. These are exploratory optimizations; do not retroactively pass the
original Q8/512 result or select on speed alone.

## Sampling-chain follow-up

FP16 KV/full replay reached 38.30 tokens/s but again scored 9/12 live
workflows: cancellation acquired an incorrect occupied slot. Preserve that
failure. Inspection of upstream `common/sampling.cpp` shows the default
native chain still applies top-k/top-p/min-p before temperature-zero greedy
selection. Test a minimal temperature/argmax chain, first keeping the JSON
grammar. These truncation filters cannot remove a unique maximum, but equal
logits and floating-point paths can still change tokens; retain quality gates.

A separate arm may remove the generation grammar to enable native GPU
sampling with the minimal greedy chain. The original prompts still demand
JSON, and all host schema/authorization checks remain unchanged. No retries
or output repair. Require all 36 replay responses to parse as JSON objects,
plus the same live quality/disclosure gate before considering promotion.
This is explicitly a changed generation constraint, not the original arm.
