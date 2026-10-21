# Independent grading of the frozen Llama-distilled NCM

Date: 2026-10-21. Hypothesis: `llama32-3b-ncm-independent-quality-v1`.
Status: prospective evaluation of one already-selected checkpoint, not training.

Question: does the existing approximately 149M-parameter NCM produce useful,
faithful short dialogue summaries on previously unused human-written sources,
and does it retain the original Llama-3.2-3B-Instruct's quality?

The previous 100/100 was a narrow synthetic final-project-decision test. It is
not evidence for the following population. Do not tune weights, select another
checkpoint, alter the rubric, or remove cases after inspecting these outcomes.

## Models and data

Use the unchanged generator `20261021_generator_v1/model.pt` (SHA256
`ff7133b2de50bc1f66ef1e490fec4c723e0fc5a5c1957107209e923b198c3e3c`)
and codec `20261021_codec_v1/model.pt` (SHA256
`b87f7288edfc0eb1f5b9122d1139cc2e61aa2331be7d6fe53f0582993086ede2`).
Use the complete original Llama-3.2-3B-Instruct at cached revision
`0cb88a4f764b7a12671c53f0838cd831a0843b95`, FP16/FlashAttention2, greedy
decoding. NCM uses its unchanged native nine-level BF16 graph inference, fixed
512 GPT-2-token prefix and 48-position output canvas. No input truncation.

Primary: 200 DialogSum test dialogues, pinned upstream commit
`848cc8c05e0a04def79326539de827f1b28786a0`. Eligibility: full source at most
512 GPT-2 tokens and at least 192 Llama tokens, corresponding to the application's
summarization threshold. Select the first 200 eligible sources sorted by
SHA256 of `ncm-quality-v1:` plus upstream ID. The other 17 eligible sources are
excluded before outputs; use two for the CUDA systems pilot. Exclude exact
source overlap with the previous 1,188 fixtures; report all eligibility counts.
Keep all three human references. They guide salient information, but source
evidence takes precedence. No reference-length filtering. This is public
dialogue research data, not production traffic, and pretrained contamination
cannot be excluded. Preserve the upstream CC BY-NC-SA 4.0 attribution.

Secondary: 40 fresh stress cases, ten fixed semantic families with four
variations each: corrected amounts, denied approval, tentative schedules,
speaker attribution, cancellation, unresolved disagreement, two subjects,
negated action, uncertainty, and an unapproved later suggestion. Keep this
cohort separate; do not mix it into primary accuracy. Sources, references, and
selection are frozen before generation. Check disjointness from training.

Both models receive the same complete source. Llama's native chat wrapper asks
for one short faithful summary of at most 32 words, preserving central facts
and decisions without inventing details. Allow 96 native Llama output tokens
to finish; enforce the common stored-summary limit of 128 Llama tokens for
both outputs. NCM's 48 GPT-2-position canvas is an architectural limitation,
not a shared tokenizer unit. Report truncation/termination separately. No
postediting, lookup, source-aware decoding changes, or response regeneration.

## Independent grading

Primary judge: configured gateway `codex_sdk/gpt-5.6-sol`; cross-family judge:
`claude_code/claude-sonnet-4-6`. Transport checks established both can return
JSON; direct `claude-opus-5` is unavailable for insufficient provider credit.
Store exact requested and returned aliases, full raw replies and request hashes.
These gateway aliases do not provide immutable weight revisions; usage fields
may be zero and are not billing measurements. No production/private data is sent.

Each request contains only source, references and two anonymous summaries in
deterministically randomized order. No checkpoint/model names or speed results.
Score each independently; both may pass or fail. Reverse the order for the
second judge on the fixed first 50 primary and first 20 stress IDs under a
separate hash order. No outcome-based selection for second judging.

For every summary return all material atomic claims labeled `supported`,
`contradicted`, or `insufficient evidence`, with exact source quotes where
available, plus 0–4 faithfulness, coverage, and coherence scores and explicit
critical omissions. Grade coverage for a short approximately 32-word summary,
not exhaustive reproduction of all reference details. A 3 captures the central
situation and outcome with no major omission; a 4 covers all essential facts.
Faithfulness 4 requires every material claim to be supported. Coherence 3 is
understandable prose with at most minor defects; incoherent/gibberish outputs
score at most 1. Quotes must occur literally in the source. References are
helpful, not an exact-match target or a substitute for the source.

Code computes a pass only if: nonempty output, within 128 common tokens,
faithfulness=4, every material claim and overall verdict supported, coverage
at least 3, coherence at least 3, and no critical omissions. Empty claims never
pass. Unknown schema, missing IDs, nonliteral evidence, malformed JSON and
unavailable judge results invalidate a judgment rather than count as quality
passes. Preserve fenced JSON transport normalization without changing content.

Before primary outputs are generated, each judge must pass 24 known-answer
calibration pairs (48 summaries): factual positives and negated, wrong-amount,
wrong-date, invented-outcome, omission and incoherence negatives, including
both-good and both-bad pairs. Require at least 90% correct summary decisions
and rejection of every contradiction/unsupported-fact negative. Failure stops
that judge's primary grading; no rubric tuning against NCM outcomes. A secondary
calibration failure does not change the primary grade, but prevents an agreement
claim. Gates and calibration cases are source-hashed prospectively.

## Endpoints, execution, and stopping

Primary endpoint: independent judge's strict summary pass fraction on 200
human-written sources, with Wilson 95% interval. Also report fully supported
summary fraction, coverage/coherence failures, claim support fraction, output
termination and limits, paired Llama-minus-NCM difference (fixed-seed paired
bootstrap, 10,000 draws), and judge agreement on the fixed audit subset.
Report the stress cohort independently. Do not interpret model-judge scores
as measured production accuracy or human gold labels.

The user's 95% target is met as a point estimate at at least 190/200 primary
passes. Separately mark whether the Wilson lower bound reaches 95%; a point
estimate alone does not establish that population guarantee. Retention requires
NCM pass rate no lower than Llama. Classify invalid evidence first, then NCM
below-95 failure, then retention failure, then point-estimate success. Report
teacher quality even if NCM fails. Do not stop early based on observed accuracy.

Use focused contract tests, then two nonprimary CUDA lifecycle cases for both
frozen models. Physical GPU0 only, selected before Python; one local worker;
no CPU fallback. Verify checkpoint hashes and original Llama physical manifest.
Independent judge requests may use four network workers; retry infrastructure,
schema, or quotation errors at most twice with the identical request, retaining
every attempt. Never retry a valid low score. Missing final judgments invalidate
the affected endpoint and must be disclosed. No speed benchmark in this study.

Fresh append-only roots: `data/experiments/ncm_quality/20261021_*_v1`.
Snapshot and hash producing sources, this document, inputs, checkpoint files
and requests. Complete all scheduled cells, audit bindings/counts, write strict
JSON aggregate, measured analysis, conservative typed registration/readback,
and commit source/tests/docs only. One checkpoint, no independent adaptation
seeds: `confirmed:false`, with deployment and mobile accuracy untested.

Reproduce with `python -m experiments.structure_net.ncm_quality --help` in
the existing `data/experiments/moba_paper/environment` environment. Preparation,
calibration, pilot, generation, judgment and aggregation are separate commands.

Sources: [DialogSum](https://github.com/cylnlp/DialogSum),
[G-Eval](https://aclanthology.org/2023.emnlp-main.153/) motivates separate
quality dimensions and explicit rubric; this is not a reproduction of its metric.
