# Measure NCM end-to-end summary task accuracy

2026-10-22. `ncm-memory-task-accuracy-v1`, exploratory, one trained checkpoint.
The user asked why task accuracy was omitted. Measure it now, rather than
substituting codec reconstruction accuracy or word overlap. The old training
campaign and report remain unchanged. No training or checkpoint selection runs.

Use the frozen original/trained NCM outputs on the same 256 held-out native
memory-summary cases from `20261022_training_v1`. Include the existing saved
summary as a third anonymous candidate, as a label-quality diagnostic, not gold.
Evaluate each against its complete original source. The saved targets inspected
in 12 disjoint calibration sources include headings/isolated sentences, so they
cannot be assumed to be good summaries. Model output texts have not been used
to design this new grader; previous word-overlap outcomes are already known.

Primary endpoint: whole-summary task pass rate, passes / 256. A pass is nonempty,
at most 128 cl100k_base tokens, all material claims supported by the source,
readable, and sufficiently salient to summarize the source's main point in
roughly 32 words. Allow paraphrases and omission of minor/list details. Do not
require exhaustive coverage or exact target wording. A heading or tangential
fact alone does not pass. Judge source faithfulness, not external world truth.
Every claim gets supported / contradicted / insufficient evidence and exact
source-line IDs; raw original evidence is reconstructed mechanically.

Primary judge: `claude_code/claude-sonnet-4-6`. Blind all three candidates and
shuffle their order deterministically. Qualification controls use 12 real
sources outside the 256 evaluation source sessions, with two source-reviewed
good paraphrases, one contradiction and one unrelated summary per source:
24 positives and 24 negatives. Author/reviewer: Codex source review, not humans.
Require at least 23/24 good controls accepted and 23/24 bad controls rejected,
and all 12 packets valid. Fix rubric and controls before their outcomes. The
previous overly strict rubric is not reused. Do not silently tune a failed gate.

Secondary judge: `codex_sdk/gpt-5.6-terra`, same controls, serial qualification;
stop its qualification on an invalid packet after allowed retries. If qualified,
audit 64 evaluation cases selected by fixed ID hash, with a different candidate
order. Report inter-judge agreement. If infrastructure/calibration fails, retain
failure and report secondary evidence unavailable; do not substitute a judge
selected for agreeing with the primary result.

Use four primary request workers, one secondary worker, a 180-second request
timeout, temperature zero and a 4,096-token response cap. Up to three identical
requests only for transport/schema/consistency failure; never retry a valid low
grade. Save each raw request/response and all invalid attempts. A missing final
judgment makes the complete-cohort endpoint invalid; report counts, not a
success percentage silently excluding those cases.

Report original NCM, trained NCM and saved-target pass rates, numerators,
denominators, Wilson 95% intervals and source-session cluster-bootstrap intervals
(10,000 draws, seed 20261022). Point target: >=95% (at least 244/256). Report the
stronger Wilson-lower-bound >=95% criterion separately. Also report fully
supported summaries, claim support, salience/readability failures and paired
changes. These are source-grounded LLM-judged task accuracy estimates, not
human-rated production accuracy. Context is still <=512 GPT-2 input tokens and
48 output positions; no full 2,048/6,300-token application or speed claim.

Stages explicitly use CPU/network only. Run focused parser/pass/control tests,
freeze source/data/config hashes and retain new append-only evidence under
`data/experiments/ncm_memory/20261022_accuracy_v1/`. Check raw-response readback
and result accounting, write a separate measured accuracy report, then register
the conservative result and verify readback. Preserve all unrelated work.
