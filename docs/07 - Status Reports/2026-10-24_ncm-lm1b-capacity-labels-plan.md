# Capacity-compatible TRAIN target revisions for native NCM

Date: 2026-10-24. Exploratory data-preparation hypothesis
`ncm-lm1b-capacity35-source-grounded-labels-v1`.

Known before generation: 25 of 690 original TRAIN summaries need more than one
native frame; 21 BERT-rendered targets exceed the 128-storage-token limit. Their
union is 35 cases, including 3 of the previous component-development 64. The
task-domain codec passed 52/64 source-grounded semantic probes, not its 61-case
gate. The previous conditional student remains 0/128. All final sources remain
closed. The latest canonical semantic cache has 2284 first decisions.

Select every original label violating at most 126 BERT lexical tokens, at most
128 storage tokens for raw text, or at most 128 storage tokens after BERT decode.
No selection by student performance or prior semantic verdict. Fix order by
SHA256 of `capacity-labels-89142:` plus case ID. Keep the other 655 target texts
unchanged. Preserve all original artifacts.

Generate one concise new summary from each complete exact public source, without
showing the old target or its verdict. Sonnet is the teacher, temperature 0,
max768 tokens, timeout180 seconds, four workers. Prompt aims at 50-60 words,
faithfulness and salience, simple punctuation and genuine capacity headroom.
Up to three transport/schema attempts; retain the first schema-valid candidate.
Capacity or semantic failure is a valid outcome, never a retry reason. Preserve
requests, responses and first candidate text. No clipping, heuristic repair,
best-of selection or fallback.

Run a four-case schema pilot and reuse those exact candidates in all35 generation.
Prepare anonymous single-candidate packets for raw and BERT-decoded wording,
using full sources and the qualified 128-token faithfulness/salience/readability
rubric. Both Terra and Sonnet grade all35 in both arms. Sonnet also generated
the text; disclose that one judge shares the teacher model. Terra supplies a
different model's assessment. No old component-model admission gate is changed.

Use four semantic workers, at most three transport/schema attempts, first valid
decision wins and exhausted invalids remain. The four-case semantic pilot checks
validity only and retains all decisions. Inherit all2284 canonical decisions from
the registered domain-codec campaign SHA256
`2afebfffbc04110a87e18e4c5c23e89738c15c3d7267f90dcaab03387155cf9a`,
cache SHA256 `d5de61efafde82d51d2d648f90605e5f577c6203f042c6d5eaf1d0e845122586`.
Identical packets share decisions; never reroll a valid negative or exhausted
invalid. New actual text is a new candidate, not an overwritten prior judgment.

Admission is strict: every one of the 35 candidates must satisfy native and
storage capacity, and all35 must pass both judges in both wordings, all valid.
Failure retains all35 outcomes and exports no replacement training dataset.
If admitted, export all690 source-identical TRAIN cases with exactly35 new texts
and655 unchanged texts. Remove old GPT2 word IDs and XSum concept IDs from ALL
rows; regenerate native targets in the subsequent CUDA stage. Do not carry stale
labels into training. Existing full-source frozen-body features stay unchanged.

The independent CPU auditor replays source selection/public proof, every teacher
attempt and first schema-valid text, capacity counts, all semantic case scores,
canonical inheritance and all actual producing child exits. Hash and register
the conservative outcome and independently read it back. Use fresh
`20261024_lm1b_capacity_labels_*_v1` roots and immutable source snapshots. The
entire study is CPU/network only, never loading a model or modifying Llama MLPs.

This prepares TRAIN labels; it does not qualify a summarizer. The objective is
still at least95% complete-summary accuracy on the full source-disjoint task.
Full128 source-only DEV and sealed320 final evaluation remain required, with
the existing joint-judge gates. No final source is opened or hashed and no speed
claim is made here.
