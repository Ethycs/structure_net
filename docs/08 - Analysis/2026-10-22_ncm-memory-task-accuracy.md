# End-to-end accuracy of the memory-trained NCM

**Trained NCM task accuracy estimate: 5/256 = 1.95%.**
The 95% target is not met.
This is source-grounded LLM-judged summary quality on held-out short fragments.
It is not the 95.46% codec reconstruction figure and is not human-rated production accuracy.

Date: 2026-10-22. Hypothesis: `ncm-memory-task-accuracy-v1`.
Classification: `ncm_memory_task_accuracy_below_95`.
Aggregate: `data/experiments/ncm_memory/20261022_accuracy_v2/campaign_results.json`.
SHA256: `5fb395ba255a4b0ea70ba77d341c2834af1462ae7cc3513b83507336c4bb42fa`.

## What counts as correct

A complete generated summary passes only if it is nonempty, within 128
cl100k_base tokens, faithful to the original source, readable and useful as a
brief summary of the main point. Every material claim must be supported.
Reasonable paraphrases and omission of minor/list details are allowed. Headings,
tangential fragments, unsupported invented facts and incoherent text fail.
Claims carry supported/contradicted/insufficient-evidence labels and source-line
IDs; the exact original evidence text is reconstructed by code.

Primary judge: **`claude_code/claude-sonnet-4-6`**. Model identities were hidden and
candidate order shuffled. All candidates were judged independently against the
complete original text. Saved targets were a third anonymous candidate, not
assumed correct. No outputs were regenerated and no model was retrained or
selected using these grades.

The judge accepted **24/24**
source-reviewed acceptable controls and rejected
**24/24** incorrect controls.
These controls contain realistic paraphrases from 12 separate source sessions,
not just easy exact-match positives. They were authored/reviewed by Codex, not
independent humans. The previous overly strict 0–4 rubric was not reused.

## Measured pass rates

| Candidate | Correct summaries | Task accuracy estimate | Wilson 95% interval | Fully source-supported |
| --- | ---: | ---: | ---: | ---: |
| original_ncm | 0/256 | 0.00% | 0.00%–1.48% | 0/256 |
| trained_ncm | 5/256 | 1.95% | 0.84%–4.49% | 41/256 |
| saved_summary | 58/256 | 22.66% | 17.95%–28.17% | 238/256 |

`original_ncm` is the NCM before the shared-data adaptation, not the full Llama
3B teacher. `saved_summary` is the supplied training-target style. A supported
snippet can still fail as a summary when it misses the main point.

| Candidate | Salience failures | Readability failures | Supported/material claims |
| --- | ---: | ---: | ---: |
| original_ncm | 256 | 210 | 16/691 |
| trained_ncm | 245 | 227 | 233/507 |
| saved_summary | 198 | 25 | 381/383 |

The trained-minus-original change is
**1.95 percentage points**.
The point target requires at least 244/256 passes. The stronger condition that
the Wilson lower confidence bound is at least 95% is
`False`. Confidence intervals describe sampling
under these judge labels; they do not include judge error or model-seed variation.
Source-session cluster-bootstrap intervals (10,000 draws, seed 20261022) are
saved in the aggregate to account for multiple fragments from one session.

## Independent audit

Secondary judge: `codex_sdk/gpt-5.6-terra`. Its control result and any completed
64-case audit are retained, including infrastructure failures. Agreement:

```json
{
  "original_ncm": {
    "cases": 64,
    "same_pass_fail": 64,
    "secondary_passes": 0
  },
  "trained_ncm": {
    "cases": 64,
    "same_pass_fail": 61,
    "secondary_passes": 1
  },
  "saved_summary": {
    "cases": 64,
    "same_pass_fail": 51,
    "secondary_passes": 17
  }
}
```

## Scope and integrity

The 256 cases are exactly the fixed held-out cases from the completed training
campaign. The train/test source conversations are disjoint. The sources fit
512 GPT-2 input tokens, with a 48-position NCM output canvas. This is not an
evaluation of the full 2,048-token fragments or 6,300-token application batches.
LongMemEval was used as training data, so this is the designated source-disjoint
test partition, not the untouched LongMemEval benchmark.

Raw requests, replies, retries, source citations, candidate mappings, producing
source and hashes are preserved under `data/experiments/ncm_memory/20261022_accuracy_v2`.
The first validator incorrectly rejected some conservative failure judgments as
format errors and retried them. The [parser recovery amendment](../07%20-%20Status%20Reports/2026-10-22_ncm-memory-accuracy-parser-recovery.md)
preserves that interrupted V1 run. V2 reuses the first usable original reply,
including recovered failures, and ignores later redundant replies. No judge
decision was changed. Under the corrected validator, valid low grades were not
retried. The aggregate was independently rebuilt from the raw responses. The
study remains exploratory with one adapted checkpoint; no speed claim is made.
The earlier codec accuracy and word-overlap report remains unchanged as a
separate measurement of different things.

## Reading the audit correctly

The second judge accepted **1/64 trained-model summaries (1.56%)**. Its 61/64
pass/fail agreements with the primary judge are dominated by shared failures;
that agreement is not 95% model accuracy. The per-case disagreement table is
retained in `data/experiments/ncm_memory/20261022_accuracy_v2/audit_confusion.json`.

Three hash-selected, post-outcome agent source checks confirmed garbled generated
language and missing main points. For example, case
`495d1cc983c105418c60c1de1236909b0799915b66377424b95279904a1aea1e`
describes venues with a $10–$20 price range; the model outputs a garbled
"a trendy shropub" sentence with "$10-$10". The other checked cases concern
Chicago water taxis and a novel about recruiting young football players.
These are qualitative agent checks, not additional human-rated accuracy labels.
Individual claim classifications can still be debatable; the two judges are
estimators rather than human gold labels. Both find task success around 2%,
and the 95% target is not demonstrated.
