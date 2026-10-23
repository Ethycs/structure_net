# Converged frozen-native summary development evaluation

Date: 2026-10-23. Hypothesis `ncm-converged-native-summary-quality-v1`.
Classification **converged_native_summary_development_failed**. Aggregate `data/experiments/ncm_paper_regime/20261023_converged_native_results_v1/campaign_results.json`,
SHA256 `de982821af286d11ce1c66fa4bf237860f78a6b47a5d40c6c72a857046ba3724`.

The converged hybrid scored **41/64 whole-summary passes
(64.06%)** under qualified Terra. All 64 original development cases remain
in the denominator. The original fixed-nine-pass model remains a separate
0/64 failure; its checkpoint and all original judgments are unchanged.

| Judge role | Whole summary | Valid | Faithful | Salient | Readable |
| --- | --- | --- | --- | --- | --- |
| secondary | 41/64 | 64/64 | 42/64 | 62/64 | 62/64 |

Whole-summary success requires faithfulness, salience, readability and at most
128 stored cl100k tokens jointly. The Terra result contains
0 stored overflows and 0 empty
summaries. Its Wilson 95% interval is `[0.518208506097791, 0.7471159770854825]`. Independent
Sonnet grading was not admitted because Terra did not reach 64 valid judgments and 61 passing summaries.
The joint development gate **failed**.
Every failed case, source text, component failure, reason and cited claim
evaluation is retained in `data/experiments/ncm_paper_regime/20261023_converged_native_results_v1/case_failures.json`. This includes invalid judgments
and format failures; no failed case is removed from the denominator.

This uses the original 2,048-update model, not the later four-case capacity
control. Original embeddings, all 28 Llama3B body blocks including MLPs, native
word head and concept books stayed frozen. Source-only five-level predictions
conditioned 128 whole-frame word updates and one additional verification pass.
All full word frames reached an exact verified fixed point; every four-update
snapshot exactly reproduced the original failed outputs and concept codes.
There was no oracle plan, gold history, external verifier, AR fallback, new
training, output repair or choice between alternatives.

The two-case inference/reload pilot completed before the remaining 62 cases;
its first outputs and all word states were retained exactly. Generation ended
before judging. Both qualified 128-token judge profiles were checked. All 192
earlier first judgments, including negatives, were audited and preserved.
Raw requests, first-valid response parsing, token counts, complete histories,
source identities and whole-summary decisions were replayed offline. New
endpoints correctly identify NCM_trained=true. All workers terminated normally.
An independent metadata audit found no overlap by case ID, source session,
group or exact source hash against all 1,231 stored training/diagnostic cases.
The new scorer also reproduced the old0/64 endpoint and its component counts
without making API requests. These controls are separate from new model quality.

This is a new causal-decoding hybrid endpoint, not a reproduction of released
NCM training and not a repair of the original nine-pass result. It uses 134
body/query passes per frame including verification. No throughput, latency,
compression or speed benefit is measured. Unchanged AR baselines under the
same 128-token rubric previously scored 52/64 for 3B and 47/64 for 1B.

The cohort is source-group disjoint from training but has been reused for
development; inputs are short fragments. The original 256 final cases and new
64 longer-source final cases remain sealed. Neither a passing development point
estimate nor numerical convergence alone establishes 95% final application
accuracy. Full input-range and final evaluation remain required for the goal.
