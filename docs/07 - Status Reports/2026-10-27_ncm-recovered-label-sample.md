# Recovered saved-label review sample

Freeze 128 existing source–summary pairs from the audited recovered review pool
before inspecting semantic scores. Select 64 raw inputs of 512–2,048 stored
tokens, then 64 of 192–511 tokens. Order candidates within each stratum by
SHA-256 of `recovered-label-review-20261027-v1:` plus their immutable pair ID.
Take at most one pair from any source component across both strata. This is a
balanced diagnostic sample with source-component diversity, not a uniformly
weighted estimate of all pair rows. Preserve every selected case and any later
failure; no resampling based on the contents or grades.

The upstream pool contains 104,503 candidate pairs, 55,701 distinct input texts
and 11,282 source components after provenance, existing source-split guards and
native token-capacity filtering. These checks do not establish label quality or
codec reconstruction quality. Merges and unproven source records are deferred.

This preparation writes source text, saved target text, pair/component identities
and provenance references in `20261027_recovered_label_sample_v1`. It validates
the upstream actual terminal and independent export audit, and freezes the exact
export, producer and selection implementation. It performs no teacher or judge
requests and does not admit labels to training or open the sealed final cohort.

The subsequent semantic review must use the already qualified 128-token profile
and both configured judges (Claude Sonnet 4.6 and GPT-5.6 Terra), with source text
as the evidence authority. Complete-summary faithfulness, salience and readability
are the endpoint; exact wording is not required. Report claim support, contradiction,
insufficient evidence and numerical errors separately. This review measures saved
label quality, not NCM model accuracy. Before any external grading, inherit the
current audited first-judgment cache and bind the exact rubric, judge identities,
packets and request lifecycle. This preparation does not itself qualify that
grading lifecycle or alter the currently scheduled model DEV evaluations.
