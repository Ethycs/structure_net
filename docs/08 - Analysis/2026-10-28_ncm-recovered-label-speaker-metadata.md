# Missing speaker metadata in recovered-label grading

The fixed 128-label review contains 57 paraphrased targets and 71 exact
substrings of their source. Of the 57 paraphrases, 46 passed both judges.
Reviewing the eleven joint failures found seven rejected solely for attributing
the text to an assistant when the normalized source packet contained no role.
This is a posthoc diagnosis of the saved judgments, not a revised score.

An exact join to the original label tables recovered source-pointer metadata
for all57 paraphrases. Every one of the seven attribution cases has an
unambiguous original `assistant` role. The join requires the same pair ID,
target hash and full source-span hash. All matching original pointers and
label rows are retained, including their timestamps and summarizer identities.
No inference from writing style supplies the role.

The other four paraphrase failures include an unsupported requested item count,
comparisons with an earlier feature list, a narrowed interpretation of future
plotting, and a claim that shopping-list categories were reduced. Restoring a
speaker role alone does not address those judgments. Preceding conversation
context might explain some claims, but it has not been recovered or admitted
by this diagnostic.

The original joint score remains54/128. No judge was called and no old decision
was changed. The broader candidate pool remains unadmitted. The appropriate
next data-quality test is to present verified original conditioning metadata
with the same complete sample and unchanged rubric, while preserving every
earlier first judgment. It must not select only previously failed cases or
count a plausible correction as a measured semantic pass.

Evidence: `data/experiments/ncm_paper_regime/20261028_recovered_label_role_diagnostic_v1`.
The CPU worker completed with actual exit zero, session93848, chunk`87c997`.
Its selected SQL rows, per-case contexts, producer, result and actual execution
receipt are saved. This is training-label provenance evidence, not NCM model
accuracy or a final-test result.
