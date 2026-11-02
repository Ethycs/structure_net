# Human-reference diagnostic for the independent quality rubric

Date: 2026-10-21. Exploratory diagnostic, not a replacement primary endpoint.

After 53 primary pairs had been graded (NCM 0/53, Llama 14/53), a low Llama
acceptance rate motivated measuring the rubric's acceptance of human reference
summaries. This choice is post-outcome and cannot rescue the 95% gate or establish
human gold accuracy for the judge.

Select the first 20 primary DialogSum cases sorted by SHA256 digest of
`['reference_audit', case_id]`, without consulting their model judgments. Grade
the first and second human references as two anonymous candidates under the
unchanged v3 rubric, primary judge, source-turn evidence, score thresholds and
128-common-token storage cap. Preserve all three references as grading context,
as in the model comparison; thus reference matching can bias this check upward.
Human references are not guaranteed correct. Their greater length may also
improve coverage relative to models asked for approximately 32 words.

Run only after the fixed model evaluation and secondary-judge attempt finish;
no concurrent judge jobs. Use the same four-worker request/retry policy. Save
all packets, exact source citations, native replies and pass reasons in a fresh
`20261021_reference_diagnostic_v1` root under the v3 campaign. Report factuality,
coverage, coherence and strict pass rates for these 40 references. This is a
secondary rubric diagnostic; retain the full primary grade and its limitations.
