# Wider training labels vetted; student quality remains unmeasured

Date: 2026-10-23. Classification `training_labels_vetted_student_quality_unmeasured`.
Aggregate `data/experiments/ncm_paper_regime/20261023_wide_vetted_labels_results_v1/campaign_results.json`, SHA256 `e184dbc1ab6f80d22bf06614fbd1e1e809387e349531e58890e892a5bb8bdf28`.

The fixed112public training sources span512–2048cl100k tokens, with28sources
per band. Every source received one Sonnet draft, one Terra critique, one
Sonnet revision and one Terra final audit. Four pilot texts and their first
judgments were retained exactly; there were no semantic rerolls or selection
between draft and revision. The4/4final pilot gate passed before main production.

First drafts passed93/112whole-label audits. Final revisions passed109/112
(97.32%), all112judgments valid. The final population retains all three
failures; one final candidate also fails the format/capacity checks. The109
accepted training labels fit both128stored cl100k tokens and128native tokens
including EOS, have exact native round trips, nonempty evidence-linked claims
and no embedded special tokens. Accepted counts by increasing source-length
band are27,28,26,28. The final Wilson95%interval is92.42–99.08%; no teacher
population95%claim is established by a97.32%point estimate.

Raw first schema-valid teacher candidates and first-valid judgments were
replayed offline against saved labels and scores. Token lengths, native
round trips, evidence quotes and exact pilot reuse were checked. Maximum
three attempts allowed transport/schema recovery only. All eight workers
reached observed terminal success. Generation and judging stayed sequential.

These are accepted teacher training labels, **not student accuracy**. Terra
both critiques and audits; this is not independent human gold. Diagnostic16,
new sealed64and original sealed256sources were excluded. No student training
is admitted by this result, and no model, speed or compression claim follows.
The separate2048-update native NCM study scored0/64development summaries.
