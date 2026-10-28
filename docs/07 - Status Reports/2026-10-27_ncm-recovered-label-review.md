# Semantic review of the recovered saved targets

Exploratory data-quality study `ncm-recovered-saved-label-review-v1`. Preserve the
completed 128-case selection and its producer unchanged. Before any new requests,
bind all exact source/target packets, current qualified judges, the 128-token
rubric and the latest audited first-judgment lineage. Use both judges on all
128 labels, irrespective of their scores; this is a label diagnostic, not a
student quality gate. Keep both 64-case length strata and one case per source
component. The sample is component-diverse and balanced, not pair-weighted.

The earlier sample note named the historical Claude primary. Inspection of the
active pipeline establishes that the user-authorized, already qualified primary
is now `codex_sdk/gpt-5.6-sol`; secondary is `codex_sdk/gpt-5.6-terra`. Both scored
50/50 on the fixed controls. This amendment uses that existing replacement and
preserves all historical Claude decisions under their original model identities.
It does not requalify a model, change the rubric, or resample cases.

One anonymous `summary` candidate is supplied with exact numbered source lines.
The source alone determines support; paraphrases are allowed. Use the existing
request/parser and first-judgment helper: at most three attempts for transport or
schema failures, never resample a valid unfavorable judgment, and never overwrite
or renew an exhausted request. Save raw attempts and fsync each new first to the
journal. Use four workers, CPU/network only, through the existing configured
gateway. Existing request-frontier uncertainty must not be given a new budget.

Before full grading, run a request-lifecycle pilot on fixed sample indices
0, 1, 64 and 65, covering two long and two short sources, under both judges.
Require eight valid outcomes, regardless of semantic pass/fail. Reuse those firsts
in full grading. Do not launch another copy when a worker is live, and record its
actual terminal before moving to the next stage. Fresh stage roots are
`20261027_recovered_label_review_{prepare,pilot,grade,audit}_v1`. Preserve failures.

Report per-judge complete-summary faithfulness AND salience AND readability,
same-case joint passes, disagreement, invalid counts and lower/upper pass bounds,
with fixed full denominators overall and per length stratum. Invalid outcomes
remain unresolved. Report supported/contradicted/insufficient-evidence claim
counts and unsupported claims containing digits separately; the latter is not
automatically a numeric-value-error rate. Replay raw firsts and all metrics in
an independent audit stage. No subset becomes model test accuracy or admission
of every saved label. No new teacher generation, model updates or final access.

The completed label branch must be included in later public and task DEV
lineages through a new consumer wrapper before further model judging. Existing
training producers and their imported source files stay unchanged. Until the
branch is complete and audited, do not initiate another judging branch that
could omit these firsts. GPU training may continue independently.
