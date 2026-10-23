# One revision of rejected NCM training labels

Date: 2026-10-23. Hypothesis `ncm-targeted-training-label-revisions-v1`.

Prepare better TRAIN targets for a possible subsequent NCM iteration while the
fixed 32,768-update reviewed-draft model continues unchanged. This study has no
student accuracy endpoint. The full goal remains 95% complete-summary semantic
accuracy on source-disjoint evaluation, with quality established before speed.

The current data contain 438 admitted first drafts and 74 rejected drafts from
the fixed 512 additional TRAIN sources. Of the 74, 73 failed semantic review and
one additional draft failed format or capacity. Their band counts are 16, 31,
7, and 20. Select exactly those 74 using the saved first judgments and capacity
decisions, before any student development outputs. Keep all 438 admitted drafts
and the running model's 690 examples unchanged.

The earlier all-512 revision workflow remains failed. Its first four batches
exhausted their transport budget, and other requests may have been in flight
when it stopped; unobserved responses are not treated as known candidates.
Both original judge aliases answered new source-free availability probes with
HTTP 200 on 2026-10-23 at 23:02 UTC. This is a fresh targeted study after that
external-state change. Preserve every old failure and request. Each selected
case gets one first-observed schema-valid revision, using its original draft,
first critique, complete source, and capacity feedback. Permit at most three
fresh transport/schema attempts, never a semantic reroll. An unchanged revision
reuses its existing judgment, including a negative one.

Use the original Sonnet teacher, Terra reviewer, 128-token semantic rubric,
temperature zero, teacher batch two with 2,048 output tokens, reviewer batch one
with 4,096 output tokens, and 180-second timeouts. Limit both pools to two workers.
All source spans must match the already verified public TRAIN data. No student
development or sealed source is sent to this label workflow.

The pilot is the first rejected source in each original length band. Require
four valid first revisions, four valid judgments, and four admitted labels under
the unchanged semantic and format checks before main production. A pilot failure
stops without a retry of its semantic outcome. Retain all four pilot candidates
and judgments exactly in the full 74-case population. Main admission is per label;
retain the complete 74-case denominator, every failure, and all prior 1,444 first
judgments. The teacher acceptance rate is not student accuracy.

Use fresh `20261023_targeted_label_*_v1` roots. A supervisor records actual child
exits. If any request group exhausts all three attempts without a schema-valid
response, the all-valid stage can no longer qualify: record that evidence and
stop the remaining queue. Archive and conservatively register the complete
outcome only after raw generation, judgment, pilot-retention, and cache replay.
No GPU work or change to the running word model occurs here. Future model or
data changes require a separate prospective study after its semantic results.
