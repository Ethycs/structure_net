# Semantic quality of oracle NCM codec reconstruction

Date: 2026-10-23. Exploratory hypothesis
`ncm-oracle-codec-semantic-quality-v1`. The fixed sample-scaled EMA transfer
completed with 94.262% token recovery versus 95.179% for the released codec;
the unscaled transfer scored 67.167%. Both transfer representation gates failed.
Those failures remain unchanged. No semantic judgments for this follow-up have
been inspected when fixing this protocol.

Question: can the released or sample-scaled codec preserve complete-summary
meaning on excluded source groups, despite differing token strings? The user's
goal is task quality, not exact output agreement. This component control cannot
establish source-to-summary accuracy: codes come from the complete target.

Select 64 of the 252 complete diagnostic targets by ascending SHA256 of
`ncm-codec-semantic-v1:` plus case ID. No selection by output, recovery, number
changes or judgment is allowed. Retain the exact full public source, target,
released reconstruction and final sample-scaled reconstruction for each case.
All targets were excluded from codec fitting by session, group, source hash
and exact text. They are training labels for a later conditional summarizer;
this is neither the reused 64-case model development cohort nor a final test.

Three arms: uncompressed target control, released-codec reconstruction, and
sample-scaled-codec reconstruction. Use single anonymous `summary` candidates
in the existing qualified 128-token rubric, judged against the full source.
Require faithful claims, salient coverage, readability, nonempty text and at
most 128 cl100k tokens. No reference target is included in a codec packet.
Preserve all 320 prior first judgments from the completed teacher audit,
including negatives; reuse any exact packet/rubric/role/model identity.
Deduplicate identical packets across arms. Never reroll valid judgments.

Both judges must have passed their existing 50-case qualification. First use
Terra (`codex_sdk/gpt-5.6-terra`); admit Sonnet
(`claude_code/claude-sonnet-4-6`) for an arm only if both that arm and the
uncompressed control pass Terra. Each gate is at least 61/64 complete-summary
passes, all 64 valid. Sonnet must independently meet the same gate. An arm that
fails Terra cannot be rescued by Sonnet. Invalid judgments count in the full
denominator and prevent qualification. Report valid counts, exact numerators,
component scores, length failures, paired gains/regressions and Wilson intervals.
These are oracle semantic pass rates; the `task_accuracy` field remains null.

Use at most four API workers and three attempts per packet for transport or
schema failures only. The budget is at most 192 distinct packets per judge,
including the pilot; conditional Sonnet stages may use fewer. Stop on identity,
public-source, qualification, first-judgment or artifact drift. A systems pilot
uses the first four selected cases in each arm, requires 12 valid arm-case
judgments, and contributes its exact first judgments to the full run. The
already qualified single-candidate format and concurrency are unchanged.

Before any external request, save the exact packets, rubric/model scope and
proof that every source span belongs to the SHA-verified public MIT LongMemEval-S
file. The payload contains public passages and locally produced summaries,
with generic instructions. It contains no private data, local paths or secrets.
The existing approved judge gateway is `https://central-dev.zt:4000`.
Authentication is used only in the normal HTTP header, never recorded in
packet JSON. Do not open either final cohort.

Roots under `data/experiments/ncm_paper_regime`:
`20261023_oracle_codec_semantic_{prepare,pilot,grade,results}_v1`.
Entry point `python -m experiments.structure_net.ncm_oracle_codec_semantic
{prepare|pilot|grade}`. Capture source/config/input hashes before each stage,
observe terminal receipts, aggregate and register the result conservatively.
No training, source-conditioned generator, speedup or compression claim follows
from this control. A successful codec arm is eligible only for a separately
specified conditional-generation experiment.
