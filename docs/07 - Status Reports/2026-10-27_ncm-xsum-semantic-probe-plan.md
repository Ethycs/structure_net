# Semantic grading of the frozen XSum adapter outputs

Exploratory hypothesis `ncm-xsum-semantic-probe-v1`: output distillation improves
whole-summary quality over the matched input-MSE continuation. Exact native
content agreement is already known: initial 16.17%, continued MSE 16.68%,
distilled 16.83%. The preceding preservation experiment failed. Two fixed
examples showed some better factual content, including retaining Burundi.
These observations motivate this separate semantic diagnostic; they do not
change the earlier failure or justify selecting favorable cases.

Use all 128 existing probe sources and all four frozen outputs: native,
initial adapter, matched MSE continuation and output-distilled adapter.
No inference, training, source truncation or output editing. Decode saved IDs
with the existing released text decoder: stop at the first EOS or pad, skip
special tokens and strip exterior whitespace. Retain raw IDs and termination
reason. Count stored text with cl100k_base; the cap remains 128 tokens.

Recover each complete source from the audited public XSum training catalog by
its hashed index record. Require exact source SHA256, provenance, GPT2 ID
round-trip, and identity correspondence with both prediction experiments.
Packets contain only that source and one anonymous candidate named `summary`.
No model identity, reference summary, metric, training target or comparison
result enters the judge packet. Every arm is assessed independently against
the source. Identical packets share the same first judgment.

These public TRAIN sources are held out from the 512-case adapter fit but may
have been seen by the released XSum teacher; the probe was already inspected
in development. This is not an unseen final test across every original model,
not application-range validation, and cannot establish the final 95% goal.
Application DEV and final320 remain excluded. The GPU1 curriculum continues
unchanged. This study is CPU/network only under locked Pixi ncm.

Use the already qualified 128-token rubric and current judge identities:
secondary Terra `codex_sdk/gpt-5.6-terra`; primary Sol
`codex_sdk/gpt-5.6-sol`. All 50 existing calibration controls and their actual
receipts must verify. Do not rerun those qualifications. Preserve the entire
4691-first epoch2 cache, including historical judge identities. New judgments
extend it without replacement. Save exact requests, responses, first rows,
packet hashes and source-line citations. Temperature 0, response cap 4096,
at most three transport/schema attempts, four workers; no retry of a valid
negative decision. Persist every completed row before updating the in-memory
cache and append a flushed first-judgment journal. Save the full cumulative
cache at stage completion; an interruption requires recovering existing first
rows/raw attempts before any further requests, never a blind restart.

Run a systems pilot on the first four fixed sources for all four arms. All
sixteen arm-case judgments must be valid; quality does not gate the pilot.
Retain those exact first judgments in the full grade. Full grading runs Terra
for all128 cases in all four arms. Sol is conditional only for an arm with
all128 valid Terra judgments and at least122 complete-summary passes. Final
qualification is never inferred from this diagnostic, even if both judges pass.

A case passes iff faithful AND salient AND readable AND nonempty AND within
128 stored tokens. Paraphrases are accepted. Report n/128 and percentage for
each arm, every component, invalid count, exact same-case distill-versus-MSE
discordances and Wilson intervals (descriptive intervals on a reused probe).
Invalid judgments make the corresponding accuracy unresolved; report lower
and upper bounds without relabeling invalid outcomes as valid failures.
The directional diagnostic requires all Terra judgments valid and at least
seven more distill passes than MSE passes (>=5 percentage points of128).
Report the >=122/128 threshold separately; preserve the distinction between
one-judge scores, joint same-case scores and application final accuracy.

CPU tests cover frozen source/output binding, anonymous single-candidate
packets, pass/length boundaries, invalid denominators, directional threshold
and unchanged inherited firsts. Before requests, freeze producer source,
configuration, input hashes and exact public payload proof. Independently
replay every raw judgment, first-decision rule, token count and aggregate.
Then report, conservatively register and independently read back the result.
No speed or efficiency claim is made.

Fresh roots are `20261027_xsum_semantic_probe_<stage>_v1` under
`data/experiments/ncm_paper_regime`, for prepare/pilot/grade/audit/results.
The new cache is an explicit child of epoch2's4691-first cache. Before later
main-branch grading, preserve this branch's new firsts through an explicit
lineage extension; do not edit the live trainer or silently discard judgments.
