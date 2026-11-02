# One critique-guided revision of failed native-NCM training labels

Date: 2026-10-24. Exploratory TRAIN-data hypothesis
`ncm-lm1b-capacity35-critique-revision-v1`.

Known before new generation: the first35-candidate study completed all8 children,
audited and registered failure. Capacity32/35. Raw passes27/35 Terra and32/35
Sonnet; BERT-rendered passes25/35 Terra and30/35 Sonnet, all valid. Exactly23
cases satisfy capacity and all4 semantic checks;12 fail at least one. No dataset
was exported. Parent campaign SHA256
`f49329b87b2635c038a3b554b4c48b2ed1977e6f282e7c0081b319d74b0c745d`;
latest2424-first cache SHA256
`b10c96895aedeaa898dca7d96ae92fe74d5c84695d0dcd226f49db4ccf95001f`.

Keep the SAME35 source cases and order. Retain all23 qualified first candidates
exactly, including their original judgments. For each of the12 failed candidates,
make exactly one critique-guided textual revision using full public source,
previous candidate, BERT rendering and all recorded reviews/capacity counts.
The source is authoritative. Prompt targets30words, at most40when possible;
omit incidental lists/values/code, spell out essential units/decimal quantities
and describe case-sensitive code operations. This addresses both capacity and
observed normalization failures. This is genuine TRAIN-label editing, not a
student prediction or a reroll of unchanged semantic judgments.

Teacher Sonnet,temperature0,max768,timeout180,four workers,at most3transport/schema
attempts. First schema-valid revised text wins. Capacity/semantic failure never
triggers a schema retry. An unchanged failed candidate retains its original
negative via canonical packet identity. Only the selected12 can cause teacher
requests; no new generation for the23 passes. Pilot first4 cases and reuse their
exact final candidates in the complete35 run.

Inherit all2424 decisions and all pilot firsts; never overwrite or reroll valid
negatives/exhausted invalids. Both Terra and Sonnet grade raw and BERT renderings
for ALL35, reusing exact prior packets. One judge shares the teacher model; the
other is a different qualified model. Keep all35 cases and all failures visible.
Admission is unchanged: all35 must fit126BERT lexical positions and128storage
tokens before/after BERT and pass both judges in both wordings, all valid. No
relaxed gate, selective export, clipping, inference repair or fallback.

If admitted export all690 source-identical TRAIN cases:35 genuinely new target
texts relative to the original dataset and655 unchanged targets. Remove stale
GPT2-word/XSum-concept arrays from all rows before fresh native encoding. Original
labels, first proposals, negative judgments and failed parent remain preserved.

Reuse the tested first-study engine with explicit new root/configuration bindings
and a separate auditor. CPU tests verify the fixed12/23 selection, exact payload
distinction and inherited2424 firsts. Independent audit replays the selection,
retained candidates, teacher payloads/first text, capacities, all4semantic score
tables and canonical inheritance, then registers and independently reads back
the conservative result. Fresh `20261024_lm1b_capacity_revision_*_v1` roots.
The source and plan freeze before new teacher calls. Entire study CPU/network.

This remains TRAIN-label preparation; it is not source-only summarizer accuracy.
The full95% task goal stays unchanged, all permanent128 task DEV remains outside
gradients, and all320 final sources remain closed/unhashed. No speed claim.
