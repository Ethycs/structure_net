# Critique every first draft, with measured capacity supplied to one revision

Date: 2026-10-23. EXPLORATORY v2 pipeline amendment. All128preselected first
drafts are now frozen:64/64schema-valid batches, no empty text/missing claims,
three drafts over both128token limits (stored/native counts141/142,132/133,
132/133). The v1 implementation stopped at draft capacity before critique, though
the intended algorithm includes critique and one revision. No main-sample
semantic judgments or final revisions have been inspected. Preserve the failed
draft-capacity status, all128first drafts and every raw response.

Allow schema-valid first drafts to reach critique even if over capacity, without
regenerating, clipping, excluding or relabeling them. Add exact stored/native
token counts, both ceilings and a fit flag to EVERY draft's critique input for
the one fixed revision. Give that revision a40–60-word preference to reduce
length overruns. Preserve the original judge's flags/claims/reason unchanged;
capacity metadata is an additional deterministic check, not a revised verdict.

This is changed revision production and needs a fresh pilot. Reuse the same
ten pilot drafts and original128-profile critiques; request exactly one v2
revision for every draft, including positives. Audit all ten v2finals. Reuse ALL
earlier exact128-profile judgments, including v1final audit positives/negatives
and exhausted invalids. Require10/10valid final passes and all10finals meeting
both capacities before the main sample proceeds. The v1pilot10/10 stays intact.

New roots: `20261023_revised_labels_pilot_{revision,audit}_v2` and
`20261023_revised_labels_main_{critique,revision,audit}_v2`. Original pilot draft/
critique and original main draft roots stay unchanged; NO draft generation
command is exposed. Main critique evaluates all128first drafts, main revision
runs once for each, and final audit evaluates all128regardless of cap overruns,
so overruns remain visible failures rather than exclusions. The overall label
gate still requires all final candidates fit both token frames,128valid audits
and at least122whole-summary passes. A failed final capacity check cannot be
rescued by this amendment. No revisions after final audit and no per-case
semantic resampling. Pipeline phases remain sequential with fourAPIworkers.

Hash the v1source, both qualification producer/results, this amendment and the
v2wrapper in every new stage. Unit-test capacity metadata preserving critique
decisions. This remains label-production evidence, not student quality or an
NCM95%result. A passing audit admits a separately specified training study.
