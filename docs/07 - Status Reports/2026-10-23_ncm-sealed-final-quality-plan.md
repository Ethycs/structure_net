# Sealed complete summary evaluation for the frozen NCM candidate

Date: 2026-10-23. Prospective hypothesis `ncm-generated-history-final-quality-v1`,
one model seed, underpowered. The concept bootstrap and teacher-label pipeline
are live. The generated-history word candidate has not trained or produced
development outputs. This protocol is fixed before those outcomes and before
opening either sealed cohort for this candidate.

First finish, audit and conservatively register the unchanged generated-history
development candidate. Admit final evaluation only if each64-case development
cohort has64valid Terra and Sonnet judgments and at least61complete-summary
passes under each judge. Require the registered campaign, word checkpoint,
forecaster checkpoint and grade/cache hashes to remain exact. An admission
failure stops before reading or hashing either sealed source file. Do not
change, retrain or select model weights after final admission.

Use all original256sealed sources from `20261022_data_v1/sealed_test_sources.json`
and all64longer sources from the original application pool's
`sealed_final_sources.json`. The latter retain16cases in each existing512–767,
768–1,023,1,024–1,535 and1,536–2,048cl100k-token band. The former retain the
original fixed case IDs and source hashes derived from the preexisting public
proof after removing its training/development identities. Read no old atom or
summary target. Verify exact public source spans, fixed membership/order,
unique cases/sessions/groups/source hashes, and disjointness among final,
expanded TRAIN and both development cohorts. Record original source-partition
proofs. Do not exclude cases based on length, content or generated output.

Prepare complete canonical native prompts and GPT-2 character alignment with
the same system/template and source conventions as training. Run the unchanged
frozen28-layer Llama body in BF16, storing all aligned source features in FP32;
original weights/MLPs must retain their recorded body digest and no gradients.
Before full feature extraction, run a repeatability/memory pilot on first and
longest native prompts, then retain those two feature files exactly. Use GPU0,
one own model at a time. Stop on invalid alignment, truncation, changed frozen
state or the existing8.6GBfeature-allocation ceiling.

Run the same frozen concept forecaster and trained independent word branch.
Inference pilot cases are the first final source and the longest GPT-2 source;
require deterministic composed inference and exact checkpoint reload. Retain
those outputs unchanged in all320outputs, in the fixed cohort/source order.
Keep all192raw word IDs and all eight predicted concept levels. Decode once at
the first global EOS. No target concepts, previous gold words, lookup, output
repair, truncation to the128stored-token cap, autoregressive fallback, rerolls
or candidate selection may enter inference.

Use the same qualified128-token complete-summary rubric, four workers and
at most three transport/schema attempts per judge packet. Keep the first valid
decision and all negatives across the complete latest development judgment
cache. Send only exact proven public sources and locally derived summaries to
the already authorized central-dev endpoint. Require supported material claims,
useful salient content, readability, nonempty text and at most128stored cl100k
tokens jointly for each judge's pass.

Grade every case with Terra. If Terra has allNvalid and at leastceil(.95N)
passes in a cohort, grade those exact texts with Sonnet. Final success requires
allNjudgments valid for both judges and at leastceil(.95N)cases passing BOTH
judges, independently in each cohort:244/256short and61/64long. This consensus
criterion is fixed prospectively; separate95%judge rates alone cannot pass it.
Invalids remain in the original denominator and prevent a success claim.
Report per-judge and joint pass counts, point estimates, Wilson95%intervals,
overflows, empty outputs, invalids and every failure. This is measured accuracy
on these sealed cohorts, not a95%population lower confidence bound or human
rating claim.

Use fresh roots `20261023_sealed_final_{data,feature_pilot,features,
inference_pilot,outputs,packets,grade,results,workflow}_v1`. Hash producing
source/plans, checkpoint identities, complete data and output manifests and raw
first judgments. Require actual child zero exits and final audit/registry exact
and semantic readback. The registry must accurately mark final_test_opened=true,
keeping confirmed=false for this one-seed exploratory experiment. Preserve all
earlier failures. A failed development gate consumes neither final cohort.
No speed or compression claim follows from this quality evaluation.

The separate conditional final supervisor waits for the existing generated-
history workflow's live kernel identity or completed result. It first audits
and registers that development outcome. An unqualified outcome ends with both
sealed files unopened. Qualified development admits the fixed source/feature/
inference/judge stages, each with an observed child exit. Before final audit it
writes an immutable `evaluation_complete.json` containing those receipts.
Finalization hashes that marker and the already completed stage receipts,
excluding its parent's still-changing operational progress journal. This
prevents circular provenance while the parent observes finalizer/registry exits.
Goal completion still requires the root agent's independent final evidence audit.
