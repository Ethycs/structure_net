# Fresh64 evidence-bound TRAIN teacher experiment

2026-10-24. Hypothesis `ncm-evidence-first-fresh64-teacher-generation-v1`.
The previous fixed64single-stage teacher scored58/64faithful passes on Terra,
below61/64admission. Preserve all of those labels and judgments. Do not regenerate
the six known failures or bulk-label10000with that failed protocol.

From the unchanged10000reserved TRAIN sources, exclude all64previous components.
Choose fresh64by this hypothesis's source-ID hash,32short+8in each long bin.
Require64distinct components and exact public-source proof. No source selection
uses teacher/model outputs or judgment quality. All DEV/final partitions stay
excluded by the existing source reservation; final files remain closed/unhashed.

Each case receives two fixed calls: Terra draft then Sonnet review, separately
with the full source in both calls. Draft768output tokens; review512; prompt
limit6300cl100k tokens. Workers4. Prefer35-55words/<=128stored tokens and126BERT
lexical tokens, keeping the original native CLS/PAD target framing. Begin 'The
passage'. Request source-line support for each final factual clause. Explicitly
preserve list/item scope, qualifications, conditional code behavior and stated
chronology; prohibit inferred recognition, added causes, order or generalized
properties. The review treats the draft as untrusted and always produces its
own final summary and supports. The two teacher models also serve as downstream
qualified judges; semantic assessment remains anonymous and independently
requested, not independent human validation.

Parse only JSON schema, literal summary-span membership and existing line IDs.
Those checks do not establish entailment. The first schema-valid response is
retained; semantic quality, capacity, emptiness or support completeness never
trigger another attempt. Up to3transport/schema/model/termination attempts per
stage, all raw attempts kept. No repair, reranking, quality-based selection or
judge-driven generation loop. If the draft has no valid response, preserve the
case failure without a review. Every valid draft receives exactly one planned
review, independent of draft quality. Preserve all64in the denominator.

First/longest2case systems pilot tests both stages and capacity. Reuse its exact
first candidates in the full64; no pilot weights or semantic judgments. Record
actual child and whole-tool completion, then independent CPU raw response and
source-binding replay. Freeze source/plan/tests/auditor before generation.

This raw generation does not touch canonical semantic firsts and may overlap
GPU training. A later separate64-case semantic qualification must inherit the
latest completed canonical cache without a concurrent branch, use the unchanged
128-token rubric and require61of the SAME64passes on both qualified judges with
all valid. No bulk10000student-training admission before that qualification.
Raw syntax/capacity are not model accuracy,95%task evidence or speed evidence.
