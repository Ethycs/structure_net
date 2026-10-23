# Source-grounded readout feasibility before a new NCM head

Date: 2026-10-23. EXPLORATORY `ncm-source-grounded-readout-feasibility-v1`.
The current free-text NCM attachment fails0/64 development summaries under
Terra. Correct-code novel reconstruction is92.6081% for its adapted codec.
Natural-only decoder training failed; vocabulary-balanced training also failed
its codec gate at97.7685% novel tokens and9/20 exact texts. Dictionary inversion
failed. Stop frozen-XSum-decoder data variants. None establishes a95% summarizer.

Next question: can a source-supported copying readout meet the **same whole
summary criterion**, before training a new NCM planning head? The user's task
requires high-quality summaries, not exact reference wording. Exact source
copying avoids re-encoding rare names through the failing learned text codec,
but can still fail salience, attribution, coherence or the128-token limit. Test
those failures explicitly. This is a representation/readout fork, not a
replication of the paper's learned text-codec architecture.

Use the unchanged64 development sources and preserved final NCM/parent drafts.
Use the separate20 teacher-pilot source/reference pairs only for a target-aware
projection diagnostic. All sources are already verified public. Final256 task
test stays sealed. Sources and denominators remain fixed; no quality exclusions.
No NCM training or final-test admission occurs in this feasibility study.

Generate candidates as one to three contiguous complete source sentences, each
rendered as one paragraph by normalizing whitespace and limited to128 cl100k
tokens. Preserve their exact source character spans and raw substrings; use
pinned pySBD with clean=false and character spans. Never invent or repair words.
Deduplicate rendered candidates. If none is eligible, record an empty failed
output rather than truncate a sentence, remove the case or ask an LLM to rewrite.

Use the cached public BAAI/bge-m3 checkpoint at revision
`5617a9f61b028005a4858fdac845db406aefb181`, after verifying official metadata
and file hashes. It is an embedding model, not a factuality judge. Dense vectors
are CLS last hidden states, L2 normalized, as in the official implementation.
Load cached weights with weights_only=true and strict architecture matching;
do not disable Transformers' pickle-loading security guard. FP16 on GPU2, SDPA,
batch8, no source truncation. Install sentence-splitting dependencies only in a
fresh isolated directory, preserving every existing training environment.

Fixed non-oracle conditions on the64 sources:
1. Lead: earliest complete eligible source sentence.
2. Source centroid: candidate cosine similarity to the normalized mean of
   embeddings of the source's individual complete sentences.
3. NCM query: candidate cosine similarity to its preserved NCM draft embedding.
4. Parent query: candidate cosine similarity to its preserved full-Llama draft.
For the20 separate teacher-pilot pairs, score candidates using the reference
summary as query; explicitly label this target-aware and unusable at inference.
Cosine ties prefer fewer output tokens, then earlier source start/end. Report
all conditions; no query instructions, score-mixture or candidate-policy search.

Require CPU span/length/identity/selection tests, a real four-source embedding
pilot at intended batch8, finite vectors, fixed source/model states and exact
prediction/embedding reload before main selection. Test the intended judge
concurrency in a fresh four-case pilot after the teacher worker is terminal;
never add primary judging to the previously failed teacher/primary concurrency.
Stop on OOM, source/hash drift, invalid model loading or pipeline-control failure.

Grade every complete candidate output under the unchanged qualified Terra
rubric, first-valid judgment preserved, at most3 transport/schema attempts,
128-token limit. Failed/empty/invalid cases remain in the denominator. Report
whole-summary faithful/salient/readable pass counts and Wilson intervals. Only
conditions at least61/64 on Terra proceed to independent qualified Sonnet
grading with the same61/64 gate. Teacher-projection20 is a separate diagnostic
requiring19/20 on Terra before using this label projection to teach a new head.
No token reconstruction or exact-reference score can pass this task gate.

Even a passing BGE or lead baseline is **not NCM success**. This study qualifies
or rejects a copying representation for a subsequent actual frozen-body NCM
coarse-to-fine pointer/planning head. That later head must generate its own
selection without teacher references or parent drafts, be trained/evaluated on
source-disjoint data, and be compared against the non-NCM selectors. Mechanism
ablations must establish NCM's contribution; do not credit it for BGE doing all
the work. The final head still needs the original actual95% summary endpoint
and full application-range evaluation before any speed claim.

If this source-copying family fails feasibility, preserve the failures and use
an explicitly declared abstractive final-token/pointer readout study. Do not
quietly weaken summary quality, restrict to easier source types, relabel the
baseline as NCM or return to closed decoder-data sweeps.

Fresh artifact family: `20261023_source_readout_*`. Freeze this plan, source,
package/model revisions, complete data/candidate/embedding hashes and commands
before each stage; preserve raw candidates, selections, first-valid judgments,
strict aggregates and conservative exact/semantic/JSON registry read-back.
