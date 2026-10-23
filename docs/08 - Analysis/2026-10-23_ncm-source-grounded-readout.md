# Source-grounded readout: whole-summary feasibility

Date: 2026-10-23. `ncm-source-grounded-readout-feasibility-v1` is complete,
classified **source_copy_readout_task_gate_failed**. Jointly passing non-oracle conditions:
`[]`. No new NCM head was trained; the final task test stays sealed and
the95% final NCM objective is not established by this comparison.

These are **whole-summary** pass counts under the unchanged faithful, salient,
readable, nonempty and128-cl100k-token criterion, not codec reconstruction or
exact-reference accuracy. Secondary is Terra; primary is Sonnet.

| Readout | Judge role | Complete summaries passing | Wilson95% | Invalid |
| --- | --- | --- | --- | --- |
|lead|secondary|8/64=12.5000%|invalid judgments; bounds retained|2|
|lead|primary|not run: ordered gate|—|—|
|source_centroid|secondary|26/64=40.6250%|29.46–52.85%|0|
|source_centroid|primary|not run: ordered gate|—|—|
|ncm_query|secondary|37/64=57.8125%|45.61–69.13%|0|
|ncm_query|primary|not run: ordered gate|—|—|
|parent_query|secondary|42/64=65.6250%|53.40–76.08%|0|
|parent_query|primary|not run: ordered gate|—|—|
|reference_query|secondary|11/20=55.0000%|34.21–74.18%|0|
|reference_query|primary|not run: ordered gate|—|—|

Lead uses the earliest complete eligible sentence. Source centroid uses mean
normalized BGE sentence vectors. NCM/parent query conditions use the preserved
draft as a semantic query; they copy source words, never repair those drafts.
Reference-query20 uses the separate teacher summary and is target-aware,
unusable at inference. Its19/20 projection gate passed=`False`.

Every condition uses the same64 original source-disjoint development cases;
reference20 is a separate source-disjoint diagnostic. Candidates are one to
three contiguous source sentences, normalized only for whitespace and capped
at128tokens. No source/case exclusion, truncation, rewritten word or LLM fallback.
All selected raw character spans and rendered text identities were verified.
This output budget is the application's128-token cap; the old free-text NCM
used a48-GPT2-token canvas, so this is not a matched generation-speed comparison.

Exact source-summary/rubric packets share their first judgment within a judge
role. The amendment was fixed before main grading; pilot negatives and exhausted
invalids cannot get another semantic draw. All four pilot judgments replay
exactly, cache requests/role/model/payloads replay, and denominators stay64.
There are261 unique judgments including the pilot, with each condition's
cache hits and raw origins retained. Role-separated caches preserve independent
Sonnet judgment. Primary is run only after61/64 on the valid secondary endpoint.
No positive/negative cherry-picking or alternate checkpoint selection occurred.

On the same secondary cases, centroid versus NCM-query paired counts are
`{'both_pass': 20, 'centroid_only': 6, 'ncm_only': 17, 'both_fail_or_invalid': 21}`. This identifies whether the NCM draft helps or hurts this selector;
it is not proof of a superior trained NCM mechanism. A BGE/lead baseline passing
would not count as NCM success. The original free-text attachment's prior0/64
is preserved; this experiment changes the readout and has no new head training.

BGE-M3 public revision is5617a9f61b028005a4858fdac845db406aefb181, verified
against official metadata and LFS hashes. Strict safe state loading, FP16,
SDPA, batch8 on GPU2, normalized CLS readout; no model weights changed. Real
worst-input memory pilot, finite vectors and exact embedding/selection reload
passed. pySBD0.3.4 supplies exact sentence character spans. Embedding preparation
and cached old drafts are offline experimental work, not inference speed evidence.

The independent broader-label audit remains113/128 and fails its training-data
quality gate. Therefore this report admits no scaled student run. A subsequent
NCM pointer/concept head needs a newly qualified training-label representation,
its own source-disjoint complete-summary evaluation and mechanism ablations.
If this copying family fails, use a new declared abstractive final-token/pointer
study; do not weaken the rubric or return to closed frozen-decoder data sweeps.

Aggregate: `data/experiments/ncm_paper_regime/20261023_source_readout_results_v1/campaign_results.json`; SHA256 `c739039f772cd807ad778e514177e206324cf2d3847793482fcd14fed690725e`. All fixed outputs,
source spans, embeddings, first raw responses, exact reuse origins, producers
and source snapshots remain append-only. Registry confirmed=false,
NCM_success_established=false, final_test_opened=false and speed_tested=false.
