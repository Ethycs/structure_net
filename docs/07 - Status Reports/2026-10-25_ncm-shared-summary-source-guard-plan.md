# Check full text against open evaluation before shared-corpus training

2026-10-25. Exploratory data audit
`ncm-shared-summary-open-source-guard-v1`. The completed inventory found
73,836 candidate TRAIN pairs among the original204,967, after current length,
native-representability and protected-component metadata guards. No quality
score has been measured on these labels, and no new model has trained on them.

Prepared but unlaunched: the user's subsequent PBX pickup delivered a broader
archive handoff beyond the500 LongMemEval stores. Prioritize that non-LongMemEval
corpus for new training, excluding all500 core stores and related source
families. This legacy-corpus audit is auxiliary and does not authorize new
LongMemEval training. Historical LongMemEval training exposure remains reported;
the old corpus cannot be relabeled an untouched benchmark. The broader archive
requires its own verified publication, reader and data-admission protocol.

Require actual inventory exit0 and the unchanged full corpus, candidate
metadata and producer hashes. Replay EVERY candidate against its original
physical JSONL line: IDs, source/summary hashes, original TRAIN split, stored
lengths and fresh pinned native target tokenization must match. Repeat the
reserved component metadata guards. Never read or hash the original old test
JSONLs or the current sealed final source files.

For every candidate compare full normalized source text to all128 open current
DEV sources and all64 already-open teacher/codec component sources. Use the
existing NFKC/casefold Unicode word normalization. Flag exact normalized
equality or at least32 shared distinct word5grams with at least80% containment
of the smaller gram set. An inverted index may skip pairs with no shared
grams; its decisions must match this fixed rule. Exclude overlaps from the
new training export while preserving the original corpus and every conflict's
IDs/counts. Do not alter any prior model or judgment result.

Deduplicate remaining exact source hashes, retaining the first pair in the
unchanged original TRAIN order. This rule does not use target quality or model
outputs. Export metadata/line numbers only. Count all reviewed, conflicting,
duplicate and retained pairs, groups and source sessions. Keep target text
unchanged. New training still needs its own fixed method and checkpoint plan.

These checks establish metadata disjointness and full-text disjointness from
the already-open cohorts under the stated rule. Full-text checks against the
sealed final cohort remain deferred to the existing final evaluation gate;
do not reconstruct its content from public files. Saved reference quality and
model task accuracy remain unmeasured. No new judges or model forwards run.

CPU only, max RSS8GB. Fresh root:
`20261025_shared_summary_open_source_guard_v1` under
`data/experiments/ncm_paper_regime/`. Preserve producer, candidates and all
conflict/exclusion counts. This is not a claim of 95% accuracy or a speedup.
