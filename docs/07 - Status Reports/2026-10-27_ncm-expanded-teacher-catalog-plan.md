# Admit the new reviewed summaries without changing the original catalog

Preparation hypothesis `ncm-expanded-reviewed-catalog-v1` supports the next
matched data-diversity training comparison. It is not an accuracy measurement.
The 1,024-source teacher collection is running; its eight-case pilot completed
with valid, fitting, independently replayed first responses. The existing
student DEV remains 0/128. The original 95% source-disjoint summary goal remains
unchanged.

Keep all 1,376 existing reviewed TRAIN rows exactly unchanged and retain the
original catalog's 73 invalid first observations. Keep the 64 component sources
reserved. Admit new candidates only from the fixed 1,024-source collection,
after its complete terminal observation and independent raw-response audit.
Do not turn a partially collected population into a finished training dataset.

Replay the original source and first draft/review responses for every new
candidate, including invalid firsts. Require exact unchanged target strings,
at most 128 storage tokens, a complete 128-position native frame, and no native
special/unknown tokens in lexical content. Record every exclusion and its
reason. Never repair, truncate, reroll or replace a first target.

Repeat protection of all 512 reserved components by metadata and the exact/
near-duplicate text guard against the 192 open DEV/component sources. Exclude
exact source and source-component duplicates deterministically. Verify the
union's source metadata and original FP32 feature manifests. Preserve the old
feature mappings, add only mappings for newly admitted cases, and independently
reconstruct the complete saved catalog from raw candidates before training.

Run focused CPU contract tests and a separate real eight-new-first catalog
pilot first. The pilot retains all old rows and reads the original features for
all admitted pilot cases plus existing first/longest controls through a 256 MiB
cache. It never authorizes student training. Full admission requires all 1,024
candidate outcomes and at least 900 eligible new pairs, plus independent replay
of every resulting union feature payload. This minimum is a data-expansion
operational gate, not the 95% summary-quality endpoint. A smaller completed
yield is retained but does not qualify this planned comparison.

Write the existing and expanded arm IDs separately. Existing = the original
1,376 rows in their original order; expanded = those exact rows followed by
all admitted new firsts in fixed collection order. Preserve rejected candidates
outside the training rows. Recipe qualification is inherited; no individual
candidate is asserted to be independent semantic gold.

Roots: `20261027_expanded_teacher_catalog_<pilot,prepare,audit>_v1` under the
NCM experiment directory. Entry point:
`experiments.structure_net.ncm_expanded_teacher_catalog`. Use repository-root
locked Pixi `ncm`, CPU-only with two OMP/MKL threads. Freeze source and inputs,
record actual successful receipts before dependent stages, and keep all old
and live producing files unchanged. The main public curriculum continues
unchanged. No new teacher/judge request, final-source payload access or model
training occurs in this preparation. A future final-test source audit must
cover every newly admitted training source.
