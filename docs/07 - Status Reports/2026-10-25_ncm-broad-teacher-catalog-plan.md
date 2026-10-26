# Consolidate complete-summary teacher firsts for broader NCM training

Exploratory preparation `ncm-broad-teacher-catalog-v1`. The preceding goal turn
made progress: completed codec diagnostics isolated imperfect reconstruction
and a harmful source-attention interaction in the latest adapted decoder.
The task endpoint remains at least 95% source-disjoint complete-summary accuracy.

Consolidate the three completed teacher inventories: 1,117, 188 and 144 first
candidates. Their 1,449 total observations include 1,376 schema-valid fitting
summaries and 73 preserved failures before further admission checks. These are
complete-summary teacher drafts with a review, not the rejected 4,949 atomic
labels. Do not reroll failures or make new teacher/judge requests.

Replay the first-valid raw draft/review responses, full source bindings,
unchanged target text, 128 stored-token limit, native CLS framing and all
special/unknown-token exclusions. Bind every candidate to the original
10,000-source reservation and cached full FP32 Llama features. Preserve all
exclusions with reasons, including malformed targets and any contamination.

Apply the existing protected-component metadata guard for all 512 reserved
components, including the 320 sealed final components, without opening or
hashing final source text. Apply the existing exact/near-duplicate text rule
against the 128 open DEV and 64 reserved component sources. Exclude exact
source duplicates deterministically. All 64 Sol component summaries remain
reserved; the recent 2019/2021 failure case and its variants are never TRAIN.

Teacher recipe qualification is inherited from the completed 64-case semantic
controls (63/64 joint for the original reviewer; 62/64 for Sol). This does not
make every new target independent semantic gold. Retain target provenance and
that limitation explicitly. No target or stored source may be rewritten.

Use CPU-only locked Pixi `ncm` for catalog preparation and metadata validation.
Then replay the selected feature payloads through the existing lossless disk
loader with a 256 MiB cache. No model or optimizer runs during preparation.
Any training will use a separate fixed protocol, full lifecycle pilot and
unchanged source-only semantic evaluation; this preparation is not accuracy.

Artifact roots are `20261025_broad_teacher_catalog_v1` and
`20261025_broad_teacher_feature_audit_v1` under
`data/experiments/ncm_paper_regime/`. Completed prior artifacts remain immutable.
