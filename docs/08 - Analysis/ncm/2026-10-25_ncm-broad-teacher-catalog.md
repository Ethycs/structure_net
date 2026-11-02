# Broader complete-summary supervision is ready for a fixed training experiment

Completed preparation `ncm-broad-teacher-catalog-v1`: **1,376 complete-summary
teacher firsts from 1,376 source components**, with all cached source features
verified. No student training or new accuracy measurement ran in this preparation.
The latest complete-summary DEV result remains 0/128; the 95% goal is unmet.

The [prospective preparation plan](../07%20-%20Status%20Reports/2026-10-25_ncm-broad-teacher-catalog-plan.md)
consolidated three completed inventories: 1,117 + 188 + 144 observations. All
1,449 firsts were accounted for. The 73 original invalid attempts remain
preserved and excluded; none was retried or replaced. All 1,376 valid fitting
firsts passed the additional admission checks. The rejected 4,949 atomic-label
set was not used.

| Coverage | Verified quantity |
| --- | ---: |
| Complete-summary TRAIN pairs | 1,376 |
| Source components / sessions | 1,376 / 1,376 |
| Native lexical target tokens | 94,158 |
| Stored target tokens | 90,394 |
| Source application tokens | 1,080,931 |
| Frozen Llama feature positions | 1,158,389 |
| Original FP32 feature bytes replayed | 14,234,284,032 |
| Peak resident feature cache | 268,431,360 bytes (within 256 MiB) |

Sources use the existing 192–2,048-token fragment range. Targets are at most
128 storage tokens and span 46–121 native lexical tokens. Full text, reviewed
target strings, native frames and source provenance were preserved. No unknown
native tokens, clipping, or truncation was admitted. The original 732-example
engineering set covered eight components; this catalog supplies much broader
source diversity, though still only 94,158 distinct target-token occurrences.

Metadata protection covered all 512 reserved components, including the 320
sealed final components. The existing text-overlap rule was applied to all
128 open DEV and 64 reserved component sources. It found no additional overlap
among the 1,376 valid candidates. Final source text was neither opened nor
hashed. The reserved date-error example and its 2019/2021 variants are not TRAIN.

Teacher recipe qualification comes from earlier complete 64-case semantic
controls, with 63/64 joint passes for the original reviewer and 62/64 for Sol.
These controls qualify the labeling recipes; the 1,376 targets are not
independently established semantic gold. No target was selected by a new
semantic judgment, and no external teacher or judge request was made here.

Raw draft/review replay initially used the pre-amendment JSON parser and failed
before catalog creation on a valid fenced response. The failed source and actual
exit-1 receipt are preserved in `20261025_broad_teacher_catalog_CPU_checks_v1`.
Replay was corrected to use the original producing v3 parser. This changed no
candidate, source, target or acceptance criterion. Six focused tests passed.
The completed catalog process exited 0; a separate CPU process then loaded and
verified all 1,376 feature payloads through the existing lossless FP32 loader.
Metadata replay checked the complete first-candidate denominator, every retained
target/frame, every feature-log binding and all aggregate byte/token counts.

The next quality experiment should use a fixed training protocol and retain
clean-code reconstruction while testing broader source conditioning. Targeted
factual contrasts are motivated by the [literature review](2026-10-25_ncm-adversarial-training-options.md),
but no contrastive or adversarial training result is claimed here. Encoder/VQ
target codes and any generated-code training cache still need to be prepared
under the next experiment's selected checkpoint.

Artifacts under `data/experiments/ncm_paper_regime/`:

- `20261025_broad_teacher_catalog_v1`: all accepted pairs, exclusions, first-response provenance, metadata and manifest.
- `20261025_broad_teacher_feature_audit_v1`: every payload replay and memory measurements.
- `20261025_broad_teacher_catalog_results_v1`: independently recomputed aggregate.

Accepted cases SHA-256:
`cd3d2a8a1fe0d398d1ca951e0df382d05c47cefe4d07892107b23d6e1c7e5914`.
