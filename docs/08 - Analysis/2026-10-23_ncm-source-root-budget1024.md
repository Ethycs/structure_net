# Longer-budget source concept capacity check

Date: 2026-10-23. Hypothesis `ncm-source-root-capacity-budget1024-v1`.
Classification **source_root_TRAIN_capacity_passed_full_hierarchy_unproven**. Aggregate `data/experiments/ncm_paper_regime/20261023_word_final_root_capacity_budget1024_results_v1/campaign_results.json`,
SHA256 `b8ed8da821a29a11eb632d648588e330db9e9952eee7ed585fdf9fa7a847aa3f`.

These measurements concern fitting the first concept level on 16 training
cases. They are not summary accuracy. The latest complete summarizer remains
at 0/64 short and 0/64 longer development passes. The 95% task goal is unmet.

| State | Active root labels | All root labels | Active labels with shuffled sources |
| --- | --- | --- | --- |
| Initial | 6/176 | 78/256 | 6/176 |
| Previous 256 | 121/176 | 200/256 | 6/176 |
| Fresh 1,024 | 173/176 | 253/256 | 2/176 |

The fixed training-capacity gate **passed**. It
requires at least 95% active and all-root recovery, with an active recovery
drop of at least 50 percentage points under cyclic source shuffling. A pass
does not qualify later concepts, text generation or generalization.

This run started fresh from the same original word-final checkpoint as the
failed 256-update control. Its initial predictions and parameter hash match
exactly. The data, objective, precision, optimizer, dropout and gradient scope
stayed fixed. The budget increased to 1,024 updates and the cosine schedule
stretched after the same 16-update warmup; this changes both budget and schedule.
Only final 1,024 was selected. The existing maximum-frame two-update CUDA pilot
was reused because operations, state shapes and initial warmup rates were exact.
No pilot or 256-update weights were used as initialization.

All 1,024 sampled source identities, full frames, rates, finite loss records
and final root counts replayed. Frozen word modules, other heads and codec
remained exact, as did model/optimizer/RNG/source-only checkpoint reload.
Original Llama weights and MLPs were frozen; the parent was not loaded for this
training. Peak allocation was 3,517,923,328 bytes.

The failed 256-update control and all 930 canonical task judgments remain
unchanged. Both final task cohorts remain closed. No compression or speedup
has been established. Further hierarchy training and source-disjoint complete
summary evaluation require their own fixed protocol.
