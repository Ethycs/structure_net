# TinyLLM HSS shared-basis and nesting result

**Status:** COMPLETED
**Hypothesis:** `tinyllm-hss-shared-basis-nesting-v1`
**Classification:** `shared_and_nested_hierarchy_supported`
**Campaign artifact:** `data/experiments/tinyllm_hss_shared_basis_nesting/20260830_registered/campaign_results.json`
**Campaign SHA-256:** `6f42a59b3a723eb4b80742e8fab8278be9b21d1db35131cc4ea81bd702e03c01`

## Outcome

A6 completed all five frozen evaluation seeds on GPU 0. All 120 cells
reproduced the A4 chronological HSS maxima at every length, every complement
peer concatenation was exact, and the exact rank-one positive control passed.

At the primary 256-token length:

| Gate | Result | Threshold | Pass |
| --- | ---: | ---: | :---: |
| median of cell-median sharing inflation | 1.0 | `<= 2.0` | yes |
| p90 of cell-p90 sharing inflation | 1.0 | `<= 3.0` | yes |
| spectrally stable cuts | 105,886 / 121,920 (86.85%) | `>= 25%` | yes |
| median stable-cut nesting defect | 0.0 | `<= 0.10` | yes |
| p90 stable-cut nesting defect | `5.92e-16` | `<= 0.25` | yes |
| median augmented/shared rank ratio | 1.0 | `<= 1.5` | yes |
| p90 augmented/shared rank ratio | 1.0 | `<= 2.0` | yes |

The chronological shared-boundary rank medians were 10, 17, 24, and 33 at
lengths 32, 64, 128, and 256. The length-64 value is the chronological-tree
median; A4's headline used the best of chronological and Q/K-PCA per cell.

## Interpretation

For the tested checkpoint, the individually low-rank dyadic interactions do
not require incompatible partner-specific bases: concatenating all peer
interactions did not inflate the typical required rank. On stable SVD cutoffs,
the restricted parent directions were already contained in the child spaces to
numerical precision and did not increase their numerical rank.

This supplies the necessary shared/nested compatibility that A7 needs, but it
does not itself construct one simultaneous H2 approximation or establish its
kernel-normalized rowwise error. A7 remains the decisive representation test.

The raw campaign retains node and parent-child records for all 480 natural
operator lengths, plus the declared analytic and permutation controls. The
evaluation seeds are input-prefix replicates from one checkpoint, not model
replicates.
