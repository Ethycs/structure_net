# TinyLLM TTNO cut-localization and parity-control result

**Status:** COMPLETED
**Hypothesis:** `tinyllm-ttno-cut-localization-parity-v1`
**Classification:** `intrinsic_operator_rank_growth`
**Campaign artifact:** `data/experiments/tinyllm_ttno_cut_localization_parity/20260830_registered/campaign_results.json`
**Campaign SHA-256:** `d9585b4d4d833e632052d2277ecdd11eb2949899aa1073e00d0f006c234f04ef`

## Outcome

A5 passed its full validity contract: all five frozen evaluation seeds completed,
all 120 layer/head/prefix cells reproduced the stored A4 ranks, and every
transformed paired tensor round-tripped exactly.

The 128-to-256 rank cliff persisted as an operator effect under the frozen
diagnostic:

| Endpoint | Result |
| --- | ---: |
| median `r128` | 32 |
| median `r256` | 90 |
| median cliff | 58 |
| median zero-pad rank | 32 |
| median duplicated-sector rank | 32 |
| median best alternate-topology rank | 90 |
| median maximum parity-artifact fraction | 0.00 |
| cells with at least 25% topology reduction | 0/120 |
| median new-root cliff fraction | 1.00 |

Neither introduction of the eighth bit by zero padding nor by exact sector
duplication raised the rank above the original 128-token value. Consequently,
the new tensor shape alone does not explain the natural operator's increase.

Gray ordering was the lowest-rank alternative in 97 cells and LSB ordering in
23, but the improvement was too small: median reduction was zero and the 90th
percentile was about 1.2%. The preregistered 25% topology-sensitivity gate was
never reached.

## Interpretation

Within the A5 fixed controls, the A4 cliff is intrinsic to the changed natural
attention operator rather than a parity-only artifact or an avoidable feature
of the three alternative paired-bit orders. The critical A7 question therefore
remains whether chronological far-field interactions admit a different shared,
nested H2 representation; A5 supplies no rescue for the strict paired-bit TTNO.

This remains evidence from one pretrained checkpoint. Evaluation seeds select
input prefixes and are not independently trained models.
