# Semantic quality of oracle codec reconstruction

Date: 2026-10-23. Hypothesis `ncm-oracle-codec-semantic-quality-recovery-v1`.
Classification **oracle_semantic_codec_quality_failed**. Aggregate `data/experiments/ncm_paper_regime/20261023_oracle_codec_semantic_recovery_results_v1/campaign_results.json`,
SHA256 `53848961898cbada4b38ea07c33334199798e880e8ae7677e437947f9c390462`.

| Arm | Judge role | Complete-summary passes | Valid | Faithful | Salient | Readable | Over 128 tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uncompressed | secondary | 63/64 | 64/64 | 63 | 64 | 64 | 0 |
| uncompressed | primary | 64/64 | 64/64 | 64 | 64 | 64 | 0 |
| original | secondary | 8/64 | 64/64 | 9 | 61 | 28 | 0 |
| trained | secondary | 6/64 | 64/64 | 7 | 59 | 29 | 0 |

These are semantic reconstruction pass rates with correct target-derived codes,
**not source-to-summary model accuracy**. The uncompressed target is a control.
Source-only conditional generation remains unproven. The model-development
teacher result remains 56/64 (87.5%), and the earlier NCM hybrid 41/64. Neither
final cohort was opened. No speedup was measured.

The sample contains 64 of 252 codec-excluded complete summaries, selected by
case-ID hash without output or judgment filtering. Sources span the training
label pool, including longer passages. Both codecs reconstruct complete targets
through four 48-token blocks. Each anonymous single-candidate packet contains
the full SHA-verified public source and one summary. Both qualified judges use
the same 128-token storage rubric. Terra is secondary; Sonnet is primary and
is admitted only when both the arm and the uncompressed control reach 61/64
valid Terra passes. Qualification requires the same threshold on both judges.
Qualified codecs: none.

On paired Terra cases, changing released to adapted reconstruction fixed
2 cases and regressed
4; both passed on
4 and both failed on
54.
Invalid pairs: 0.

All first valid raw judgments, including negatives, replayed exactly. The
12 arm-case systems pilot contributes its unchanged first judgments. All 320
prior first judgments remain exact. Transport/schema retries alone are allowed,
with a maximum of three attempts. The failed 99.5% literal-reconstruction
gates remain failed; this prospective semantic audit does not amend them.
Raw failures and claim verdicts are retained in `semantic_failures.json`.


## Recovery of the incomplete cache

The preceding semantic pilot/full run was stopped at observed exit 143 when
identical control packets were found in two omitted label-preparation caches.
Its incomplete full audit is invalid. Later duplicate control judgments were
preserved as evidence and excluded regardless of outcome. The recovery retains
all 606 canonical earlier judgments plus
27 new first judgments saved by the stopped
run. It ignores 50 later duplicates. For
51 identities with queued directories
but no persisted response, at most two further transport attempts were allowed.
No earlier valid reply was rerolled, and the original 320 model-output entries
remain exact. The cases, external packets, rubric and gates did not change.
See the [prospective recovery amendment](../07%20-%20Status%20Reports/2026-10-23_ncm-oracle-codec-semantic-cache-recovery.md).
