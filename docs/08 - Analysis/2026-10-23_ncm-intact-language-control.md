# Intact language passes; summary quality remains below target

Date: 2026-10-23. `ncm-intact-language-control-v1`, **intact_language_pass_summary_quality_failed**.
The untouched Llama-3.2-1B-Instruct scored **50/64 = 78.1%**
whole-summary accuracy on the fixed development sources under qualified Terra.
All 64/64 judgments were valid. Component passes: faithful 53/64,
salient 61/64, readable 64/64. The necessary
61/64 readable language gate passes; the complete-summary gate fails. This model
has no NCM attached and does not establish the 95% task objective.

The official cached revision and all six files were verified against pinned
Hugging Face LFS/Git identities. Its native vocabulary and all64 prompt IDs match
the 3B parent. All16 pretrained blocks and the tied native word head remained
unchanged. Greedy output repetition, serialization and frozen-weight checks
passed. The measured peak allocation was 2,634,609,152 bytes on GPU2 in FP16.
This is memory evidence, not a matched speed comparison with the BF16/GPU0
five-block readout, which scored 0/64 and failed every readability judgment.

The fixed64-source outputs were frozen before judging. Four first pilot
judgments (including negatives) were reused and60 new judgments completed the
endpoint. No semantic retries, clipping, output repair or post-hoc exclusions.
Sonnet was not run after the failed secondary summary-quality gate. The final256
remain sealed. Current short-source evaluation does not establish the full
2,048-token fragment / 6,300-token batch operating range.

The contrast supports retaining a usable pretrained language path. It does not
isolate pruning, bridge initialization, model size or training-data effects.
Faithfulness is still the main failure: 11 cases fail that
component. A new NCM training protocol must qualify its labels, preserve native
initialization and evaluate actual source-disjoint summaries. Training fit and
an intact AR baseline cannot substitute for that result.

The grading worker's terminal output was unavailable after context rollover.
Its result is complete, the exec session expired and an exact process check
found no live worker; its unknown exit code is recorded as null. Other worker
exit codes were directly observed. All source fingerprints and raw judgments
were replayed before finalization.

Aggregate `data/experiments/ncm_paper_regime/20261023_intact_decoder_results_v1/campaign_results.json`, SHA256 `5f45a58144ae9d51e702294701307c41d15b682a6f182308fb49c08160be58c9`. Confirmed=false;
no NCM-success, final-test or speed claim. Reproduce the read-only evidence
validation with `python -m experiments.neural_architecture_lab.finalize_ncm_intact_language`;
fresh finalization is intentionally refused once this aggregate exists.
