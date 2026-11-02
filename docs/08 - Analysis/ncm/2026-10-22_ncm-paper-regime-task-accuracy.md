# Actual summary accuracy after the NCM paper-style training run

Date:2026-10-22. Hypothesis `ncm-paper-noisy-pairs-v1`; exploratory, one model seed.
Classification **`development_gate_failed`**. These are complete-summary judgments,
not token accuracy, code prediction, ROUGE or training loss.

|Model|Independent judge|Passes|Accuracy|Invalid|
|---|---|---|---|---|
|native|claude_code/claude-sonnet-4-6|0/64|0.00%|0|
|native|codex_sdk/gpt-5.6-terra|0/64|0.00%|0|
|hybrid|claude_code/claude-sonnet-4-6|0/64|0.00%|0|
|hybrid|codex_sdk/gpt-5.6-terra|0/64|0.00%|0|

All64fixed source-disjoint development passages were evaluated under each
judge. Passing requires source-grounded faithfulness, salience, readability,
nonempty text and <=128cl100ktokens. The61/64admission threshold was fixed before
grading. Each zero result has Wilson95%interval0–5.66%, excluding judge error.
Neither arm qualifies for the sealed256-case final test. No speedup is measured.
The user's95% goal remains unmet by these models.

The native arm used NCM source embeddings; the hybrid used frozen full-Llama
hidden states projected through a new bridge. Both shared the same512-update
unpaired checkpoint and received2,048conditional updates, batch8xaccumulation2.
Hybrid additionally received512bridge-only updates. Both used the exact newly
trained codec and identical conditional minibatch/augmentation schedules. All
1,999labels were used with disclosed label noise: the independent audit passed
30/32. The diagnostic was explicitly declared to test noisy-pair learning and
did not retroactively rescue the failed quality-gated earlier studies.

The full codec encoder, codebooks and projections were frozen during generator
training; original Llama layers, including MLPs, were never trained. Targets
were reencoded from the accepted codec, and tied-head codebook buffers matched
that codec. Loss fell from4.24to2.03(native) and4.21to2.23(hybrid), but the output
still invented names and scrambled wording. Training loss did not predict
useful task quality. The released native baseline on these same64sources had
also scored0/64under the qualified secondary judge.

Both original judge endpoints initially failed transport/infrastructure checks.
After recovery, each passed a fresh4-concurrent-request systems pilot before
full grading. Systems decisions were not pooled into task evaluation. The
original48-control qualification, source-grounded rubric and conservative parser
were retained. Each request contained one candidate; all raw payloads/replies
and valid negative judgments remain preserved. No semantic failure was retried
to obtain a pass. Controls are source-reviewed by Codex, not human gold.

Separate follow-ups have demonstrated32-case training competence after a
coarse-code curriculum and improved full-cutoff decoder reconstruction. Those
results are in the [competence report](2026-10-22_ncm-training-competence.md),
and cannot reclassify this failed full-data campaign. A separate full-data
curriculum run is underway and must earn its own held-out result.

This is an adaptation of the paper's tokenizer-first, frozen-codec,
unpaired-then-conditional staging, at a much smaller training budget. It does
not establish that the published NCM architecture cannot work. The prototype
covers only512GPT2source tokens and48target tokens, rather than the full
application input/batch range. [NCM paper](https://arxiv.org/html/2610.14162v1).

Aggregate:`data/experiments/ncm_paper_regime/20261022_noisy_quality_v1/campaign_results.json`;
SHA256`5ee0bb574163f38f6153c408d65627655c19cb3f59a783c14604881eec5416b9`.
Reproduce with `ncm_noisy_pairs_campaign` and `ncm_recovered_grade_campaign`
under `experiments.structure_net`. Checkpoints, optimizer/RNG states, source
snapshots and64-case raw scores remain in the declared roots. Registry keeps
confirmed=false, underpowered=true and final_test_opened=false.
