# Diagnose the failed reviewed-draft word model on TRAIN

Date: 2026-10-23. Prospective exploratory mechanism study after the fixed
32,768-update/690-case model received zero of 64 short and zero of 64 long
complete-summary passes, with all 128 Terra judgments valid. The failure is
registered under `ncm-reviewed-draft-history-word-v1`; its canonical aggregate
SHA256 is `9560dd529786efa588cd9416a6112481b51694831a7c1ddab0bfc0fb5127abd4`.
No further model training is selected from this result before this diagnosis.

Keep the current model, forecaster, original codec, source features, labels,
tokenizer, precision and code entirely fixed. Select the exact original sixteen
TRAIN identities in `20261023_word_final_generation_gap_v1/selection.json`,
requiring their complete word targets to match the earlier measurement exactly.
Add four additional TRAIN examples in each source-length band 0–511, 512–1023,
1024–1535 and 1536–2048 stored tokens. Exclude all original 252 cases from these
additional selections. Sort each band by SHA256 of
`ncm-reviewed-word-train-mechanism-v1:<case id>` and take its first four.
The combined set has 32 unique cases and contains no development or final cases.

For each case compare the current word branch with its frozen forecaster's own
eight coarse levels against the same branch with correct TRAIN coarse levels.
No finest target codes or target word IDs enter word prediction. Correct-history
conditioning is explicitly an oracle component diagnostic and cannot be used
in production evaluation. Verify every fresh source-only forecast exactly
matches its original TRAIN cache and each complete target decodes exactly.
Save all 192 raw word IDs, text through one global EOS, active-position token
agreement, cross entropy, own-versus-correct output agreement and per-level
active concept agreement. Preserve complete source frames and all cases.

Report token-weighted results separately for the old sixteen, additional
sixteen and all thirty-two. Token agreement and exact target text counts test
component fitting; neither is semantic summary accuracy. No new judge calls,
training updates, output repair, parameter update or held-out source access
occurs. Semantic development accuracy remains the measured zero percent.

First run the first selected case and the longest selected source as a CUDA
pilot, repeating them exactly. Then a fresh worker measures all 32, reproducing
and retaining the pilot measurements exactly. FP32 parameters/loss and BF16
forward computation remain unchanged; TF32 is disabled. All parameters and the
codec must remain byte-identical; peak allocation must remain below 8 GB.
Failure of data identity, cache parity, finiteness, reload parity or memory
stops the full study. Preserve the failure; do not reroll selections.

Use fresh roots `20261023_reviewed_word_train_diagnostic_pilot_v1`,
`20261023_reviewed_word_train_diagnostic_v1`, and supervising
`20261023_reviewed_word_train_diagnostic_workflow_v1`. Hypothesis/diagnostic ID
is `ncm-reviewed-word-train-mechanism-v1`. Test selection, masked counting and
aggregation, check the actual selection before launch, then commit source,
tests and this plan. The supervisor records actual child waits and zero exits.
The diagnostic guides the next prospective training choice without rescuing
the failed model, opening sealed sources or claiming speed or 95% task quality.
