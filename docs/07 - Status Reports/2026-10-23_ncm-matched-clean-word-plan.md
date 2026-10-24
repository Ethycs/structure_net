# Restore correct-history word fitting before matched NCM forecasting

Date: 2026-10-23. Prospective component study
ncm-matched-population-clean-word-v1. The overall goal remains at least95%
complete-summary semantic task accuracy on the existing source-disjoint
evaluation, with the original Llama body and codec frozen.

The previous 32,768-update model scored zero of64 short and zero of64 long
semantic passes, with128 valid Terra judgments. Its registered campaign is
20261023_reviewed_draft_history_results_terminal_v2, SHA256
9560dd529786efa588cd9416a6112481b51694831a7c1ddab0bfc0fb5127abd4.
The subsequent32TRAIN diagnostic found nearly perfect active plans on the
original16 probes but only81.41% correct-history word-token fitting. The
additional16 probes had1.64% active-code agreement; the forecaster had trained
on252cases while the word branch trained on690. Preserve both findings.

This first repair establishes word fitting with correct histories on the same
complete690TRAIN population needed by the future forecaster. No new targets,
source features, concept IDs, tokenization, semantic judges or evaluation cases
are introduced. The64 extra targeted accepted labels remain outside this study.
Forecaster training on all690 and later predicted-history word adaptation are
required future stages before any new end-to-end semantic claim.

Initialize the word branch from the exact registered checkpoint SHA256
e608f416e2915204a56d5f892d27087dc5cc2ff0e7b0b4cc24dbc44fe272acb3.
These are learned weights, with a fresh optimizer. Keep the original codec,
unused word-branch roots/coarse heads and current concept forecaster frozen.
The forecaster checkpoint remains
27f835d1391d7e14439716a0514a86d5b301bee79341199ab9dcfe88748e454d.
Train only the existing word-branch source norm/projection, transformer, word
projection and decoder. Preserve the NCM architecture and original dropout.

Use32,768updates, batch1, no accumulation, AdamW betas0.9/0.95, weight decay0.01,
gradient clipping1, 256-step warmup to0.0003 then cosine decay to0.00003.
Model/dropout seed24142 and independent sampler seed25142 are fixed. Use FP32
master parameters/optimizer/loss, BF16 forward and disabled matmul/cuDNN TF32.
The single loss is complete nonpadding word cross entropy conditioned on all
and only the608 correct earlier-level TRAIN codes. No finest target code or
target word enters the prediction inputs. Ignore padding50257; include EOS50256.
Use complete native source frames, with3328 as the maximum-frame pilot probe.

First run two CUDA optimizer updates at the declared maximum learning rate on
the first and longest TRAIN source, padding only the second source to3328.
Require finite loss, positive source/word/decoder gradients, correct frozen
gradient routing, memory below8GB, exact optimizer/sampler/CPU/CUDA RNG reload
and identical composed source-only outputs after reload. A fresh worker then
checks the full32-case fit path on this pilot model as a systems test, with no
scientific fitting gate. Only these two actual zero exits admit full training.
Full training starts fresh from the same initial learned weights as the pilot,
uses the fixed warmup/cosine schedule and saves recovery snapshots every4096
updates. Select only update32768.

A fresh worker evaluates the exact same original16 and additional16 TRAIN
identities in the completed diagnostic. Recompute both correct-history and
frozen-forecaster-history word outputs; verify every fresh forecast equals its
original source-only TRAIN cache. Report all192 raw output IDs, global-EOS text,
token-weighted fitting and active concept counts. Require unchanged model and
codec parameters and allocation below8GB. All32 targets and sources stay intact.

The primary component gate is correct-history token agreement at least199/200
in each16-case group and in their32-case union, using inclusive integer counts.
This is a necessary fitting check and establishes no semantic summary accuracy.
A valid miss is recorded as clean_word_fit_gate_failed and stops admission to
the next training stage. Invalid systems/data/identity checks stop the workflow.
No failed seed, output or target is rerolled or replaced.

Artifact roots are20261023_matched_clean_word_pilot_v1, fit_pilot_v1,
train_v1, fit_v1, workflow_v1 and results_v1 under data/experiments/ncm_paper_regime,
with the common20261023_matched_clean_word_ prefix. Implement and test objective
and gradient routing, finest-label exclusion, inclusive group gates, schedule
and checkpoint contracts. Check actual data and CUDA availability, commit the
source/tests/plan, then launch one GPU0 worker at a time.

The supervisor records actual child waits and terminal receipts. The immutable
audit replays all32768 sampled IDs/frames/actual correct histories/word targets/rates, all690-case
coverage, optimizer steps and sampler state, unused-head equality, raw32-case
fitting and the fixed gate. Hash only immutable producer/numerical receipts,
never mutable parent progress or future receipts, in the canonical aggregate.
Create a measured report and conservative exact registry readback.

Neither a fitted TRAIN decoder nor this component gate admits sealed source
access. Preserve the original128-case development and256short+64long final
protocol: valid64 and at least61 passes per development cohort/judge, then all
final judgments valid with the same cases passing both judges at least244/256
short and61/64 long. No speed claim precedes that measured task quality.
