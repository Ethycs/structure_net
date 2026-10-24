# Train the concept forecaster on the matched TRAIN population

Date: 2026-10-23. Prospective conditional component study
ncm-matched-population-forecaster-v1. The clean-word study is running, and no
new complete-summary outcome exists. The last completed semantic candidate
scored0/64short and0/64long, with128valid Terra judgments. Preserve that failure.

The preceding fixed32TRAIN diagnostic found6,683/6,688 correct active codes
on the original16probes and100/6,080 on the additional16. The old forecaster
trained on252examples while the word branch trained on690. This study repairs
that population mismatch, conditional on successful clean-history word fitting.
It does not change the original95% complete-summary semantic target.

## Admission and immutable data

Before creating any model stage, require the existing clean-word workflow to
complete with six actual zero child exits, its component gate true, and exact
registered campaign-hash/classification readback. Capture its original PID and
Linux start tick while waiting; never restart it after an observation timeout.
Wait for that supervisor to become terminal so GPU workers cannot overlap.
A valid clean-word gate miss records not_admitted_clean_word_fit_gate_failed
with no model stage or tested-model claim. An upstream systems failure stops
and is preserved. No time passage substitutes for successful admission.

Use all and only the same690 current TRAIN examples, exact word/concept targets,
full frozen parent features and original codec. The64 future targeted labels
remain outside this experiment. Select the same32 TRAIN probes as the preceding
diagnostic:16fixed original cases and4additional cases per floor(tokens/512)
band, excluding the original252from additional selection. No held-out example
enters training, prediction or scoring. Both sealed source files stay unopened.

## Training configuration

Initialize from registered learned forecaster checkpoint
27f835d1391d7e14439716a0514a86d5b301bee79341199ab9dcfe88748e454d.
Use a fresh AdamW optimizer. Train the source norm/projection, generator
transformer, learned root queries and all eight coarse heads. Freeze the
initializer's word projection/decoder, finest head, original codec, original
Llama body and original body MLPs.

Use49,152updates, batch1, no accumulation, AdamW betas0.9/0.95, epsilon1e-8,
weight decay0.01, gradient clipping1, warmup256to0.0003 then cosine decay to
0.00003. Model/dropout seed26142, sampler seed27142 and independent corruption
seed28142 are fixed. This budget gives about71case exposures, matching the
order of the earlier16,384/252source-concept regime. FP32 parameters,
optimizer/loss/reconstruction and BF16 forward remain; bothTF32 modes are off.
Preserve the architecture, original dropout and complete native source frames.

Keep the existing eight-level concept objective exactly: root weight4,
other-level weight1, active-block CE plus0.25padded-block CE. The active mask
uses target words only for loss weighting. Earlier coarse history is correct
with probability0.5; otherwise apply the existing uniform corruption with
rates descending0.35to0 across the eight coarse levels. These rates vary by
level, not by training time. Finest codes and target words never enter coarse
prediction. The quantizer interface's unused192finest positions receive constant
zero placeholders instead of target labels; targets themselves remain intact.
Keep all target positions; no clipping, oracle evaluation or repair.

First perform two CUDA optimizer updates at the maximum learning rate on the
first and longest source, padding the second to3328. Force a clean then a
corrupted history, while consuming the declared RNG draw. Require positive
root/source/all-coarse gradients, frozen gradient routing, finite values,
allocation below8GB and exact model/optimizer/sampler/noise/CPU/CUDA RNG/native
reload checks. A fresh32-case fit-path pilot then checks finite source-only
rollouts, exact repeats and native coarse-generation parity on its first and
longest probes. This pilot has no scientific fitting gate.

Only successful actual pilot exits admit the full run, starting fresh from
the same learned initializer. Save recovery snapshots every8192updates and
select only the fixed final update49152. No seed, target or output rerolls.

## Fitting, reporting and next stages

A fresh worker predicts all eight coarse levels from source-only features on
all32fixed TRAIN probes. Compute active code agreement per level, separately
for each16-case group and their32-case union. Require at least99/100 at every
level in every group, using inclusive integer counts. A valid miss is
matched_forecaster_fit_gate_failed; it does not admit later word adaptation.
This is a component-fitting check, never semantic summary accuracy.

Use fresh20261023_matched_forecaster_ roots with suffixes pilot_v1,
fit_pilot_v1, train_v1, fit_v1, workflow_v1 and results_v1 under
data/experiments/ncm_paper_regime. The report is
2026-10-23_ncm-matched-forecaster.md. Test admission consistency, source-only
label exclusion, real gradient scope, active/padded counting, fixed group
gates, RNG/history/objective replay and supervisor terminal behavior. Check the
actual source set, config and sampling coverage; commit source/tests/plan before
launch. One GPU0 worker runs at a time.

The supervisor owns all actual child waits, raw audit and exact registry
readback. Replay all49,152sampled IDs/full frames/actual and target histories/
word masks/rates/weighted losses and sampler/noise/optimizer state. Verify
frozen parameter values against the registered initializer, raw32case counts,
native reload probes and all five recovery snapshots. Hash only immutable
producer and numerical receipts, excluding mutable parent progress/result and
future audit/register receipts. Preserve every failed conditional stage.

A passing component still requires predicted-history word adaptation and new
semantic evaluation. Keep the original128development cases and conditional
256short+64long final cases, qualified Terra/Sonnet judges, all-valid
denominators and same-case joint final thresholds244/256and61/64. No95%success,
speed, compression or paper-training-replication claim follows from TRAIN fit.
