# Diagnose the trained predictor on fixed training controls

The existing supervision candidate's four DEV pilot summaries were repetitive,
often unrelated to the source and failed all three semantic criteria. Those
outcomes are preserved; they are not a measurement of the newly completed
recovered-data predictor. Before making further architecture changes, inspect
the latter's predicted codes and its two already declared readouts on training
examples only.

After the recovered predictor's full independent audit passes, use its exact
final checkpoint from all 6,416 updates. Reuse the same four original TRAIN
controls and inference procedure from the completed paired-readout pilot.
These controls were fixed before this training, including source/target length
extremes. Keep original FP32 features and inference, TF32 disabled, batch one,
physical GPU0 RTX3060, a 256 MiB feature cache and the 4.9 GB allocation ceiling.
Require the existing paired-readout qualification and three new focused tests.

Generate source-predicted codes with the trained predictor. Compare the shared
prediction helper against the original sequential function for numerical parity,
then decode those same saved codes with the original source-attending readout
and fixed step6212 decoder alone. Preserve the trained predictor's exact model
hash, protected tensors and exact restoration after the decoder swap.

After predictions are complete, report code agreement with the training targets
at every native hierarchy level and compare it with the saved initializer
predictions on the same controls. Gold codes enter this comparison only. Preserve
all decoded texts for inspection. Exact code agreement is a mechanism diagnostic,
not summary accuracy; four training cases cannot establish generalization.

Use fresh root `20261028_recovered_trained_control_readouts_v1`. Freeze and bind
the completed training and audit, original control outputs and new diagnostic
source. No optimizer update, judge call, DEV generation or Final320 access is
allowed. This diagnostic does not alter the ongoing semantic evaluations, their
first judgments, candidate order or selection gates.
