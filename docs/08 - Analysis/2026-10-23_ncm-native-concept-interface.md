# Native residual concepts pass representation controls

Date: 2026-10-23. `ncm-native-concept-interface-v1` is complete,
**native_residual_representation_invariants_passed**. This is not a trained
summarizer. No task accuracy,95%NCM result, decoder quality or speed is measured.

Frozen native word embeddings are represented at4,8,16,32,64 cells using256
FP32 residual centers per level, with an exact zero entry. Native word prediction
is a later language stage; these books do not decode text. This is a new
NCM-inspired interface, not the paper's unchanged learned VQ-VAE.

| Concept cells | Raw training embedding MSE | Diagnostic embedding MSE | Diagnostic active entries |
| --- | --- | --- | --- |
|4|0.0003456069|0.0003475974|76|
|8|0.0003426718|0.0003458579|13|
|16|0.0003374162|0.0003415830|30|
|32|0.0003232230|0.0003287824|115|
|64|0.0002886096|0.0002957212|169|

These are squared embedding errors, not token or summary accuracies. The
initial diagnostic error is0.0003660070. Public
training sources:512 for fit,64 separate diagnostic groups. Raw windows are
the first128 native tokens with EOS padding; task examples are never truncated
or evaluated here. No summary labels/dev/final-test cases enter fit.

Zero-code error monotonicity, static/clean-history label parity, experts no
worse than static labels at the same corrupted history, finite centers and
exact center/assignment save/reload passed. Diagnostic same-history repair
gain is0.0000013566 squared
embedding units. This does not establish better generated summaries. K-means
ran32 iterations per book with fixed initialization indices; GPU index-add
floating reductions are not claimed bitwise reproducible across fits.

FP32, physicalGPU2, fixed source/distance chunks. Representative pilot and fit
terminated successfully; own peak allocation4336910336bytes.
The original embedding file stayed unchanged. No student run is admitted by
representation MSE alone. Broad training still needs qualified labels and its
own prospective protocol; final256 task cases remain sealed.

Aggregate `data/experiments/ncm_paper_regime/20261023_native_concepts_results_v1/campaign_results.json`; SHA256`a8e025c4903cc005d44a41da7e22d4e867791897b20b3f6e2348714ecff9e2b4`. Actual centers, raw ID
windows, assignments, source snapshots and terminal receipts remain append-only.
Registry confirmed=false, task_accuracy_measured=false.
