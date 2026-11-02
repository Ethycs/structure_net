# Measure unseen summary quality after same-model word convergence

Date: 2026-10-23. Prospective exploratory hypothesis:
`ncm-converged-native-summary-quality-v1`. Status: planned, not executed.

The original five-concept/four-word-pass NCM endpoint failed 0/64 whole-summary
judgments. A target-free control on four training sources found that the same
unchanged model reaches verified word fixed points and produces substantially
more coherent text when allowed to converge. Training examples and numerical
convergence cannot establish task quality. The separate 128-update paired-loss
control failed its oracle decoding-capacity gate and is not selected here.

Restore the original 2048-update checkpoint, with all original 28 Llama3B
blocks, embeddings, MLPs and native word head frozen. Use its source-only five
coarse predictions and fixed 128 whole-frame word updates, followed by one
extra verification pass. This is a new NCM/causal-decoding hybrid endpoint,
not a rescue or relabeling of the failed fixed-nine-pass endpoint and not a
replication of the NCM or JacobiForcing authors' training recipes. It uses no
external verifier, gold previous words, oracle plan, AR fallback, extra
training, selection between output variants, or inference-time teacher call.

Use all 64 original development sources in their original order, with original
complete public text and unmodified native chat instructions. These are reused
development cases, not an untouched final test. Verify training/development
group disjointness and source identities. Generate before judging; never select
or repair an output after judgment. Retain every case, empty or overflowing
summary, raw native IDs through first stop, all concept codes and word states.
The four-update snapshot must reproduce the original failed outputs exactly
under the same batch two/precision. Require the final verification pass to
leave every full 128-position word state unchanged. Preserve systems failures
and stop rather than hiding them as quality exclusions.

Use qualified Terra first under the application 128-cl100k-token rubric.
The whole-summary pass requires faithfulness, salience, readability and token
capacity together. Retain all 64 in the denominator. Require 64 valid and at
least 61 whole-summary passes before independent qualified Sonnet grading of
the identical frozen texts. Require the same gate from Sonnet. Invalids,
component failures, native/storage counts, exact first-judgment reuse and Wilson
intervals must be reported. No retuning judges, multiple semantic attempts or
regeneration for better scores. Transport/schema recovery may use the existing
maximum three attempts and must retain first valid judgments.

Before new API requests, import and audit all 192 existing exact first Terra
judgments from baseline and failed-student grading, including every negative.
Identity includes complete packet, profile, role and model; an identical
candidate must retain its first decision. Preserve all prior raw results.
Report NCM_trained=true accurately in new endpoints; do not inherit the old
generic grader's stale NCM_trained=false placeholder.

Pass focused CPU lifecycle checks and a real two-case CUDA inference/reload
pilot before the full run. Reuse the identical qualified rubric and existing
CUDA word-kernel memory controls where justified; no new memory-concurrency
assumptions. Physical GPU0 only, BF16 native/FP32 new weights, TF32 disabled.
The pilot uses the first two development sources, checks exact inference after
deliberately perturbing and reloading new parameters, and retains its first
outputs and complete word states unchanged in the full 64-case generation.
It makes no API requests or quality decisions. The remaining 62 cases are
generated once after the pilot's observed terminal success.
Freeze producer source, original checkpoint, complete source identities,
tokenizer/profile, prior caches and qualified judge evidence under fresh roots:
`20261023_converged_native_{pilot,outputs,grade,results}_v1`.

Do not open the original sealed 256 or longer-source sealed 64 cases, and do
not benchmark speed in this study. A passing short-source development gate
only admits separately planned application-range and final evaluation. A
failure requires reporting the new complete model accuracy before deciding on
more training. Adaptive/block decoding and its possible acceleration remain
separate work, with equivalence and task-quality checks and all compute costs.
