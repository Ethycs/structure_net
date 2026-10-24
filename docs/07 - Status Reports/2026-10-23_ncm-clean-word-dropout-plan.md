# Matched training-dropout diagnostic after a clean-word fitting failure

Date: 2026-10-23. Prospective exploratory hypothesis
`ncm-clean-word-dropout-fit-v1`. The full frozen-body summarization goal remains
at least95% complete-summary semantic accuracy on source-disjoint evaluation.
This study is a TRAIN fitting diagnostic, never a substitute for that goal.

The current clean-word run has not finished or received its final fitting
measurement. Its preceding model scored0/128 semantic passes. The completed
32TRAIN diagnosis found81.41% correct-history word agreement on original probes
and70.81% on added probes. The official pinned XSum configuration retains .3
generator dropout and .1 codec-decoder dropout:
https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/xsum.yaml
The actual attention implementations use both numeric SDPA dropout rates and
nn.Dropout modules. Disabling only the modules would leave attention noise.
Whether this regularization contributes to the fitting failure is unmeasured.

Admit GPU work only after the existing clean-word workflow has actually
completed its six zero-exit children, finalized and registered an exact
clean_word_fit_gate_failed result, and the clean-word supervisor kernel is
terminal. Require the already queued matched-forecaster supervisor to terminate
with not_admitted_clean_word_fit_gate_failed and no executed stages. Bind both
original PID/start-time identities while waiting. Any invalid upstream result
stops this study. If the clean-word gate passes, this study is not admitted;
the existing forecaster workflow remains responsible for its positive branch.
Never restart either upstream worker or overlap GPU training.

Compare two arms sequentially on physicalGPU0: original_dropout, then
zero_dropout. Initialize each independently from the exact final registered
clean-word checkpoint with a fresh AdamW optimizer. Preserve identical690TRAIN
rows, targets, source features, source frames, original body/MLPs, frozen codec,
old252-case forecaster, unused word roots and coarse/fine heads. Both arms train
the same existing source norm/projection, word transformer, word projection and
word decoder. No architecture, vocabulary, finest code or target word is added
to prediction inputs.

The only arm difference is training dropout: retain every original rate versus
set to zero every nn.Dropout.p and numeric attention.dropout in the trainable
generator transformer and copied word decoder. Record every original/applied
rate, require a nonempty original map containing .3 and .1, preserve all model
parameter bytes at this intervention, and reconstruct/verify the map during
fresh checkpoint loading. The separate frozen forecaster and codec retain their
original rates and eval mode. Both arms use eval mode for all fitting outputs.

Use8192updates per arm, batch1, no accumulation,128-step linear warmup to3e-5
then constant3e-5. This maximum equals the preceding run's final rate. AdamW
betas(.9,.95),epsilon1e-8,weight decay.01,clip1. Model/dropout seed30142 and
independent sampler seed31142 are identical across arms. Sampler identities
must match step-for-step and cover every690TRAIN row. Removing dropout changes
its RNG consumption intentionally; no other stochastic intervention is used.
Use FP32 master parameters/optimizer/loss, BF16 forward, both TF32 modes off.
The only loss is correct-coarse-history nonpadding word cross entropy, with
EOS50256 included and PAD50257 ignored. Do not change the original192-position
word canvas or source-frame maximum3328.

Before either full arm, run that arm's two-update maximum-LR/max-frame CUDA
pilot, using the first and longest TRAIN sources, with only the second padded
to3328. Require finite loss, positive source/projection/decoder gradients,
correct frozen gradient routing, peak allocation below8GB, exact model,
optimizer, sampler, CPU/CUDA RNG reload, identical composed source-only output
and actual original forecaster/cache parity. Pilot updates are not retained in
full training. Preserve a4096-update recovery checkpoint; select only the fixed
final8192-update checkpoint. Full workers run only after both pilots have
actually exited zero. A fresh fitting worker for each full arm then runs the
same complete32 probes used by the original diagnosis and current clean-word
component. Require native cache parity and unchanged parameters/codec. Save all
192 raw word IDs, targets, text and active counts in both correct- and actual-
history arms. Original history remains the old252-case forecaster; these
actual-history outputs do not establish matched-population generation quality.

The fitting gate remains199/200 correct-history active word-token agreement
in EACH original16, additional16 and complete32 group, using inclusive integer
counts. Compare both arms with the fixed upstream final result and report raw
paired count differences. If only zero_dropout passes, classify
zero_dropout_fit_passed_control_failed. If both pass, classify
both_continuations_fit_passed; extra training and dropout cannot be conflated.
If only the original-rate control passes, classify
original_dropout_only_fit_passed. Otherwise classify
both_dropout_fit_gates_failed. These are one-seed exploratory component results,
not semantic accuracy, a training replication, or a proven generalization gain.
Do not automatically promote any arm into the old queued forecaster workflow.
Further matched-forecaster and generated-history work requires a fresh plan.

Implement source/tests/plan/auditor in new files. Before launch pass focused
tests for numeric and module dropout coverage, parameter/eval invariance,
gradient routing, fixed schedule, sampler pairing, exact metadata reload,
upstream positive/negative/pending gates and raw fitting counts. Check actual
690-data coverage and CLI paths, then commit all producing files. The fixed
root prefix is20261023_clean_word_dropout_ with workflow_v1, and
{original_dropout,zero_dropout}_{pilot,train,fit}_v1, plus results_v1, under
data/experiments/ncm_paper_regime. Keep all earlier artifacts and source intact.

The canonical audit must replay all8192 sampled IDs, complete frames, actual
correct-history/target hashes, learning rates, optimizer/sampler states and
unchanged frozen parameters for EACH arm; verify paired sampler identities,
identical initial parameters, original/applied dropout maps and checkpoint
configurations, recovery snapshots, actual zero child receipts, raw32-case
counts, complete output decoding and inclusive fitting classifications.
Hash immutable numerical evidence and the workflow producer/first six child
receipts, excluding mutable parent progress and future finalization/registration
receipts. Write a measured report and conservative exact registry readback.

Keep semantic task_accuracy null and both sealed source cohorts unopened.
The95% goal still requires qualifying the unchanged128-case semantic development
evaluation and then the original256short+64long final evaluation under both
qualified judges, with no exclusions, repair, fallback or best-of selection.
No speed/compression claim is admitted by this diagnostic.
