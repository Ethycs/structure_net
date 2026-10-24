# Source disjoint semantic development evaluation of the dense NCM student

Date: 2026-10-23. Prospective exploratory hypothesis
`ncm-dense-source-semantic-development-v1`. Evaluate the fixed final 2048-update
student checkpoint 0b8afde259c1165ddc4ebb14110568b0695228850dd09eaeadf645095550e266
on the unchanged 64 short and 64 long development sources. The parent TRAIN
study reached 3186/3197 token agreement and 24/32 exact texts, but original16
missed its literal fitting threshold by one token. Preserve its registered
dense_source_TRAIN_fit_failed classification. This independent semantic study
asks about complete summaries, not literal reconstruction, and does not rescue
or amend that earlier numerical gate. No unseen output from this checkpoint
has been inspected when this plan is written.

Use exactly the existing source-only cases in
20261023_word_final_evaluation_sources_v1/cases.json, SHA
07786d65a4d29201ba7be302d8c4fcc9af56a5f1bb48beec55a9cddcd756d3e1.
Retain their original frozen-body features, manifest SHA
4d1f62db7d7e5ed12f288059fc3b74c3f4ca44067a094d7cc208301e0284b6af,
and original 28-layer Llama body hash. Verify every selected feature's bytes,
complete token count, source/group/session/hash disjointness from all 690 TRAIN
sources and between development cohorts. These are reused development cases,
not a new blind final test. Neither sealed final source file may be read or
hashed by this study.

Load exact forecaster and student weights with the verified direct loader,
then strictly load the final 2048 student. Verify all 213 parameter tensors,
the original codec and frozen word parameters. Fresh inference runs the fixed
forecaster's eight levels from source features/IDs/mask only, reconstructs the
word conditioning, and runs the dense head and original frozen decoder. Keep
all 192 raw word IDs, eight full concept levels and source metadata. Decode
once at the first global EOS. Never use teacher words/latents or correct
concepts in prediction; never repair, clip, filter, reroll, look up or select
among candidate outputs. Use deterministic argmax throughout.

Use physical GPU 0 after the completed fitting supervisor is terminal. Weights
and decoder stay FP32; source/transformer computations use BF16 autocast, with
both TF32 modes disabled. Cached original-body features are used; the parent
body is not loaded. This is a quality evaluation, not an end-to-end timing
benchmark. Constructor seed 48142 identifies restored weights, not random
model initialization. Require peak allocation below 8 GB.

Before all 128 outputs, pass focused tests and actual-data/cache preflight.
The CUDA lifecycle pilot must reproduce the registered fitting outputs of the
first and longest TRAIN probes exactly. It also generates the first short and
longest long development case, then destroys/reloads fresh models and requires
identical full raw outputs and latent digests. The full worker recomputes and
retains both development pilot outputs exactly, in original 128-case order.
Model and codec hashes stay unchanged before/after generation.

Keep the qualified 128-token semantic rubric and both 50/50 qualification
records unchanged. Send only one anonymous candidate and its exact public
source, with supported/contradicted/insufficient-evidence claims and source
line citations. Useful paraphrases are allowed. Passing requires faithful,
salient, readable, nonempty text within 128 stored cl100k tokens. All 64 cases
remain in each cohort denominator; invalid judgments are retained and prevent
a qualified accuracy claim. Terra is secondary and Sonnet primary. Admit
Sonnet for a cohort only after 64 valid Terra judgments and at least 61 passes.
Require at least 61/64 for each judge in each cohort AND 61 of the SAME 64 cases
to pass both judges, with all valid, before declaring development qualified.
Report same-case joint counts and Wilson intervals; do not claim human gold
or a 95% population lower confidence bound.

Preserve all 1669 canonical first judgments from the registered TRAIN semantic
cache, archive SHA c88711fd3af4d65c887c8aac4aeb092c82adb117f6dcbf374b2cd42a47cfbcec.
A CPU judge-path pilot uses the first and longest nonfirst case of each cohort,
four total, with Terra only. Require valid transport/schema, never positive
quality, for pilot admission. Retain those first decisions in full grading.
Identical packets reuse their original decision. Permit only the existing
maximum three transport/schema attempts; no valid negative or exhausted invalid
decision is rerolled. Replay every raw request/response and preserve all failures.

Use fresh roots 20261023_dense_semantic_dev_{workflow,inference_pilot,outputs,
packets,judge_pilot,grade,results}_v1 under data/experiments/ncm_paper_regime.
Freeze source/tests/plan/auditor before launch. One supervisor owns pilot,
generation, packets, judge pilot, full grading, CPU audit and exact registry
stages, with actual child exit receipts. Independently replay source and output
bindings, full concept/word/EOS decoding, first caches and qualified conditional
judge admission, complete endpoint/component/token/Wilson and same-case joint
counts. Classify judgments_invalid, quality_failed, or passed_final_unproven.
Passing supports a separate final-evaluation plan; this study never opens the
original sealed 256 short + 64 long cohorts and cannot complete the 95% goal by itself.
