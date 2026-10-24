# Semantic accuracy of the full population NCM student on fresh sources

Date: 2026-10-24. Prospective exploratory hypothesis
`ncm-clean-full690-semantic-development-v1`.

Measure complete-summary quality of the fixed final32768-update full690 dense
word student on all64 short and64 long development sources reserved before
model outputs. Earlier semantic failures, including the TRAIN-overlapping lone
pass, remain recorded. This study measures source-conditioned task accuracy;
exact reference-summary matching is not an accuracy metric.

Prepare source-only evaluation metadata from the complete fresh DEV records and
retain their exact public text, spans, order and128 distinct components. Verify
literal public provenance and repeat actual text/family/turn overlap checks
against all690 unchanged TRAIN sources and within DEV. Use the source reservation
guard: family and group identities, normalized word equality, and at least32
shared exact word5grams covering80% of the shorter text. Shared full-turn aliases
require at least192 cl100k tokens; all text and provenance remain unchanged.
Open no sealed final source file and never filter cases by model quality.

The CPU supervisor may prepare these source records while the already queued
word continuation runs. Capture that supervisor's PID and Linux start ticks.
Admit generation only after its full five-stage lifecycle has actual zero exits,
the exact campaign is registered and that same process is terminal. The final
32768 word checkpoint is the only candidate; the literal TRAIN fit gate is a
diagnostic and does not block this separately specified semantic measurement.
Bind the exact checkpoint and feature manifest at admission, before inference.

Use the original frozen28-layer Llama body features from the new128-case cache,
the new fixed matched forecaster and the selected dense word branch. Run eight
coarse concept passes and one dense word pass, with original decoder and codec
unchanged. Full192 GPT2 word positions use one global EOS. Select one deterministic
argmax output, with no target concepts, teacher words/latents, output repair,
source clipping, lookup, autoregressive fallback or best-of selection.
Use FP32 weights/decoder/reconstruction and BF16 transformer forwards on GPU0,
both TF32 modes off, constructor seed57142 and at most8GB peak allocation.

Before full generation, compare both actual full-population TRAIN reload probes
against their registered raw words,608 coarse codes and latent hashes. These
two probes need not belong to the separate fixed32 fitting set. The inference
pilot contains the first short and longest GPT2 long source. Require exact
predictions after a fresh model load, unchanged parameter/codec hashes and the
memory limit. Retain those pilot outputs exactly in full128 inference and check
their agreement again. Any invalid model, feature or identity control stops work.

Judges receive the full public source and one anonymous candidate only. Keep
the qualified128-token rubric: faithful, salient, readable, nonempty, at most128
stored cl100k tokens, with supported/contradicted/insufficient-evidence claim
verdicts and source citations. Paraphrases are allowed. Qualified Terra is the
secondary judge and Sonnet the primary; each retains its50/50 control result.

Inherit all1797 canonical first judgments from the registered last dense semantic
campaign, cache SHA256
`77f3cda9d75c808780168b4b59d96af7182ea56bb9899fef812b1c577165748b`.
Never reroll a valid negative, overwrite a first judgment, or restart an
exhausted-invalid packet. Permit at most three transport/schema attempts, with
four concurrent requests. The four-case judge pilot uses first and longest
other source per cohort. It checks schema/transport only, permits zero quality
passes and contributes its unchanged first decisions to the full evaluation.

Run Terra on all64 cases in each cohort. Admit Sonnet for that cohort only after
64 valid Terra decisions and at least61 passes. A cohort qualifies only when at
least61 of the SAME64 cases pass both judges, with every decision valid. Require
both cohorts to qualify. Retain invalids and all failures in their original
denominators, raw predictions, requests, responses and first caches. Report
per-judge and joint semantic pass rates and Wilson intervals; these are model
judgments, not human ratings or a95% population confidence lower bound.

The independent CPU auditor replays source metadata, overlap exclusion, raw
output decoding, constructor parity, pilot retention, exact anonymous packets,
all1797 baseline first decisions, new firsts, citations/token counts, endpoint
components, conditional judge admission and same-case joint gates. Actual child
exits, frozen source snapshots and exact registry readback complete the study.
Passing DEV permits a separately specified sealed320-case evaluation and frozen
body integration checks; it does not establish final95% accuracy or speedup.

Use fresh `20261024_clean_semantic_dev_*_v1` roots. Reproduce with the existing
NCM Python environment, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=''`
and `python -m experiments.structure_net.ncm_clean_semantic_development workflow`.
