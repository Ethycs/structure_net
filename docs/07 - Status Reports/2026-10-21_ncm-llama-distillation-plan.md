# Llama-to-NCM summarization distillation pilot

Prospective exploratory hypothesis `llama32-3b-ncm-distillation-v1`.
Try sequence-level distillation from the cached original Llama-3.2-3B-Instruct
into HazyResearch's released XSum Next-Concept Model. This is a new architecture,
not a weight conversion or a continuation of the completed Q4-kernel study.
One adaptation seed (42), underpowered; physical GPU0 only, one worker.

Primary sources: [NCM paper](https://arxiv.org/abs/2610.14162),
[official code](https://github.com/HazyResearch/ncm), commit
`c70274273f489abc73832b2b98584a8483b6b203`, and
[released checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints),
revision `d342ce8aaea58830eef3b8ad473664673e41f692`.
The downloaded tokenizer/generator SHA-256 values are
`1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7` and
`f152f390253de886e46303654f01b9e7a2586e609ba3d6b5e86d5969903b1fd4`.
Use strict state loading and the actual checkpoint configuration. The released
model has a 512 GPT-2-token source prefix and 48-token output canvas. Do not
truncate longer sources to make a cell fit. This pilot does not test the
application's full 6300-token batches or 128-token output capacity.

Create 1024 training, 64 development and 100 held-out single-fragment sources
with disjoint seeds starting at 610000, 620000 and 630000. Four equally balanced
families: raw source lengths at least 256/384/480 GPT-2 tokens and hierarchy-like
384-token sources. Use the existing final-decision generator, retain explicit
superseded proposals and filler, and pad with whole filler sentences. Require
every source <=512 GPT-2 tokens and >=192 Llama tokens, no overlapping source
hashes. Final sources are fixed before adaptation; never use final examples for
training or checkpoint selection. These are synthetic source-fidelity probes,
not production summaries or general Llama capability transfer.

The frozen original FP16 Llama teacher generates actual summaries through its
chat template with a short, factual-summary instruction, greedy through EOS,
cap128. Use the existing CUDA-graph/FlashAttention/current-prompt lookup8
runtime at context1024. Save every source, teacher output, token sequence and
capacity/field check. Training labels must preserve final project, owner,
approved budget and deadline, omit superseded values, and fit 47 GPT-2 tokens
plus EOS. Reject invalid training labels without resampling; require >=95%
usable labels. Ground-truth facts are used for checks, never substituted for
Llama targets. The teacher and student have different vocabularies, so this is
text/sequence distillation followed by code cross-entropy, not logit KL.

First test the pretrained text codec's reconstruction on development teacher
summaries. If it preserves all required facts on >=95% of cases, leave it
unchanged. Otherwise, adapt only its decoder on training teacher summaries:
freeze encoder, codebooks and reconstruction operators, precompute clean VQ
codes, and train decoder token CE for 512 updates, batch16, AdamW lr2e-4,
betas(.9,.95), weight decay.01, clip1. Evaluate steps256/512 and select by most
development fact passes, then lowest development token CE, then earlier step.
All optimizable weights start from the released model, with no random weight
reinitialization. If the selected codec still misses >=95% development fact
fidelity, stop the conditional generator stage as `codec_fidelity_failed`.

With the codec frozen, adapt the released NCM generator for 1024 optimizer
updates, batch8, AdamW lr1e-4, betas(.9,.95), weight decay.01, clip1, 50 warmup
steps followed by cosine decay to1e-5. Use dropout.1, BF16 autocast with FP32
master weights/optimizer, clean teacher-forced prior-level code reconstructions,
and equal code-position CE, including the learned level0 queries. No target
level may see its own or later target codes. Evaluate steps256/512/1024 and
select by most development fact passes, then lowest code CE, then earlier step.
Use deterministic greedy code argmax and token argmax, no search or verifier.
Save model, optimizer and RNG state for traceable continuation. Decoder
adaptation and generator adaptation are separate training stages.

Before full execution, pass focused CPU split/configuration/gradient/leakage/
fingerprint tests; a real GPU forward/backward, finite-update, checkpoint-reload
and batch1 inference lifecycle; and representative batch8 generator and batch16
decoder memory pilots. Label these systems-only and keep their data separate.
One visible GPU and one worker is the intended-concurrency pilot. Do not launch
a larger configuration on OOM; preserve the failed root and amend explicitly.
Freeze/hash all producing code, vendor sources, data, checkpoint and library
identities before each producing worker. Earlier completed evidence is append-only.

After selection, evaluate the frozen adapted NCM, original teacher, previously
optimized factorized rank1024 FP16 Llama, and unadapted released NCM in isolated
blocks in that order. Use all100 fresh cases, three timing repeats after an
excluded warmup per case. Measure from CPU input token IDs through decoded CPU
text, including NCM's entire hierarchy and text decoder. Exclude load,
compilation and warmups. Both model families use their native input encodings
of the same complete source; Llama additionally uses its chat instruction.
Count output tokens under the common Llama tokenizer, not latent-code count;
report request latency and token rate together. Record weight/buffer payload,
allocated/reserved peaks, actual generation steps and output termination.

Final quality requires >=95/100 source-reviewed adapted summaries and no fewer
passes than the original teacher. Positive-control teacher requires >=95/100.
Use deterministic project/owner/budget/deadline/supersession/length diagnostics
and Codex source review; no independent judge model. Missing facts are
insufficient, conflicting facts contradicted, and extra assertions require
source support. A factual failure cannot be rescued by fast generation.
Primary speed goal is >=2x both sum-of-case-median request latency and common
output token rate versus original Llama. Report both ratios against optimized
factorized FP16 separately. Report released-NCM control to distinguish
distillation gains from pretrained behavior. No test-driven tuning or retries.

Classifications: training-label capacity/control failed; codec fidelity failed;
teacher control failed; distilled quality failed; distilled quality retained
but speed target unmet; task-specific NCM prototype meets speed and quality.
All remain exploratory and `confirmed:false`. No mobile, production95%, general
capability, or full application-context claim. Root
`data/experiments/ncm_llama/20261021_*`; retain raw outputs and all misses,
aggregate, report, register/read back and commit only this new study.
