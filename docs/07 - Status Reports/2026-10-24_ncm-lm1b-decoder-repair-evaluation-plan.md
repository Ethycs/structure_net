# Fixed released-NCM decoder repair evaluation

Date: 2026-10-24. Hypothesis `ncm-lm1b-fixed1024-decoder-repair-semantic-v1`.
EXPLORATORY component evaluation. Freeze this evaluator before reading repaired
application predictions. It uses only the fixed final update 1024 from the
decoder-only fine-tune. The step-512 recovery checkpoint is ineligible.

Require the real training workflow to terminate successfully, including actual
gradient/reload pilot and all 1024 updates. Load the released encoder and books
strictly, then replace only the decoder from the final checkpoint. Verify its
full model hash and every frozen encoder/book/EMA state. A new two-case CUDA
pilot repeats first/longest reconstructions and retains exact raw outputs; the
fresh full evaluation must reproduce those outputs.

Reconstruct every unchanged original 690 TRAIN target, including all 25 targets
that require multiple native 128-position frames. Preserve raw argmax predictions
across all 30,528 logits, native code hierarchies, boundary positions, text,
complete denominators and failures. Use the baseline PAD0 input format and
unchanged renderer. Every code hierarchy must equal its original released-codec
hierarchy. No target/output clipping, learned target latent, reranking, correction,
fallback or output repair is allowed.

Semantic cases are exactly the earlier prospectively selected 64 public sources
(32 original and 32 reviewed targets) in their original order. Anonymous single
candidates have four arms: approved target, unquantized BERT roundtrip, released
PAD0 codec and repaired codec. Use the full exact source, qualified Terra and
Sonnet 50/50 controls, and the unchanged meaning/salience/readability/nonempty/
128-storage-token rubric. Accept paraphrases; token equality is diagnostic only.

Inherit all 2,239 canonical first decisions, cache SHA256
`df7de0055109fccc6fed26aba6fb4f3dbf0df7a29f520da77dcb2460bc60eab9`,
from the registered padding campaign `af2335b3a87c83509673f8e693aabf5682a71364f5350d5ee637a10ce9416337`.
Identical packets across arms share first decisions. A four-case semantic pilot
tests validity only, retains every first decision, and may not select examples.
Four workers, at most three transport/schema attempts; never reroll a valid
negative or exhausted invalid. Admit Sonnet for an arm only if both that arm
and the approved target pass at least 61 of all 64 cases under Terra, all valid.
Qualification requires at least 61 of the SAME 64 cases passing BOTH judges for
the repaired codec and approved target. All invalids and all cases stay visible.

An independent CPU auditor replays every raw reconstruction metric, the full
training sample/augmentation/schedule trace, source identities and public proof,
every first-decision score, inherited-cache preservation, admission gates and
joint counts. Bind hashes of producer source, data, checkpoints, requests, actual
child exits and reports; register the result and independently read it back.

This is gold-code reconstruction of application TRAIN targets after unpaired
author-TRAIN decoder adaptation. It is not held-out source-only summarization.
The previous conditional student remains 0/128. Training loss and native token
agreement cannot establish 95% task accuracy. All final sources remain sealed;
conditional training and speed measurement require appropriate quality checks.
