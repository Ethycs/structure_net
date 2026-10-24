# Task-domain decoder: fixed final checkpoint and held-out semantics

Date: 2026-10-24. Exploratory, underpowered hypothesis
`ncm-lm1b-domain-decoder-heldout-semantics-v1`. The domain corpus and training
protocol are frozen in `2026-10-24_ncm-lm1b-domain-decoder-plan.md`. This evaluator
is frozen before reading application predictions from the new checkpoint.

Require actual successful terminal training, including the real CUDA pilot,
fixed final 1024 updates and strict decoder/optimizer/RNG reload. Check every
encoder/codebook/EMA state hash and the final full codec hash. Use only the fixed
final checkpoint, never step 512. Reuse the frozen evaluation engine with new
roots and explicit configuration bindings; no outcome-dependent branch is added.

Reconstruct all unchanged 690 targets without clipping, including 25 needing
multiple native windows. A first/longest CUDA pilot repeats complete raw logits'
argmax outputs; a fresh full process must reproduce them. Verify all concept
hierarchies still equal their original released-codec hierarchies. Independently
replay each raw metric and report gradient-fit 626 versus held-out 64 separately.
All-690 numerical agreement combines fit and held-out data and is diagnostic.

Semantic evaluation uses exactly the same 64 earlier hash-selected sources,
excluded from gradient fitting by case, session, group, source hash and turn hash.
They were inspected in earlier studies, so call them gradient-disjoint component
development, not a fresh test set. The arms remain approved original target,
BERT roundtrip, released PAD0 codec and the new task-domain codec. The failed
generic adaptation remains fully preserved in its earlier campaign.

Use the same anonymous single-candidate packets, full public sources, qualified
Terra/Sonnet controls and unchanged rubric: faithfulness, salience, readability,
nonempty output and at most 128 storage tokens. Accept paraphrases. Inherit all
2266 first decisions from the last registered campaign, SHA256
`9fcefdba0ef580599b0b226d762b898b180601756ebfd57513ba92e7c7441075`;
cache SHA256 `883d00182e9614ba999e0d1ab4182988cd519f7c956ec5ae7d23075415fc40fb`.
Identical packets share decisions. Four-case schema pilot, four workers and at
most three transport/schema attempts. Retain every first decision, including
valid negatives and exhausted invalids; none may be rerolled.

Admit Sonnet for an arm only after that arm and the approved reference pass at
least 61 of 64 under Terra, all valid. Qualify the codec only when at least 61
of the SAME 64 pass BOTH judges, with a qualified original reference. Report all
invalids and complete denominators. A numeric gain cannot rescue semantic failure.

The independent CPU auditor must reconstruct the corpus mixture from source text
and seeds, prove held-out exclusion, replay every sampled update/augmentation and
every raw prediction and semantic score, verify all first-decision inheritance,
check actual child/whole-tool exits, bind source/input/artifact hashes, then
register the conservative result with independent backend readback.

Even qualification cannot establish source-only summarization or 95% task
accuracy. Original Llama weights and MLPs remain frozen and absent from codec
training. Source-conditioned NCM training and full 128-case task development
qualification are still required before the sealed 320-case final evaluation.
No final source is opened or hashed, and no speed claim is made in this study.

Fresh roots: `20261024_lm1b_domain_decoder_{reconstruction_pilot,reconstruction,
packets,judge_pilot,grade,results,eval_workflow}_v1`.
