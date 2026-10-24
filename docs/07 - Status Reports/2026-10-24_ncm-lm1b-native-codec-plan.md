# Native LM1B codec compatibility before NCM retraining

Date: 2026-10-24. Hypothesis `ncm-lm1b-native-codec-compatibility-v1`.
Conformance: EXPLORATORY representation control, no training and no task-success
claim. This protocol precedes loading either LM1B weight file or observing its
reconstruction outputs. Source and labels are the SAME complete690 TRAIN cases.

The current expanded-XSum dense student failed all128 development summaries.
Its date-matched evaluation is running separately, with its source frozen.
The existing original48-token released-XSum control works on its small native
benchmark. Current full192-position adaptation changes both geometry and the
word prediction mechanism. The alternative released LM1B codec natively processes
one128-position BERT-wordpiece frame, with12 levels and a deeper text decoder.
This can test a pretrained global text representation before training a new
conditional generator. It is not a replication of conditional summarization.

Pin author weights and configs to
`hazyresearch/ncm-checkpoints@d342ce8aaea58830eef3b8ad473664673e41f692`.
Shared `lm1b/tokenizer.pt` SHA256 is
`d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c`;
small generator SHA256 is
`b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe`.
The inventory and local BERT tokenizer are already pinned. OpenWebText's native
1024-position config exists, but this pinned HF release has no OWT weights.

CPU tokenizer coverage is already known:670/690 targets fit128 wordpieces with
EOS,20 require129–149 positions, zero unknown tokens. BERT lowercases and
normalizes text, so exact equality to the original cased target is inappropriate.
These facts motivate checking the codec; they are not quality measurements.

Retain ALL690 cases. Reconstruct complete native tokenizations, appending one
genuine EOS101 and padding with0. For the20 overlength targets use TWO native
windows for this diagnostic, retain both, concatenate raw IDs, then stop at the
first global EOS. Do not truncate or remove those targets from token/semantic
denominators. These windows do not prove that a single128-token output frame can
produce an overlength reference. Any later generator requires a separate protocol
to adapt those20 training labels or expand the output geometry.

Load the native codec strictly, including every buffer, from the exact author
state. Preserve its128 context,12 unmodified levels, vocab30528, all pretrained
encoder, quantizer and four decoder blocks. No blocks are split or tiled. Run
FP32 inference with TF32 off, eval/no gradients, one deterministic full-vocabulary
argmax. Report non-PAD token agreement against native tokenization, exact native
sequences, first EOS, active invalid IDs and one-frame coverage, separately from
semantic summary accuracy. Source-only accuracy stays unmeasured.

Before full measurement, use the fixed first and longest TRAIN target as systems
probes. Require repeated encode/decode equality and unchanged full codec hash;
the full fresh-process load must reproduce both outputs exactly. GPU2 only,
one worker,4GB free before loading, peak at most4.7GB. Original Llama and all its
MLPs remain frozen and absent. Stop on systems, provenance or memory failure;
preserve its root. All final source contents remain closed and unhashed.

Download, CPU preparation, actual CUDA pilot and full690 reconstruction each
receive fresh roots, source fingerprints and observed subprocess exit receipts.
Numerical reconstruction cannot pass the95% task goal or authorize a speed
claim. A later separately specified semantic audit must compare reconstructed
summaries and baseline targets against their complete sources, preserving every
canonical first decision from the latest completed semantic study. No concurrent
fork of the first-judgment cache is permitted.

Entry point: `python -m experiments.structure_net.ncm_lm1b_codec workflow`.
Roots: `data/experiments/ncm_paper_regime/20261024_lm1b_*_v1`.
