# Generate the fixed128source-only predictions independently of judge transport

2026-10-24. Hypothesis `ncm-level-conditioning-fixed128-source-only-inference-v1`.
The new FP16native655candidate completed1024updates, but semantic requests
are delayed by observed gateway read timeouts. Local inference has no dependency
on an external judge or its canonical first cache. This prospective split
preserves the original checkpoint,128development sources, full feature range,
output policy and subsequent same-case semantic gates.

Require actual whole training0and independent CPU training audit0. Copy the
existing strict checkpoint contract into an independent frozen helper, with
no semantic cache references. Load the ENTIRE saved model strictly, verify
new generator/word-decoder/attention hashes, FP16scaler state,1024updates,
original warm state and frozen encoder/VQ; reject any old/pilot checkpoint.

Use all unchanged64short+64long permanent DEV source records and full FP32
features. Check every identity, alignment, prompt, feature hash and source
position. Prediction receives source-only metadata: no target text, teacher
word IDs, oracle codes or quality labels. No parent body or MLP is loaded or
trained. Native deterministic generation predicts all641codes and128raw word
IDs; use full30528vocabulary and the native CLSrenderer, with no repair,
clipping, masking, reroll or output selection.

PhysicalGPU2RTX2060SUPER, FP32hard inference, TF32off, max allocated3GB.
First/longest-source lifecycle pilot repeats exactly. Fresh full inference
must retain every pilot first, cover all128in fixed order, preserve model
hashes and produce no gradients. Independent CPU audit replays all source/
feature/checkpoint/code/output bindings and native rendering after whole
actual0. No semantic request or canonical cache is read or written.

Fresh `20261024_level_conditioning_numeric_*_v1` roots. The later complete
source-only semantic DEV must reproduce all128archived outputs exactly and
inherit the final completed parent judgment lineage. Its launch guard and
both-cohort61same-case/both-judge/all-valid gates stay intact. This split is
not a new accuracy endpoint;95percent remains unestablished until measured.
All sealed final320source files stay unopened and unhashed.
