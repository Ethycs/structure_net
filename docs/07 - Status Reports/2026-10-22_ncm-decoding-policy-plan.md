# Paired decoding-policy test on generated NCM codes

Date:2026-10-22. EXPLORATORY `ncm-decoding-policy-v1`.
The completed noisy-pair models scored0/64under both original judges. The new
coarse-curriculum model is training and broader teacher data is being collected.
The released evaluator uses frequency/deficit and adjacency penalties and skips
PADbeforeEOS. Local runs used raw argmax and EOS-or-PAD termination. An earlier
test found the XSum prior harmed *oracle-code* reconstruction; that does not
establish what it does to generated, potentially inconsistent code hierarchies.

Question: do published decoding rules improve complete-summary quality on the
same generated hierarchies? Use the completed noisy native generator, the new
polished decoder, and all64fixed development sources. Verify unchanged
non-decoder state and original generator/codec pairing before constructing the
new composite candidate. This is a new candidate, never a rescue/relabel of the
prior0/64models. It bundles decoder polish with a new precision cell; matched
policy comparisons use identical cached logits/codes within this candidate.

Fixed placement: physicalGPU2, RTX2060Super8GB, one inference worker, batch4,
FP32 parameters with explicitFP16autocast, FP32 residual-code reconstruction,
seed42. No BF16 or silentCPU fallback. No parent Llama loading or training.
Keep resident services and ongoingGPU0training unchanged. Source <=512GPT2tokens,
canvas48. Generate each hierarchy exactly once with the released temperature1,
top_k1, top_p0 and CFG0 settings. Freeze and save the resulting codes/logits.

Compare these3predeclared policies:

1. Local control: raw argmax, stop at firstEOSorPAD.
2. Released XSum: frequency2, deficit1, adjacency(5,4,.5), released XSum token
   unigram, cut atEOSand removePAD. Preserve the evaluator'sFP64count arithmetic.
3. Domain prior: exactly policy2with a unigram computed from all48positions of
   the1,999training targets, includingEOSandPAD, normalized with no smoothing.

Also save raw argmax with the released EOS/PAD rendering as a diagnostic to
separate termination from penalties; it is not an additional selected endpoint.
The released prior'sPADmass is.4403935andEOSmass1/48; includingPAD/EOSin the
domain prior mirrors that convention. Never use development labels or test
data to calculate the prior. All3policies share exact logits and model weights.

A separate CUDA pilot on4training passages must pass finite logits, memory,
non-decoder identity, unchanged parameters, upstream decoder-algorithm parity
and exact checkpoint reload/generation with restoredRNG. CPUtests compare the
policy implementation against the function extracted from the vendored author
script. Pilot outputs are not pooled into development quality. Stop on OOM,
changed dependencies/weights, nonfinite values or parity/reload failure.

After generation, grade every policy on all64development cases under both
qualified original judges with the same one-candidate rubric/parser,4workers
per judge and preserved raw replies. Schedule primary grading after the broader
teacher job or validate combined concurrency separately; secondary can run
while the primary teacher is active. No valid failure retries. Report all6
policy/judge results, denominators and Wilson intervals; no cherry-picking the
best judge or treating invalid judgments as passes. Require>=61/64under both
for any later final-test admission. The256-case test remains sealed.

Artifacts:`20261022_decode_pilot_v1`, `20261022_decode_outputs_v1` and distinct
policy/role grade roots under `data/experiments/ncm_paper_regime`. Preserve the
old failures and this single-seed/precision limitation. No inference-speed or
95%model-quality claim from inspection, training fit or decoding eligibility.
