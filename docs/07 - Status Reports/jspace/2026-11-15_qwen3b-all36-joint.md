# Concurrent all-at-once 3B distillation on the 6 GB RTX 2060

Prospective study: `JSPACE-3B-ALL36-JOINT-01`. The user requested this separate
run concurrently with progressive compression. It has its own reservation,
initializer, checkpoints and results. The recovered fourteen-block model is
not overwritten or used to initialize its learned parameters.

The original model is Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Freshly project its parameters into
all 36 existing task-Jacobian residual/head bases: residual rank 1,024 and head
width 64, preserving original attention scaling and softmax. The polynomial
relations compress the feed-forward layers; this is not quadratic attention.
Use the existing 34 relations and explicitly complete the two missing ones:
15 anchors for layer 2, as admitted by its original TRAIN numerical-rank
diagnostic, and 24 for layer 5. Keep the original cutoff, TRAIN pivots, ridge
and 1,024 removals. Development is diagnostic for these fixed-width fits.
Historical calibration teachers and data are recorded; this is not a claim
that all 36 bases were fitted against the new recovered model.

All learned stack/norm parameters and the original tied embedding/readout use
BF16. Geometry retains the existing BF16-mode compiler convention. The runtime
constructs its nonpersistent rotary buffers directly on the execution GPU and
pins their resulting whole-state fingerprint. Do not substitute CPU rotary
bits or silently change precision. Use the unfused Torch executor on this
Turing card; Ampere BF16 Triton kernels and a second GPU-resident packed copy
are unnecessary for this training experiment.

Use the already frozen 4,608 TRAIN cases and 256 DEV cases, in the same common
order and option permutations. The input boundary is the original BF16
embedding lookup, captured on CPU without any learned prefix. Reuse the
original native teacher's full-vocabulary logits at content slots 4–7. The
objective is only `0.7 * T^2 * KL(native || student) + 0.3 * CE`, with T=2.
There is no auxiliary compressed-parent feature loss. Thus this is a practical
independent experiment, not a one-variable ablation of feature-assisted
progressive recovery; initialization, depth, hardware, microbatching and that
auxiliary objective differ.

Run one pass: 288 updates, effective batch 16, microbatch 1 and accumulation
16, constant LR 5e-6, clipping 1, seed 20861. Train every compressed block and
the final norm jointly. Keep bases, bridges, geometry, tied E and rotary fixed.
Use the existing CPU-gradient-bank Adam implementation: BF16 accumulated leaf
gradients and FP32 masters/moments in CPU RAM. Do not retain all parameter
gradients on the GPU. Maximum CPU optimizer/gradient allocation is 32 GiB;
check available host RAM with an 8 GiB reserve, including snapshot overhead.

A discarded pilot scores the initializer, then updates the sixteen longest
TRAIN examples once. Require finite losses/gradients, all 36 blocks and the
norm receiving every microbatch gradient, FP32 master changes in every block,
and unchanged complete fixed state. The scientific run starts in a fresh
process from the untouched initializer and zero optimizer state and must
reproduce the pilot's initial records and master fingerprints. A failed pilot
is retained; it is not counted as scientific training or silently retried
under different ranks/precision.

Retain actual weights and complete DEV predictions at updates 0, 72, 144, 216
and 288. Eligibility is the same absolute original-native reference: native
195/256, maximum 12 net lost overall and 6 per family. Choose highest correct
among eligible checkpoints, earlier update on ties. Keep a separately labelled
diagnostic best if none qualifies. No extra epochs or changed thresholds in
response to observed quality. Freshly reload selected and distinct diagnostic
weights on the 2060 and require exact Torch DEV records and whole fixed-state
hashes. These are development findings, not held-out confirmation or inference
throughput measurements.

Completed phase receipts bind pilot → training → fresh verification by their
manifest hashes. Admission recomputes candidate accuracy, family counts and
eligibility from the retained DEV records, checks report/checkpoint linkage,
and requires separate exact replay proofs for the selected and distinct
diagnostic checkpoints. A bare success flag cannot establish verification.

The recovery128 audit has already been exposed by the preceding study and
failed its five-point screen (native 103/128, recovered14 93/128). It is not a
new sealed test for this experiment and must not influence its checkpoint
selection or enter training. The final512 remains sealed. Any later final
confirmation requires a separately authenticated frozen full-model candidate.

Reserve only `GPU-8cf440df` at 5,632 MiB with a 5,376 MiB allocator cap. Keep
the 3060 and 2060 SUPER available to the independent progressive run. Choose
the training lease duration from the discarded pilot timing, with margin for
evaluation and checkpoint I/O. Store large files on the data volume. Every
completed phase is immutable and failures retain their original evidence.
