# Llama 3.2 3B tensorization and physical-model packing

Date: 2026-10-10. Hypothesis: `llama32-3b-tensorized-cotenancy-v1`.
Profile: EXPLORATORY with fixed execution and quality gates. No model outcome has
been inspected at this registration. The four tensor primitive CPU tests passed.

## Question and scope

Can complete, independently stored tensorized Llama-3.2-3B-Instruct replicas,
stacked on a leading model axis and executed with dense batched GEMMs, provide
a useful memory/concurrency or throughput improvement over the deployed Q4_K_M
reader while preserving summary quality? Packing error and tensorization error
are separate measurements. No shared weights across models, multiple KV slots
on one model, continuous batching, or block-diagonal substitute is permitted.
Within each replica, the checkpoint's tied embedding/output weights stay tied.
No training is included in this initial post-hoc compression study.

The complete cached Meta checkpoint is revision
`0cb88a4f764b7a12671c53f0838cd831a0843b95`, with 28 decoder layers, hidden size
3072, MLP size 8192, 24 query heads, eight KV heads, head size 128 and vocabulary
128256. Preserve Llama 3 RoPE scaling, all blocks, residuals, normalization and
the output head. Model conversion and factorization run on CUDA in FP32;
inference cores and activations are FP16, softmax/normalization accumulation FP32.

Baseline: `bartowski/Llama-3.2-3B-Instruct-GGUF`, revision
`5ab33fa94d1d04e903623ae72c95d1696f09f9e8`,
`Llama-3.2-3B-Instruct-Q4_K_M.gguf`, 2,019,377,696 bytes, SHA-256
`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`.
The sending PBX agent supplied the same filename, byte count and digest for
its deployed file. The local download matches. This establishes artifact
identity, not identity of the Windows runtime or application data.
Pin llama.cpp at `23b0202a189c44a54625aadcb37a946dd1d6278d`.

## Decision order

1. CPU tests: TT contractions equal reconstructed matrices; model slices do not
   overlap; changing one replica leaves the others unchanged; save/reload works.
2. Small complete-model lifecycle and HF graph controls, plus a real projection
   CUDA pilot. Preserve failures and revise producing sources only in new roots.
3. Reproduce the sender's projection gate on GGUF `blk.0.attn_q.weight`: CUDA
   SVD ranks 64, 128, 215, 256, 384; 128 probe tokens; two physical FP16 copies.
   Measure reconstruction error, energy, packed/separate error, timing and
   memory. Two rank-215 factor copies approximately match the storage of one
   Q4_K query projection. This is a projection diagnostic, never full-model proof.
4. Convert complete models with two-factor SVD ranks 215 and 768, and four-core
   TT-matrix ranks 64 and 256. Factor embeddings/output heads as well as all
   attention and MLP projections. Dense retention is allowed only when a factor
   representation would use at least as many parameters, and must be recorded.
   Deeper TT runs are mandatory even if both two-factor variants fail quality.
   Use per-matrix deterministic seeds and randomized SVD with 16 extra range
   vectors and two power iterations; use exact SVD when the matrix is small or
   full rank. These are approximation choices, not packing losses.
5. Evaluate full-model validation quality for all four variants against both
   unmodified FP16 and Q4_K_M. Separately benchmark packed versus separate
   execution with 1, 2 and 4 resident physical models where memory permits.
6. Only candidates passing validation quality and numerical packing controls
   advance to the untouched ten-example summary evaluation. Stop quality
   promotion if none pass. Complete packing/runtime diagnostics even for failed
   quality candidates, clearly labeling them as such.

## Data and quality

Use deterministic synthetic dialogue/note fixtures with explicit superseded
proposals and final owner/date/location decisions. The task asks for exactly
three source sentences, one per line, retaining final decisions. Four validation
examples use seeds 1963 and 1964. Ten final examples use seeds 7, 17, 29, 41 and 53, one
work and one personal example per seed. Generated data and encoded prompts,
references and fragment identities are hashed before model evaluation.

Greedy generation, maximum 96 new tokens, common chat-template token IDs and
EOS policy. Teacher-forced reference-token NLL is a secondary diagnostic.
Primary quality: exact per-fragment coverage, exact-source support of output
lines (allow stripping a bullet or numeric list prefix only), and complete
three-fragment success per example. These are deliberately narrow synthetic
summary diagnostics, not application accuracy, open-ended summarization quality,
or a general 95% accuracy claim. Store complete generated outputs.

The Q4 baseline must cover >=90% of gold fragments with >=95% supported output
lines on validation; otherwise classify the summary task as an invalid quality
assay and revise it prospectively. Candidate promotion requires no loss relative
to Q4 in aggregate fragment coverage, supported-line rate or complete-summary
count, and no individual example losing more than one fragment. The final gate
requires no worse aggregate quality on either family and no individual seed
losing more than one fragment. Report exact counts, not only percentages.

## Packing, performance, and memory

Stack every corresponding tensor/core into allocated contiguous storage with
a real leading model dimension. Identical initial checkpoint values do not imply
shared storage. Separate execution uses disjoint model slices and independent
KV states. Test unequal inputs and perturbation isolation. Compare logits and
KV caches under the same token inputs; mathematical equivalence permits only
FP16 roundoff (logit relative L2 <=0.002, maximum absolute error <=0.05 and
top-token agreement >=0.99). Bitwise equivalence is recorded if observed, not
assumed. A failure is an implementation/numerical failure, not compression loss.

Run on the otherwise idle RTX 3060, preserving unrelated resident services.
Benchmark complete-model prefills of 128 and 512 tokens and 32 forced-length
greedy decode steps after a 128-token prefill. Each model receives a different
document. One warmup and five rotating/alternating samples, synchronized GPU
timing and wall timing, compilation/load/decomposition excluded and reported
separately. Record parameter storage, process/device resident memory, allocator
allocated/reserved memory, peak workspace and KV cache bytes. Allocation failures
are capacity outcomes; no silent offload or context reduction is allowed.

Load 1, 2 and 4 separate Q4 model objects with separate contexts. Benchmark
both sequential and concurrent independent-model execution when supported by
the pinned runtime. A single shared-model multi-slot server is not the baseline.
Record cross-runtime differences, CUDA offload and actual resident memory.

Useful packing requires >=1.10x throughput over separate execution of the same
tensorized replicas. A practical overall win additionally requires preserved
quality and either >=1.10x throughput over the matched physical-Q4 configuration
or strictly higher resident-model capacity at the same context and memory cap.
Also report the stricter ratio of two tensorized models to one Q4 model's memory;
do not confuse beating a FP16 baseline with beating Q4_K_M.

## Evidence and interpretation

New roots: `data/experiments/llama_tensorized_cotenancy/20261010_<stage>_v1`.
Snapshot and hash source, protocol, model inputs, fixtures and environment;
write strict atomic JSON and checkpoints with digests. Register and read back
the completed conservative result, and commit only this study's durable files.
Seeds are evaluation inputs, not independently trained model replicates.

Possible outcomes: packing execution invalid; quality assay invalid; packing
works but tensorization loses quality; quality retained but no practical win;
or practical co-tenancy supported under these fixed conditions. Failed direct
factorizations do not disprove learned/calibrated tensor networks, other ranks,
or different hardware. A projection result cannot rescue a full-model failure.
