# Q4 residual kernel optimization

Prospective exploratory hypothesis `llama32-3b-q4-kernel-runtime-v1`.
Optimize execution of the fixed rank64 FP16-core/Q4-tail checkpoint whose
SHA256 is `8937d83adfa9c77e065f1c4918bea90a20ee9e1695ffc281561646e0f949a8ff`.
The parent study finished at 61.80 tokens/s versus optimized FP16's 80.45 on
used synthetic requests 25000–25019. Preserve all completed evidence. No new
training, quantizer changes or weight changes. Physical GPU0 only, one worker.

Test paired-nibble unpacking, exact FP16 bit decoding of signed Q4 values,
grouped M-tile scheduling, shape-independent M, and a fused FP16 core/output
epilogue. Keep FP16 core intermediates, tail outputs and output additions at
the same representation boundaries. Decode uses FP32 split-K partials with
one fused reduction/core/add epilogue. Precision variants are full FP32 tensor
accumulation and short FP16 tensor-core partial products converted into FP32
for accumulation across K tiles. The latter changes rounding and requires
explicit numerical and quality checks. An alternative prefill path may
transiently dequantize one weight matrix for cuBLAS; never retain a dense tail
in the module or across requests, and include that temporary in peak memory.

Before model outcomes, run CPU nibble/bit-decoding, configuration and storage
identity tests. A fixed finite operator tuning list is frozen in its producer.
Use seed51015, both MLP shapes (N8192/K3072 and N3072/K8192), rank64, and
M=1/9 for decode, M=512/1980/6300 for prefill. Compare full projection outputs
against the old packed projection and a dequantized FP16 reference. Require
finite values, relative RMS <=0.002 and cosine >=0.9999. Expected hardware
resource-limit failures may exclude a tile with the exception recorded;
implementation/numerical failures stop the run and preserve its root.
Use three event-timed repetitions of twenty CUDA-graph replays after warmup.
Select one decode layout and one prefill layout per precision family by the
smallest sum of median times across its shapes/widths; tie configuration order.
Record all tested layouts, controls, selected layouts, compilation and memory.
Recheck selected layouts at irregular M=513/6234 before model evaluation.

Compare up to three optimized execution families on the known adaptive eight
development requests 22000–22007: packed FP32, staged FP16/FP32, and transient
cuBLAS prefill (with the selected fused packed decode). First each must pass
the real longest-request cache, graph/eager and request-reset memory pilot.
Use three timed repetitions after excluded per-case warmup. Require 8/8 strict
source reviews, correct labels/cardinality, <=128 tokens per summary and EOS
within raw768/hierarchy512. Select lowest sum of per-case median full request
times among eligible families; ties packed FP32, staged, transient. If none
pass, stop. No quality-threshold retry or tuning after final outcomes.

Generate fresh final requests 26000–26019 before development outcomes, using
the existing four families, five each, no outcome filtering and no overlap
with earlier prompt hashes. If a candidate qualifies, compare it against the
unchanged old packed runtime and rank1024 FP16 runtime on these exact prompts,
five technical repetitions each. Run separate isolated blocks, selected then
old packed then FP16; exclude loading, compilation, graph setup and per-case
warmup. Record complete CPU-input/output request time, TTFT, output IDs,
non-EOS output tokens/s and both allocated/reserved GPU memory. Existing
lookup policy stays current-prompt ngrams1–4, maxdraft8, context8192.

Primary quality is >=19/20 strict source-reviewed requests and no fewer than
old packed. Judge: Codex reading source quotations with deterministic checks;
no independent judge or production data. Primary optimization success is
>=1.05x in both paired complete-request latency and output-token rate versus
old packed, with unchanged unique persistent weight payload (3,441,776,640
bytes, plus the existing 256-byte nonpersistent RoPE buffer). Report the same
speed ratios versus fresh FP16 separately; do not claim longer outputs alone
as acceleration. The broader 2x native target is unmeasured here.

Classifications: development quality failed; final quality failed; kernel
speed improved; speed target unmet. State whether FP16 speed was recovered.
One frozen trained model and technical repetitions make this underpowered and
exploratory. No mobile, production accuracy, energy or co-tenancy claim.
Stop on OOM/provenance/implementation failures, preserve each failed root and
use a new version for a real correction. Root:
`data/experiments/llama_q4_kernel/20261014_*`. Hash/snapshot producers, audit
all cells and physical state, report, register/read back and commit only this
study. Do not modify the parent's runtime modules or checkpoints.
