# FP16 tensor core with a Q4 residual

Hypothesis: `llama32-3b-fp16-core-q4-tail-v1`. Prospective exploratory mixed
precision study, one frozen trained model, no new optimizer updates.

Interpret the requested “FP16 tensors and Q4 noise around it” as a fixed
quantized residual around a smaller high-precision tensor core. Gaussian
training noise remains part of the parent checkpoint's history; no random
noise is injected at inference. Preserve that trained student's effective
weights, rather than correcting back toward the original pretrained teacher.

For each trained rank-1024 projection `W*=L R`, take the first `r` existing
factor components as the FP16 core and quantize the remaining components:

```
core = L[:, :r] R[:r, :]
tail = L[:, r:] R[r:, :]
output(x) = FP16-core(x) + Q4-tail(x)
```

Test `r=64` and `r=128` against the unchanged rank-1024 FP16 student. This is
not a random reinitialization or a lossless representation. It uses the
existing ordered SVD-derived, subsequently trained factor basis. Tail
construction uses FP32 products with TF32 disabled. Signed uniform Q4 weights
use one FP16 max-absolute scale per output channel and 64 input channels;
packed pairs occupy one byte. Activations, attention, embedding/output weights,
normalization weights, residual additions and KV remain FP16, with their
existing higher-precision accumulation. Q4 storage is not GGUF Q4_K_M.

The residual matrix is dense, so it can add arithmetic despite reducing stored
bytes. A fused Triton kernel dequantizes tiles into FP16 registers for tensor
core multiplication, with FP32 accumulation. It must execute packed weights
directly at both one-token decode and nine-position verification; do not keep
a dense FP16 residual hidden in a runtime cache.

Before model outcomes: CPU packing, signed decode, zero-range, shape/precision
and state reload tests; real CUDA full-shape numerical/performance operator
pilot for M=1,9,512 against multiplication by the same dequantized FP16 weights.
Require finite outputs, relative RMS error <=0.002 and cosine >=0.9999 for the
operator implementation. Serialized physical byte counts must include scales
and unique tied storage. Full-size longest-prompt pilot must pass graph/eager
and fresh-request reset controls. Physical GPU0 only, one process at a time.
Separate compilation and model loading from steady-state timing.

Development reuses known adaptive seeds 22000–22007, three timed repetitions
per request after excluded per-case warmup. Each candidate requires all eight
source reviews, correct labeled summary cardinality, no unsupported or
superseded facts, <=128 tokens per summary and EOS within raw768/hierarchy512.
Select the eligible candidate with lowest summed median complete-request
latency; tie smaller byte count, then rank64. No eligible candidate ends the
study without a new final test. A quality miss is not a retry opportunity.

If a candidate passes, compare it with optimized FP16 and pinned native Q4 plus
lookup8 on fresh seeds 25000–25019, five requests from each existing family:
raw ~500, raw ~1980, eight fragments ~6220 and hierarchy ~1000 prompt tokens.
Generate before development outcomes; no filtering. Earlier 23000 and 24000
sets are used evidence, not new final tests. Run five technical repetitions
per case with per-case excluded warmup. Student conditions run as separate
isolated blocks because memory does not allow all models resident together;
disclose block order. Native uses its existing short warmup plus five excluded
longest-request cells, as in the prior runtime study.

Primary quality: >=19/20 strict source-reviewed requests, no fewer than optimized
FP16. Source-factual scores separated from formatting may be reported as an
exploratory diagnostic. Judge is Codex source review plus deterministic checks;
there is no independent model or production data. Repeated distinct outputs
require separate reviews; worst-case decision per request.

Primary size: full serialized unique model weight/buffer payload at least 10%
below the 4,135,409,664-byte FP16 student. Include Q4 metadata, no hidden dense
tail; report runtime peak memory separately. Primary speed: complete-request
median-latency sum and end-to-end output tokens/s (exclude EOS), matching input
limits/caps/greedy decoding, >=1.05x FP16 in both to call it smaller and faster.
The broader 2x native Q4 target is a separate unmet-until-measured endpoint.
Do not use longer outputs alone to claim a speed gain.

Classifications: development quality failed; final quality failed; smaller
and faster with proxy quality retained; smaller with speed unproven/unmet; or
no useful compression. Mark all exploratory/underpowered; one trained model,
technical timing repetitions, no mobile/energy/co-tenancy/application claim.
Stop on implementation/provenance errors or OOM, preserve failed roots and
use a new producer version for an actual correction. Save packed checkpoints,
hash source/config/inputs, audit, report, register/read back and commit only
this study. Root: `data/experiments/llama_fp16_core_q4_tail/20261014_*_v1`.
