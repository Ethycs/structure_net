# Local judge recovery

Date: 2026-10-22. EXPLORATORY infrastructure/qualification study
`ncm-local-judge-qualification-v1`. Full NCM training continues separately.
Existing CLI-backed judge calls time out. Direct Anthropic requests return
insufficient-credit errors; advertised Qwen/Phi routes report missing Triton
models. None of these errors is a semantic calibration failure or model accuracy.

Prepare an isolated local judge using the official `Qwen/Qwen3-8B-GGUF` Q6_K
weights. Read current Hugging Face metadata, pin its commit and LFS SHA, download
under `data/experiments/ncm_local_judge`, and verify every byte. No global cache,
router, billing or resident Triton service changes. The official model card
supports llama.cpp: https://huggingface.co/Qwen/Qwen3-8B-GGUF.

Build a server from the existing vendored llama.cpp source using a copied build
tree under the new study. Preserve original libraries, binaries and sources;
verify their hashes before/after preparation. Reconfigure only the copied tree.
Use the existing CUDA architecture86 for physical GPU0. Building/downloading
can overlap training; local GPU judge inference starts only after the NCM
training worker finishes. Bind a separate loopback-only port, use the fixed
Q6_K model and FP16 KV, and disable thinking for structured grading. No external
publication or model-service reconfiguration.

Before grading outputs, run real serial and intended-concurrency memory pilots
on physical GPU0 with all resident services preserved. Start with one request
and context4096, then test two concurrent requests/context8192 only if memory
fits. Stop on OOM, nonfinite output, model/hash/alias mismatch or malformed
results after3infrastructure/schema attempts. No CPU fallback for GPU inference.
Retain raw JSON, exact command, binary/library/model hashes and environment.

Qualification uses the unchanged48source-reviewed positive/negative controls,
verified against public LongMemEval spans and the same semantic rubric/parser.
Require24/24positive and24/24negative decisions, complete valid grading. Never
retry a valid miss. This is a model qualification test, not task accuracy or
proof of human agreement. If it fails, report failure and do not use this judge
to declare95%. If it passes, its results are explicitly attributed to a new
judge; do not overwrite earlier judgments or silently substitute it into an
older confirmatory endpoint. Any final-test protocol receives its own declared
judge identities before opening the sealed256cases. No speed claim here.
