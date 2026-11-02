# Local judge placement amendment

Date: 2026-10-22, before local model inference or qualification outcomes.
The official Qwen Q6_K file has passed its published SHA check. The isolated
CUDA86 server build completed with original vendor sources/binaries unchanged.
Native NCM training has completed; the frozen-Llama arm is running on GPU0.

Use a new CUDA75 build for physical GPU2 (RTX2060Super8GB) so evaluation can
overlap training without using GPU0. Fixed judge configuration: Qwen3-8B-Q6_K,
28transformer layers offloaded to GPU2, remaining layers/output head on CPU,
FP16 KV, context4096, one slot, four CPU threads. This explicit placement is part
of the experimental cell. Preserve the resident~2.6GB GPU2 service allocation.
Abort on OOM or CPU-only execution; do not change placement silently. No timing
or model-speed inference follows from this shared-resource configuration.

Prepare a separate build root with CUDA architecture75; preserve original
libraries and the completed CUDA86 build. After build and memory checks, start
an isolated loopback-only server at127.0.0.1:18081 with one request at a time.
Amend the local judge client concurrency from2 to1 for this fixed cell. Use
distinct source versions/roots, recording all binary/model/config hashes. The
same48-control qualification and24/24positive+24/24negative rule remain fixed.
No valid failure retries. No existing model service is changed.
