# Runtime shakedown correction

The initial campaign `20261010_campaign_v1` stopped at its first single-model
GGUF pilot, before model loading or quality evaluation. The process exited with
SIGILL (-4). Its runtime was built with AVX2, FMA and BMI2 enabled, while this
host is an Ivy Bridge Xeon E5-2697 v2 with AVX/F16C but none of those three
extensions. Preserve that failed pilot, its CMake configuration and CPU library.

Rebuild the same pinned llama.cpp commit with `GGML_AVX2=OFF`, `GGML_FMA=OFF`
and `GGML_BMI2=OFF`. CUDA architecture 86, full GPU offload, FP16 KV caches,
two host threads per physical model, model/data/quality gates and tensorizations
are unchanged. Successful runtime producers additionally snapshot the linked
llama/ggml libraries. Restart under a new campaign root, with fresh GGUF pilots.

Reuse the already completed full-model HF/CUDA control and five-rank projection
diagnostic by artifact path and SHA-256; neither depends on the CPU llama.cpp
backend. At this correction, the projection results are known (rank-215 weight
error 44.42%; packing CUDA speed 2.41×), but no full-model quality or co-tenancy
outcome has been inspected. No failed measurement is replaced or pooled with a
successful retry.

Implementation clarification: Python benchmark cells reserve the unused portion
of a 1024-token KV capacity in addition to their dynamic live caches. This keeps
the declared context budget comparable to each private llama.cpp context.
