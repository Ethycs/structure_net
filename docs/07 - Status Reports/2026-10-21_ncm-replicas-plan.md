# NCM replicas under the Llama memory budget

Exploratory hypothesis `llama32-3b-ncm-replicas-v1`, written before replica
timings. Parent studies are completed and unchanged. Known: selected NCM graph
81.89 common Llama output tokens/s, 713294288 runtime model bytes; original
Llama 77.21 tokens/s, 6425499904 model bytes, 6667318272 peak allocated and
6794772480 peak reserved bytes. Nine NCM weight copies fit Llama's model-state
budget; working memory may change that answer.

Use literal independent selected NCM/codec copies on physical GPU0 RTX3060,
one Python process, one CUDA serving stream and graph per copy. No weight,
content, or KV sharing across copies/requests. Same FP32 state, BF16 autocast,
512-source/48-output shape, nine-level generation and native sampling law.
Weights SHA-256 generator ff7133b2de50bc1f66ef1e490fec4c723e0fc5a5c1957107209e923b198c3e3c,
codec b87f7288edfc0eb1f5b9122d1139cc2e61aa2331be7d6fe53f0582993086ede2.

First test CPU schedule/aggregation contracts and a separate nine-copy CUDA
systems pilot, with independent-storage equality and concurrent/single-stream
native-output parity on development sources, including repeated A/B/A inputs.
No training/save-reload is needed: frozen parent checkpoints reload strictly.
Stop on OOM, nonfinite output, aliasing, or pilot parity failure. Preserve
failed roots; implementation corrections need new roots/amendments.

After pilot passes, isolated conditions run in fixed order 1,9,2,8,4,7 copies,
each in a fresh process. All100 parent runtime-validation sources are served
once per repetition, three repetitions, in waves of up to N distinct sources.
Final partial waves contain no duplicate/padded requests. Each condition has
one excluded development-source warmup wave using all copies. Queue each
prefix copy and graph on its own stream before synchronizing the device and
decoding every output. Wall time includes CPU IDs to decoded CPU text, excludes
loading, capture, token counting, artifact I/O and warmup. Record each wave and
request, all48 native IDs, source IDs/hashes, request-equivalent wave latency,
sum of actual common Llama-token counts, and allocated/reserved device memory.

Primary throughput is total actual output tokens divided by sum of all timed
wave durations, with per-repetition rates, not N times a single-copy estimate.
Per-copy effective throughput is aggregate/N; it is not single-request speed.
Require all1800 request outputs equal the parent's native IDs and text, no
failed/missing cells, and no weight changes. Quality is inherited from the
used synthetic source review by exact parity, not a new holdout. Report all
conditions, model-state budget eligibility, and separate full-runtime peak
allocated/reserved eligibility against original Llama. Original Llama timing
and memory are explicitly reused sequential controls, not a fresh matched
multi-request Llama serving benchmark. No claim of optimal equal-memory Llama
batching. Directional engineering prediction: replicas improve aggregate
throughput, but linear9x scaling is not assumed. Operational throughput goal
is >=2x the reused single-request Llama rate, under its peak-reserved budget
with complete parity; otherwise retain the narrower measured result.

One frozen model, technical repeats, UNDERPOWERED, confirmed:false. No mobile,
general capability, full6300/128 range, or production quality claim. New roots
`data/experiments/ncm_replicas/20261021_*`; freeze producers, preserve all raw
evidence, audit, report, register/read back, and commit source/tests/docs.
