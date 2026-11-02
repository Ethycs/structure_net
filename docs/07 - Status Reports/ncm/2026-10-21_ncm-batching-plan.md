# Shared-weight NCM batching, separate from independent replicas

Exploratory hypothesis `llama32-3b-ncm-batching-v1`, UNDERPOWERED. Known before
batch timings: literal1/2/9-copy conditions completed about76.5/77.9/78.3
aggregate common tokens/s, with exact output parity. This does not test a
single shared model processing a batch. The factorization study is queued;
do not overlap GPU work with either preceding experiment.

Freeze the original selected NCM/codec and inference precision/sampling. Add
only a batch dimension to the established full9-level CUDA graph, decoding
all outputs. Pilot batch9 on nine development sources in A/reversed/A waves;
require all27 native48-ID/text comparisons to original batch1 generation,
fresh-input reset, finite outputs and memory below Llama's6794772480-byte peak
reserved budget. Stop on pilot failure, preserve failures, amend code only
under a new root. No training or model selection.

Then test fixed batch sizes1,4,9, each a fresh process and graph. One excluded
development warmup wave, then all100 parent used runtime-validation sources,
three repetitions. Last partial waves duplicate the first source to fill the
static batch; those extra computations remain inside timed wall time but
their outputs never contribute request/token counts. Include fresh CPU source
IDs, device input copy, all generation, CPU output copy and text decoding;
exclude loading/capture, token counting and evidence writes. Count actual
non-EOS output text with the common Llama tokenizer, never latent codes or
padding. Record raw per-wave times and per-source outputs, memory and setup.

Primary throughput is actual text tokens divided by sum of timed wave wall
times; report each repetition, wave latency, aggregate and effective per-slot
rates. All900 actual request outputs must equal the original saved native IDs
and text to inherit its100/100 source fidelity. This is runtime validation on
used synthetic sources, not a fresh quality holdout. Operational success is
>=2x fresh batch1 aggregate throughput within the original Llama reserved
budget, with complete parity. Report all conditions, not just the maximum.
Original Llama77.21 tokens/s is an explicitly reused single-request secondary
control; its own batching was not optimized/tested here. No claim of optimal
equal-memory serving efficiency, improved single-request latency, general
capability, production95%, mobile or fullapplication context.

Use append-only `data/experiments/ncm_batching/20261021_*`, content-hash sources,
checkpoint and source identities, CPU shape/tail-count tests, separate CUDA
pilot, aggregate/audit/report and conservative registry/readback. Keep both
previous study decisions. Optional factorized batching is unexecuted unless
separately specified before its timings; this study measures dense NCM only.
