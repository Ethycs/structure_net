# Tensor student runtime follow-up

Hypothesis: `llama32-3b-tensor-runtime-v1`. Status: prospective exploratory
systems study, one fixed trained checkpoint; not independent training seeds.

The completed noise continuation recovered synthetic summarization quality but
its Python prototype remained slower than native Q4 plus prompt lookup. Test
whether execution changes alone improve complete-request latency and throughput.
No training, weight perturbation, quantization or cross-request KV reuse occurs.
Use all fourteen SHA-verified selected groups from
`data/experiments/llama_neighbourhood_continuation/20261014_noise_global_v2`.

Development compares the previous HF/512-token prefill implementation with:
full-prompt HF prefill plus fused FlashAttention KV updates, the same decode
captured in CUDA graphs, and those graphs with eight-token prompt lookup.
Lookup proposes only tokens already in the current prompt. Every proposed
token is verified causally by the complete student; rejected KV entries are
excluded by rolling back the logical cache length. Only cache allocations and
captured executable graphs persist between requests, never evidence KV.
Fixed-size verification pads to nine inputs; causal masking excludes padding
from earlier positions. All weights, tokenizer, greedy decoding, EOS handling,
caps and source prompts are fixed. Custom kernels may differ in FP16 rounding;
task quality, rather than bitwise equality with HF, is the requirement.

Before full execution: CPU tests of lookup, rejection/rollback, EOS/caps and
quality cardinality; CUDA longest-prompt pilot; teacher-forced numerical check
(finite logits, max absolute difference <=0.25, cosine >=0.999); captured versus
uncaptured custom decode and repeated-request output equality. Failures remain
recorded and require a new implementation root. Physical GPU0 only, one worker.

Known adaptive development seeds 22000–22007 are used solely for runtime choice.
Three timed repetitions per condition and prompt, after one excluded warmup
per condition on the longest development request. All eight complete source
reviews must pass to qualify a candidate. Choose the eligible candidate with
lowest summed per-prompt median latency; ties prefer graphs without lookup,
then eager. No eligible candidate stops the final stage.

Fresh final seeds 24000–24019 are generated prospectively by the frozen parent
generator and instruction template: five each of raw ~500, raw ~1980, eight
fragments ~6220 and hierarchy ~1000 tokens. No seed/outcome filtering. Old
23000–23019 final cases are now used evidence and excluded. The selected runtime,
old student runtime and pinned native Q4_K_M plus lookup8 each run five timed
fresh requests per case after one excluded longest-request warmup. Student
arms alternate order within each case/repetition. Native baseline runs separately
on the same isolated GPU; disclose the runtime/block-order difference.

Primary quality: whole-request source review, exactly one labeled line per
fragment, approved owner/budget/deadline/project attribution, no unsupported or
superseded facts, <=128 tokens per summary, EOS before raw768/hierarchy512 cap.
Assistant reviews the raw source and output; deterministic checks assist but
there is no independent LLM judge or production data. Require >=19/20 and no
fewer passes than either baseline. All five repeated outputs must agree within
each arm/case, otherwise review every distinct output and use worst-case quality.

Primary speed: sum of twenty per-case median complete-request seconds (input
on CPU through final output IDs on CPU, excludes model load, graph capture and
warmup), old/native divided by candidate. Also report per-family ratios and
total output tokens divided by summed median seconds; varying output lengths
cannot alone establish useful speed. Success requires quality gate, >=2x native
Q4 whole-request speed and >=2x its end-to-end tokens/s. Partial success means
quality gate with >=2x old prototype speed but unmet native target. Otherwise
report the measured quality or speed failure without promoting secondary gains.
Repeat timings are technical repetitions, not training replicates. No phone,
two-model co-tenancy, application accuracy or general summarization claim.

Artifact root: `data/experiments/llama_tensor_runtime/20261014_*_v1`.
Freeze source/config/checkpoint/fixture hashes and preserve source snapshots.
Stop on nonfinite output, provenance mismatch, OOM or invalid cache controls;
do not retry a quality/latency miss. Follow the master guide through audit,
report, conservative registration/readback and a commit of this study only.
