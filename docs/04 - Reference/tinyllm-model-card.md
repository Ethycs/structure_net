# TinyLLM Engineering Model Card

**Status:** EXPERIMENTAL LIBRARY COMPONENT  
**Date:** 2026-08-06  
**Implementation:** `structure_net.components.models.tinyllm_model`  
**Upstream family:** [`weiserlab/TinyLLM`](https://github.com/weiserlab/TinyLLM) / GPT-2  
**Pinned architecture reference:** TinyLLM's `llm.c` commit `d3ce15420e94ac84c92e6aeeb236ad5153bd7624`  
**Code license:** MIT; external checkpoints, tokenizers, and datasets retain their own terms

## Summary

Structure Net's TinyLLM component is a GPT-2-compatible model builder for the
30M--124M TinyLLM family. It supports native TinyLLM/`llm.c` checkpoint import,
Hugging Face GPT-2 state-dictionary interchange, causal language-model
training, intermediate-depth evaluation, and an optional research extension
for delayed feedback between transformer blocks.

This is an **architecture and tooling release, not a pretrained model
release**. Constructing `create_tinyllm_model("d6")` returns a randomly
initialized model unless the caller explicitly loads weights. The repository
does not supply a tokenizer, general-purpose instruction tuning, safety tuning,
or established language-model quality benchmarks.

Use the baseline when you need a compact, inspectable GPT-2-family model or
must consume TinyLLM checkpoints inside Structure Net. Treat delayed feedback,
continuous-depth analysis, and the sensor-front-end studies as research APIs,
not production capabilities.

## Model variants

All standard presets use a 50,257-token vocabulary and a maximum context of
1,024 tokens unless overridden. Parameter counts below are reported by the
current implementation and include tied token-embedding/output-head weights
only once.

| Preset | Aliases | Blocks | Heads | Width | Parameters |
| --- | --- | ---: | ---: | ---: | ---: |
| `d6` | `30m` | 6 | 6 | 384 | 30,339,456 |
| `d8` | `51m` | 8 | 8 | 512 | 51,475,968 |
| `d10` | `82m` | 10 | 10 | 640 | 82,056,320 |
| `d11` | `101m`, `102m` | 11 | 11 | 704 | 101,625,216 |
| `d12` | `124m`, `gpt2` | 12 | 12 | 768 | 124,439,808 |

The architecture uses learned token and position embeddings, pre-layer-normalized
causal self-attention and MLP blocks, GPT-2 tanh GELU, depth-scaled residual
projection initialization, a final layer norm, and tied token embedding and LM
head weights. Attention uses PyTorch causal scaled-dot-product attention.

## Intended uses

Suitable uses include:

- loading and studying compatible TinyLLM/`llm.c` checkpoints;
- compact causal-LM experiments where GPT-2 behavior is appropriate;
- controlled architecture studies using fixed initialization and explicit
  checkpoints;
- probing residual representations at integer or fractional transformer depth;
- research on delayed low-rank feedback with topology-aware persistence;
- synthetic sensor and representation-geometry experiments whose assumptions
  are explicitly tested.

The component is not presently validated for:

- production chat, instruction following, or autonomous agents;
- safety-critical, legal, medical, or financial decisions;
- multilingual or domain-specific quality claims;
- long-context use beyond the configured positional-embedding limit;
- stock GPT-2, GGUF, or `llama.cpp` deployment after feedback edges are added;
- efficient autoregressive generation with feedback, because there is no
  feedback-aware KV cache;
- sparse inference: feedback masks select active connections but storage and
  kernels remain dense.

## Quick start

### Construct a baseline

```python
import torch

from structure_net.components.models import create_tinyllm_model

model = create_tinyllm_model("d6").eval()
input_ids = torch.randint(0, model.config.vocab_size, (2, 32))

with torch.no_grad():
    logits, loss = model(input_ids)

assert logits.shape == (2, 1, 50_257)
assert loss is None
```

Without targets, the default output contains logits for only the final token.
Pass `return_full_logits=True` when every position is required.

### Compute a next-token loss

```python
tokens = torch.randint(0, model.config.vocab_size, (2, 33))
inputs = tokens[:, :-1]
targets = tokens[:, 1:]

logits, loss = model(inputs, targets)
loss.backward()
```

Targets must have the same shape as `input_ids`. Target value `-1` is ignored
by cross-entropy. The canonical `CausalLanguageModelTrainer` can perform this
shift automatically for an unlabelled `[batch, sequence]` token tensor. Its
defaults are AdamW with learning rate `3e-4`, zero weight decay, and gradient
norm clipping at `1.0`; choose training settings for the actual dataset rather
than assuming those defaults are a validated pretraining recipe.

### Load a Structure Net checkpoint

```python
from structure_net.components.models import TinyLLMModel

model = TinyLLMModel.from_checkpoint(
    "tinyllm.pt",
    map_location="cuda",
)
print(model.get_architecture_summary())
```

Save with `model.save_checkpoint(path, metadata={...})`. This self-describing
format preserves configuration, weights, feedback topology, masks, gates, and
refinement count. Metadata must be JSON-compatible. It does **not** preserve
optimizer, scheduler, dataloader, gradient-scaler, or RNG state, so it is not a
mid-training-resume format.

### Load a native TinyLLM checkpoint

```python
import torch

from structure_net.components.models import TinyLLMModel

model = TinyLLMModel.from_llmc_checkpoint(
    "gpt2_30M_bf16.bin",
    dtype=torch.bfloat16,
    device="cuda",
)
```

The importer supports native `llm.c` version-3 FP32 and version-5 BF16 model
checkpoints using the pinned TinyLLM tensor ordering. It handles CUDA's padded
vocabulary rows while exposing the declared vocabulary size.

## Input and output contract

`TinyLLMModel.forward` accepts:

```text
input_ids:              int32 or int64 [batch, sequence]
targets:                optional [batch, sequence]
refinement_steps:       optional non-negative integer
return_full_logits:     optional boolean
output_hidden_states:   optional boolean
return_dict:            optional boolean
```

`sequence` must not exceed `config.block_size`. The model does not expose a
padding or attention-mask argument; batch construction must use a fixed causal
token layout, with unwanted loss positions marked `-1` in `targets`.

The default tuple is `(logits, loss)`. With `return_dict=True`, the output is a
`TinyLLMOutput` containing logits, loss, optional per-block hidden states, and
the number of refinement steps. Hidden states and `capture_activations=True`
can retain large tensors; enable them only for analysis.

For a baseline model, `residual_at_depth(input_ids, depth)` and
`forward_at_depth(input_ids, depth)` evaluate an exact block prefix plus a
fractionally gated next residual block. Valid depths are from zero through the
number of blocks. These APIs intentionally reject models containing feedback.

## Checkpoint and deployment compatibility

| Path | Support | Engineering note |
| --- | --- | --- |
| TinyLLM/`llm.c` v3 FP32 import | Yes | Native model weights |
| TinyLLM/`llm.c` v5 BF16 import | Yes | Native model weights |
| Hugging Face GPT-2 state-dict import | Yes | Conv1D projections are transposed automatically |
| Hugging Face GPT-2 state-dict export | Baseline only | Returns a state dict, not a model directory or tokenizer |
| Structure Net self-describing checkpoint | Yes | Baseline and feedback graphs |
| Raw PyTorch `state_dict` | Conditional | Caller must first construct the exact graph |
| GGUF / `llama.cpp` export | Baseline through external TinyLLM tooling | Feedback export is rejected |
| Standard KV-cached generation | Baseline integration only | No generation wrapper is supplied here |
| Feedback-aware KV cache | No | Each refinement recomputes the prefix |
| Mid-training resume | No | Training state is not included |

For stock GPT-2 deployment, export only a baseline model. The implementation
raises an error instead of silently dropping feedback behavior.

## Experimental delayed feedback

The optional extension adds a low-rank connection from a later source block to
an earlier target block. The source state is read from the previous refinement
pass, so execution remains explicit and token-causal:

```python
model = create_tinyllm_model(
    "d6",
    feedback_connections=1,
    feedback_width=16,
    feedback_connection_density=0.05,
    feedback_seed=7,
    initial_feedback_gate=1e-3,
)

logits, _ = model(input_ids, refinement_steps=1)
```

Endpoints satisfy `0 <= target_block < source_block < n_layer`. The gate is
bounded by `tanh`; the default initial gate is zero and therefore initially
preserves the baseline function. Use a small nonzero gate if patch weights must
receive gradients immediately.

`feedback_connection_density` is the active fraction of persistent read/write
masks, not the fraction removed and not a compression ratio. Feedback adds
stored parameters and refinement compute even at low density. Current evidence
shows that the mechanism executes, trains, restores, and remains causal; it
does not show a real-task quality or edge-efficiency advantage.

## Evidence and measured behavior

### Engineering validation

The acceptance audit verified preset shapes, tied weights, causal behavior,
GPT-2 projection translation, native FP32/BF16 import, masked gradients,
zero-gate identity, deterministic topology placement, online optimizer growth,
topology-aware restoration, and explicit export refusal. At the recorded
2026-08-04 gate, the full repository suite passed 319 tests and the focused
TinyLLM suite passed 36 tests.

A three-arm CPU shakedown and two fixed-stack CUDA repetitions restored every
checkpoint with exact logits. The CUDA repetitions used an RTX 2060 SUPER,
driver 575.57.08, and PyTorch 2.5.1+cu121 and produced byte-identical result
files. This is reproducibility evidence for that fixed stack, not a portability
guarantee across devices, drivers, or PyTorch releases.

### Performance observations

The repository has not run a controlled general-language pretraining or
inference benchmark. In one d8 synthetic sensor campaign on an RTX 2060 SUPER,
shared-run bookkeeping reported roughly 10.5k--12.7k input tokens/s for raw
27-token contexts and 2.4k--2.8k input tokens/s for structured 3-token front
ends. Those numbers include different per-example work and concurrent campaign
conditions; do not use them for capacity planning or comparisons with other
models. Benchmark the intended sequence length, precision, batch size, and
software stack directly.

### Task-quality evidence

No general NLP quality, perplexity, instruction-following, toxicity, fairness,
or downstream benchmark is reported for this implementation. The completed
research campaigns use a narrow synthetic sensor task, not natural language.

The latest gauge-repaired I/O study found that calibrated analytic and learned
front ends met its joint internal-representation gate in 5/5 seeds, while a
declared target-side equivariant repair passed in 0/5. This result supports a
narrow identifiability and representation-geometry conclusion. It must not be
read as evidence that TinyLLM is broadly accurate, robust, invariant, or safer
than other language models.

## Limitations and risks

- **No bundled capability:** a fresh instance has random weights. Capability
  depends on the loaded checkpoint and its training data.
- **Tokenizer is external:** token IDs must follow the tokenizer used by the
  checkpoint; vocabulary size alone does not establish compatibility.
- **Short fixed context:** learned positions default to 1,024 and do not imply
  extrapolation beyond that limit.
- **No padding mask:** mixed-length batching requires careful preprocessing and
  loss masking.
- **No generation policy:** sampling, repetition control, stopping, and content
  safeguards are application responsibilities.
- **Checkpoint provenance matters:** validate licenses, data lineage, hashes,
  and compatibility for every externally obtained weight file.
- **Feedback changes execution semantics:** it prevents stock GPT-2 export and
  increases latency and memory use.
- **Training restoration is incomplete:** model checkpoints alone cannot resume
  a run bit-for-bit at an arbitrary optimizer step.
- **Research APIs may change:** the component contract is marked experimental.

The architecture inherits the usual risks of autoregressive language models,
including fabricated text, memorization, bias, prompt sensitivity, and unsafe
continuations. This repository has not measured those risks for any particular
TinyLLM checkpoint.

## Production-readiness checklist

Before using a TinyLLM model in an engineering system:

1. Record the exact preset, code revision, checkpoint hash, checkpoint format,
   tokenizer revision, license, and training-data provenance.
2. Verify checkpoint restoration and deterministic evaluation on the target
   hardware/software stack.
3. Evaluate task quality and failure modes on held-out, shifted, malformed, and
   adversarial inputs representative of production.
4. Benchmark latency, throughput, peak memory, and energy at the real batch
   size, context length, precision, and generation strategy.
5. Add application-level input validation, output constraints, monitoring,
   rollback, and human review appropriate to the risk.
6. Keep feedback disabled unless the system can run Structure Net's iterative
   graph and the deployment benefit has been measured against a matched
   baseline.
7. Store model and evaluation artifacts in the project's versioned data path;
   do not rely on an untracked local weights file.

## Verification

Run the focused implementation and lifecycle checks:

```bash
pixi run pytest -q \
    tests/structure_net/components/test_tinyllm_model.py \
    tests/structure_net/test_tinyllm_shakedown.py
```

Inspect a concrete model rather than relying on descriptor names:

```python
from structure_net.components.models import create_tinyllm_model

print(create_tinyllm_model("d6").get_architecture_summary())
```

## Related documentation

- [TinyLLM feedback adapter architecture](../03%20-%20Architecture/tinyllm-feedback-adapter.md)
- [TinyLLM adapter acceptance audit](../08%20-%20Analysis/2026-08-04_tinyllm-adapter-acceptance.md)
- [Latest gauge-repaired I/O correspondence report](../08%20-%20Analysis/2026-08-06_tinyllm-io-correspondence.md)
- [Developer guide](../02%20-%20Implementation/developer-guide.md)
- [Experiment and report authoring guide](../02%20-%20Implementation/experiment-and-report-authoring-guide.md)
