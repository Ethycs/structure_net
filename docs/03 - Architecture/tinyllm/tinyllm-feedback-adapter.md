# TinyLLM Feedback Adapter

**Status:** EXPERIMENTAL  
**Date:** 2026-08-04  
**Applies to:** TinyLLM model, causal-LM trainer, feedback strategy/evolver, and shakedown runner  
**External target:** [`weiserlab/TinyLLM`](https://github.com/weiserlab/TinyLLM), pinned `llm.c` architecture at commit `d3ce15420e94ac84c92e6aeeb236ad5153bd7624`

Structure Net can now construct the GPT-2 family used by TinyLLM and grow explicitly delayed backward connections between transformer blocks. The baseline remains checkpoint-compatible; the feedback extension is a research graph and is not silently exported as ordinary GPT-2.

## Baseline compatibility

TinyLLM delegates pre-training to a pinned `llm.c` fork. Its published descriptors use the following shapes:

| Descriptor | Blocks | Heads | Embedding width | Approximate class |
| --- | ---: | ---: | ---: | ---: |
| `d6` | 6 | 6 | 384 | 30M |
| `d8` | 8 | 8 | 512 | 51M |
| `d10` | 10 | 10 | 640 | 82M |
| `d11` | 11 | 11 | 704 | 101–102M |
| `d12` | 12 | 12 | 768 | 124M |

The adapter preserves:

- learned token and positional embeddings;
- pre-layer-normalized causal attention and MLP blocks;
- GPT-2's tanh GELU;
- residual-projection initialization scaled by depth;
- tied token-embedding and language-model-head weights;
- Hugging Face GPT-2 projection transposition rules;
- TinyLLM/`llm.c` version-3 FP32 and version-5 BF16 checkpoint ordering.

`TinyLLMModel.from_llmc_checkpoint(...)` loads native pre-training checkpoints directly. Both FP32 and BF16 streams are reconstructed in tests using the pinned header and all 16 upstream parameter groups. `load_huggingface_state_dict(...)` consumes weights after TinyLLM's existing export step. A baseline model can export a Hugging Face-layout state dictionary; a model with feedback refuses that export because GPT-2 and `llama.cpp` cannot express its iterative execution.

Dynamic models use `save_checkpoint(...)` and `from_checkpoint(...)`. That envelope records configuration, feedback topology, refinement count, dtype, masks, gates, and weights so callers do not have to reconstruct modules before loading a raw `state_dict`. Loading uses PyTorch's `weights_only=True` safety boundary, and user metadata is restricted to JSON-compatible values. Raw state dictionaries remain supported only when the caller has already built an identical graph.

## Feedback execution

A backward edge would create an undefined cycle in an ordinary forward pass. The adapter therefore uses delayed refinement:

```text
pass 0: embeddings -> block 0 -> ... -> block j -> logits
                         ^                 |
                         |                 |
pass 1: embeddings ------- gated patch <--+
```

A patch reads block `source_block` from the previous pass, transforms it through `width` feedback neurons, and injects a correction before the earlier `target_block` on the next pass. Valid endpoints satisfy:

```text
0 <= target_block < source_block < number of blocks
```

The gate is bounded with `tanh`. Its default value is zero, so adding a patch initially preserves the baseline function exactly. Set a small non-zero gate for immediate gradients through the patch weights, or allow the gate to learn first.

`connection_density` controls persistent read/write masks. It means active fraction, not sparsity: `0.05` selects approximately 5% of channels on each side of every feedback neuron. Masked dense tensors still consume their full stored parameter memory and use dense kernels. Reports therefore keep `feedback_parameters` separate from `active_connections`; neither is presented as compressed inference cost.

Feedback is causal across tokens. A later transformer block at token position `t` has only attended to positions `<= t`, and the feedback patch is position-wise. Refinement does increase inference work and currently requires recomputing the prefix; it is not compatible with the standard GPT-2 KV-cache path.

## Component composition

```text
RandomFeedbackGrowthStrategy
        | plans.feedback_growth
        v
FeedbackGrowthEvolver ------> optimizer.add_param_group(new patch parameters)
        |
        v
TinyLLMModel.add_feedback_connection(...)
        |
        v
CausalLanguageModelTrainer.train_step(...)
```

- `TinyLLMModel` owns tensor execution, checkpoint translation, topology, and architecture reporting.
- `RandomFeedbackGrowthStrategy` emits a seeded random-placement control plan.
- `FeedbackGrowthEvolver` mutates the model and registers new parameters with an optimizer that already exists.
- `CausalLanguageModelTrainer` supplies next-token batch shifting, gradient clipping, optimizer ownership, and online-growth compatibility.
- NAL remains responsible for real datasets, multi-seed campaigns, and evidence aggregation.

Example:

```python
from structure_net.components.models import create_tinyllm_model

model = create_tinyllm_model(
    "d6",
    feedback_connections=2,
    feedback_width=16,
    feedback_connection_density=0.05,
    feedback_seed=7,
    initial_feedback_gate=1e-3,
)

logits, loss = model(input_ids, targets, refinement_steps=1)
print(model.get_architecture_summary())
model.save_checkpoint("feedback-model.pt")
```

## Runnable systems shakedown

The canonical local runner executes matched baseline, recompute-only, and random-feedback arms from identical initial weights and token batches. Random growth travels through the real strategy, evolver, trainer-owned optimizer, and model interfaces:

```bash
pixi run python experiments/structure_net/tinyllm_feedback_shakedown.py \
    --steps 8 \
    --seed 7 \
    --output data/experiments/tinyllm_feedback_shakedown
```

For a host with several busy GPUs, pin one physical device and address the
single visible card as `cuda:0`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python experiments/structure_net/tinyllm_feedback_shakedown.py \
    --device cuda:0 --steps 8 --seed 7 \
    --output data/experiments/tinyllm_feedback_shakedown_cuda
```

It writes a versioned `results.json` plus restorable weights for every arm. On a fixed CPU or CUDA software/hardware stack, the result JSON is reproducible for a fixed configuration; the test suite runs the CPU path twice and compares the artifacts. CUDA devices are selected with `--device cuda` or `--device cuda:N`. Every checkpoint is reloaded and required to reproduce logits and topology exactly. Its arithmetic-token task validates construction, training, dynamic growth, measurement, and checkpoint restoration. The result declares `systems_lifecycle_only_not_quality_evidence`; it must not enter the meta-hypothesis system as evidence that feedback improves real tasks. Timing is deliberately excluded because this sequential smoke run is not a controlled benchmark.

For several seeds, use `tinyllm_feedback_nal_campaign.py`. One seed is one NAL
job containing all three sequential matched arms; independent seed jobs may
share a GPU through fixed or memory-calibrated slots. Completed seeds have
fingerprinted resume records. Shared-run wall times are excluded from benchmark
claims; `--isolated-timing` forces one job at a time.

## Explicit boundaries

| Boundary | Status |
| --- | --- |
| TinyLLM `llm.c` v3/v5 baseline import | Supported |
| Hugging Face GPT-2 state-dictionary translation | Supported without requiring `transformers` at runtime |
| Hugging Face model directory/tokenizer creation | Not implemented; use TinyLLM's exporter for baseline models |
| Feedback-aware Structure Net checkpoint | Supported; model graph and tensors only |
| Completed-seed campaign resume | Supported through NAL's fingerprinted result ledger |
| Mid-training resume | Not implemented; optimizer, scheduler, dataloader, scaler, and RNG state are not stored |
| GPT-2/GGUF/`llama.cpp` feedback export | Unsupported and rejected rather than silently flattened |
| KV-cached feedback generation | Unsupported; refinement recomputes the prefix |
| Padding/attention-mask API | Unsupported; batches must follow TinyLLM's fixed causal token layout |
| Sparse storage or sparse kernels | Unsupported; masks constrain gradients and active connections but weights remain dense tensors |
| Fixed-stack deterministic shakedown | Verified on CPU and CUDA; portability across GPU/driver stacks is not claimed |
| Quality or edge-efficiency claim | Open; the synthetic shakedown is systems evidence only |

## Evidence boundary and first experiment

This implementation makes the hypothesis testable; it is not evidence that backward growth improves TinyLLM. The first useful comparison should use the same checkpoint, data order, optimizer steps, and added stored-parameter budget for:

1. continuation only;
2. random forward low-rank patches;
3. random delayed feedback patches;
4. analysis-selected delayed feedback patches;
5. a parameter-matched baseline widening or adapter.

Use at least five seeds. Select topology using training/validation data only, keep a final test split untouched, and report task accuracy, loss, active connections, stored parameters, refinement latency, peak memory, and tokens per second. TinyLLM's sensor tasks are the natural target, but a tiny vocabulary/config smoke test should precede any 30M run.

## Verification

Run:

```bash
pixi run pytest -q \
    tests/structure_net/components/test_tinyllm_model.py \
    tests/structure_net/test_tinyllm_shakedown.py
```

The focused suite covers preset mapping, tied weights, token causality, masked gradients, zero-gate identity, deterministic random placement, optimizer registration, topology-aware restoration, online training after growth, Hugging Face round trips, export refusal, and native `llm.c` FP32/BF16 checkpoint equivalence.
