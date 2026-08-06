# TinyLLM Adapter Acceptance Audit

**Status:** IMPLEMENTATION ACCEPTED; SCIENTIFIC CLAIM OPEN  
**Date:** 2026-08-04  
**Target:** `weiserlab/TinyLLM` at `74bf91e`, pinned `llm.c` at `d3ce154`  
**Architecture:** `../03 - Architecture/tinyllm-feedback-adapter.md`

## Outcome

The adapter's systems claims are now backed by executable tests and a matched-control shakedown. No experiment in this acceptance pass establishes that delayed feedback improves language modeling, sensing accuracy, parameter efficiency, or edge inference.

## Acceptance matrix

| Cheque | Status | Executable evidence |
| --- | --- | --- |
| TinyLLM `d6/d8/d10/d11/d12` shapes | Cashed | preset and `C = 64L` assertions |
| GPT-2 causal baseline and tied head | Cashed | independent upstream-math equivalence, prefix-causality, initialization, and parameter-identity tests |
| Hugging Face weight translation | Cashed | explicit Conv1D shape/transposition checks, exact state/logit round trip, and feedback export refusal |
| Native `llm.c` v3 FP32 loading | Cashed | synthetic upstream-order checkpoint with exact logits |
| Native `llm.c` v5 BF16 loading | Cashed | bit-preserving BF16 tensor comparison |
| Persistent sparse masks | Cashed | checkpoint mask round trip and zero masked gradients |
| Delayed backward execution | Cashed | zero-gate identity, nonzero-gate behavioral change, and multi-pass token-causality test |
| Deterministic random placement | Cashed | same-seed topology/mask equality |
| Growth after optimizer creation | Cashed | optimizer group registration and updated patch weights |
| Self-describing feedback checkpoints | Cashed | safe topology reconstruction with exact refined logits |
| End-to-end training lifecycle | Cashed | causal trainer plus byte-reproducible three-arm shakedown, component-driven growth, and exact restored logits |
| Stock GGUF/`llama.cpp` feedback export | Explicitly unsupported | export raises instead of dropping behavior |
| Real-task quality improvement | Open | requires the multi-seed controlled experiment below |
| Edge efficiency improvement | Open | masked dense tensors and refinement currently add compute |

## Upstream reconciliation

The pinned `llm.c` source confirms:

- descriptors `d6`, `d8`, `d10`, `d11`, and `d12` map to widths 384, 512, 640, 704, and 768;
- vocabulary size is 50,257 and CUDA storage pads it to 50,304;
- checkpoint magic is `20240326`, with version 3 for FP32 and version 5 for BF16;
- the parameter stream contains 16 tensor groups in the order implemented by `from_llmc_checkpoint`;
- GPT-2 Conv1D projections require transposition at the Hugging Face boundary.

The audit also found and repaired a framework-wide collision: `BaseComponent._version` overwrote PyTorch's integer `nn.Module._version`. This made safe state-dict metadata contain a custom `ComponentVersion` object. Component semantic versions now use `_component_version`, and safe checkpoint loading is regression-tested.

## Verification record

The repository gate completed with **319 passed, 0 failed** and 23 warnings in
397.42 seconds. The focused TinyLLM acceptance gate completed with **36 passed,
0 failed** and 9 warnings in 3.16 seconds. `git diff --check` was clean. The
canonical CPU shakedown also completed all three arms and restored each
topology-aware checkpoint with exact logits. The same runner supports `cuda`
and indexed devices such as `cuda:2`; deterministic reproducibility is scoped
to a fixed software and hardware stack:

```bash
pixi run python experiments/structure_net/tinyllm_feedback_shakedown.py \
  --steps 8 --seed 7 --output /tmp/structure-net-tinyllm-shakedown
```

| Arm | Initial loss | Final loss | Parameters | Active feedback connections |
| --- | ---: | ---: | ---: | ---: |
| baseline | 4.204499 | 3.081642 | 40,608 | 0 |
| recompute control | 4.204499 | 3.081642 | 40,608 | 0 |
| random feedback | 4.204499 | 3.081623 | 41,122 | 128 |

The recompute control matched the baseline exactly before and after training.
The feedback arm's final-loss delta versus baseline was `-0.00001884`. That
number is deliberately **not** treated as evidence of an improvement: it is a
single seed on a tiny synthetic arithmetic task. The task confirms execution,
growth, training, persistence, and controlled initialization only. Timing is
excluded because sequential arm order and runtime warm-up would make it
misleading. Two independent processes produced byte-identical `results.json`
files with SHA-256
`7be50fa8c3462cface0bd79e12d6ae9796a95f90e46e436943f7fe6210b63dc1`.
The CPU artifact above is byte-reproducible across independent processes. CUDA
runs use deterministic PyTorch/cuDNN settings and preserve CUDA RNG state; their
reproducibility contract is limited to the same GPU architecture, driver, CUDA,
and PyTorch stack.

On 2026-08-05, two independent runs completed on a PCI-pinned NVIDIA GeForce
RTX 2060 SUPER using driver 575.57.08 and PyTorch 2.5.1+cu121. Both passed the
matched-initialization and exact-checkpoint controls and produced byte-identical
result files with SHA-256
`dd978d593d1f653dc9169e716e03bf0997025afef506585286ce9d985cf673f0`.

## Scientific acceptance still required

Run at least five seeds on a real TinyLLM task with identical initial checkpoint, data order, training tokens, and added stored-parameter budget across continuation-only, recompute-only, random forward adapter, random feedback, analysis-selected feedback, and parameter-matched baseline growth. Report task loss/accuracy/F1, stored and active parameters, peak memory, latency, and tokens per second. Selection uses train/validation data only; the final test split remains untouched.
