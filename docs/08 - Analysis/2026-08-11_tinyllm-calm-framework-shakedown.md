# TinyLLM CALM framework shakedown

**Status:** FRAMEWORK LIFECYCLE VALIDATED; NO QUALITY CLAIM  
**Date:** 2026-08-11  
**Conformance:** NAL-STD-EXPERIMENT `SHAKEDOWN`  
**Hypothesis:** `tinyllm-calm-framework-lifecycle-v1`  
**Preregistration:** [`../07 - Status Reports/2026-08-11_tinyllm-calm-framework-shakedown-preregistration.md`](../07%20-%20Status%20Reports/2026-08-11_tinyllm-calm-framework-shakedown-preregistration.md)

## Verdict

Structure Net can express and persist the central CALM computation. Both the
CPU run and a representative RTX 3060 run copied every source TinyLLM backbone
tensor exactly, executed real autoencoder and energy-model optimizer steps,
kept the autoencoder byte-identical during the frozen energy stage, generated
one complete in-vocabulary chunk, and restored state and fixed-noise outputs
exactly.

This result validates an engineering path, not an LLM conversion result. The
toy autoencoder reached only `3.6%--6.3%` token reconstruction after 32 steps,
and no BrierLM, perplexity, downstream task, throughput, or human generation
quality was measured.

## Campaign integrity

| Run | Requested | Completed | Failed | Reused | Lifecycle gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, seed 7 | 1 | 1 | 0 | 0 | 7/7 |
| RTX 3060, seed 7 | 1 | 1 | 0 | 0 | 7/7 |

Both runs used the preregistered configuration: a 2-layer, 2-head, width-32
GPT-2 backbone; 32-token synthetic vocabulary; `K=2`; latent width 8; 4 model
samples and 8 target-posterior samples per energy score; 4 source-model, 32
autoencoder, and 16 energy updates. Every producing source file is content
hashed in each `result.json` because the exercise ran from an intentionally
uncommitted implementation tree.

The first CPU/CUDA shakedowns exposed a component-level seed-ownership issue:
new one-dimensional biases followed the globally seeded PyTorch stream rather
than the component's own initialization rule. Those internally restorable
artifacts were preserved, the preregistration received a post-outcome
amendment, and the corrected implementation was rerun into new `v2` roots.
Only the `v2` results below enter the verdict. A focused regression test now
constructs the same component under different ambient RNG states and requires
every tensor to remain identical.

## Lifecycle gates

| Gate | CPU | CUDA |
| --- | --- | --- |
| source backbone tensors copied exactly | pass | pass |
| finite nonzero autoencoder gradients | pass | pass |
| finite nonzero energy-model gradients | pass | pass |
| frozen autoencoder state unchanged | pass | pass |
| exactly K in-vocabulary tokens generated | pass | pass |
| checkpoint state digest exact | pass | pass |
| fixed-noise output exact after restore | pass | pass |

The CPU and CUDA checkpoints each restored with zero loss delta and zero
maximum latent-prediction delta.

## Diagnostic measurements

These values diagnose execution only. The run was not powered or designed to
test improvement.

| Device | AE reconstruction CE, before -> after | token accuracy, before -> after | exact chunks after | energy loss, before -> after |
| --- | ---: | ---: | ---: | ---: |
| CPU | `3.4686 -> 3.5754` | `.0313 -> .0365` | `.0000` | `10.3536 -> 7.4704` |
| RTX 3060 | `3.4686 -> 3.2137` | `.0313 -> .0625` | `.0104` | `9.9126 -> 8.1886` |

The same seeded CPU and CUDA optimization paths diverged numerically, as is
normal across device kernels. Cross-device identity was not a gate. Each
checkpoint is exact on its own restoration path.

## Architecture and resource result

| Component | Parameters |
| --- | ---: |
| copied TinyLLM backbone | 26,880 |
| robust chunk autoencoder | 23,184 |
| K-token input adapter | 6,304 |
| stochastic energy head | 8,168 |
| total | 64,536 |

The CUDA run used PyTorch `2.5.1+cu121` on an NVIDIA GeForce RTX 3060 and
allocated a peak of 70,772,224 bytes (`67.5 MiB`). Its wall time is not a
benchmark: this graph is too small, includes checkpoint I/O and Python
bookkeeping, and had no isolated warm-up/repetition protocol.

## What the framework exercise establishes

- The paper's autoencoder, discrete chunk input, stochastic continuous head,
  and energy objective map cleanly to existing `BaseModel`, `BaseLayer`, and
  `BaseTrainer` contracts.
- Ordinary TinyLLM checkpoint tensors can seed the new backbone without a
  format translation.
- Optimizer ownership can enforce the two-stage boundary rather than relying
  on an informal freeze convention.
- The self-describing checkpoint contains enough graph information to restore
  deterministic fixed-noise behavior.

## What remains open

- Copying tensors does not preserve the source function: K tokens now occupy
  one residual position and the output factorization is different.
- High-fidelity autoencoding has not been demonstrated at TinyLLM vocabulary
  scale.
- Continued pretraining has not shown retention of natural-language quality.
- The paper's Llama backbone, BrierLM evaluator, low-temperature sampling, and
  throughput frontier have not been reproduced.
- The method has not been compared against multi-token prediction,
  speculative decoding, or a parameter/compute-matched TinyLLM baseline.

The next licensed experiment is a quality-retention pilot, not more systems
plumbing: use a real tokenizer and corpus, train a K=2 autoencoder to a frozen
reconstruction gate, and only then compare a checkpoint-initialized CALM arm
against matched continued TinyLLM training and a from-scratch CALM control.

## Artifacts and reproduction

Local artifacts:

```text
data/experiments/tinyllm_calm_framework_shakedown/20260811_cpu_seed7_v2/
data/experiments/tinyllm_calm_framework_shakedown/20260811_cuda_seed7_v2/
```

Each root contains `campaign_results.json`, a strict per-run `result.json`, and
a 280 KiB model checkpoint. The artifacts are under the repository's
DVC-managed `data/` tree, but the shared `data.dvc` manifest was not advanced:
`dvc status` also reported unrelated existing data changes, so folding the
entire tree into this exercise would have exceeded its scope.

Verification completed with `1,807 passed, 1 skipped` across the full
repository suite in 952.25 seconds. The 23 emitted warnings were existing
deprecation, snapshot-loading, and tensor-layout warnings; no test failed.

Reproduce:

```bash
pixi run pytest -q \
  tests/structure_net/components/test_calm_tinyllm_model.py \
  tests/structure_net/test_tinyllm_calm_framework_shakedown.py

pixi run python -m experiments.structure_net.tinyllm_calm_framework_shakedown \
  --device cpu \
  --output data/experiments/tinyllm_calm_framework_shakedown/20260811_cpu_seed7_v2

CUDA_VISIBLE_DEVICES=0 pixi run python \
  -m experiments.structure_net.tinyllm_calm_framework_shakedown \
  --device cuda:0 \
  --output data/experiments/tinyllm_calm_framework_shakedown/20260811_cuda_seed7_v2
```
