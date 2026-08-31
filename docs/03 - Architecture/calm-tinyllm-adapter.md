# CALM-style TinyLLM adapter

**Status:** EXPERIMENTAL  
**Date:** 2026-08-11  
**Implementation:** `structure_net.components.models.calm_tinyllm_model`  
**External method:** [Continuous Autoregressive Language Models](https://arxiv.org/abs/2510.27688)

## Purpose

The adapter expresses continuous autoregressive language modeling through
Structure Net's component contracts while retaining an inspectable
GPT-2-compatible TinyLLM backbone. It supports a two-stage training lifecycle,
initialization from an ordinary TinyLLM checkpoint, base-temperature chunk
generation, and self-describing checkpoint restoration.

It does not claim that copying an existing backbone preserves its language
modeling function. The chunk interface changes both its inputs and prediction
factorization, so a converted instance is a new model requiring continued
training.

## Component graph

```text
ordinary TinyLLM checkpoint
          |
          | exact tensor copy
          v
  TinyLLM transformer blocks <-----------------------------+
          ^                                                |
          |                                                |
previous K discrete tokens                                 |
          |                                                |
token embedding -> discrete patch adapter -> chunk position+
                                                           |
                                                           v
                                                stochastic energy head
                                                           |
                                                           v
                                                   predicted latent z
                                                           |
                         +---------------------------------+----------------+
                         |                                                  |
                         v                                                  v
               energy-score training                              frozen AE decoder
                         ^                                                  |
                         |                                                  v
next K tokens -> robust chunk AE encoder                         next K token IDs
                 -> target Gaussian
```

The model deliberately follows the paper's discrete feedback loop: a generated
latent is decoded immediately, and the resulting K discrete tokens form the
next transformer input. It does not feed the compact latent directly back into
the backbone.

## Components

| Component | Framework role | Contract boundary |
| --- | --- | --- |
| `RobustChunkAutoencoder` | `BaseModel` | K tokens to diagonal-Gaussian latent and back |
| `ChunkEncoder` / `ChunkDecoder` | `BaseLayer` | tied token weights, variational parameters, reconstruction logits |
| `PatchInputAdapter` | `BaseLayer` | K token embeddings to one GPT-2 residual position |
| `EnergyBasedGenerativeHead` | `BaseLayer` | context plus sampled noise to one latent prediction |
| `EnergyScoreObjective` | `BaseLayer` | sample-based proper scoring loss |
| `CALMTinyLLMModel` | `BaseModel` | conversion, continuous training, generation, checkpointing |
| `ContinuousAutoregressiveTrainer` | `BaseTrainer` | isolated `autoencoder` or `energy` optimizer phase |

The autoencoder trainer optimizes only autoencoder parameters. The energy
trainer freezes the autoencoder and excludes it from its optimizer. Separate
trainer objects make the phase boundary visible in optimizer ownership rather
than relying on a caller to maintain parameter groups correctly.

## Minimal construction

```python
from structure_net.components.models import (
    CALMTinyLLMConfig,
    CALMTinyLLMModel,
    TinyLLMModel,
)

source = TinyLLMModel.from_checkpoint("tinyllm.pt")
config = CALMTinyLLMConfig(
    backbone=source.config,
    patch_size=2,
    latent_size=64,
    autoencoder_hidden_size=source.config.n_embd,
)
model = CALMTinyLLMModel.from_tinyllm(source, config)
```

The source must be an ordinary TinyLLM without delayed feedback, and its
configuration must exactly match `config.backbone`. Matching autoencoder and
backbone embedding widths also permits a warm start of the autoencoder's tied
token table.

`config.backbone.block_size` counts chunk positions after conversion. The
maximum token context is therefore `block_size * patch_size`.

## Training

```python
from structure_net.components.trainers import ContinuousAutoregressiveTrainer
from structure_net.core import EvolutionContext

ae_trainer = ContinuousAutoregressiveTrainer(phase="autoencoder")
ae_trainer.train_step(model, token_batch, EvolutionContext(device="cuda:0"))

energy_trainer = ContinuousAutoregressiveTrainer(phase="energy")
energy_trainer.train_step(model, token_batch, EvolutionContext(device="cuda:0"))
```

Token batches must contain complete chunks. The energy model uses chunk `i` as
context for chunk `i+1`; at least two chunks are therefore required.

## Checkpoint contract

`save_checkpoint` preserves:

- complete nested configuration;
- backbone, adapter, generator, and autoencoder tensors;
- the autoencoder frozen state;
- JSON-compatible provenance metadata.

It does not preserve optimizer, scheduler, dataloader, gradient scaler, or RNG
state. The checkpoint is a model restoration format, not mid-training resume.

## Deliberate deviations from the paper

- The backbone is GPT-2-style TinyLLM rather than Llama-style.
- Learned absolute positions are used instead of RoPE.
- The compact exercise uses one encoder and decoder residual MLP per stage.
- Only base-temperature single-step sampling is implemented.
- BrierLM evaluation and the approximate low-temperature candidate-pool
  sampler are not yet implemented.

These differences preserve the central two-stage continuous autoregressive
construction but make this a method adaptation rather than an exact
reproduction.

## Verification

```bash
pixi run pytest -q \
  tests/structure_net/components/test_calm_tinyllm_model.py \
  tests/structure_net/test_tinyllm_calm_framework_shakedown.py

pixi run python -m experiments.structure_net.tinyllm_calm_framework_shakedown \
  --device cpu \
  --output data/experiments/tinyllm_calm_framework_shakedown/20260811_cpu_seed7
```

The measured CPU and CUDA lifecycle result is reported in
`docs/08 - Analysis/2026-08-11_tinyllm-calm-framework-shakedown.md`.
