# TinyLLM CALM framework shakedown — preregistration

**Status:** PRE-OUTCOME SHAKEDOWN SPECIFICATION  
**Date:** 2026-08-11  
**Conformance:** NAL-STD-EXPERIMENT `SHAKEDOWN`  
**Hypothesis:** `tinyllm-calm-framework-lifecycle-v1`  
**External method:** [Continuous Autoregressive Language Models](https://arxiv.org/abs/2510.27688)

## Question

Can Structure Net express a CALM-style language model as inspectable components,
initialize its GPT-2 backbone from an ordinary TinyLLM checkpoint, execute the
autoencoder and energy-model training stages, generate a token chunk, and
restore the resulting graph exactly?

This is an engineering lifecycle question. It is not a test of language-model
quality, paper reproduction, inference speed, or the claim that a pretrained
TinyLLM retains its capability after conversion.

## Fixed design

The shakedown uses a deliberately tiny GPT-2-compatible backbone and a synthetic
arithmetic-token stream. The source TinyLLM receives a small real next-token
training update before conversion. The converted model then uses:

- chunk size `K=2`;
- a variational token-chunk autoencoder with tied encoder/decoder token weights;
- a discrete K-token input compressor;
- the copied TinyLLM transformer blocks and learned position table;
- a stochastic single-step continuous generator;
- an energy-score objective;
- separate framework trainer phases for the autoencoder and energy model.

The CPU and CUDA runs use the same declared configuration and seed. Numerical
outcomes are not required to be byte-identical across devices.

## Lifecycle gates

The shakedown passes only if all of the following hold in each run:

1. every source backbone tensor is copied exactly at conversion;
2. the autoencoder performs a real backward/optimizer step and receives finite
   gradients;
3. the continuous model performs a real energy-loss backward/optimizer step and
   receives finite gradients;
4. autoencoder parameters remain byte-identical throughout the frozen energy
   stage;
5. generation returns exactly `K` in-vocabulary token IDs per sequence;
6. the saved model restores with an identical state digest and identical
   fixed-noise energy output;
7. the result and campaign envelopes serialize as strict JSON and retain the
   complete configuration, implementation digests, checkpoint path, and method
   boundaries.

Reconstruction loss and energy loss are recorded before and after their small
training stages, but improvement is not a gate. These underpowered values may
diagnose execution and may not be interpreted as quality evidence.

## Artifact plan

```text
data/experiments/tinyllm_calm_framework_shakedown/<run>/
├── campaign_results.json
└── runs/framework_lifecycle/seed_7/
    ├── result.json
    └── model.pt
```

Planned entry point:

```bash
pixi run python -m experiments.structure_net.tinyllm_calm_framework_shakedown \
  --device cpu \
  --output data/experiments/tinyllm_calm_framework_shakedown/20260811_cpu_seed7
```

The representative CUDA run changes only `--device` and the artifact root.

## Outcome meanings

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| all lifecycle gates pass | the framework can host and persist the CALM computation | design a matched quality-retention pilot |
| backbone copy or restoration fails | conversion/checkpoint contract is incomplete | repair the component boundary before training |
| either training phase fails | objective or gradient routing is incomplete | repair and rerun under a new implementation digest |
| CPU passes but CUDA fails | the implementation is not yet device-portable | localize the CUDA incompatibility |

## Method boundaries

- The paper uses a Llama-style backbone; this exercise deliberately adapts the
  method to TinyLLM's GPT-2 blocks.
- The synthetic vocabulary, tiny model, and short run cannot estimate BrierLM,
  perplexity retention, downstream capability, throughput, or sampling quality.
- The run implements base-temperature single-step sampling. It does not certify
  the paper's approximate low-temperature candidate-pool algorithm.
- A high-fidelity chunk autoencoder is necessary but not sufficient for a useful
  continuous language model.

## Post-outcome amendment — deterministic component initialization

**Recorded:** 2026-08-11, after the first CPU and CUDA lifecycle outcomes were
inspected.

The first implementation explicitly seeded newly introduced matrix weights,
but one-dimensional bias parameters retained their constructor initialization
from PyTorch's ambient RNG stream. The complete experiment was reproducible
because that stream was seeded before construction, and both original
checkpoints restored exactly; nevertheless, a component's declared
`initialization_seed` did not independently determine every new parameter.

The implementation was corrected to initialize every non-backbone matrix,
normalization weight, and bias from an explicit deterministic rule. No
lifecycle gate, configuration value, threshold, or interpretation changed.
The original CPU/CUDA roots remain preserved. Corrected runs use new
`20260811_*_seed7_v2` roots and a new producing-code digest; only those runs
enter the final shakedown verdict.
