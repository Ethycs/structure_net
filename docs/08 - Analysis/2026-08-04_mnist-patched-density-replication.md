# MNIST Patched-Density Replication and Fashion-MNIST Transfer

**Status:** PROVISIONAL SINGLE-SEED SIDECAR ABLATION  
**Date:** 2026-08-04  
**Applies to:** `src/neural_architecture_lab/experiments/patched_density_replication.py`, `src/structure_net/data_factory`, archived MNIST claims and weights  
**Depends on:** `2026-08-03_runner-experiment-data-modernization.md`, `../03 - Architecture/structure-net-overview.md`

## Executive verdict

The qualitative sidecar finding held at seed 4: five extrema-selected dense paths improved a 2%-connected sparse scaffold on both datasets relative to a matched continuation-only control. This is an ablation of added sidecar capacity, not a replication of the mature embedded variable-density growth mechanism described later in the archive. The archived 97.67% MNIST headline did **not** replicate. The sidecar protocol reached 62.05% MNIST and 69.68% Fashion-MNIST; the paired improvements were +14.64 and +7.41 percentage points respectively.

The two stored checkpoints named for 95% and 100% MNIST accuracy also failed canonical reevaluation. They are dense `[784, 64, 10]` models with no sparse masks or patch weights, and score 14.94% and 8.17% on the full 10,000-example MNIST test set. Their filename/stored metrics are not valid full-test evidence.

| Dataset | Scaffold final (best) | Continued control final (best) | Patched final (best) | Patched − control |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 47.37% (47.37%) | 47.41% (47.45%) | 62.05% (62.05%) | **+14.64 pp** |
| Fashion-MNIST | 62.28% (64.55%) | 62.27% (62.33%) | 69.68% (69.76%) | **+7.41 pp** |

**observed:** sidecar patching added paired value on both datasets for this seed.  
**not validated:** the mature StructureNet patch/growth mechanism, 95%+ MNIST accuracy, extrema selection versus random placement, parameter-storage reduction, statistical generality, or a production compute advantage.

## 1. What the archive actually contains

The historical evidence has three incompatible forms:

| Evidence | Claim | Audit result |
| --- | --- | --- |
| `archive/old_research/notes/Breakthroughs.md` | one 97.67% and one 76.12% patched MNIST run | transcript only; no result envelope or matching checkpoint |
| earliest `patched_density_experiment.py` commit | `[784, 128, 10]`, 2% scaffold, high-extrema patches | patch output width does not match logits, so the recorded code cannot produce the claimed patch effect |
| later archived script | `[784, 256, 128, 10]`, high/low patches | low patches resample input indices on every forward; `patch_density` is ignored |
| two `test_simple_models/*.pt` weights | filenames/stored fields say 95% and 100% | dense models; canonical full-test scores are 14.94% and 8.17% |

The high-accuracy claim is therefore **unverified archival evidence**, not a baseline the current code can reproduce exactly. The new protocol repairs one early sidecar interpretation and marks that scope explicitly; it does not implement the later embedded-mask, progressive-growth, vertical-cloning, or dead-neuron-routing design.

## 2. Replication protocol

The paired experiment uses:

| Field | Value |
| --- | --- |
| Architecture | `[784, 128, 10]` |
| Scaffold connection density | 2% Bernoulli masks through canonical `StandardSparseLayer` |
| Seed | 4 |
| Scaffold phase | 20 epochs, Adam, learning rate `1e-3` |
| Extrema probe | first 1,000 test inputs, hidden mean greater than mean + 2 standard deviations |
| Patch budget | top five high extrema; each `1 → 10 → 10`, scaled by `0.1` |
| Continuation phase | 30 epochs; scaffold learning rate `1e-4`, patch learning rate `5e-4` |
| Control | deep copy of the same trained scaffold, identical minibatch order, no patches |
| Batch/evaluation | 256 / 1,000; full 60,000 train and 10,000 test sets |
| Runtime | CPU, one PyTorch thread, transformed datasets cached in RAM |

The batch size is 256 rather than the archived script's 64 to make the CPU shakedown bounded. This is a documented protocol difference. Caching evaluates deterministic normalization once and does not change sample values or minibatch order.

**known rough edge:** retaining the archive's test-input extrema probe leaks test-distribution information into architecture selection, although it does not use test labels. A confirmatory run must select extrema from training inputs and reserve the test split for final evaluation.

## 3. Parameter and compute accounting

Both runs selected five patches and have the same accounting:

| Quantity | Count |
| --- | ---: |
| Active scaffold weights | 2,005 |
| Scaffold biases | 138 |
| Dense patch parameters | 650 |
| Effective parameters | 2,793 |
| Fully dense scaffold parameters | 101,770 |
| Effective fraction of dense scaffold | 2.744% |
| Stored trainable parameters | 102,420 |

The old notes described masked-out weights as if they were absent parameters. They are not: `StandardSparseLayer` stores dense weight tensors and masks them during its dense linear operation. The model has fewer active connections/effective degrees of freedom, but slightly **more** stored trainable values than the dense scaffold after adding patches. No memory or kernel-speed claim follows from this experiment.

## 4. Dataset transfer result

Fashion-MNIST is now a first-class data-factory dataset named `fashion_mnist`, with the same `(28, 28)` input shape and ten-class output contract as MNIST. Its loader uses Fashion-MNIST normalization `(0.2860, 0.3530)` and reproducible subsetting without mutating NumPy's global random state.

The same seed selected related but non-identical extrema:

| Dataset | Selected hidden neurons |
| --- | --- |
| MNIST | `126, 108, 71, 29, 116` |
| Fashion-MNIST | `108, 71, 26, 126, 22` |

Three of five selected neurons overlap: `{126, 108, 71}`. The important measured result is the paired gain, not neuron identity: the patch mechanism transferred directionally, with roughly half the MNIST gain on clothing.

## 5. Stored-weight audit

| Archived checkpoint | Stored accuracy | Canonical full-test accuracy | Correct / 10,000 | Sparse masks? |
| --- | ---: | ---: | ---: | --- |
| seed 4 `[784,64,10]` | 95.00% | 14.94% | 1,494 | No |
| seed 5 `[784,64,10]` | 100.00% | 8.17% | 817 | No |

The audit loads only tensor/checkpoint-safe globals, reconstructs the sequential dense architecture from metadata, applies the current canonical MNIST normalization, and performs no training. SHA-256 hashes and state-dict keys are stored in `data/experiments/mnist_checkpoint_audit/20260804/results.json`.

## 6. Interpretation and next experiments

1. **Run at least five seeds.** The archive itself reports high variance, and `n=1` cannot support a population or significance claim.
2. **Move extrema selection to training data.** This removes test-distribution leakage and creates a defensible final-test contract.
3. **Add a random-patch placement control.** The current pair proves added capacity helps; it does not yet prove extrema selection beats the same patch budget placed randomly.
4. **Add a parameter-matched dense control.** This tests whether sparse-plus-patches is better than an ordinary model with approximately 2,793 effective parameters.
5. **Only then test broader clothing architectures.** The Fashion-MNIST direction is encouraging, but 69.68% is a mechanism shakedown rather than a competitive classifier.

## Artifacts

| Path | Contents |
| --- | --- |
| `data/experiments/patched_density_replication/20260804_seed4/results.json` | protocol, environment, epoch histories, extrema, paired metrics, parameter accounting |
| `data/experiments/mnist_checkpoint_audit/20260804/results.json` | checkpoint hashes, stored claims, canonical reevaluation |
| `data/meta_hypotheses/meta-extrema-sidecar-transfer-mnist-fashion-v1.json` | conservative aggregate hypothesis, direct evidence, provenance audits, excluded scope, and follow-ups |
| `data/chroma_db` | searchable `hypotheses` parent plus four linked `experiments` evidence records |
| `data/datasets/fashion_mnist/FashionMNIST/raw/` | cached Fashion-MNIST source files |

## Verification

```bash
env PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache \
    UV_CACHE_DIR=/tmp/structure-net-uv-cache \
    pixi run python experiments/neural_architecture_lab/patched_density_replication.py \
      --datasets mnist,fashion_mnist \
      --seeds 4 \
      --scaffold-epochs 20 \
      --continuation-epochs 30 \
      --batch-size 256 \
      --torch-threads 1 \
      --output-dir data/experiments/patched_density_replication/20260804_seed4

env PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache \
    UV_CACHE_DIR=/tmp/structure-net-uv-cache \
    pixi run python experiments/neural_architecture_lab/audit_mnist_checkpoints.py \
      --output data/experiments/mnist_checkpoint_audit/20260804/results.json

pixi run python \
  experiments/neural_architecture_lab/store_replication_meta_hypothesis.py
```

Run the focused tests below to confirm loader registration, patch routing/gradients, parameter accounting, and checkpoint reconstruction; then decide whether the next budget goes to multi-seed confirmation or the random-placement control.

```bash
pixi run pytest -q \
  tests/structure_net/test_fashion_mnist_data.py \
  tests/neural_architecture_lab/test_patched_density_replication.py \
  tests/neural_architecture_lab/test_checkpoint_audit.py \
  tests/neural_architecture_lab/test_meta_hypothesis.py
```
