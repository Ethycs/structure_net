# Sequential NCM fits the training codes; frozen rendering still changes terms

Date: 2026-10-24 (America/Los_Angeles). Status: completed exploratory control.
Hypothesis: `ncm-sequential-native-engineering16-v1`.
Classification: `native_generator_fits_TRAIN_codes_frozen_rendering_quality_failed`.

The native generator completed 512 sequential training updates and reproduced
10,254/10,256 reference code positions in source-only generation. All 16 decoded
texts exactly matched the corresponding reference-code rendering. This isolates
the remaining observed textual defects to the frozen rendering path on these
TRAIN examples. It does not establish held-out summarization accuracy.

| TRAIN diagnostic | Before | After |
| --- | ---: | ---: |
| Mean clean-history code cross-entropy | 5.415732 | 0.002255 |
| Source-only correct code positions | 699/10,256 | 10,254/10,256 |
| Active reference word-ID agreement | 47/672 (7.0%) | 650/672 (96.7%) |

All 16 roots and all code positions in the first 11 levels were correct. The two
remaining code errors were in the last level and did not change the rendered
text. Fifteen full 128-position word-ID arrays matched the oracle arrays exactly;
all 16 decoded texts matched. Shuffling source inputs reduced word agreement to
52/672 (7.7%), supporting source dependence on this memorized subset.

The numeric learnability gate passed. The text-quality gate failed: all 16 pairs
were inspected against their supplied references, and consequential defects
remain. Examples include `tensor products` becoming `aggregation products`,
`dependency-aware` becoming `perspectives-aware`, `manual pinning` becoming
`manual dashboard`, and loss of the entropy qualifier in entropy measurement.
These same defects occur with reference codes. This inspection is not an
independent semantic-judge evaluation or an accuracy percentage.

The experiment trained 105,323,136 native-generator and concept-source-attention
parameters. Llama, the entire codec, and word-source attention stayed frozen;
their protected tensors were hash-verified unchanged. Each of 12 forwards used
only the cumulative reconstruction from earlier levels. Clean-history and
corrupted-history microbatches used clean categorical targets; inference used
only generated histories. The source features were previously audited full
FP32 tensors. The 16 examples each appeared 64 times, totaling 1,024 exposures.

The fixed run used FP32 weights, FP16 training, FP32 inference, microbatch 1,
accumulation 2, AdamW, 32-step warmup, and cosine decay from 1e-4 to 1e-5 on physical
GPU 2. The optimizer loop took 1,569.20 s (26.15 min); training and postflight took
1,633.28 s. Peak allocated GPU memory was 2,832,838,656 bytes. These timings are
training-operation observations, not inference-throughput benchmarks.

Five CPU tests passed. The two-update CUDA pilot verified native/explicit
sequential logits, identical greedy codes, real gradients, frozen codec state,
and full checkpoint reload. The 512-update worker and independent arithmetic
audit both exited 0. Final model, optimizer, scaler, RNG, and decoded outputs
passed strict reload verification. One pilot and one training run completed;
there were no retries or excluded runs.

Earlier native 655 experiments already used coarse-to-fine generation. This
experiment also changes trainable capacity, objective, optimizer, and example
exposures relative to the 3.8M simultaneous-repair head. Its improvement cannot
be attributed solely to sequential execution. It is not a full paper replication.
The trace's corruption counter includes terminal-level replacements, which
condition no later prediction; this does not affect the losses.

Full 732-example training was not launched because the decoded-text gate failed.
Held-out semantic quality, the 95% task-accuracy target, and inference speedup
remain unestablished. Sealed final evaluation was not opened. The next relevant
experiment is correction of the frozen rendering defects with the generator
held fixed, followed by source-disjoint quality evaluation.

Plan: [sequential control](../07%20-%20Status%20Reports/2026-10-24_ncm-sequential-levels-plan.md).
Producing implementation commit: `48d1ad3`.

Aggregate:
`data/experiments/ncm_paper_regime/20261024_sequential_engineering16_review_v1/campaign_results.json`
SHA256 `8c86edbf15bb04b7fd04c95c3e964eabac472dd1818d7cd6ab49b363db7cd000`.
Reviewed text pairs are stored beside the aggregate in `reviewed_pairs.json`.

Checkpoint:
`data/experiments/ncm_paper_regime/20261024_sequential_engineering16_capacity_v1/model.pt`
SHA256 `d27af43e7bb8f8a80d81e2c19c51e12ddb8bcc08154dd0540303862ecd81f841`.

Recorded execution command (the completed root is immutable and rejects reruns):

```bash
env LD_LIBRARY_PATH='' CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=src:. \
  data/experiments/moba_paper/environment/bin/python \
  -m experiments.structure_net.ncm_sequential_engineering capacity
```
