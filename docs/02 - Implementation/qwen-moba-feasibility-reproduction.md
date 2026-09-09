# Reproduce the Qwen-prefix MoBA experiment

The [protocol](../07%20-%20Status%20Reports/2026-09-09_qwen8b-moba-feasibility.md)
defines the model graph, data splits, metrics and gates. The
[shared-GPU amendment](../07%20-%20Status%20Reports/2026-09-09_qwen8b-moba-shared-gpu-amendment.md)
defines the measured auxiliary runtime assessment. Run from the repository root
on an Ampere-or-newer NVIDIA GPU. The observed environment used an RTX 3060 and
Python 3.11. The available Turing GPUs could not run this FlashAttention 2 path.

The experiment used an isolated environment with shared repository packages:

```bash
.pixi/envs/default/bin/python -m venv --system-site-packages data/experiments/moba_paper/environment
data/experiments/moba_paper/environment/bin/python -m pip install \
  torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
data/experiments/moba_paper/environment/bin/python -m pip install \
  numpy==1.26.4 einops==0.8.1
data/experiments/moba_paper/environment/bin/python -m pip install \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
git clone https://github.com/MoonshotAI/MoBA.git data/experiments/moba_paper/upstream
git -C data/experiments/moba_paper/upstream checkout b5d58363311d3ca946f1ec444182727c15e338b5
```

The inherited Transformers version was 4.53.3, tokenizers 0.21.4.dev0 and
safetensors 0.8.0. The inherited repository environment also supplies the
analysis/registry dependencies. Complete producing library versions and source
hashes are in every run's `producer.json`. Do not replace the shared repository's
PyTorch installation to reproduce this isolated experiment.

Download/verify the pinned public weights and tokenizer:

```bash
data/experiments/moba_paper/environment/bin/python -m \
  experiments.structure_net.prepare_qwen_moba_weights
```

This verifies five pinned files, including the approximately 4 GB first shard.
It downloads no remaining model shards. It creates the 67-key prefix plan and
content-hash manifest used by the model loader. Existing mismatched inputs are
rejected. The existing corpus inputs are
`data/corpora/babylm_10M/wikipedia.txt` and `bnc_spoken.txt`; their exact hashes,
Qwen-tokenized arrays and predetermined schedules are preserved in
`data/experiments/qwen_moba_prefix/inputs_v2/` and each producer. Supply those
same artifacts for an exact reproduction; different text is a new experiment.

Run numerical/selection/artifact contract tests:

```bash
env LD_LIBRARY_PATH= PYTHONPATH="$PWD:$PWD/src" \
  data/experiments/moba_paper/environment/bin/python -m pytest -q \
  tests/structure_net/test_qwen_moba_prefix.py \
  tests/structure_net/test_qwen_moba_shared_assessment.py \
  tests/neural_architecture_lab/test_qwen_moba_audit.py \
  tests/neural_architecture_lab/test_qwen_moba_feasibility_hypothesis.py
```

For primary uncontended work, use `run_qwen_moba_feasibility --stage pilot`,
then `--stage screen`, with `--head-chunk-size 32`, and a new output directory
per stage. Check available GPU capacity immediately before launching.

For the separately scoped shared-GPU assessment:

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH= \
  PYTHONPATH="$PWD:$PWD/src" \
  data/experiments/moba_paper/environment/bin/python -m \
  experiments.structure_net.qwen_moba_shared_assessment \
  --output data/experiments/qwen_moba_prefix/NEW_shared_screen
```

Only a screen producing `shared_gpu_candidate_selected` authorizes its matching
held-out stage: add `--selection <screen>/campaign_results.json` and use a new
final output directory. This enforces the unchanged source, settings, model,
inputs, windows and timing configuration. It performs zero optimizer updates.

The independent auditor accepts a campaign path and required `--sha256`. The
finalizer combines the readout, kernel, shared screen and optional selected
final evidence; it produces a report and a conservative registry with exact-ID
and semantic read-back. Completed raw evidence and existing reports are
append-only: use new dated paths for new measurements or interpretations.
