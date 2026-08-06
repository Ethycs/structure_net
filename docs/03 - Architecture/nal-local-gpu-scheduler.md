# NAL Local GPU Scheduler

**Status:** CURRENT

**Date:** 2026-08-06

**Applies to:** canonical NAL experiments executed on one local host

**Backend:** spawned Python workers; Ray is not in the canonical path

NAL schedules independent experiments onto explicit logical CUDA slots. It is
experiment parallelism, not distributed training of one model.

## Device contract

Physical GPU selection happens before Python starts:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 pixi run python ...
```

Inside that process, the selected card is logical `cuda:0`. `LabConfig.device_ids`
and `--nal-gpus` always contain logical ordinals after visibility has been
applied. Importing Structure Net or NAL no longer silently sets
`CUDA_VISIBLE_DEVICES`.

## Scheduling contract

```text
visible logical GPUs
        |
        v
fixed slots or free-memory calibration
        |
        v
[gpu 0, gpu 0, gpu 1, ...]
        |
        v
CUDA-safe spawned ExperimentWorker processes
        |
        v
atomic per-experiment completion ledger
```

`gpu_slots_per_device > 0` sets a fixed slot count. A value of zero divides
currently free GPU memory, after a safety fraction, by
`gpu_memory_per_experiment_gb` and caps the result at
`max_gpu_slots_per_device`. This is admission control, not memory isolation;
the estimate must come from a representative pilot. Invalid or invisible
logical IDs fail before workers start.

Each failed result or raised worker exception may be retried up to
`max_experiment_retries`. When `resume_completed_experiments` is enabled, NAL
writes an atomic fingerprinted result under `.nal_runner/completed/`. A later
run reuses only a successful result whose experiment ID, hypothesis ID, seed,
and parameters have the same fingerprint. This resumes completed experiments;
it does not restore an optimizer halfway through an experiment.

## TinyLLM campaign

The TinyLLM campaign makes one seed one NAL experiment. Its baseline,
recompute-only, and random-feedback arms remain sequential inside the seed, so
independent seeds can share a GPU without changing matched-arm initialization.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python experiments/structure_net/tinyllm_feedback_nal_campaign.py \
  --preset d6 --seeds 7,17,29,41,53 \
  --slots-per-gpu 0 --memory-per-seed-gb 1.5 \
  --max-slots-per-gpu 3 \
  --output data/experiments/tinyllm_feedback_nal_campaign
```

The campaign stores each seed's three model checkpoints and deterministic
shakedown bundle, NAL's completion record, and one aggregate
`campaign_results.json`. Shared-run wall times are explicitly not benchmark
measurements. Use `--isolated-timing`, one visible GPU, and one seed when timing
is the target.

## Boundaries

| Capability | Status |
| --- | --- |
| Multiple experiments per local GPU | Supported |
| Multiple visible local GPUs | Supported by logical slot assignment |
| Completed-experiment retry/resume | Supported |
| Mid-training optimizer/dataloader resume | Not supported by this layer |
| GPU memory enforcement | Not supported; slots use an estimate |
| Single-model DDP | Not implemented by the canonical NAL runner |
| Multi-node scheduling | Not implemented |
| Ray | Retained only as non-canonical historical compatibility code |
