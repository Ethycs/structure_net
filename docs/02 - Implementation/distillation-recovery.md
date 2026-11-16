# Distillation recovery implementation

Date: 2026-11-15. Governing contract:
[NAL-STD-DISTILLATION v1](../05%20-%20Standards/NAL-STD-DISTILLATION-v1.md).

This implementation builds the next recovery workflow separately from the
running Qwen 15-block progressive and 36-block all-at-once experiments. It does
not change those experiments' sparse-position objectives or promote their
results. A CPU lifecycle proves execution, not Qwen recovery.

## Components and ownership

| Component | Implementation | Responsibility |
| --- | --- | --- |
| Response data | `src/neural_architecture_lab/distillation_data.py` | Verified token boundaries, complete response/EOS targets, split guards, length buckets and authenticated mmap teacher targets |
| Recovery training | `src/neural_architecture_lab/distillation_training.py` | Exact full-vocabulary forward KL in bounded token chunks, explicit CE/JSD alternatives, optimizer updates, token budgets, rollouts and checkpoints |
| NAL lifecycle | `src/neural_architecture_lab/distillation_recovery.py` | Frozen per-arm/seed specifications, generation-aware evidence gates, existing NAL experiment/result types and result ledger |
| CPU entry point | `experiments/neural_architecture_lab/distillation_recovery.py` | Prepare, actual CPU shakedown, aggregate and authenticated completed-result readback |
| Qwen adapter | `experiments/structure_net/jspace_3b_dense_adapter.py` | Complete compact-stack hidden/readout callbacks and frozen original Qwen teacher targets |

No additional generic scheduler was introduced. The lightweight entry point
loads the same NAL core and ledger files without importing optional analyzer
dependencies from the canonical package initializer. The normal NAL package
remains unchanged.

## Execute the CPU lifecycle

Use a new artifact directory. The following runs real forward/backward updates,
saves initial and final checkpoints, reloads into a separate model object,
checks teacher/fixed-state preservation and reads back the existing NAL ledger:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache UV_CACHE_DIR=/tmp/structure-net-uv-cache \
  /home/rabbit/.pixi/bin/pixi run --locked --offline -e ncm python \
  -m experiments.neural_architecture_lab.distillation_recovery cpu-shakedown \
  --root /tmp/nal-distillation-cpu-shakedown
```

Repeating the command at a completed matching root checks receipts instead of
training again. A failed or incomplete root is preserved. Inspect its failure
before creating a new attempt; never overwrite it. `aggregate` requires a new
`--output` filename. Completed readback authenticates archived producing source;
new worker admission also checks the live producing source.

This fixture uses synthetic tokens and reports `systems_passed`. It contains
no semantic accuracy result and must never be counted as quality evidence.

## Assemble a study-specific Qwen producer

1. Freeze teacher, tokenizer/template, admitted TRAIN/DEV manifests, initializer,
   geometry, objectives, token budgets, evaluator, checkpoint-selection rule and
   precision in a dated study. Use separate source groups for each split. Keep
   the final test corpus unavailable to the training process. The study adapter
   must authenticate corpus membership against its manifest; a supplied data
   digest in the generic trainer does not by itself prove that membership.
2. Construct the matching **unpacked** full compact stack once and strictly load
   the authenticated initializer. Wrap stack, normalization, original tied
   embedding and rotary module in `DenseCompactQwen`. The first adapter requires
   all original layer indices starting at zero; it does not silently omit a
   native prefix. Existing progressive suffix studies keep their own adapter.
3. Build `ResponseRecord`s with the pinned template. `response_batch` returns
   full-length tensors. `loss_mask[b,t]` selects the logit predicting token
   `t+1`; `targets[b,t]` is already shifted. Do not shift them a second time.
   Right-padded batches keep documents independent. Packing is not implemented.
4. Supply `HFTeacherTargets.prepare(batch)` or an authenticated
   `TeacherTargetStore`. `prepare` runs before student activations are created.
   The HF adapter caches one batch of original-teacher backbone hidden states on
   CPU. An optional teacher residency context owns activation/offload. Its
   readout can share the student's frozen embedding after exact equality
   admission, avoiding a second vocabulary matrix and CPU vocabulary projection.
   A later target call receives `[row, predicting_position]` pairs.
5. Create a fresh optimizer over exactly the intended trainable parameters.
   Pass `student.hidden_forward`, `student.readout`, the teacher provider and
   explicit `DistillationConfig` to `DistillationTrainer`. The exact loss
   computes bounded token chunks, accumulates hidden adjoints, then backpropagates
   the trunk once. Token-weighted accumulation handles unequal response lengths.
   A token budget stops at complete batches; it does not cut off EOS targets.
6. For the rollout arm, `collect_rollouts` calls `student.next_logits` on TRAIN
   prompts, records the exact generating checkpoint and prefixes, and retains
   every generated token. An observed EOS is supervised; a token cap invents no
   EOS. Refresh teacher scores for those prefixes. The initial implementation is
   cache-free, greedy and bounded; this is not a decode-speed optimization.
   Captured rounds use the explicit `offline_round` policy and a maximum stale
   update count. Each trainer starts with a fresh optimizer; the implementation
   does not yet provide persistent optimizer state across an adaptive on-policy
   refresh loop. A study must name an optimizer reset as part of its recipe.
7. The study producer performs paired generation and task evaluation, then
   supplies raw outputs, judgments and judge provenance to `RecoveryWorker`.
   A generation failure cannot be rescued by multiple-choice accuracy. The
   reusable trainer itself does not select a scientifically eligible checkpoint.
   Early, equivalent eligible checkpoints should win ties.

`RecoveryStudy` creates an existing NAL `Experiment` per arm/seed.
`RecoveryWorker` has the existing `(Experiment, logical_device_id)` signature
and writes an `ExperimentResult` plus authenticated evidence artifacts. CPU is
`-1`. GPU execution must enter `inference-ctl lease run` **before Python**;
the worker checks the active lease and uses logical ordinals after visibility
has been set. Model constructors must not allocate CUDA tensors before that
admission. NAL scheduling slots do not reserve hardware.

The generic confirmation adapter currently implements a conservative paired
Hoeffding bound. Its sample and seed minima are explicit configuration. This
is one supported decision rule, not a universal sample-size prescription; it
can require substantially more cases than a variance-aware paired interval.
Exploratory observations do not need to pass that confirmation bound. A different
confirmatory estimator must be specified and tested before its outcomes are used.

## Initial scientific sequence

The proposed first comparison holds the original teacher, initializer and
fixed J-space/polynomial geometry constant:

- Dense complete-response forward KL, temperature 1, versus a separately named
  CE mixture if there is budget for a second arm.
- Short student-rollout recovery from an eligible dense checkpoint, compared
  with equal-supervised-token continued teacher-forced recovery.
- Evaluate useful generation, stopping/repetition, factual/numerical summary
  preservation and task benchmarks at each frozen evaluation boundary. Keep
  native failures visible and use sufficient matched output budgets.

The illustrative 100k → 1M → 10M supervised-token ladder **does not start
automatically after a passing shakedown**. Freeze admitted data, transition
rules and the selected initializer after inventory, and run a representative
reserved CUDA memory/time pilot before the first scientific cell. Neither the
generic optimizer nor teacher residency has yet been qualified for the actual
3B workload on the 6 GB card. Existing CPU-offloaded optimizer machinery is a
candidate integration, not a result established by these tests.

The current Qwen compressed geometry supports **416 total positions**. The
adapter rejects oversize sequences and clamped anchors. A 128-token summary
limit leaves room only for the remaining prompt within that total; it does not
establish support for 2,048-token fragments or 6,300-token batches. Measure how
many real pairs fit before selecting a corpus. Longer inputs require a separately
validated geometry/length adaptation, not silent truncation.

Full teacher logits cost `tokens × vocabulary × element_size`; at Qwen's
151,936-token vocabulary, 100,000 FP32 target rows occupy about **60.8 GB** before
metadata. Do not blindly cache every vocabulary row. The hidden-state/readout
adapter is one bounded alternative. Top-k plus an explicit tail bucket is
available as a separately labelled approximation and must be compared with the
exact objective before use. The current exact store intentionally rejects a
top-k cache masquerading as full logits.

Before a recovery or speed claim, finish the original full-vocabulary generation
path diagnostic, actual GPU pilot, representative-corpus admission, generation-aware
DEV selection and fresh confirmation. Export/parity and isolated prefill/decode
measurements form a later deployment gate.
