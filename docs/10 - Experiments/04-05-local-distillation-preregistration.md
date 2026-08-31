# Experiments 04–05 — Local Distillation Preregistration

**Status:** LOCKED BEFORE DISTILLATION OUTCOMES  
**Date:** 2026-08-16

## Arms

For B (Experiment 04, SmolLM2-360M-Instruct) and A (Experiment 05, TinyLLM
d8), freeze the pretrained transformer and train a two-class linear semantic
head on its final prompt representation. This is coordinate distillation into
the same finite task, not full language-model fine-tuning. The head emits the
exact labels `DIFFERENT` and `PARAPHRASE`, so there is no post-hoc label repair.

Each model has two arms with seeds 7, 17, and 29:

- `label_only`: human-label cross entropy;
- `teacher_assisted`: human-label cross entropy on every row; on the frozen
  2,048-row Qwen audit subset only, combine 0.75 human-label cross entropy with
  0.25 Qwen hard-label cross entropy.

Both models use the same eligible training pair-groups in source order. Their
token tensors differ because their frozen tokenizers differ. Training uses 50
epochs, AdamW, learning rate 0.001, weight decay 0.0001, and batch size 512.

## Selection and gates

Checkpoint selection uses only development groups assigned to partition 0 by
`SHA256("paws-dev-partition-v1:" + group_id)[0] mod 4`. Other development
partitions are reserved for the competence atlas, router calibration, and
construction audit. The official test split remains unread.

For each model and arm, choose the seed with highest checkpoint-partition
balanced accuracy, breaking ties by lower seed. Report all seeds. An experiment
passes when the feature cache matches its model/data hashes, all six NAL runs
finish, checkpoints are loadable, and both arms cover identical group IDs.
Teacher assistance is retained only as an experimental arm; human labels remain
the primary objective and correctness oracle.
