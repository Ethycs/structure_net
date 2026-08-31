# Experiment 04 — SmolLM Distillation Preregistration

**Status:** LOCKED BEFORE OUTCOMES  
**Shared protocol:** [`04-05-local-distillation-preregistration.md`](04-05-local-distillation-preregistration.md)

Train the SmolLM2-360M-Instruct two-class semantic head in label-only and
teacher-assisted arms at seeds 7, 17, and 29. The frozen representation cache
must contain exactly the 49,259 eligible unique training groups and 7,983
eligible unique development groups, with model, tokenizer, dataset, cache, and
code hashes in its manifest. Selection uses development partition 0 only.

The canonical outputs are six NAL results, six loadable head checkpoints, full
training histories, an arm/seed comparison, and the selected frozen checkpoint.
No test file is available to this experiment.
