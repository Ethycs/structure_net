# Experiment 05 — TinyLLM Distillation Preregistration

**Status:** LOCKED BEFORE OUTCOMES  
**Shared protocol:** [`04-05-local-distillation-preregistration.md`](04-05-local-distillation-preregistration.md)

Train the TinyLLM d8 two-class semantic head in label-only and
teacher-assisted arms at seeds 7, 17, and 29, initialized from the registered
BabyLM step-12000 checkpoint. The semantic group order and objectives match
Experiment 04; token tensors and representation dimension are model-native.
The feature manifest must bind the 49,259/7,983 eligible train/development
groups to the TinyLLM checkpoint and tokenizer hashes.

The canonical outputs and partition-0 selection rules match Experiment 04. No
test file is available to this experiment.
