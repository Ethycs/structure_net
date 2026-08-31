# Experiment 02 — Undistilled Model Baselines

**Verdict:** COMPLETE  
**Runtime:** canonical NAL GPU/API campaign  
**Date:** 2026-08-16

The locked 256-example PAWS development sample was balanced by human label and
identical across A, B, and C. Its ordered selection digest is
`4b74e7953c4a22a8cd7e76019b330282596d3c57557ff4ce18b740decdc7ba61`.
The official test split was not evaluated.

| Model | Accuracy | Balanced accuracy | Malformed |
| --- | ---: | ---: | ---: |
| TinyLLM d8 (A) | 0.00% | 0.00% | 100.00% |
| SmolLM2-360M-Instruct (B) | 0.00% | 0.00% | 100.00% |
| Qwen3-8B (C) | 76.95% | 76.95% | 0.00% |

All three received the same prompt and used greedy exact generation. No label
ranking, extraction, or repair was applied. Thus the zero local baselines mean
that the pretrained models did not satisfy this task's exact-output contract;
they do not establish zero latent paraphrase knowledge. This is precisely the
instruction-following deficit Experiments 04 and 05 must address.

The canonical campaign, per-example raw outputs, dataset digest, NAL ledgers,
and retry counts are under
`data/experiments/paws_abc_routing/2026-08-16_experiment_02/`. The three run
files contain 256 predictions each and agree on both dataset and selection
digests.

Experiment 03 is licensed to annotate eligible training examples with the
frozen Qwen teacher while retaining human labels as the correctness oracle.
