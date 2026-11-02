# TinyLLM temporal-language identifiability preregistration (ladder stage L1)

**Status:** LOCKED BEFORE ANY FINE-TUNE OUTCOME  
**Date:** 2026-08-12  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, training campaign with frozen generator contracts  
**Hypothesis:** `tinyllm-temporal-language-identifiability-v1`  
**Planned schema:** `nal.tinyllm-temporal-language-identifiability.v1`  
**Design:** [temporal-phase language task](../01%20-%20Design/temporal-phase-language-task.md)

## Provenance boundary

At lock time the following exist and are frozen; no task fine-tuning outcome
has been produced or inspected:

- BabyLM strict-small cleaned corpus under `data/corpora/babylm_10M/`;
- BPE tokenizer `data/corpora/babylm_10M_bpe16k.tokenizer.json`, SHA-256
  `ffb45dbe848de6ab2bdfc40c55e577a429e45791d6047c1fd0401b2b3311e0cf`;
- task generator `experiments/structure_net/tinyllm_temporal_language_task.py`
  with 10 passing generator-contract tests;
- a BabyLM pretraining run (`d8`, seed 7, 12,000 steps) in progress at
  `data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/`. Its loss
  curve is systems information; the campaign must record the final
  checkpoint's SHA-256 before the first fine-tune and may not change it
  afterward. If pretraining fails to complete, the campaign is blocked, not
  reconfigured silently.

## Question

Does an in-context gauge reference (the stated UTC offset clause) make the
UTC time-of-day target identifiable and learnable for a TinyLLM reading
templated English — reproducing the circle task's calibrated-identifiability
result in a language modality — and does BabyLM pretraining change that?

## Arms

`initialization x mode`, five fine-tune seeds each (`7, 17, 29, 41, 53`):

| Axis | Levels |
| --- | --- |
| initialization | `babylm_pretrained` (seed-7 pretraining checkpoint), `scratch` (random init, same tokenizer) |
| mode | `calibrated_text`, `uncalibrated_text`, `utc_oracle` |

Thirty cells total. The fine-tune seed controls task data sampling, batch
order, and any newly initialized state; the pretraining seed is not varied at
L1 (declared scope boundary).

## Fixed training protocol

- generator: `TemporalLanguageTaskConfig()` defaults (16 bins, sequence 96,
  train offsets = whole hours −11..+11 excluding {−7, +3}, templates 0–5,
  train person/event pools);
- 4,096 exact fibers (8,192 examples) for training; evaluation cohorts of
  1,024 examples per regime (`interpolation`, `composition`,
  `extrapolation`) at locked seeds;
- objective: soft-target cross-entropy of the final-position answer logits
  over the 16 answer tokens (identical to the circle campaigns);
- optimizer AdamW, lr `3e-4`, weight decay `0.01`, clip `1.0`, batch 64
  examples (32 fibers, both sheets), 600 steps, no early stopping;
- all parameters trainable in both initialization arms.

## Primary endpoints and gates

Exact-bin accuracy; chance is `1/16 = 0.0625`. A gate passes in an arm when
at least four of five seeds satisfy it.

| Gate | Cell | Requirement |
| --- | --- | --- |
| G1 oracle competence | `utc_oracle`, each init arm | interpolation >= 0.375 and composition >= 0.25 |
| G2 identifiability control | `uncalibrated_text`, each init arm | interpolation <= 0.125 |
| G3 calibration use | `calibrated_text`, each init arm | interpolation >= 0.30 and (calibrated − uncalibrated, seed-paired mean) >= 0.15 |

Secondary, descriptive only (no gate): the pretraining advantage
`babylm_pretrained − scratch` per mode and regime, and extrapolation
accuracies (extrapolation offsets are off the training grid; no L1 gate is
set on them).

## Outcome meanings

| Outcome | Classification | Next action |
| --- | --- | --- |
| G1–G3 pass in `babylm_pretrained` | `identifiable_and_learnable` | proceed to ladder L2 (calibration titration) on the retained checkpoints |
| G1 fails in both init arms | `oracle_task_too_hard` | stop; amend the task (budget or template complexity) with a new dated preregistration |
| G1 passes, G3 fails in both init arms | `in_context_calibration_unused` | stop the ladder; this bounds the external-validity claim and is reported as such |
| G2 fails in any arm | `identifiability_control_leak` | invalid; repair the generator, do not interpret G1/G3 |
| G1–G3 pass only in `scratch` | `pretraining_interference` | proceed with the scratch arm; report the inversion prominently |

## Validity contracts

The campaign is invalid unless: the tokenizer SHA above matches; the
pretraining checkpoint SHA is recorded before the first fine-tune and every
`babylm_pretrained` cell loads a state dict with that digest; generated
dataset digests are recorded per cell and identical across arms that share a
(mode, regime, seed); train/composition offset and template pools are
disjoint in the generated data; every metric is finite; per-cell results are
written atomically with scientific fingerprints and an exact resume returns
completed cells unchanged.

No outcome licenses changing thresholds, pooling shakedown cells, or
reinterpreting the uncalibrated control as anything but an identifiability
floor. Ladder stages L2–L6 each require their own preregistration.

## Expected artifacts and command

```bash
MPLCONFIGDIR=/tmp/matplotlib-temporal-l1 \
pixi run python -m experiments.structure_net.tinyllm_temporal_language_identifiability \
  --device cuda:1 \
  --output data/experiments/tinyllm_temporal_language_identifiability/20260812_l1_preregistered
```

Aggregate `campaign_results.json`, per-cell `result.json` and model
checkpoints under `runs/<init>/<mode>/seed_*/`, and a measured report in
`docs/08 - Analysis/` written from the aggregate.
