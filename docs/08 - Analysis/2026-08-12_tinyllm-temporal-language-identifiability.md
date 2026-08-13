# TinyLLM temporal-language identifiability (ladder L1)

**Status:** PREREGISTERED STOP CONDITION FIRED — `in_context_calibration_unused`; oracle competence and identifiability control both pass  
**Date:** 2026-08-12  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, training campaign with frozen generator and provenance contracts  
**Hypothesis:** `tinyllm-temporal-language-identifiability-v1`  
**Preregistration:** [L1 preregistration](../07%20-%20Status%20Reports/2026-08-12_tinyllm-temporal-language-identifiability-preregistration.md)  
**Design:** [temporal-phase language task](../01%20-%20Design/temporal-phase-language-task.md)

## Verdict

The temporal-phase language task is **readable but not yet gauge-usable** at
the locked training budget. Reading a templated English time expression is
learned almost perfectly: the UTC-oracle arms reach mean exact-bin accuracy
`0.935` (BabyLM-pretrained) and `0.836` (scratch) against a `0.375` gate and
a `0.0625` chance floor, with the pretraining advantage visible at about ten
points and stable across all three regimes. The identifiability control is
clean: with no offset clause, every seed in both arms sits at chance
(`0.061--0.068`), exactly as the construction requires.

The in-context gauge is not used. In all ten calibrated cells (five seeds ×
two initializations) accuracy is indistinguishable from the uncalibrated
control — seed-paired margins are `-0.004` to `+0.023` against a required
`0.15` — so gate G3 passes `0/5` in both arms. The locked classification is:

```text
in_context_calibration_unused
```

Per the preregistration, this outcome **stops the ladder** and bounds the
external-validity claim rather than extending it: at this scale and budget,
the mechanism the circle program studied (a calibrated reference making a
gauge-ambiguous target usable) does not spontaneously emerge when the
reference arrives in-band as language. Composing a parsed time with a parsed
offset through modular arithmetic is the failure point, not parsing, not
identifiability, and not optimization health.

## Preregistered gates

An arm passes a gate when at least four of five seeds satisfy it.

| Gate | Requirement | BabyLM-pretrained | Scratch | Verdict |
| --- | --- | ---: | ---: | --- |
| G1 oracle competence | interp >= 0.375 and comp >= 0.25 | 5/5 | 5/5 | pass |
| G2 identifiability control | uncalibrated interp <= 0.125 | 5/5 | 5/5 | pass |
| G3 calibration use | calibrated interp >= 0.30 and margin >= 0.15 | **0/5** | **0/5** | **fail** |

## Mean exact-bin accuracy (five seeds)

| Initialization | Mode | Interpolation | Composition | Extrapolation |
| --- | --- | ---: | ---: | ---: |
| babylm_pretrained | utc_oracle | **0.935** | **0.885** | **0.916** |
| babylm_pretrained | calibrated | 0.071 | 0.070 | 0.069 |
| babylm_pretrained | uncalibrated | 0.068 | 0.066 | 0.062 |
| scratch | utc_oracle | 0.836 | 0.809 | 0.830 |
| scratch | calibrated | 0.061 | 0.059 | 0.062 |
| scratch | uncalibrated | 0.061 | 0.066 | 0.066 |

Chance is `0.0625`. The oracle arms generalize across held-out templates,
names, and off-grid offsets (composition and extrapolation stay within a few
points of interpolation), so the language surface itself — including the C2
"past/to" double cover — is handled robustly once the gauge burden is
removed.

## Interpretation

Three separate claims come apart cleanly:

1. **Parsing transfers.** A 51M-parameter model, BabyLM-pretrained or not,
   maps forty-token templated English time reports onto the 16-bin circle
   almost perfectly, across both C2 sheets and held-out surface variation.
   BabyLM pretraining is worth about ten points and faster acquisition (the
   systems shakedown showed `0.60` vs `0.10` at 60 steps).
2. **The construction is sound.** The uncalibrated control sits exactly at
   chance, confirming the offset genuinely destroys identifiability — the
   task cannot be gamed from the report clause alone.
3. **Gauge composition does not emerge.** The calibrated arm has all the
   information (G2 proves the offset is necessary; the clause states it) and
   an optimization path that works for the oracle arm, yet after 600 steps it
   has not even begun to move off chance. The bottleneck is learning the
   two-argument modular composition `UTC = local − offset`, a qualitatively
   different circuit from the one-argument mapping the oracle arm learns.

This is the language analog's version of the circle program's earliest
lesson — identifiability must be *engineered into training*, not assumed —
now with a sharper edge: in the sensor task a calibrated front end could
apply the inverse transform analytically; in language there is no analytic
arm, so the composition must be learned, and at this budget it is not.

## Evidence pedigree

The preregistration was locked before any fine-tune outcome. A systems-only
shakedown (seed 7, six cells, 60 steps, 256 fibers, separate root) ran before
the campaign and exposed reduced-budget seed-7 outcomes; no gate, threshold,
budget, or protocol constant changed afterward. The campaign is therefore
preregistered on its declared protocol with seed-7 outcome exposure at a
different (non-pooled) budget.

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed | 30 / 30 / 0 cells |
| trained models | 30 (all parameters trainable) |
| training protocol | 4,096 fibers, 600 steps, batch 32 fibers (64 examples), AdamW 3e-4 |
| evaluation | 1,024 examples per regime at locked cohort seeds |
| tokenizer SHA-256 | `ffb45dbe848de6ab2bdfc40c55e577a429e45791d6047c1fd0401b2b3311e0cf` |
| pretraining checkpoint SHA-256 | `5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09` (verified per cell) |
| dataset digest parity across init arms | pass (all 15 mode×seed pairs identical) |
| finite numerical contract | pass |
| device | NVIDIA GeForce RTX 2060 SUPER, `cuda:1` |
| analysis time | 7,406 seconds |

## Decision

The ladder stops here per the locked outcome table. Stages L2–L6 (calibration
titration through the rank-ladder replay) are not licensed, because they all
presuppose a system that uses the gauge.

Two preregisterable follow-ups exist, in increasing cost order, and each
requires its own dated preregistration rather than an amendment of this
campaign:

1. **Budget escalation (L1b):** same task, same gates, training steps raised
   (with a declared schedule, e.g. 6,000) and optionally a mixed curriculum
   (oracle and calibrated examples interleaved). Tests whether gauge
   composition is slow rather than absent.
2. **Curriculum or scale change:** offset-magnitude curriculum (start with
   zero/small offsets), or a larger preset. Tests whether the composition
   needs shaping, mirroring how the circle program's learned front end
   required its own training signal.

Neither is launched by this result. If both later fail, the honest summary is
that the program's calibrated-quotient mechanism has a real language-modality
precondition — learnable gauge composition — that the synthetic sensor
setting hid, and that boundary is itself the external-validity finding.

## Artifacts and reproduction

- campaign:
  `data/experiments/tinyllm_temporal_language_identifiability/20260812_l1_preregistered/campaign_results.json`
- per-cell records and checkpoints:
  `data/experiments/tinyllm_temporal_language_identifiability/20260812_l1_preregistered/runs/<init>/<mode>/seed_*/`
- systems-only shakedown (never pooled):
  `data/experiments/tinyllm_temporal_language_identifiability/20260812_shakedown/`
- pretraining:
  `data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/pretrain_summary.json`
  (val loss 4.893, perplexity 133.4)
- runner:
  `experiments/structure_net/tinyllm_temporal_language_identifiability.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-temporal-l1 \
pixi run python -m experiments.structure_net.tinyllm_temporal_language_identifiability \
  --device cuda:1 \
  --output data/experiments/tinyllm_temporal_language_identifiability/20260812_l1_preregistered
```

## Scope boundary

Templated language with a closed grammar, one pretraining seed, one model
scale, one budget. The negative G3 result is a statement about this budget
and scale, not about the task's learnability in general; the positive G1/G2
results establish that the task construction and its identifiability
structure are sound carriers for the ladder if gauge composition is unlocked.
