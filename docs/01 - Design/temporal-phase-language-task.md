# Temporal-Phase Language Task

**Status:** PROPOSED DESIGN — external-validity rung 1 for the TinyLLM quotient program  
**Date:** 2026-08-12  
**Depends on:** `../08 - Analysis/2026-08-06_tinyllm-unifying-interpretability-direction.md`; `../02 - Implementation/experiment-and-report-authoring-guide.md`  
**Goal:** replay the program's key causal results on a *language* task whose latent geometry matches the circle task exactly, so that every frozen-checkpoint intervention, control, and gate ports without redefinition.

## Why this task

The circle-task results are nontrivial inside their sandbox; their external
validity is untested. The cheapest decisive test is not a natural corpus — it
is a **synthetic language task with exact group structure**, because the
program's evidentiary machinery (exact fibers, orbit averaging, shuffled-fiber
controls, calibration titration) requires generative control that wild text
cannot provide. Rung 1 therefore asks:

> Do the mechanisms survive the move from continuous sensor tokens to
> discrete linguistic tokenization, lexical variation, and compositional
> syntax — holding the latent geometry fixed?

A later rung 2 (pretrained small LM, natural temporal expressions) is out of
scope here and only justified if rung 1 transfers.

## The task

The model reads templated English describing an event time in a **local
frame**, plus a **calibration clause** stating that frame's UTC offset, and
must answer with the event's UTC time-of-day, quantized into 16 ordered
90-minute bins (ordered answer tokens, exactly as the circle task).

```text
"despite the rain , Priya recorded the start of the rehearsal as
 twenty five minutes past seven in the evening . local clocks run
 four hours and thirty minutes ahead of coordinated universal time .
 <query>"                                  ->  UTC bin for 15:!5 - er, 14:55
```

## Structure mapping (circle -> language)

| Circle-task element | Language analog | Exactness |
| --- | --- | --- |
| latent phase `phi` on S^1 | UTC minute-of-day on the 24 h circle | exact (identical geometry) |
| 16 ordered answer bins, wrapped soft targets | 16 UTC bins of 90 min, same wrapped von-Mises targets | identical code |
| orientation gauge (rotation of the circle) | the frame's UTC offset — a literal rotation of the time circle | exact group action |
| calibration packet (observed reference) | the calibration clause stating the offset, in text | in-context gauge |
| calibration noise `sigma` (radians) | perturb the *stated* offset by `sigma` minutes; 2 pi rad = 1440 min | continuous titration |
| C2 deck / branch (two sheets over one base) | "`m` minutes past `H`" vs "`60-m` minutes to `H+1`" — an exact double cover of the dial; branch = anchor choice, task-irrelevant | exact fiber pairs |
| N3 nuisance support | template identity, names, event nouns, filler clauses, clause order | exact orbits (templated) |
| repeated acquisition (m sensor repeats) | m witness clauses with independent clock error ("Ana's watch showed ...; Ben's read ...") | same inverse-square prediction |
| composition shift | in-range offsets x held-out templates and name/event pools | held-out combinations |
| extrapolation shift | offsets off the training resolution grid (half/quarter-hour frames) + unseen templates | outside-range |
| analytic calibrated arm | `utc_oracle`: text pre-normalized to UTC, no clause (positive control) | oracle canonicalization |
| learned calibrated arm | `calibrated_text`: local text + true clause; model must *use* the in-context gauge | the language-natural condition |
| uncalibrated arm | `uncalibrated_text`: local text, no clause; offset uniform -> target unidentifiable by construction | identifiability negative control |

Two deliberate differences, stated up front: (1) the gauge reference arrives
in-band (in text) rather than as a separate metadata tensor — this is the
language-natural formulation and is itself part of what rung 1 tests; (2) the
task text is templated, so the *latent* stays exactly controlled even though
tokenization and pretraining are natural (below).

## BabyLM pretraining

Rung 1 uses real language pretraining rather than a from-scratch closed
lexicon. The **BabyLM strict-small corpus** (~10M words of child-directed
speech, dialogue, children's stories, simple Wikipedia; cleaned split of
`cambridge-climb/BabyLM`) supplies:

- a **BPE tokenizer trained on BabyLM text** (specials at ids 0-3, text
  vocabulary below 32,000, the 16 answer bins pinned as added tokens at
  32,000-32,015 so the circle-task answer-interface layout is preserved
  exactly and answer ids can never be emitted by text tokenization);
- a **pretrained TinyLLM d8** (causal LM on BabyLM) as the initialization for
  task fine-tuning.

Arms:

| Arm | Initialization | Purpose |
| --- | --- | --- |
| `babylm_pretrained` | BabyLM-pretrained TinyLLM d8 | primary: mechanisms in a model with real linguistic representations |
| `scratch` | random init, same tokenizer | ablation: isolates what pretraining contributes |
| `utc_oracle` variant of each | task text pre-normalized to UTC | positive control |
| `uncalibrated` variant of each | no offset clause | identifiability negative control |

Both arms share one tokenizer and one task generator, so every fiber, orbit,
and calibration intervention is bit-identical across arms.

## Replication ladder

Ordered by dependency; each stage preregisters separately and reuses the
frozen checkpoints of stage L1.

| Stage | Replays | Circle-task source result | Directional prediction |
| --- | --- | --- | --- |
| L0 | generator validity (fibers exact, orbits closed, identifiability by construction) | — | systems only |
| L1 | calibrated identifiability | calibrated front-end 5/5 vs uncalibrated failure | oracle and calibrated arms pass; uncalibrated near chance |
| L2 | calibration-offset noise titration + readout-only repair | `reference_precision_critical`; two-regime ordering; arm-stratified readout repair | representation gates outlast exact-bin utility; refit head repairs mild noise |
| L3 | repeated acquisition | inverse-square recovery at m=64 | slope in [-0.6,-0.4]; recovery at preregistered m |
| L4 | Reynolds orbit-averaging depth scan | cover-to-quotient causal front | early averaging destroys task, late preserves |
| L5 | posterior-coordinate rank ladder | `high_rank_answer_chart` | the headline transfer test: is the interface again high-rank? |
| L6 | cross-seed feature swap | equivariance does not fix the gauge | swaps fail without a declared anchor convention |

L5 is the decision point named in the program synthesis: if the
interface-dimensionality phenomenon reproduces under linguistic tokenization,
the sandbox result generalizes at least across observation modalities; if it
fails, the failure mode (compact chart? nonintegrable?) is itself the finding.

## Task parameters (locked at L1 preregistration)

- times on a 5-minute grid with minutes in {5,...,55} (both sheets nondegenerate);
- train offsets: whole-hour offsets {-11..+11} h excluding held-out {-7, +3} h;
- composition: held-out whole-hour offsets {-7, +3} x held-out templates/names;
- extrapolation: half-hour offsets {-9.5, -3.5, +4.5, +10.5} h x all templates;
- 8 report templates (6 train / 2 held out), disjoint train/eval name and event pools;
- answer tokens 32000..32015 over a 50,257 vocabulary; word lexicon from id 100;
- fixed sequence length with mid-sequence padding so the query token is always final
  (keeps every downstream final-position intervention identical);
- TinyLLM d8 preset, 5 seeds (7, 17, 29, 41, 53), same trainer shape as the
  calibrated-front-end campaign.

## What would count as failure to transfer

Each stage's preregistration carries its own gates, but the design-level
falsifier is: if L1's `calibrated_text` arm cannot exceed the uncalibrated
control under any registered training budget, the in-context gauge is not
learnable at this scale and the ladder stops — that outcome would bound the
program's external validity claim rather than extend it, and must be reported
as such.
