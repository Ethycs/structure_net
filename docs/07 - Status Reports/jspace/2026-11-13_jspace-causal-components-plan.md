# Causal edits from learned concepts and their J-derived components

Date: 2026-11-13. ID: `JSPACE-CAUSAL-COMPONENTS-01`.
Prospective bounded pilot on one frozen Qwen2.5-3B-Instruct checkpoint;
underpowered for a model-population claim. Earlier exploratory probe results
motivate the design but are not pooled with these new evaluation outcomes.

Question: can a learned intermediate-country edit redirect the model's colour
answer, and is that effect concentrated in its sparse nonnegative J component?
The prediction is greater target-specific swap success for that component than
for a norm-matched random edit and a norm-matched remainder edit. First require
competent clean behavior and a working dense donor control.

## Fresh task population

Keep the two-hop person → country → colour task and use the earlier JSON
presentation for **every** split. This deliberately tests one consistent format,
not generalization across templates. Generate 256 training, 32 development,
96 evaluation and 16 shakedown source worlds with seed 151; shakedown uses
seed 1000154. Use the parent generator's disjoint name pools and balanced
country/answer combinations. Reject every named world from the parent relation
study, plus any world already emitted in this study. Anonymous permutation
shapes can recur; the unit is a named world, not a unique abstract permutation.

For each development/evaluation source, construct a paired donor question about
the other person whose country is `swap_to`, using exactly the same two tables.
The expected counterfactual is that country's assigned colour. Source and donor
are one paired unit; donor questions are never counted as independent examples.
All four countries and colours appear in every prompt. Preserve the parent
greedy complete-first-word parser and eight-token cap. No semantic rescoring.

## Representation and edit

Observe final-prompt-token block outputs at zero-based layers 18 and 24.
On training data, compute each country's conditional mean minus the global
mean. These four vectors are candidate CAVs; the name does not presume a causal
effect. Use all 256 training examples without filtering by generated accuracy.

Decompose each vector with the existing normalized 94-token J dictionary from
the residual-expansion study. Use its NNLS-refitted positive matching pursuit,
initialized with the four country atoms and bounded to 25 selected atoms;
retain the full-94 NNLS endpoint as a diagnostic. This is not the paper's
gradient-pursuit implementation or a full-vocabulary J-space. Define
`C = J_part + remainder`; the remainder need not be orthogonal to every atom.
Reuse the dictionary unchanged to isolate the learned-direction intervention.

For each source/target pair let `B` contain their two full CAV rows and let
`c = (h - training_mean) @ pinv(B)`. Swap those two coefficients. Define
`delta_full = (swap(c)-c) @ B`; substitute the J rows or remainder rows, using
the **same full-vector coefficients**, to obtain `delta_J` and `delta_R`.
Thus `delta_full = delta_J + delta_R` before numerical rounding. This is an
affine coordinate edit, not an exact replication of the paper's swap rule.

Test six arms:

| Arm | Edit |
| --- | --- |
| full | `delta_full` |
| J | `delta_J` |
| remainder | `delta_R` |
| random_J | Fixed seeded signed permutation of `delta_J` |
| random_full | Same isometry applied to `delta_full` |
| remainder_norm_J | `delta_R` rescaled to `delta_J`'s norm at each position |

The signed permutation uses seed `211 + layer`, shared across cases. Norms
match in FP32 before casting to the model's BF16; report actual cast norms.
A zero J edit remains a valid no-op in its matched controls. Reject an
unmatchable nonzero J/zero-remainder case rather than silently changing norms.

Compare prefill-only editing scopes `final` and `query_suffix`, where the latter
starts at the question after the JSON facts and continues through the generation
prefix. Locate it using the tokenizer's character offsets. Do not edit the facts
or cached generation tokens. The CAVs and centering origin are learned at the
final position; applying them across the suffix is an explicit transport
hypothesis. No layer band or all-fact-token intervention runs in this pilot.

## Selection, gates, and endpoints

1. Shakedown: finite real model outputs, identity-hook behavior, capture
   save/reload, and peak memory below the reservation. These 16 cases are
   systems checks only.
2. Development source and donor answer accuracies must each be at least 90%,
   with at least 80% of paired units correct on both. Otherwise stop before
   training capture. Training answer accuracy must also be at least 90%.
3. Fit and freeze all CAVs/components on training data. On development only,
   evaluate `full` and `random_full` for four configurations, in tie order
   `(24, final), (24, query_suffix), (18, final), (18, query_suffix)`.
   Select the greatest full-minus-random target-answer success rate among
   clean-source-and-donor-correct pairs, then the greater full rate, then the
   declared tie order. Selection always chooses one configuration; a weak
   development edit is reported and does not trigger parameter retries.
4. Freeze selection before evaluation model calls. Evaluation source/donor
   accuracies must each reach 90%, with 80% jointly correct; otherwise stop.
5. At the selected layer, replace the final source activation with the saved
   donor's final activation. This dense positive control must redirect at
   least 80% of jointly correct pairs; otherwise stop before the six-arm test.
6. Run all six arms on all 96 evaluation source cases. The primary denominator
   is the jointly correct source/donor subset determined before any edits.
   Also report all-case results, original-answer retention, and other outputs.

Primary causal assay gate: full-edit target success at least 20%, with greater
success than random_full and paired two-sided exact McNemar p ≤ .05. If this
fails, classify `learned_CAV_causality_not_established`; it is inconclusive about
whether a good CAV's effect could be concentrated in J-space.

Conditional J endpoint: J target success at least 20%, advantages of at least
10 percentage points over **both** random_J and remainder_norm_J, and paired
exact p ≤ .05 for each. If both full and J gates pass, classify
`preliminary_J_component_causal_advantage`; otherwise, after a passing full
gate, classify `full_CAV_causal_J_advantage_not_established`.
Keep `confirmed: false` for this single-checkpoint bounded pilot. Include Wilson
intervals, paired success counts and full raw outputs. Unmatched remainder is
secondary: its amplitude differs. No test-tuned strength, layer, position,
dictionary, prompt, or exclusion is allowed.

## Execution and provenance

Use root locked Pixi. CPU preparation/fitting/analysis run with CUDA hidden.
GPU capture and interventions run in the `ncm` environment under
`inference-ctl lease run --allocation 3060=10240 --minutes 60`, one worker.
Verify a live lease before any CUDA query or model load and every 32 cases.
The supervisor releases the lease on child exit. No deployed services restart.

Freeze configuration, complete cases, model identity, parent dictionary hashes,
protocol, producing source and dependencies before new outcomes. Copy the
producer into a new append-only root:
`data/experiments/jspace_causal_components/20261113_v1/`.
Save stage manifests and partial events on failure. No scientific retry.
CPU tests cover task/donor integrity, split and parent exclusions, exact edit
decomposition, norm controls, scope masking, pair selection, and gate arithmetic.

This experiment neither compresses the network nor fits a Gröbner basis.
If causal coordinates become usable, the next study can test continuous
relations and their transport. Broader dictionary/calibration remains a separate
follow-up; no negative conclusion here invalidates the full paper.
