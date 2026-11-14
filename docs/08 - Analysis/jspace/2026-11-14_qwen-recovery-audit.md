# Qwen quadratic recipe: why recovery is not established

Read-only diagnostic audit, 2026-11-14. This interprets the completed training
and held-out predictions; it does not change the frozen experiment or claim a
new successful intervention.

Run: `data/experiments/jspace_quadratic_recipe/20261114_v1`.
Fingerprint: `83f82e89e25ce8c47f5c1b4b35c7da6a94207d939434c2848365ec0f1c856503`.

## What the evidence establishes

| Model | Held-out correct / 512 | Accuracy |
|---|---:|---:|
| Native | 396 | 77.34375% |
| Compressed softmax | 211 | 41.2109375% |
| Compressed quadratic | 220 | 42.96875% |

The shared compression/training path fails even with softmax. Quadratic wins
86 and loses 77 cases against compressed softmax (paired exact binomial
p approximately 0.531); the nine-case difference does not isolate a successful
attention change. Native cached development predictions are 48/64 correct;
the selected final students are 24/64 and 26/64.

All 72 layer fits completed 12 epochs. This rules out missing training stages,
but does not establish adequate recovery. Layers 0–34 use only projected
feature MSE; output-distribution KL is introduced at layer 35, after the earlier
blocks have been frozen. Only 512 distinct training prompts (69,130 input
tokens) supply recovery data, with basis calibration using 128 prompts. Twelve
epochs repeat those examples; they do not expand coverage.

Local losses contained warnings. At layer 30, selected normalized feature MSE
increases from the preceding layer's approximately 0.079 to approximately 0.218
in both arms. At the final block it is 0.2792/0.2884, equivalent to relative RMS
error of approximately 52.8%/53.7% against the fixed train-target-energy scale.
This is not a direct percentage of semantic information lost.

Normalized losses are not comparable without their denominators. From layer 1
to 2, target energy rises from 0.425 to 33.737. Softmax raw feature MSE worsens
from 0.04746 to 0.06407 while the normalized value falls from 0.1117 to 0.001899.
That apparent cross-layer improvement cannot be read as recovered decisions.

Checkpoint selection also demonstrates objective mismatch. Quadratic final
development accuracy peaks at 30/64 in epoch 2, but the prescribed minimum-loss
selection chooses epoch 10 at 26/64. Changing selection alone would not explain
or repair the full deficit, and the existing held-out set must not be used to
tune a replacement.

## A concrete gap in the recovery objective

Let `U_i` be the basis fitted to native input boundary i and `P_i=U_i U_i^T`.
The current block targets the *output* boundary h in the same basis `U_i`, with
global anchor `a_i`. Even perfect projected recovery reconstructs only

`h_hat = a_i + P_i (h - a_i)`.

The next block uses independently fitted `U_(i+1)`. After the affine bridge,
its input error relative to directly projecting the teacher is

`U_(i+1)^T (h_hat - h) = -U_(i+1)^T (I-P_i) (h-a_i)`.

The current projected loss does not measure this term. Thus zero local loss
does not imply a correct next-block input. The code implements its chosen
projection algebra consistently, but this is a design gap in what recovery
certifies. Its actual contribution to the observed accuracy loss has not yet
been measured by an intervention.

The omitted complement is represented only by a fixed calibration anchor;
there is no input-dependent residual correction for everything discarded by
the half-width bottleneck. Local Jacobian energy retention (99.45–99.80%) does
not establish recoverability after these finite, repeated changes. In layers
30–35, only 146, 135, 120, 92, 79 and 35 retained directions exceed the numerical
functional threshold. The other retained directions are weak/null completion
under that threshold, not proven task-irrelevant coordinates. Separately, V
projection retains only approximately 76.7–94.4% of uncentered value energy.

## Experiments that should have preceded full-stack confidence

1. Score each compressed prefix through the original remaining blocks/readout
   on development data, to locate the first decision-level failure.
2. Measure the oracle handoff error above using native cached boundaries. Then
   compare supervision in the actual next-block basis and retention of the
   omitted residual. This separates a bad handoff from a bad local fit.
3. Feed teacher-derived final coordinates through original and fitted readouts
   to separate basis/readout capacity from accumulated student-prefix damage.
4. Only after identifying the failing interface, compare a small jointly
   recovered window or a residual correction, with downstream decision loss.
   More data and wider bases are hypotheses to test, not automatic remedies.

The methodological failure was treating implementation checks and declining
local loss as sufficient evidence of functional recovery. The two-block
schedule was extended to 36 blocks without testing that assumption at the
interfaces. The current result rejects this adaptation at the chosen budget;
it does not establish that task-derived compression or the original example
cannot work.

Sources: the run's immutable plan, per-layer `training.json`, `basis/*/fit.json`,
and three `evaluation/*/result.json` files; implementation in
`jspace_quadratic_recipe.py`, `jspace_quadratic_train.py`, and
`jspace_qwen_affine_block.py`. No new GPU work was launched during the CPU
benchmark, and no frozen producer files were changed by this audit.
