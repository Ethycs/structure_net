# Causal component pilot stopped at the dense donor control

Date: 2026-11-13. Study: `JSPACE-CAUSAL-COMPONENTS-01`.
Status: completed bounded pilot, stopped by its declared positive-control gate.
Classification: `dense_donor_control_failed`. Confirmed: false.

The immediate next experiment should locate a reliable causal intervention site.
Qwen answered the fresh source tasks correctly on **91/96 cases (94.79%)** and
the paired donor questions on **93/96 (96.88%)**. But replacing the entire final
prompt-token activation at zero-based block 24 with the donor's activation
redirected only **1/88 jointly correct pairs (1.14%)**. The remaining 87 kept
the original answer. The declared minimum donor success was 80%, so the
evaluation comparison of J components and remainders did not run.

This result does not establish that the J component has zero causal effect.
The primary component endpoint is **null / not measured because the gate
stopped the experiment**. It also does not establish that the model lacks the
intermediate concept. The tested single-token replacement is insufficient to
reliably install the donor's counterfactual computation at this depth.

- [Frozen protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-causal-components-plan.md).
- [Aggregate](../../../data/experiments/jspace_causal_components/20261113_v1/campaign_results.json).
- [Direct result](../../../data/experiments/jspace_causal_components/20261113_v1/result.json).
- Scientific fingerprint: `31a52b4dc6b3a534807ef4948333297da64c4e49756ebd2415806272260ecac5`.
- Result SHA-256: `3901095fefbe11ffcd2757cf51879731d4112ce6861a76f80de327285b8875a8`.
- Aggregate SHA-256: `1d3d8331124fc390e15afb7182c94a26d972bdfceaebac867efec6f0a0345770`.

## What ran

The model remained frozen Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, BF16, 36 blocks of width 2048.
All new tasks used one JSON template, disjoint name pools between splits,
balanced country/colour combinations, and fresh named worlds excluding all
worlds in the earlier relation study. Anonymous mapping shapes can recur.
Each donor used the same facts and asked about the person assigned the target
country. Source/donor questions form one paired unit.

| Measurement | Correct / total | Accuracy |
| --- | ---: | ---: |
| Training source answers | 247/256 | 96.48% |
| Development source answers | 30/32 | 93.75% |
| Development donor answers | 32/32 | 100.00% |
| Evaluation source answers | 91/96 | 94.79% |
| Evaluation donor answers | 93/96 | 96.88% |
| Evaluation pairs correct on both questions | 88/96 | 91.67% |

These are strict, complete-first-word model answers with an eight-token cap,
not probe scores or semantic-judge scores. The fresh evaluation source result
is below 95%; rounding it to 95% would hide the actual count.

The 16 separate shakedown examples checked finite computation, exact capture
save/reload, an identity intervention, and representative memory. They are not
quality evidence. All stated competence gates passed.

Candidate CAVs were conditional training means minus the overall training mean
at blocks 18 and 24. Each was split into a positive combination of at most 25
of the existing 94 dictionary atoms, plus an exact numerical remainder. The
J component accounted for 0.40–1.00% of CAV squared norm at block 18 and
0.85–1.81% at block 24. These fractions describe fitted CAVs, not activation
energy, task accuracy, or effects on generated answers. The sparse solver is
NNLS-refitted matching pursuit, not the paper's gradient-pursuit implementation.

On development data, full-CAV coefficient swaps and norm-matched signed-random
controls were tested at blocks 18 and 24, for the final position and the whole
question suffix. Each configuration had 30 eligible pairs. All four full edits
and all four matched controls achieved **0/30** target-answer changes. The
predeclared tie order selected block 24 / final position. No setting was changed
after inspecting this result; selection was saved before evaluation calls.

## Positive-control result

The dense control replaced the entire final-position state using the paired
donor capture. It did not restrict the edit to a CAV or J-derived subspace.

| Dense donor patch outcome | Eligible pairs |
| --- | ---: |
| Correct counterfactual answer | 1/88 |
| Original answer retained | 87/88 |
| Other output | 0/88 |

The target-success Wilson interval is 0.20–6.16%. Across all 96 cases, including
those excluded by the clean-pair competence condition, target success was 2/96,
original retention 92/96, and other output 2/96. The primary denominator remains
the 88 pairs fixed by clean outputs before any evaluation edit.

A separately labeled post-hoc diagnostic found a median donor-minus-source
activation change of **15.87%** of source norm among eligible pairs, ranging
from 10.53% to 20.99%. The substitution was a substantial state change, not a
numerically vanishing edit. This diagnostic is saved in
[dense_control_diagnostic.json](../../../data/experiments/jspace_causal_components/20261113_v1/dense_control_diagnostic.json)
and does not change the stopped primary endpoint.

## Interpretation and next test

One plausible explanation is that later attention reads the unchanged source
question from other token states and restores the source answer. Another is
that the relevant decision has not yet been consolidated at the final position.
These are hypotheses: the current test did not isolate either mechanism.
An implementation or alignment problem also warrants an explicit terminal-layer
positive control, even though the identity-hook and stored-case checks passed.

Run a new, bounded **dense activation-patching sweep** before expanding the
dictionary or interpreting J-component failures:

1. Check a terminal-block donor replacement. Its next-token distribution should
   match the donor after the shared final normalization and output projection.
   Compare logits directly to validate patch placement and case alignment.
2. Sweep final-token donor replacements through later layers, for example
   blocks 24, 29, 34, and 35. Record the first region where donor changes control
   the intended answer.
3. At matched-token-length source/donor prompts, replace the whole question
   suffix. This tests distributed question state while preserving the identical
   facts. Match lengths before outcomes; do not silently align unlike positions.
4. Once dense patching is reliable, decompose the effective donor-minus-source
   changes into J-derived components and remainders, and compare them with
   learned shared CAVs and matched controls. Only then interpret component
   differences as evidence about the selected J representation.

This is a revised next-step priority, not a claim that those additional tests
have already run. Continuous polynomial relations and Gröbner transport remain
downstream; no compression was attempted.

The paper's concept-swap examples operate across token positions, and its
probe-component comparisons ask about causal effects rather than just energy
or decoding. Our final-position donor failure is a reason to validate spatial
and layer coverage before treating this pilot as evidence against that proposal.
[Primary methods](https://www.transformer-circuits.pub/2026/workspace/index.html).

## Execution and reproduction

Nine CPU tests passed, covering fresh worlds/donor integrity, balanced splits,
edit reconstruction, coordinate swapping, scope masks, norm controls, zero
components, selection and paired statistics, and inclusive gate boundaries.
The reserved GPU captures passed the identity-hook and save/reload shakedown.

The training-capture lease `1ecdf39d0616` took 59.88 seconds inside its GPU stage.
The intervention lease `0e5e7ba455ed` took 115.08 seconds. Both used root Pixi's
locked `ncm` environment, Torch 2.4.1/CUDA 12.1, on the RTX 3060 with a 10,240 MiB
reservation. Peak allocated/reserved memory was 5.88/6.03 GiB. These stage times
are not throughput benchmarks. CPU decomposition used root default Pixi with
CUDA hidden. Both leases exited successfully and were released; no GPU booking
remained and no regular model service was loaded.

One prospective pilot ran, with no scientific retry. Partial/development
results, dictionaries, fitted CAVs, activations, raw outputs, exact selections,
stage manifests and producing source are retained under the artifact root.

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_causal_components prepare --output NEW_ROOT
/usr/bin/python3 /usr/local/bin/inference-ctl lease run --allocation 3060=10240 --minutes 60 --reason 'CAV training captures' -- pixi run --locked -e ncm python -m experiments.structure_net.jspace_causal_components train --output NEW_ROOT
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 pixi run --locked python -m experiments.structure_net.jspace_causal_components fit --output NEW_ROOT
/usr/bin/python3 /usr/local/bin/inference-ctl lease run --allocation 3060=10240 --minutes 60 --reason 'CAV causal tests' -- pixi run --locked -e ncm python -m experiments.structure_net.jspace_causal_components intervene --output NEW_ROOT
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_causal_components analyze --output NEW_ROOT
```

Stop at any declared gate and analyze the stopped result; do not invoke a
conditional downstream stage after its prerequisite fails.
