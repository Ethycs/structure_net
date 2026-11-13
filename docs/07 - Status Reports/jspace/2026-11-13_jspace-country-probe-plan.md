# Task information in the J-derived span and its remainder

Date: 2026-11-13. ID: `JSPACE-COUNTRY-PROBE-01`.

This is a CPU-only exploratory follow-up. The question is whether the small
activation-energy fraction found in the bounded J dictionary nevertheless
preserves decodable information about the task's intermediate country.
Compare both directions of the difference; there is no confirmatory pass gate.
The primary endpoint is J94 minus full-state country-probe accuracy, at a
common layer selected using development data. This is probe accuracy, not
Qwen answer accuracy, causal use, mutual information, or compression quality.

Reuse the frozen Qwen2.5-3B captures from
`jspace_relation_propagation/20261113_neural_v1`: 512 training, 128 development,
and 128 evaluation examples. The country label is `case.intermediate`, one of
China, France, Germany, Russia; each label is balanced independently of the
final colour answer. Every prompt contains all four country names. Splits have
different names, worlds, and templates. The evaluation examples have already
been inspected in other studies, so these are descriptive reused-test results.
No fresh confirmation or recovery of the parent hypothesis is possible here.

Use the saved final-prompt-token states at zero-based blocks 11, 12, 17, 18,
23, 24. Reuse and hash-check the training-fitted orthonormal expansions from
`jspace_residual_expansion/20261113_v1`. Arms:

| Arm | Probe input |
| --- | --- |
| full | Original 2048-dimensional state |
| J4 | Coordinates in the four country-vector span |
| J94 | Coordinates in the full 94-vector J-derived span |
| remainder | Original state minus its J94 orthogonal projection |
| PCA94 | Coordinates in the existing 94-dimensional PCA control |
| random94 | Coordinates in the existing 94-dimensional random control |

Both controls include the same four initial country directions, followed by
90 PCA or random directions. They are anchored controls, not pure PCA or an
entirely independent random subspace. The remainder has ambient width 2048
and rank at most 1954. J4/J94 are **signed linear spans**, not the paper's sparse
nonnegative J-space. The dictionary has only 94 selected tokens, calibrated on
16 paragraphs, and is not a full-vocabulary dictionary.

Fit four-output ridge regressions to one-hot country labels. Center features
using training means only; fit an unpenalized label-prior intercept. The ridge
penalty is lambda times the mean diagonal of the centered training Gram
matrix. Use lambda in `[0.0001,0.001,0.01,0.1,1,10,100]`. For each arm/layer,
select lambda by development accuracy, then lower development mean squared
error, then smaller lambda. Select one common primary layer by full-state
development accuracy, then mean squared error, then smaller layer index.
Freeze all selected heads, development scores, and that layer before loading
evaluation activations for scoring. Save predictions for every layer and arm.

At the primary layer include 20 training-label permutations, generated once
with RNG seed 137 and shared across arms. Each shuffled fit gets the same
development selection procedure against true development labels. The layer
stays fixed. Freeze these heads before evaluation. Report their mean and
range; they are a sanity control, not a calibrated permutation p-value.

Report integer correct/total, accuracy, four-class confusion matrices, and
Wilson 95% intervals, plus template-specific results and empirical majority
chance. Report a paired 2000-resample bootstrap interval for the J94-minus-full
accuracy difference, seed 149. Intervals describe this reused sample and do
not correct for the prior exploratory work or shared generated-task structure.
No model weights, prompts, generated answers, or capture locations change.

Use root locked Pixi, FP64 NumPy, four BLAS threads, and CUDA hidden. Verify
parent plan/capture identities, expansion hashes, finite arrays, label counts,
case-ID disjointness, saved/reloaded heads, and orthonormal projectors. Stop
and record an invalid run if these checks fail; never retry an unfavourable
scientific result. Source, dependencies, protocol, lockfile, inputs, and runtime
are fingerprinted and archived before fitting. New append-only artifact root:
`data/experiments/jspace_country_probe/20261113_v1/`.

Reproduce with:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
pixi run --locked python -m experiments.structure_net.jspace_country_probe \
  --output data/experiments/jspace_country_probe/NEW_RUN
```

Whatever the result, the next causal experiment requires fresh tasks and a
reserved GPU: compare full concept directions, their sparse nonnegative
J-dictionary component, the remainder, and norm-matched controls. A useful
probe alone does not establish a causal concept activation vector. Relation
transport and compression remain separate downstream experiments.
