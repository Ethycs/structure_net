# Country information in the J-derived span and its remainder

Date: 2026-11-13. Study: `JSPACE-COUNTRY-PROBE-01`.
Status: completed CPU diagnostic; exploratory reuse, unconfirmed.
Classification: `exploratory_country_decodability_measured_no_causal_claim`.

The current 94-direction J-derived span is a weaker source of linearly
decodable intermediate-country information than the full activation, its
remainder, or the matched-width controls. At the development-selected block
24, evaluation country-probe accuracy was **43/128 (33.59%)** from J94,
**78/128 (60.94%)** from the full state, and **72/128 (56.25%)** from the J94
remainder. Chance is 25%. This identifies useful task information outside our
selected span; it does not establish where the model causally uses it.

These scores measure a newly fitted readout of the intermediate country.
They are **not Qwen answer accuracy**, and must not be combined with the
earlier 47.66% strict colour-answer score. No model inference ran here.

- [Protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-country-probe-plan.md).
- [Aggregate](../../../data/experiments/jspace_country_probe/20261113_v1/campaign_results.json).
- [Direct result](../../../data/experiments/jspace_country_probe/20261113_v1/result.json).
- Scientific fingerprint: `694e6608a6b8c855d6f3c76c6757d82014791bf76093ef74c7eb6e4c0a99c097`.
- Result SHA-256: `7b3999f3b7fcc40d95471440e2a0cf2fe7d8e70a742bb4c8f6f1bcdcec9efdd3`.
- Aggregate SHA-256: `113919b9084f95c2e14b094f80e38e0ffebf22f93addd103a5c8d083c83816f7`.

## Matched comparison

Each arm has its own ridge readout trained on the same 512 examples and
regularization selected using the same 128 development examples. Full-state
development accuracy selected a common layer before evaluation scoring.
The table evaluates all arms on the same 128 previously inspected examples.
Label classes are China, France, Germany, Russia, balanced independently of
the final colour. Every source prompt contains all four countries.

| Readout input | Correct / 128 | Probe accuracy | Wilson 95% interval | Development accuracy |
| --- | ---: | ---: | ---: | ---: |
| Full activation | 78 | 60.94% | 52.28–68.95% | 63.28% |
| Four country J directions | 39 | 30.47% | 23.16–38.92% | 28.91% |
| 94 J-derived directions | 43 | 33.59% | 26.00–42.15% | 57.03% |
| Remainder after J94 removal | 72 | 56.25% | 47.60–64.54% | 61.72% |
| Anchored PCA, 94 directions | 75 | 58.59% | 49.93–66.75% | 63.28% |
| Anchored random, 94 directions | 62 | 48.44% | 39.95–57.01% | 53.12% |

The primary J94-minus-full difference is **−27.34 percentage points**.
Its paired bootstrap interval is −35.16 to −19.53 points. This interval is
descriptive of the reused sample: it does not account for earlier exploration,
shared task structure, or other trained checkpoints.

Both PCA and random controls contain the same four initial J country
directions, followed by 90 additional directions. They are not unanchored
controls. The remainder has ambient width 2048 and maximum rank 1954, so it
is not a parameter-budget-matched compression arm. Ridge penalties were
normalized by training feature energy, with training-only feature means.

Twenty shared training-label permutations per arm, with development selection
repeated, produced mean accuracies between 24.53% and 26.48%. For J94 the
shuffle mean was 26.48%, with range 20.31–37.50%; the real 33.59% result lies
within that range. This is a sanity control, not a permutation significance
test, and it gives no strong positive evidence for the present J94 readout.

## Depth and formatting

All layer/arm/penalty choices were frozen before evaluation scoring. Other
layers are secondary diagnostics and do not replace block 24 as the primary
comparison.

| Zero-based block | Full state | J94 | Remainder | PCA94 | Random94 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 32/128 | 32/128 | 32/128 | 30/128 | 32/128 |
| 12 | 30/128 | 32/128 | 32/128 | 32/128 | 32/128 |
| 17 | 32/128 | 26/128 | 32/128 | 32/128 | 32/128 |
| 18 | 32/128 | 29/128 | 35/128 | 32/128 | 32/128 |
| 23 | 45/128 | 33/128 | 43/128 | 44/128 | 43/128 |
| 24 | 78/128 | 43/128 | 72/128 | 75/128 | 62/128 |

This readout begins generalizing better at the later measured blocks; it
cannot establish information absence at earlier blocks or other positions.
Only the final prompt token was captured, and layers after block 24 are not
included because the reused 94-token dictionary was unavailable there.

At block 24 the full-state probe scored 49/64 on JSON prompts and 29/64 on
reverse-roster prompts; J94 scored 27/64 and 16/64, respectively. The remainder
scored 50/64 and 22/64. Thus format generalization affects this probe even
without parsing a generated answer. This differs from the earlier model-output
formatting failure and does not prove either failure explains the other.

## What this changes about the next experiments

The earlier energy diagnostic found that J94 captures only 5.36% of raw
activation energy at this block. Small energy alone was not a reason to reject
the span. The new result adds a task-specific limitation: its learned readout
also generalizes substantially worse than the full state and its remainder.
No speedup or model-quality preservation has been demonstrated.

The most useful next experiments, in order, are:

1. **Find and causally test the concept directions.** Train country concept
   directions on fresh, consistent-format tasks and compare full directions,
   their sparse nonnegative J component, the remainder, and norm-matched
   controls. Test final-token interventions against all relevant prompt
   positions and a bounded band of layers. The earlier single-position swap
   can miss a distributed computation. Check clean task competence first.
2. **Test dictionary and calibration coverage.** Compare the current 94-token,
   16-paragraph estimate with a broader dictionary and more independent
   calibration. Hold the model, tasks, and sparse budget fixed. This separates
   a weak approximation from a failure of the underlying proposal. The present
   signed span is not the paper's sparse nonnegative representation.
3. **Return to relation transport with meaningful coordinates.** Once concept
   directions have measurable effects, fit source and cross-layer polynomial
   relations using training data, validate them on fresh tasks, and test their
   transport under interventions. Include a known neural positive control.
   Merely discovering destination relations does not show propagation of a
   source ideal. Compression remains a separate later test.

The paper's closest causal comparison decomposes concept probes and intervenes
using the full probe, its sparse J component, and the remainder. It also uses
interventions across token positions. That is a closer next comparison than
continuing to maximize activation-energy capture.
[Primary methods](https://www.transformer-circuits.pub/2026/workspace/index.html).

These are proposed next studies, not additional completed experiments or
confirmed causes of the current shortfall. All GPU work must run through the
reservation controller; this diagnostic required no GPU allocation.

## Execution and evidence

One run completed in 3.87 seconds of measured CPU analysis after process start,
using locked root Pixi, FP64 NumPy, and four BLAS threads. No GPU was queried or
allocated. No model weights were trained; the temporary ridge readouts were
fitted and saved. Parent capture hashes, expansion hashes, row/case alignment,
orthonormal bases, label balance, and saved/reloaded predictions were checked.

Six focused tests passed: direct ridge-solve agreement, train-only centering,
known-signal projection/removal, layer selection, score/interval arithmetic,
checkpoint round-trip, and invalid input handling. The first `pytest` console
invocation could not import the repository package; using root
`pixi run --locked python -m pytest` resolved collection before the experiment
launched. There was one scientific run, no retry or excluded outcome.

`heads.pt` contains every selected readout and the shuffled-label heads;
`selection.json` contains development selection and shuffle permutations;
`predictions.pt` contains per-example scores and identities. Original evidence
is append-only. The model/checkpoint, task data, parent dictionaries, and
previous results remain unchanged.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
pixi run --locked python -m experiments.structure_net.jspace_country_probe \
  --output data/experiments/jspace_country_probe/NEW_RUN
```
