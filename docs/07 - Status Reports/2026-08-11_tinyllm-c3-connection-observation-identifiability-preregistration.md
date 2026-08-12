# TinyLLM C3 connection-observation identifiability audit preregistration

**Status:** PREREGISTERED — NO TRAINING AUTHORIZED

**Date:** 2026-08-11

**Hypothesis ID:** `tinyllm-c3-connection-observation-identifiability-v1`

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`; deterministic
identifiability and frozen-artifact audit

## Question

Do the missing, partial, or known-noisy connection variants named by the
relational-connection acquisition report create a learning problem that can
pay rent over fixed analytic transport under the current C3 generator?

The directional prediction is negative:

> The endpoint task uses the seven observed edges only through total C3
> holonomy. That statistic is sufficient, erasing any one edge destroys point
> identifiability because the interior phases are independent, and independent
> symmetric connection noise has a closed-form Bayes attenuation. These scope
> changes therefore do not license TinyLLM or another acquisition campaign.

No result from this audit may alter the failed `1/5` primary acquisition gate
or upgrade the registered `4/5` artifact-only readout diagnostic.

## Sources and unit of replication

The frozen source is the five-seed relational-connection acquisition campaign:

```text
data/experiments/tinyllm_c3_relational_connection_acquisition/
  20260811_preregistered/campaign_results.json
SHA-256: b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a
```

The replication unit is one of the five independently initialized source
models `(1453, 1471, 1483, 1531, 1543)`. Each model is paired with fresh
composition and extrapolation cohorts generated from seed bases `1_173_107`
and `1_175_107`, respectively. Each cohort contains `1,024` examples. These
streams were not used for source training, source evaluation, pilot selection,
or the corrective readout audit.

All five `learned_true` final checkpoints and the fixed six-weight analytic
witness are read-only sources. Optimizer steps, probe fits, threshold tuning,
and TinyLLM instantiation are forbidden.

## Declared observation algebra

For edge connection values `A_t in C3`, define total holonomy

```text
H(A) = sum_t A_t mod 3.
```

The source module transports the final charged endpoint by `H(A)` before
forming its neutral product with the initial endpoint. A canonical total-only
representative is

```text
A_total = (0, 0, 0, 0, 0, 0, H(A)).
```

For independent symmetric additive edge error with probability `p`,

```text
P(E_t = 0)  = 1 - p,
P(E_t = +1) = p / 2,
P(E_t = -1) = p / 2.
```

The charge-one Fourier coefficient of the total error is declared to be

```text
lambda(p) = (1 - 3p/2)^7.
```

For an ideal continuous charged observation with noisy holonomy, the declared
conditional-mean predictor is `lambda(p)` times the naively transported cosine.
Its irreducible RMSE is

```text
sqrt((1 - lambda(p)^2) / 2),
```

and its correlation with the true cosine is `abs(lambda(p))` under the uniform
phase law. The audit evaluates `p in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2)`.

## Exact single-edge erasure witness

For each of the seven edge positions, construct two latent sequences as
follows:

1. begin with zero physical phases and zero local gauges;
2. in the second sequence, add `2pi/3` to every phase and add one modulo three
   to every gauge strictly after the erased edge;
3. keep amplitude, offset, and drift identical.

This suffix transformation leaves all observed tokens unchanged. It leaves all
visible connections unchanged and changes only the erased edge. The endpoint
targets are `1` and `-1/2`, so their absolute separation is `1.5`.

This is an observation-level collision, not a sampled failure of an estimator.

## Primary endpoints and joint gate

The hypothesis is supported only if every item below passes:

1. **Total-holonomy sufficiency.** On both fresh shifts, replacing every full
   connection by `A_total` changes analytic and frozen learned predictions by
   at most `1e-6` in all five source seeds. The analytic witness must retain its
   existing scalar/task endpoint gate in all ten seed-shift cells.
2. **Single-edge erasure non-identifiability.** All seven erased-edge witnesses
   have bit-identical quantized tokens, identical calibration, identical visible
   connections, and target separation at least `1.49`.
3. **Known-noise law.** Exhaustive enumeration of all `3^7 = 2,187` edge-error
   patterns matches `lambda(p)` and the conditional-mean attenuation formula to
   maximum absolute error `1e-12` at every declared `p`.
4. **Current-gate consequence.** The analytic continuous ceiling fails at least
   one current learned scalar gate (`correlation >= .999`, `RMSE <= .01`) at
   every declared nonzero `p >= 1e-4`. The exact threshold probabilities for
   both gates must be reported rather than inferred from the sampled cohorts.
5. **Integrity.** All source result and checkpoint hashes match, both fresh
   generator families have zero saturation and 16-bin target coverage, all
   numbers are finite, and the audit performs zero optimizer steps and
   instantiates zero TinyLLM models.

The audit is a single deterministic campaign. Seedwise counts are retained for
the frozen prediction comparisons; the algebraic collision and enumeration
gates must pass universally rather than by majority vote.

## Secondary measurements

- clean analytic and learned metrics on the fresh cohorts;
- the analytic correlation/RMSE ceiling at every declared noise probability;
- the maximum frozen-prediction difference between full and canonical-total
  connections;
- the largest token, calibration, and visible-connection discrepancy in the
  erasure witnesses;
- current-gate noise tolerances derived from the closed form.

These diagnostics cannot rescue a failed primary endpoint.

## Outcome meanings

| Outcome | Interpretation | Program action |
| --- | --- | --- |
| all gates pass | total holonomy is the minimal point-identifying statistic; erasure and known symmetric noise have no learned advantage under this generator | close missing/partial and known-noise acquisition; require unknown observation-dependent uncertainty or a different dynamics law before training |
| total-only equivalence fails | the implemented architecture uses connection detail beyond holonomy | repair the algebraic claim before any new campaign |
| an erasure witness fails | the claimed non-identifiability construction is wrong | keep partial-connection scope open and diagnose the generator |
| noise enumeration fails | the closed-form uncertainty model is wrong | correct the analytic ceiling; do not train |
| source or data integrity fails | evidence is not comparable | stop without interpreting task metrics |

## Method boundaries

The audit does not cover observation-dependent error, repeated calibration
measurements, correlated errors whose law is unknown, temporal dynamics that
make a missing edge inferable, or a new task invariant to connection error. It
does not claim that learning can never help with connections. It asks whether
the two cheapest proposed changes—erasure under the present independent-phase
generator and known independent symmetric noise—create value beyond analytic
transport.

## Planned artifacts

```text
experiments/structure_net/
  tinyllm_c3_connection_observation_identifiability.py
tests/structure_net/
  test_tinyllm_c3_connection_observation_identifiability.py
data/experiments/tinyllm_c3_connection_observation_identifiability/
  20260811_preregistered/result.json
docs/08 - Analysis/
  2026-08-11_tinyllm-c3-connection-observation-identifiability.md
```

Planned command:

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_c3_connection_observation_identifiability
```
