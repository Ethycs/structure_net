# TinyLLM shared-bias reference recentering

**Verdict:** CONFIRMED  
**Date:** 2026-08-10  
**Hypothesis:** `tinyllm-bias-reference-recentering-v2`  
**Evidence role:** `preregistered_pre_model_numerical_corrective_bias_reference_intervention`

An observed zero-signal pilot exposed to the same persistent sensor bias repairs
all ten frozen structured TinyLLM systems without retraining. Correct
recentering passes all five seeds in both calibrated arms. Wrong-sign
recentering passes only one analytic seed and no learned seeds, while the
target-changing observed-action control passes none. The registered
classification is `observed_bias_reference_repair_specific`.

This closes the exact-pilot positive-control branch: the previously localized
signed sensor defect is removable at the observed calibration interface. It
does not yet show that the bias can be estimated accurately from a finite noisy
pilot.

## Intervention

The selected biased observation was

\[
x_+=x+\epsilon_c+\mu,\qquad \mu=0.03125e_x.
\]

A known zero-signal pilot observed through the same persistent lab-frame bias
returns `p=mu`. It contains no latent phase, target, or answer label. Both
structured front ends subtract the observed calibration offset before forming
their invariant feature, so the registered repair changes only

\[
o_{\mathrm{repair}}=o+p.
\]

No model, front end, task head, observer, denoiser, action, or probe was trained
or fitted, and no new random draw was introduced.

Four variants were evaluated on composition and extrapolation:

- the sealed uncorrected `source_full_plus` condition;
- `recenter_correct`;
- `recenter_wrong_sign`, which leaves centered error plus `2 mu`;
- `recenter_target_changing`, which applies correct recentering and then the
  declared observed orthogonal-axis reflection.

## Preregistration correction

Version 1 stopped during full-cohort model-independent preflight, before any
primary checkpoint was loaded. Its float32 corrected-planar audit missed the
frozen `2e-7` tolerance by one unit (`2.3841858e-7`) on extrapolation. The
threshold was not relaxed.

Version 2, frozen before primary model evaluation, audits the algebraic
corrected-planar construction in float64 and retains separate realized float32
feature (`1e-6`) and posterior (`2e-6`) equivalence gates. The invalid v1 record
is preserved in
[`2026-08-10_tinyllm-bias-reference-recentering-preregistration.md`](../07%20-%20Status%20Reports/2026-08-10_tinyllm-bias-reference-recentering-preregistration.md);
the evidentiary preregistration is
[`2026-08-10_tinyllm-bias-reference-recentering-v2-preregistration.md`](../07%20-%20Status%20Reports/2026-08-10_tinyllm-bias-reference-recentering-v2-preregistration.md).

## Primary results

| Arm | Uncorrected | Correct recenter | Wrong sign | Target-changing |
| --- | ---: | ---: | ---: | ---: |
| Analytic calibrated | 1/5 | 5/5 | 1/5 | 0/5 |
| Learned calibrated equivariant | 3/5 | 5/5 | 0/5 | 0/5 |

Counts are seeds passing the natural-utility gate on both composition and
extrapolation. Success required correct recentering at least `4/5` in both arms
and each control at most `1/5`.

Median exact-bin accuracy losses illustrate the effect:

| Arm / shift | Uncorrected | Correct | Wrong sign | Target-changing |
| --- | ---: | ---: | ---: | ---: |
| Analytic / composition | 7.23 pp | 1.86 pp | 16.21 pp | 71.78 pp |
| Analytic / extrapolation | 2.44 pp | 1.17 pp | 8.11 pp | 58.89 pp |
| Learned / composition | 4.98 pp | 1.07 pp | 10.06 pp | 70.12 pp |
| Learned / extrapolation | 0.88 pp | 0.10 pp | 4.10 pp | 47.85 pp |

Correct recentering recovers exactly the utility profile of the sealed
centered-only intervention. The wrong-sign result establishes signed
specificity; the target-changing result establishes task specificity.

## Integrity

- All ten cells are valid, finite, and state-unchanged.
- Clean posterior and source metric replay errors are exactly zero.
- Maximum realized repaired-versus-centered feature error is `2.09e-7` against
  the `1e-6` gate.
- Maximum repaired-versus-centered posterior error is `1.01e-6` against the
  `2e-6` gate.
- Maximum float64 corrected-planar construction error is `1.79e-7` against the
  unchanged `2e-7` gate.
- The observed target-changing action is involutive within `8.35e-7` and has
  analytic-feature RMS effect above `1.07` on both shifts.
- The complete artifact tree is byte-identical on resume; its file-manifest
  hash is `ace12a74b1187d534775471bd90477017d7b5670bded5d202857d510e6e42e8f`.

## Interpretation

The constructive picture now has three experimentally separated layers:

1. calibration fixes the observation gauge and makes the absolute target
   identifiable;
2. analytic/equivariant front ends and observed group averaging create a
   quotient-sufficient computation;
3. a persistent signed calibration displacement can still damage the frozen
   answer interface, but an observed shared-bias reference removes it exactly.

The result strengthens rather than restores the old whole-residual theory. The
successful mechanism is an explicit observation-side construction with a
causal frozen-continuation endpoint. It is not evidence that TinyLLM naturally
erases every nuisance direction internally.

## Boundary and next decision

The pilot is an exact positive control. Its bias value is known because the
synthetic zero signal is measured under the declared persistent additive law.
This experiment does not establish robustness to pilot noise, finite pilot
count, bias drift, or bias varying across examples.

Do not retrain TinyLLM or fit a denoiser. The only licensed continuation is a
nested, frozen pilot-acquisition titration: add noise to repeated zero-signal
pilot measurements, estimate only their mean, and determine the smallest count
that preserves the complete repair gate. If the first nonzero pilot-noise level
fails, close the practical recentering branch rather than weakening the endpoint.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/matplotlib-bias-reference-primary-v2 \
pixi run python -m experiments.structure_net.tinyllm_bias_reference_recentering \
  --device cuda:2 \
  --output data/experiments/tinyllm_bias_reference_recentering/20260810_d10_preregistered_v2
```

- Raw artifact root:
  `data/experiments/tinyllm_bias_reference_recentering/20260810_d10_preregistered_v2`
- Campaign SHA-256:
  `1996ac4c2534b62a25a2f52ceadfd21055a91bdadc81f38ccf01c6855da2b7d0`
- Result-manifest SHA-256:
  `7dbbe3a49f4e3ebac36e891ec63d5336ff3be2e176e26f1a610cbfceecaabb4e`
- Intervention-contract SHA-256:
  `6bed75b6cd9a15be35f21e53463efa28bcc2f775f1490f31414e005398894004`
- Implementation SHA-256:
  `059d4ace65402fb296bcf35bf614aa411e085cc34abaf8513e077678a4828e15`
- Runner SHA-256:
  `fd6ea5108ccd733e360010c83a1a4a411512cbed239e3c9356a6a6bb77a6996a`
- V2 preregistration SHA-256:
  `e2d04b2852bffaac0ce190a4245f54a0e587a6d8323468afa91b922ef7c2c86b`
- Source component campaign SHA-256:
  `9f7fdf98e83a320d5d49d9191e6a0f0cd6f872f32f406381c5a290f517dbed4b`
- Source DVC root: `e3bfc6a9401916ffc7f942678044fb0a.dir`
- Source lakeFS commit:
  `a0f6b67d7aad58dc96de58406abf7064728613e73134ba4959e18dd46c0cc92a`
- Meta-hypothesis SHA-256:
  `6c3541989bc1d04cef0fce62228c206efbf59603550312fd630ce2c90c03823e`
- DVC root after sealing: `1de07aeb227a8093fa5973d37d63f9a6.dir`
- lakeFS commit after sealing:
  `23a11ba9918f2adcf4397c619e8b942f7539e1f98bb52962f98be6f520e7c181`
