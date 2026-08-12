# TinyLLM bias-component causal decomposition

**Verdict:** CONFIRMED  
**Date:** 2026-08-10  
**Hypothesis:** `tinyllm-bias-component-causal-decomposition-v1`  
**Evidence role:** `preregistered_frozen_bias_component_intervention`

The fixed positive lab-frame mean is sufficient to reproduce the selected-dose
utility failure. Centered stochastic error passes all five seeds in both
calibrated arms, while the deterministic positive mean fails the four-of-five
population gate in both. Reversing the mean restores the population gate in
both arms. The registered classification is therefore
`deterministic_mean_sufficient`, with secondary sign classification
`positive_direction_specific`.

This localizes the previous biased-law failure to a directional calibration or
readout defect. It does not weaken the independently established observed-action
and Reynolds-twirl closure: those are conditional functional properties, while
this experiment measures natural utility under a perturbed sensor coordinate.

## Registered question

At the common isotropic-valid dose selected by the predecessor study,
`sigma = 0.03125`, decompose its frozen biased planar error into:

\[
\epsilon_{\mathrm{centered}}=\sigma z/\sqrt{2},\qquad
\epsilon_{\mathrm{mean+}}=\sigma e_x,
\]

\[
\epsilon_{\mathrm{full+}}=\epsilon_{\mathrm{centered}}
 +\epsilon_{\mathrm{mean+}},\qquad
\epsilon_{\mathrm{full-}}=\epsilon_{\mathrm{centered}}
 -\epsilon_{\mathrm{mean+}}.
\]

The primary prediction was that centered noise would pass at least four of five
seeds in both arms and mean-only would pass fewer than four of five in both.
The sign-reversed full law was a secondary directional diagnostic.

## Design

The study used the ten sealed d8/N3 systems from the calibrated analytic and
learned-equivariant front-end study: five seeds per arm, evaluated on
composition and extrapolation. Architecture, checkpoints, examples, and the
selected source draw were fixed.

No new random variable was sampled. The clean and original full-positive-bias
posteriors were replayed from the sealed dose-localization campaign. Only the
centered, positive-mean-only, and negative-full variants required new frozen
forwards. No model, front end, head, denoiser, action, observer, probe, or noise
process was trained or fitted.

Natural utility passed only when both shifts simultaneously met:

- exact-bin accuracy loss at most `0.05`;
- circular-error increase at most `pi/16`;
- target cross-entropy increase at most `0.10`.

The preregistration is
[`2026-08-10_tinyllm-bias-component-causal-decomposition-preregistration.md`](../07%20-%20Status%20Reports/2026-08-10_tinyllm-bias-component-causal-decomposition-preregistration.md).

## Results

| Arm | centered | mean+ | full+ | full- |
| --- | ---: | ---: | ---: | ---: |
| Analytic calibrated | 5/5 | 2/5 | 1/5 | 4/5 |
| Learned calibrated equivariant | 5/5 | 3/5 | 3/5 | 4/5 |

All counts are seeds that pass all natural-utility gates on both composition and
extrapolation.

The failure was again driven primarily by composition exact-bin accuracy. The
median composition accuracy losses were:

| Arm | centered | mean+ | full+ | full- |
| --- | ---: | ---: | ---: | ---: |
| Analytic calibrated | 1.86 pp | 5.86 pp | 7.23 pp | 2.34 pp |
| Learned calibrated equivariant | 1.07 pp | 4.10 pp | 4.98 pp | -1.46 pp |

For extrapolation the corresponding medians were `1.17`, `2.15`, `2.44`, and
`3.12` percentage points in the analytic arm, and `0.10`, `0.49`, `0.88`, and
`0.29` percentage points in the learned arm. The population verdict therefore
does not arise from a general collapse under centered noise; it is concentrated
in the fixed positive sensor direction and the composition exact-bin boundary.

## Integrity and controls

- All ten result cells are valid, finite, and state-unchanged.
- Clean posterior replay and source full-positive metric replay are exact.
- The component reconstruction contract passes on both shifts; maximum
  full-positive reconstruction error is `1.49e-8`.
- Full-positive and full-negative empirical planar energies differ by only
  `0.532%` on composition and `0.248%` on extrapolation, below the registered
  `2%` tolerance.
- The complete output tree is byte-stable on resume; its manifest-of-files hash
  is `a4028a03e10724467b8d5e56ea231d2013c2cb9fc71b1cbb92242243c99cbafb`.

## Interpretation

The deterministic positive mean is causally sufficient at lower energy than
the full biased law. That establishes a directional vulnerability, not an
equal-energy effect size. The fact that the negative full law passes four of
five seeds in both arms identifies the sign, rather than bias magnitude alone,
as mechanistically important.

This result sharpens the current architecture story:

1. Calibrated symmetry creates an identifiable quotient-sufficient input.
2. Observed group actions and within-example twirls preserve the frozen task.
3. Natural task utility can nevertheless be fragile to a persistent signed
   calibration displacement.

Group closure and measurement robustness are therefore distinct contracts. A
system may be functionally closed under its declared symmetry while remaining
poorly calibrated against an out-of-family lab-frame offset.

## Boundaries and next decision

Mean-only has less expected energy than the full law, so this study does not
estimate a matched-energy dose response. It also inherits rather than reruns
the predecessor action/twirl controls. The conclusion is limited to the frozen
selected draw, declared direction, ten retained systems, and two registered
shifts.

Do not retrain the TinyLLM body or add another representation penalty. If this
branch continues, the shortest useful test is a frozen observed-reference
recentering or signed calibration correction with a target-changing and
wrong-sign control. That would ask whether the identified defect can be removed
at the sensor interface without disturbing the already validated quotient
closure.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/matplotlib-bias-component-primary \
pixi run python -m experiments.structure_net.tinyllm_bias_component_causal_decomposition \
  --device cuda:2 \
  --output data/experiments/tinyllm_bias_component_causal_decomposition/20260810_d10_preregistered
```

- Raw artifact root:
  `data/experiments/tinyllm_bias_component_causal_decomposition/20260810_d10_preregistered`
- Campaign SHA-256:
  `9f7fdf98e83a320d5d49d9191e6a0f0cd6f872f32f406381c5a290f517dbed4b`
- Result-manifest SHA-256:
  `17d614cadfeca5e019258578ad9abe8dc269f899f7144e712e8154f7988ce07b`
- Component-contract SHA-256:
  `26b8ad368fe8d1af811f2ff62d4874545c6d90b3aa5d376a9a59002092342b2f`
- Implementation SHA-256:
  `c1b340bf1e29e485d2c254902ac6aaab87abdd594e2f46875a5e828fff415c98`
- Runner SHA-256:
  `eba5182082d8604fba47d65fc0f64706b00ac9f4fde6dbf45c63fca56ed44bb5`
- Preregistration SHA-256:
  `a3052f55181f72fb9b53d4bc8ad7a42fe28d5762acd6ee4ffbb6a0d31e81d85e`
- Source dose campaign SHA-256:
  `9b05823ebdb88bd828f27699da596dc5e7dcf0c4af5e13f1664fa70e5111f9bd`
- Source DVC root:
  `c07286d2b9710cd68228cd21f487e425.dir`
- Source lakeFS commit:
  `d4fb92ef41e39d0cc672d672e55c9192ea0e9dcf01597b1a549efcf973577061`
- Meta-hypothesis SHA-256:
  `5bcd85d82511b0d2da1bb3f4599872fb07f94ef8734fc516046e61e412512f90`
- DVC root after sealing: `e3bfc6a9401916ffc7f942678044fb0a.dir`
- lakeFS commit after sealing:
  `a0f6b67d7aad58dc96de58406abf7064728613e73134ba4959e18dd46c0cc92a`
