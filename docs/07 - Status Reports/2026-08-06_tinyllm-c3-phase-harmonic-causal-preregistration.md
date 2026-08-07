# TinyLLM `C3` phase-harmonic causal decomposition preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-c3-phase-harmonic-fusion-v1`  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`

## Question

When a frozen degree-three quotient-synthesis front is sensitive to the phase
of its charged deck carrier, is that causal dependence carried primarily by
the first symmetry-allowed phase harmonic `3 theta`?

This study decomposes a previously observed mechanism. It reuses the same
checkpoints, orbit cohort, and frozen cuts and therefore is not an independent
replication of phase sensitivity.

## Frozen sources and strata

Reuse the five retained d6 `k=3` checkpoints for seeds
`7,17,29,41,53`, with 64 exact orbits under composition and extrapolation.
Regenerate exactly the cohort used by the Reynolds character-coupling and
deck-irrep fusion campaigns. Validate the checkpoint, predecessor result, and
implementation digests before analysis.

The prior irrep campaign classified seeds `17,29,53` as
`finite_group_phase_sensitive` under both shifts. These three seeds form the
frozen primary mechanistic stratum. Seeds `7,41` remain declared descriptive
comparators. No result in the current experiment may change this membership or
move the frozen synthesis transition.

## Exact phase orbit and Fourier decomposition

At the source of the frozen synthesis sublayer `F`, write the real `C3`
activation fiber as

`h_j = c0 + c1 omega^j + c2 omega^(2j)`, with `c2=conj(c1)`.

Rotate only the carrier phase:

`c1(theta)=exp(i theta)c1`, `c2(theta)=exp(-i theta)c2`.

For the frozen 24-point grid `theta_m=2 pi m/24`, compute

`q(theta_m)=mean_j F(h_j(theta_m))`.

Exact `C3` symmetry requires `q(theta+2 pi/3)=q(theta)`. Therefore its discrete
Fourier spectrum may contain only frequencies divisible by three. Let `Q_n`
be the 24-point DFT coefficients.

At the natural phase `theta=0`, define causal reconstructions:

- continuous phase twirl: `q_bar=Q_0`;
- first harmonic: `q_bar+Q_3+Q_-3`;
- nested allowed prefixes additionally including pairs `n=6`, then `n=9`;
- final allowed prefix additionally including the Nyquist term `n=12`.

Patch every reconstruction at the frozen target cut and run the unchanged
continuation. Also patch each individual allowed pair on top of `q_bar` as a
descriptive channel ablation.

The phrase "first harmonic" refers to phase frequency, not uniquely to Taylor
order. A `3 theta` response is compatible with cubic generators such as
`c1^3+c2^3`, but higher nonlinear orders can contribute to the same frequency.

## Frozen endpoints

Use the unchanged deck causal conjunction: circular alignment at least `0.90`,
resolved sampling, winding degree within `0.10` of three, and exact-bin
accuracy loss no more than `0.03` relative to the untouched checkpoint.

Define the finite-phase task effect as the mean squared Fisher--Rao distance
between the phase-twirled and exact `theta=0` posteriors. Effects below `1e-6`
are degenerate.

For a harmonic reconstruction `p_h`, define effect explained as

`1-d_FR^2(p_h,p_exact)/d_FR^2(p_twirl,p_exact)`.

Values remain untruncated. A harmonic is causally sufficient only when its
patch passes the full task gate and explains at least `0.70` of the finite-phase
effect.

## Contract and primary gates

The spectral implementation contract requires, in every seed and shift:

- maximum relative state discrepancy across exact `2 pi/3` phase translations
  at most `1e-5`;
- forbidden-frequency variation energy fraction at most `1e-8`;
- full DFT reconstruction relative error at `theta=0` at most `1e-5`.

For the three previously phase-sensitive seeds, require all three seeds to pass
each mechanistic gate jointly across both shifts:

1. **Eligible endpoint:** exact `theta=0` passes and the finite-phase Fisher
   effect is nondegenerate.
2. **Finite-phase necessity:** the continuous phase-twirl patch fails while the
   exact patch passes.
3. **First-harmonic sufficiency:** `Q_0+Q_3+Q_-3` passes and explains at least
   `0.70` of the finite-phase effect.
4. **Shift-stable minimal prefix:** the first sufficient allowed harmonic
   prefix is identical under composition and extrapolation.

The full hypothesis is confirmed only if the spectral contract passes in 5/5
seeds and all four mechanistic gates pass in 3/3 selected seeds. The two
descriptive seeds cannot promote it.

## Outcome meanings

- Twirl fails and the first harmonic passes: exact `C3` phase fusion is causal
  and its lowest allowed phase frequency is sufficient.
- Twirl fails but a higher prefix is required: an exact `C3` architecture must
  expose higher harmonic fusion channels.
- Twirl passes despite phase sensitivity: phase changes the posterior but is
  not necessary for the coarse quotient-sufficiency gate.
- Exact endpoint failure: the frozen front is support-relative even on the
  reused cohort or the implementation does not reproduce its source.
- Forbidden spectral leakage or failed periodicity: invalidate the
  implementation rather than interpret the causal result.

## Boundaries and artifacts

The phase rotation is an off-orbit intervention within the observed real
isotypic carrier. DFT components describe this declared intervention path, not
a globally linear representation. Same-cohort reuse makes the study a causal
mechanism decomposition rather than new generalization evidence.

Confirmatory artifacts will be written to
`data/experiments/tinyllm_c3_phase_harmonic/20260806_d6_preregistered`.
Disposable shakedowns use a separate root and cannot enter evidence.
