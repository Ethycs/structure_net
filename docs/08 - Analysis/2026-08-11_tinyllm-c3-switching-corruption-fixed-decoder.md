# TinyLLM C3 switching-law corruption fixed-decoder result

**Status:** VALID PREREGISTERED NEGATIVE; ORACLE/FIXED GAP ESTABLISHED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-switching-corruption-fixed-decoder-corrective-v1`

**Classification:** `recoverable_switching_exceeds_fixed_change_point_decoder`

**Preregistrations:** [original switching preflight](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder-preregistration.md); [fresh numerical corrective](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder-corrective-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_switching_corruption_fixed_decoder_corrective/20260811_preregistered/result.json`

## Verdict

The target is exactly recoverable when the acquisition horizon supplies at
least three post-change frames, but the registered fixed change-point decoder
does not meet the population gate after quantization:

```text
exact repaired-support identifiability:  pass, code distance 3
oracle switch/drop ceiling:              5/5
fixed switch/drop ceiling:               3/5
oracle fidelity:                         1/5
required:                               >=4/5
```

The old global quadratic chart and the corrupted known-switch/no-drop
comparator pass `0/5`, while dynamics and corruption materiality pass `5/5`.
The scope is therefore new, material, and recoverable; the failure is not an
impossible target or a weak perturbation.

The registered result licenses one compact typed continuation comparison. It
does **not** license unrestricted TinyLLM training.

## The identifiability boundary is exact

For the switching design

```text
X_s(t) = [1, t, max(t-s, 0)],
```

an exact rational nullspace audit gives:

| Allowed switches | Maximum equal observed coordinates for a future-different pair | Minimum code distance | One arbitrary corruption |
| --- | ---: | ---: | --- |
| `{2,3,4,5}` | `6` | `2` | not correctable |
| `{2,3,4}` | `5` | `3` | correctable |

The late-inclusive negative is not a sampled miss. Two in-range trajectories
with switches `4` and `5` agree at clean token frames `0,1,2,3,4,6`. Replacing
frame `5` in the first and frame `7` in the second produces exactly the same
eight-frame token observation, while their future cosine targets differ by
`.295520`. Architectural invariance or additional optimization cannot recover
a target that does not descend through that observation relation.

The primary population therefore used only switches `{2,3,4}`, guaranteeing
three observed post-change frames and exact minimum distance `3`.

## Fresh corrective population

The first primary execution is preserved as invalid. Its corrupted no-drop
comparator crossed a principal-angle branch under deck-action roundoff and
violated the `2e-12` action gate in two cells. No scientific outcome from that
artifact is pooled.

The fresh corrective changed only one numerical operation: each invariant
carrier was rounded to twelve decimal places and renormalized before phase
unwrapping. The maximum displacement was `7.032e-13`, and the maximum action
error across every arm, seed, and shift fell to `1.893e-12`. All ten fresh
cells are valid.

## Primary arm results

Means over five fresh seeds:

| Arm | Composition RMSE / accuracy | Extrapolation RMSE / accuracy | Joint seed ceiling |
| --- | ---: | ---: | ---: |
| clean known switch | `.003603 / .9808` | `.003853 / .9812` | `5/5` |
| clean global quadratic | `.082655 / .5102` | `.134520 / .3251` | `0/5` |
| corrupted global quadratic | `.133651 / .3871` | `.225206 / .2310` | `0/5` |
| corrupted known switch, no drop | `.467724 / .3252` | `.508722 / .3144` | `0/5` |
| oracle switch/drop | `.004093 / .9792` | `.004356 / .9787` | `5/5` |
| fixed switch/drop | `.004133 / .9792` | `.017447 / .9783` | `3/5` |

The fixed decoder is excellent on average and on every composition cell. Its
outside-range RMSE ranges from `.004259` to `.027298`; seeds `433` and `467`
cross the locked `.020` ceiling. Seed `401` remains below the absolute ceiling
but misses the stricter oracle-fidelity comparison. Consequently:

```text
dynamics materiality:    5/5
corruption materiality:  5/5
oracle recoverability:   5/5
fixed closure:           3/5
Pareto repair:           3/5
oracle fidelity:         1/5
```

No endpoint was relaxed and the valid negative is not rescued by mean
accuracy.

## Mechanistic localization

On the exact continuous carrier, exhaustive minimum-residual decoding predicts
the future within `1.704e-12` for all `40,960` examples. The exact code-distance
contract is therefore realized by the implementation.

After token quantization, the selected switch/corruption labels agree with the
latent pair on roughly `.9502-.9583` of each cell. Label recovery is not itself
the endpoint—several alternative candidates predict the same future—but rare
quantization-induced candidate choices create large phase errors and dominate
RMSE in three extrapolation cells. The oracle uses the same quantized carrier
and stays near `.0044` RMSE, so observation precision remains sufficient when
the correct discrete chart is selected.

This establishes the narrow learned-model job that the preceding polynomial
and nested-law studies lacked:

> infer a quantization-robust discrete change-point/error chart, not rediscover
> the `C3` action, physical target, or within-chart dynamics.

## Controls and accounting

| Contract | Result |
| --- | ---: |
| fresh requested/completed/invalid cells | `10 / 10 / 0` |
| fresh base / corrupted examples | `40,960 / 40,960` |
| invalid-primary examples pooled | `0` |
| exact continuous future equivalents | `40,960 / 40,960` |
| minimum switch / frame count | `1,306 / 439` |
| minimum phase-chart margin | `.874651` |
| maximum carrier stabilization displacement | `7.032e-13` |
| maximum deck-action prediction error | `1.893e-12` |
| models / checkpoints / optimizer steps | `0 / 0 / 0` |
| reusable or target-using fits | `0 / 0` |

All target shuffles fail the complete task gate. Donor and target
derangements have zero fixed points, corruption commutes exactly with the deck
action, generated observations replay bitwise, and every exact group and
target-invariance contract passes.

## Program decision

Do not launch an unrestricted TinyLLM. First test the strongest symmetry-typed
fixed alternative: fit the charged first `C3` character rather than its cubed
invariant carrier. The charged phase transforms by a global deck rotation that
the intercept absorbs; cubing only the forecast restores invariance. This
reduces phase-noise amplification and increases the unwrap margin without
learning.

Use a fresh sequential comparison:

1. current fixed invariant-carrier decoder as the locked control;
2. fixed charged-character change-point decoder;
3. only if the charged decoder still fails while the oracle passes, one small
   typed selector/continuation trained against the same physical decoder.

The learned arm must predict only the discrete chart or a chart mixture. It
must not replace the exact `C3` sensor, known switching dynamics, or metric
decoder. This keeps the experiment focused on the demonstrated failure.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-switch-corrective-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_switching_corruption_fixed_decoder_corrective
```

| Artifact | SHA-256 |
| --- | --- |
| valid corrective result | `20d40a54ce42a904766cab9eb533f6e2baccff79f75a2703e6bd57ff9638675b` |
| corrective runner | `452b0fa8f8ff54ddb0afa98e32eef9dc38da9f240eccf6a867386d9b65197939` |
| corrective preregistration | `46e40a526c4adf71410beff4d7333da97b4829c2e03f14923744508da602c828` |
| preserved invalid result | `2dd8f23a2eb7bd5ad7e2c224ee8c08201f648ce905e2fd7d806d7d676badf20c` |
| preserved invalid runner | `5ebe462c27989e0f09f38bba8ee0e885ea5fb76190bc86c1d7143d79376e2128` |
| original preregistration | `d7e7b0cd3774a0b661e331c0c6bf56631734152e05fdc156584952f56d6b6ee2` |

The producing tests revalidate the immutable source lineage, exact rational
distance audit, explicit token collision, invalid-run correction, fresh
generator, action bound, classification logic, and authoritative result.
