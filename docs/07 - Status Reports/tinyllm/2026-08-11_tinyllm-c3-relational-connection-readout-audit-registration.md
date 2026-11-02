# TinyLLM C3 relational connection readout audit registration

**Status:** POST-OUTCOME CORRECTIVE ARTIFACT AUDIT — CANNOT RESCUE PRIMARY

**Date:** 2026-08-11

The primary acquisition campaign is valid and negative:
`learned_true=1/5`, every control `0/5`, classification
`exact_function_class_but_population_acquisition_unreliable`.

This diagnostic was registered after inspecting the primary scalar metrics.
It asks only whether the four failed final encoders contain a linearly readable
target whose jointly trained scalar head remained miscalibrated. It cannot
alter the primary classification, gates, or stop rule.

## Frozen sources

| Source | SHA-256 |
| --- | --- |
| primary runner | `cf425970a3424a32e410492ea79d7d17fd579a83cea78bacea9b8a58674116f0` |
| campaign result | `b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a` |

Every per-seed result and final checkpoint must match the hashes recorded in
that sealed campaign.

## Fixed analysis

For every arm and seed, reload the final checkpoint without an optimizer and
regenerate the sealed training and evaluation cohorts. On the training cohort
only, fit by unregularized least squares:

1. an affine map from the stored scalar output to the true target;
2. a linear map with intercept from the frozen two-dimensional neutral carrier
   to the true target.

Apply both maps unchanged to held-out composition and extrapolation. Report
the original registered metrics, exact replay error, fit coefficients,
condition number, and the same learned endpoint gates. No seed, threshold,
rank, ridge, split, or nonlinear-map sweep is permitted.

The two information-removal controls are retained as specificity checks:
`learned_no_connection` and `learned_connection_shuffled`. The
target-shuffled arm is descriptive only because it still receives the correct
connection and may retain an accidentally readable carrier.

## Interpretation

- If neither fit repairs a failed seed, the frozen encoder lacks the task
  relation under the declared linear interface.
- If the neutral fit repairs but scalar affine calibration does not, the
  learned head selected the wrong carrier direction.
- If affine scalar calibration repairs, the learned computation has the right
  one-dimensional ordering but the joint optimizer did not finish its public
  scale/offset convention.

Regardless of outcome, do not extend steps or tune the optimizer. The fixed
analytic six-weight solution remains the economic baseline.
