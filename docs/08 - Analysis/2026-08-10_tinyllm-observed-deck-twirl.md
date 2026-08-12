# TinyLLM observed-deck twirl causal closure

**Status:** VALID PREREGISTERED RESULT — ORACLE-MEMBERSHIP GAP CLOSED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-system,
no-fit observed-action intervention  
**Hypothesis:** `tinyllm-observed-deck-twirl-causal-closure-v1`  
**Schema:** `nal.tinyllm-observed-deck-twirl-causal-closure.v1`  
**Preregistration:** [observed-deck twirl preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-observed-deck-twirl-preregistration.md)

## Verdict

The calibrated `C2` quotient can be projected from one observed structured
input without latent phase, target labels, branch labels, fiber IDs, or a
second nuisance draw. The observable action and its within-example Reynolds
twirl preserve the frozen task on composition and extrapolation in **5/5**
analytic and **5/5** learned-equivariant checkpoints. The matched
orthogonal-axis action and twirl pass **0/5** in both arms.

The locked classification is:

```text
observable_twirl_closed_action_invariant
```

The result closes the oracle-membership gap left by the preceding exact-fiber
barycenter experiment. No model, front end, task head, group action, probe, or
observer was trained or fit.

## What was made observable

The prior causal intervention averaged two target-equivalent rows with
independently sampled nuisance. It proved sufficiency, but that pair is not the
orbit of a deterministic action on one observation.

For the structured planar interface, the new action uses only the decoded
history and observed calibration:

```text
remove observed offset and drift
divide by observed amplitude
reflect the planar trajectory across the observed orientation axis
restore amplitude, offset, and drift
flip observed signed speed
```

After calibration this maps `(phi, direction)` to `(-phi, -direction)`, so the
future phase is negated and absolute cosine is unchanged. The third harmonic
channel is carried through but is not consumed by either structured front end.
The transform is internal continuous preprocessing after token decoding and is
not re-quantized.

The intervention constructor never reads:

```text
latent phase, target posterior, target bin, branch, fiber ID,
or an independent nuisance draw.
```

## Primary endpoint

A checkpoint passes only when both held-out shifts meet all three locked task
ceilings. Every correct-action, correct-twirl, and block-0 closure population
passes; every matched semantic control population fails.

| Frozen arm | Correct action | Pre-block twirl | Twirl at all cuts | Attention closed | MLP closed | Orthogonal action | Orthogonal twirl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| analytic canonicalizer | **5/5** | **5/5** | **5/5** | **5/5** | **5/5** | **0/5** | **0/5** |
| learned equivariant encoder | **5/5** | **5/5** | **5/5** | **5/5** | **5/5** | **0/5** | **0/5** |

The orthogonal-axis control is an equally norm-preserving involution. In
calibrated coordinates it sends future cosine to its negative. Its failure
shows that the result is specific to the correct target-preserving group
action, not generic reflection, averaging, or residual smoothing.

## Action invariance and twirl effects

The analytic canonicalizer is invariant to numerical precision. Across both
shifts and all five analytic checkpoints, the transformed observation changes
no exact-bin prediction; posterior Jensen--Shannon divergence is at most
`3.57e-14`. Its pre-block residual changes by only about `5e-8` relative RMS.

The learned front end is approximately, not exactly, action-invariant:

| Shift | Correct-action accuracy change | Correct-action maximum JS | Twirl accuracy gain | Twirl maximum JS |
| --- | ---: | ---: | ---: | ---: |
| composition | `-0.10` to `+0.78` pp | `0.000864` | `+0.10` to `+0.88` pp | `0.000215` |
| extrapolation | `0.00` to `+0.78` pp | `0.000974` | `+0.20` to `+0.78` pp | `0.000243` |

Positive values denote accuracy improvement. The learned correct-action state
differs from identity by `0.16--0.68%` relative RMS before block 0 and
`0.87--1.66%` at full depth, yet the task remains unchanged. Averaging the two
action states reduces posterior deviation and slightly improves accuracy in
every learned checkpoint/shift cell.

The first block is essentially closed on this observed twirl. The largest
relative Reynolds/Jensen residual defect is `2.42e-4`, and the largest
downstream posterior JS between propagated and actual next twirls is
`2.76e-7`.

## What the oracle average was also doing

The observable twirl and the older independent-nuisance oracle barycenter
answer different questions:

- the observed twirl proves that one input has a constructible, task-preserving
  `C2` projection;
- the oracle barycenter additionally averages two independent nuisance
  realizations.

That additional nuisance averaging explains the much larger accuracy gains in
the earlier experiment. At the pre-block cut, five-checkpoint median gains were:

| Front end | Shift | Observable twirl | Independent-nuisance oracle average |
| --- | --- | ---: | ---: |
| analytic | composition | `0.00` pp | `5.08` pp |
| analytic | extrapolation | `0.00` pp | `8.01` pp |
| learned equivariant | composition | `0.39` pp | `1.46` pp |
| learned equivariant | extrapolation | `0.29` pp | `10.74` pp |

The oracle intervention remains valid evidence for causal sufficiency, but its
performance improvement must not be attributed solely to branch projection.
It combines projection with independent-nuisance denoising. The present result
supplies the missing deployable projection control and keeps those mechanisms
separate.

## Action contract

The input-level contract passes before and during the campaign:

| Contract | Composition | Extrapolation |
| --- | ---: | ---: |
| sensor involution maximum error | `8.35e-7` | `1.08e-6` |
| calibration involution maximum error | `0.0` | `0.0` |
| target-cosine maximum error | `0.0` | `0.0` |
| corrected-norm maximum error | `4.77e-7` | `4.77e-7` |
| analytic-feature maximum error | `1.79e-7` | `1.79e-7` |
| transformed planar maximum absolute value, including control | `1.384` | `1.993` |
| relative RMS from independent-nuisance oracle mate | `1.093` | `1.292` |

The final row proves that the new action is not silently replaying the old
paired row. The large separation is expected because the older mate resamples
amplitude, orientation, offset, speed, harmonic, noise, drift, and direction.

## Controls and integrity

- all ten selected systems were state-validated before the first action
  outcome;
- the source campaign, result and diagnostic manifests, checkpoint states,
  task configuration, preregistration, and cohort hashes are locked;
- continuation from every identity, correct-action, and control-action cut
  reproduces its captured full posterior exactly (`0.0` maximum error);
- baseline task metrics replay the source causal-closure campaign exactly;
- every model and system state remains unchanged;
- all numeric results are finite;
- 10/10 requested cells complete with no retry, exclusion, or failure;
- all 80 correct-action and all 80 correct-twirl checkpoint-shift-cut cells
  pass;
- all 160 orthogonal action/twirl checkpoint-shift-cut cells fail;
- trained or fitted parameters: **zero**; and
- exact resume leaves both primary and shakedown trees byte-identical.

| Item | SHA-256 / value |
| --- | --- |
| campaign | `79c3e27374d8b6f4611552595de5852ace940204bda825e64cf80eff6ab2050d` |
| implementation | `c970fe8801524f5248a9314e821b6783127596d05a2f206325ed85deb42f9629` |
| ten-result manifest | `b91af38162fbf45e29348fbdf583cb676660d68cf22e5a795b438fd8cd015db3` |
| action contract | `b20fca24168f4e9386e9afdbd3d1980ab100d44b6787bfa48ac1f6ef8de34d60` |
| exact-resume primary tree | `9fe9422fa89b2c4563dce9aeac0ba9f09ab6a1404f1bc367ec94e7ab39416a5a` |
| exact-resume shakedown tree | `ca03ae9b66e73849915c45b8579797e372a899bdbb238e066958191d04dfa75b` |
| source causal-closure campaign | `1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14` |
| source preflight manifest | `fe2eae2d6b297e87f67e5ce40a7c572a6df80d0d24068063d21c2d3027625922` |
| preregistration | `9c397fe42b6bdcb1952d9ed7a5865889e0f919bc5dd0bffce1cb0ef56e484030` |
| meta-hypothesis record | `a87ab9f41d2eec6b4e6df51b5a94b250a78937ef6b3aaa9ad43cadf3c07ee3ff` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA allocation | `351,758,848` bytes |
| analysis time | `182.33` seconds |
| DVC data root | `cb3bd8bc25b8134f791c455f40b9c483.dir` |
| lakeFS commit | `8d10621e08af60a245fdd3b570e3fc9364b473e207e9ae18546e8cfb5b4b81d0` |

## Program decision

Close the calibrated `C2` oracle-membership gap. The current structured system
does not need an additional reflection-equivariant front end, group loss,
observer, probe, or model retraining. It already supports a deterministic
observed action and a causally sufficient within-example Reynolds twirl.

The shortest same-scope checks are exhausted. A next experiment must change a
real assumption: an unknown or noisy calibration axis, a nonsymmetric noise
law, a richer group whose action cannot be generated from one input, or a
checkpoint/architecture population. Do not use the larger oracle-denoising
gain as evidence that the deployable twirl itself improves accuracy by the
same amount.

## Artifacts and reproduction

- primary campaign:
  `data/experiments/tinyllm_observed_deck_twirl/20260810_d10_preregistered/campaign_results.json`
- per-checkpoint records and posterior diagnostics:
  `data/experiments/tinyllm_observed_deck_twirl/20260810_d10_preregistered/runs/*/seed_*/`
- analytic lifecycle shakedown:
  `data/experiments/tinyllm_observed_deck_twirl/20260810_shakedown_analytic_cuda/`
- runner and tests:
  `experiments/structure_net/tinyllm_observed_deck_twirl.py`,
  `tests/structure_net/test_tinyllm_observed_deck_twirl.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-observed-deck \
pixi run python -m \
  experiments.structure_net.tinyllm_observed_deck_twirl \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_observed_deck_twirl/20260810_d10_preregistered
```

## Scope boundary

The observed action assumes the calibration orientation and signed speed are
available, planar noise is compatible with reflection, and the structured
front end consumes continuous decoded planar values. It does not solve raw
three-channel token pairing, infer an unknown group, remove calibration cost,
or establish behavior under biased/anisotropic noise, richer groups, natural
language, real sensors, or an architecture population.
