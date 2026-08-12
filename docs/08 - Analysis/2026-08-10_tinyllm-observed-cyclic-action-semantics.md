# TinyLLM observed cyclic action-semantics decomposition

**Status:** VALID PREREGISTERED STAGE-A NULL — CAUSAL STAGE CORRECTLY NOT RUN  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, staged input-only
falsification before frozen-checkpoint evaluation  
**Hypothesis:** `tinyllm-observed-cyclic-action-semantics-front-v1`  
**Schema:** `nal.tinyllm-observed-cyclic-action-semantics-front.v1`  
**Preregistration:** [observed cyclic action-semantics preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-observed-cyclic-action-semantics-preregistration.md)

## Verdict

The hypothesis is **not confirmed**. Neither registered observable
residual-fixed action passed the input-only eligibility gate across `C2`,
`C3`, composition, and extrapolation. The locked stop rule therefore forbade
the causal model stage.

This is a completed falsification, not an incomplete model experiment:

| Lifecycle item | Outcome |
| --- | --- |
| input cells | `2` degrees x `5` cohorts x `2` shifts = **20** |
| observable candidates eligible | **0/2** |
| requantization-alone alternative | **fail** |
| selected action | **none** |
| TinyLLM checkpoints loaded | **0** |
| causal cells run | **0** |
| models, probes, observers, or action parameters fit | **0** |
| raw classification | `no_observable_action_semantics_candidate` |

The most important attribution is that the latent-phase oracle also fails the
joint action contract. Full-history phase estimation is therefore not the sole
cause. A decoded anchor does not expose the pre-quantization noise realization
needed to reconstruct the older generator's shared-noise, separately
quantized sheets.

The conservative conclusion is:

> The mature observable `C2`/`C3` quotient result remains valid, but the older
> generator-defined first-preserved cut cannot be promoted to an operationally
> observable layer constant. It depends on a hidden coupling between noise,
> phase rotation, and per-sheet quantization.

No threshold was relaxed and no model outcome was inspected to choose an
action.

## Registered decomposition

The historical exact-orbit generator and the one-observation continuous action
differ in two declared ways:

1. the generator holds pre-quantization sensor-frame noise fixed while changing
   phase, whereas the observed action rotates the already realized planar
   observation; and
2. the generator quantizes each sheet separately, whereas the observed action
   operates after token decoding.

Stage A compared six fixed constructors:

| Constructor | Residual transport | Quantization | Selection role |
| --- | --- | --- | --- |
| rotate-all continuous | rotate with signal | none | locked baseline |
| rotate-all requantized | rotate with signal | decode-grid round trip | alternative explanation |
| residual-fixed continuous | hold observation-derived residual fixed | none | selectable |
| residual-fixed requantized | hold observation-derived residual fixed | decode-grid round trip | selectable |
| oracle residual-fixed continuous | hold residual around latent-phase carrier | none | attribution only |
| oracle residual-fixed requantized | hold residual around latent-phase carrier | decode-grid round trip | attribution only |

The observable residual estimator demodulates the full calibrated planar
history, reconstructs its coherent sinusoid, and assigns the remainder to the
sensor-frame residual. It reads only the decoded history, observed calibration,
fixed time grid, and declared group element. Latent phase, future phase,
targets, branch/fiber identity, nuisance dictionaries, noise realizations, and
non-anchor generator rows are forbidden.

The oracle substitutes latent phase only to ask whether phase-estimation error
explains failure. It is never selectable.

## Primary distance result

The table reports median reduction in planar relative RMS from the locked
rotate-all continuous action. The gate required at least `50%` in **every**
degree/shift cell. Negative values are worse than the baseline.

| Constructor | `C2` composition | `C2` extrapolation | `C3` composition | `C3` extrapolation | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| residual-fixed continuous | `-737.4%` | `45.2%` | `8.8%` | `47.6%` | **no** |
| residual-fixed requantized | `100.0%` | `100.0%` | `-4.9%` | `38.0%` | **no** |
| oracle residual-fixed continuous | `-648.6%` | `50.4%` | `9.1%` | `52.2%` | attribution-only; fails |
| oracle residual-fixed requantized | `100.0%` | `100.0%` | `-5.5%` | `42.1%` | attribution-only; fails |
| rotate-all requantized | `100.0%` | `-27.5%` | `-14.5%` | `-5.9%` | alternative fails |

The exact zero medians for some requantized `C2` cells are real but not a
general mechanism. They do not transfer to `C3`, and the same constructors
fail character, composition, or norm gates.

## Observable-candidate gate audit

The character columns show `(median / p95)` angular error in radians. The
registered ceilings were `0.05 / 0.20`. Composition ceilings were `0.02`
continuous and `0.05` requantized; the corrected-norm p95 ceiling was `0.05`.

| Candidate and cell | Distance reduction | Character median / p95 | Composition max | Norm p95 | Cell result |
| --- | ---: | ---: | ---: | ---: | --- |
| continuous, `C2` composition | `-737.4%` | `0.0002 / 0.2442` | `7.36e-7` | `0.1862` | fail |
| continuous, `C2` extrapolation | `45.2%` | `0.0002 / 0.5218` | `7.46e-7` | `0.5530` | fail |
| continuous, `C3` composition | `8.8%` | `0.0721 / 0.3457` | `0.0771` | `0.1702` | fail |
| continuous, `C3` extrapolation | `47.6%` | `0.1084 / 0.7795` | `0.3502` | `0.5025` | fail |
| requantized, `C2` composition | `100.0%` | `0.00005 / 0.2718` | `0.1129` | `0.2112` | fail |
| requantized, `C2` extrapolation | `100.0%` | `0.00002 / 0.5394` | `0.2611` | `0.5851` | fail |
| requantized, `C3` composition | `-4.9%` | `0.0727 / 0.4030` | `0.1890` | `0.1986` | fail |
| requantized, `C3` extrapolation | `38.0%` | `0.1155 / 0.8365` | `0.5222` | `0.5132` | fail |

All observable candidates remained finite and within planar support `2.0`.
Direct closure also passed: continuous maximum error was below `5e-7`, and
requantized closure was exactly zero on the decoded grid. Those positive
contracts cannot rescue failures of distance, character fidelity, group
composition, and corrected norm.

## Why the oracle matters

The oracle knows the anchor's latent phase, so it removes demodulation error.
It still does not know the generator's pre-quantization noise. Its residual is
computed from the decoded anchor:

```text
decoded anchor - exact clean anchor
    = physical sensor noise + anchor quantization error.
```

Holding that combined residual fixed while rotating the clean signal is not
the same operation as holding the original continuous noise fixed and then
quantizing a newly rotated sheet:

```text
Q(rotated clean + pre-quantization noise)
    != Q(rotated clean + decoded-anchor residual)
```

in general. Requantizing the right side does not recover the discarded
continuous value. The oracle's failure—particularly `C3` composition and the
character/norm tails—shows that the discrepancy survives perfect phase
knowledge under the registered carrier.

Thus the historical generator front depends on counterfactual coupling data
that one decoded observation does not contain. The result is an
identifiability boundary for that intervention, not evidence that the mature
network quotient disappeared.

## Relationship to the prior causal result

The prior observed cyclic experiment remains the causal evidence:

- one-observation `C2` and `C3` twirls preserve the frozen task at full depth
  in `5/5` checkpoints under both shifts;
- both destroy the task at the early cover in `5/5`;
- matched target-changing controls are specific in `5/5`; and
- exact generator-front agreement is weaker, especially `C3` at `2/5`.

The present result explains why a same-scope attempt to force the observed
action toward the generator action is not justified. The two interventions
do not differ by one recoverable nuisance toggle after decoding.

Accordingly:

```text
mature observable quotient sufficiency: established
exact first-preserved cut: conditional on declared action semantics
generator-defined cut from one decoded sheet: not identified here
```

## Scientific decision

Stop the same-scope action fitting, causal-front reruns, and retraining branch.
There is no registered candidate to evaluate, and the oracle says a better
phase estimator alone cannot create one.

Use one of two clean designs if this question is reopened:

1. Treat the already validated continuous one-observation action as the
   operational causal definition and report its front as action-qualified.
2. Prospectively expose a pre-quantized sensor/noise reference in the data
   contract, then test generator-front recovery. This changes the observation
   model explicitly instead of inferring discarded values.

The first option requires no further experiment. The second tests a new
identifiability assumption and still needs no TinyLLM retraining.

## Campaign integrity

| Item | Value |
| --- | --- |
| cohorts | d6 `C2/C3`, seeds `7,17,29,41,53`, composition and extrapolation |
| anchors per cell | `256` |
| map points declared for authorized Stage B | `192` (unused) |
| environment | Python `3.11.13`, PyTorch `2.5.1+cu121`, CPU |
| GPU allocation | `0` bytes |
| primary analysis time | `2.65` seconds after source validation/cache warmup |
| exact resume | second invocation left all three artifact hashes unchanged |
| trained/fitted parameters | `0` |

All three locked source campaigns, their schemas and hashes, the
preregistration, runner digest, finite-data contract, and empty causal result
manifest validate. The canonical campaign reports `valid: true`; the nested
Stage-A object reports `valid: false` because no candidate was eligible, which
is the intended scientific stop condition rather than a data-validity error.

## Artifacts and reproduction

| Item | SHA-256 / value |
| --- | --- |
| campaign | `022fa4256c37555dd267f2be94d6a3eec2f50dc03e2f1b1e35e166b3d64e1815` |
| implementation | `dd7bbf10594cf5129acb65977f48108916304087acf7201527e5bee681679f4b` |
| runner | `f8342f897b7e151949c9e7f682f4125644bb05af276ff517022310616e7574e8` |
| Stage-A JSON | `652bf9d082cfe8b8cd997724f4b1d36b82e46aead2283354fc6e376c981c138e` |
| Stage-A arrays | `7f0997be3741a58feebb003d453f0aa81b9e2800627aa47209307ee087128a6d` |
| empty causal manifest | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| preregistration | `8c649c67500fecd3bf672e7a879f83c8af316f838237ca051db7539e2d25977d` |
| meta-hypothesis record | `d6082112885032c9b547968c285627b0d111c98fbac0dcb09fca6b66d6960993` |
| DVC data root | `4cb5acacdbc68f5d3273d2110028679a.dir` |
| lakeFS commit | `365782635729910cabb14921424db2bcc4a4cb7306be73e13aeb344a62521953` |

- primary campaign:
  `data/experiments/tinyllm_observed_cyclic_action_semantics/20260810_d6_preregistered/`
- aggregate meta record:
  `data/meta_hypotheses/tinyllm-observed-cyclic-action-semantics-front-v1.json`

```bash
MPLCONFIGDIR=/tmp/matplotlib-action-semantics-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_observed_cyclic_action_semantics \
  --device cpu \
  --output \
  data/experiments/tinyllm_observed_cyclic_action_semantics/20260810_d6_preregistered
```

## Boundaries

This result covers the declared sinusoidal planar carrier, its calibrated
decoded observation, one fixed residual estimator, one requantization rule,
`C2/C3`, twenty N3 input cohorts, and the historical shared-noise generator
coupling. It does not prove that every conceivable observable action fails,
that the mature front is absent, or that real sensors share this quantization
boundary. No checkpoint activation or task output from a candidate action was
evaluated.
