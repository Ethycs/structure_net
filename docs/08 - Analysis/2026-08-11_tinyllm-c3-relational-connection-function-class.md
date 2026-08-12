# TinyLLM C3 relational connection function-class result

**Status:** VALID NO-TRAINING RESULT — EXACT FUNCTION CLASS AND TASK GRADIENT CONFIRMED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-relational-connection-function-class-v1`

**Classification:** `connection_invariant_function_class_contains_transport_and_task_gradient`

**Preregistration:** [connection function-class preflight](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-relational-connection-function-class-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_relational_connection_function_class/20260811_preregistered/result.json`

## Verdict

The smallest declared connection-conditioned learned interface clears every
prerequisite without training. Its function class is invariant under the full
observed-pair local action for every parameter state by construction, contains
the analytic endpoint-transport solution in a six-nonzero-parameter state,
receives a finite and control-specific true-task gradient, and passes exact CPU
plus real CUDA checkpoint lifecycle tests.

```text
analytic witness:                pass on 10/10 sealed cells
all-parameter invariance law:    architectural identity
sampled implementation states:  3/3 pass
true/control gradient route:     pass
CPU / CUDA lifecycle:            pass / pass
optimizer steps / TinyLLM:       0 / 0
```

The result licenses one matched learned sensor/readout acquisition campaign.
It does not license an unrestricted transformer or imply that learning is
preferable to the fixed analytic solution.

## The architecture

The module has one raw-to-output path:

```text
calibrated three-channel tokens
  -> shared 1-16-8 GELU channel map
  -> charge-one C3 Fourier character
  -> learned complex mixing and normalization
  -> exact transport by the observed edge connection
  -> neutral endpoint product (real, imag)
  -> two-input scalar head.
```

| Component | Parameters |
| --- | ---: |
| charged encoder | `184` |
| invariant scalar head | `3` |
| total | `187` |

Because the scalar map is shared over sensor channels, channel rolls permute
its outputs before the fixed first-character projection. The resulting
charged feature transforms in the charge-one representation for any weights.
The edge coboundary then cancels the two endpoint frame changes. The head sees
only the neutral product and has no bypass through which it could reconstruct
or exploit an absolute gauge.

This is stronger than training-time invariance regularization: violating the
declared local action is absent from the function class.

## Closed-form witness

The exact identity

```text
GELU(x) - GELU(-x) = x
```

sets the shared scalar map to the identity using four weights. One real
character mixer and one real neutral-product head weight complete the analytic
solution. Every other parameter is zero.

Across all ten predecessor cells:

| Witness measurement | Worst cell | Gate |
| --- | ---: | ---: |
| charged-character error | `2.600e-7` | `<=2e-6` |
| scalar RMSE | `.0018143` | `<=.01` |
| scalar correlation | `.9999967` minimum | `>=.999` |
| exact-bin accuracy | `.98950` minimum | `>=.98` |
| target cross-entropy | `1.28644` maximum | `<=1.35` |
| predicted-bin coverage | `16` | `16` |
| local-action output error | `4.172e-7` | `<=2e-5` |

Every regenerated dataset hash matches the sealed relational preflight. No
example was substituted and no witness coefficient was fitted.

## All-parameter symmetry audit

The implementation was evaluated at deterministic random initialization,
after a nonzero deterministic perturbation of every parameter tensor, and at
the analytic witness:

| State | Charged covariance error | Output invariance error |
| --- | ---: | ---: |
| random | `7.177e-7` | `6.258e-7` |
| perturbed | `3.372e-7` | `1.937e-7` |
| analytic witness | `2.533e-7` | `2.608e-7` |

All are more than an order of magnitude inside the registered `2e-5` limits.
These measurements audit floating implementation. The universal parameter
claim follows from the shared-map/Fourier/neutral-product construction rather
than extrapolating from the three states.

## Gradient route and specificity

On the wholly fresh `512`-example diagnostic cohort:

| Measurement | Result | Gate |
| --- | ---: | ---: |
| true-task gradient norm | `6.9125` | `>=1e-6` |
| nonzero scalar gradients | `183/187` (`.9786`) | `>=.90` |
| local-action loss error | `6.847e-10` | `<=5e-6` |
| local-action gradient error | `7.153e-7` | `<=2e-5` |
| normalized `1e-3` downhill loss decrease | `.006890` | `>=1e-4` |

The diagnostic perturbation was restored exactly; the before/after state
digests match. It is a local derivative check, not an optimizer step.

The gradient is not merely a generic consequence of the architecture:

| Control | Cosine with true gradient | Relative vector difference |
| --- | ---: | ---: |
| connection shuffled | `.4296` | `.9037` |
| target shuffled | `.5316` | `.8797` |

Both controls clear the frozen cosine `<=.95` and relative-difference `>=.20`
requirements with zero permutation fixed points.

## Checkpoint and CUDA lifecycle

The random module was serialized to a temporary `4,628`-byte checkpoint and
loaded with `weights_only=True`:

- schema and parameter count match;
- CPU state digest is exact;
- CPU output replay is tensor-exact;
- CUDA sees three devices and uses the RTX 3060 on device zero;
- CPU/CUDA output difference is `7.153e-7`;
- CUDA local-action output error is `3.874e-7`;
- the CUDA-round-tripped state digest is exact.

The temporary checkpoint is lifecycle evidence only. No historical or trained
checkpoint was loaded.

## Program decision

The predecessor showed that the connection supplies necessary information.
This experiment now shows that a compact, exactly typed learned interface can
represent and receive gradients for that computation without introducing a
gauge-violating shortcut.

One next experiment is licensed:

```text
fixed analytic connection solution
vs learned 187-parameter connection-invariant module
vs no-connection / connection-shuffled / target-shuffled controls,
five seeds, matched examples/minibatches/steps/optimizer.
```

The learned module should start independently in every seed. Success must be a
joint composition/extrapolation task gate in at least four of five seeds, with
exact action invariance checked before, during, and after optimization.

The acquisition campaign tests whether the known computation can be learned
inside the restricted class. It does not test whether a 187-parameter module
beats its six-weight analytic state in deployment. The analytic implementation
remains the price/performance baseline.

`unrestricted_tinyllm_training_licensed=false` remains locked. TinyLLM should
enter only after a later observation law makes a fixed connection unavailable
or the learned typed interface demonstrates a distinct transferable benefit.

## Scope boundary

This result establishes exact architectural symmetry, function-class
containment, local target-gradient access, and executable CPU/CUDA lifecycle.
It does not establish:

- global trainability or convergence frequency;
- sample efficiency;
- robustness to noisy, missing, or unknown connections;
- learning the group law itself;
- TinyLLM utility;
- superiority over fixed analytic transport.

## Reproduction and provenance

The registered primary requires host CUDA visibility:

```bash
MPLCONFIGDIR=/tmp/mpl-c3-connection-function-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_relational_connection_function_class
```

| Artifact | SHA-256 |
| --- | --- |
| primary result | `2292e971bb655db246565675fece8dfd9e1546692b9b782e940a8bbef49de82c` |
| runner | `7010794c3be5fda05a035e5a0b4a178aacd40c934dc1f5f7b36eb2eb03ea96b1` |
| preregistration | `421573a12a09a6782bd44b16e0f57e5e13a158f5bc740d19b8d4447964c5ae86` |
| predecessor result | `ae0321b3c3d5b7e8c80ebbd57bfaea10c0522c4b2241082daab007a54e55984e` |
