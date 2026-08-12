# TinyLLM final-query task-kernel barycenter decomposition

**Status:** VALID PREREGISTERED UNDERPOWERED NULL  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`,
frozen-checkpoint no-fit causal decomposition  
**Hypothesis:** `tinyllm-final-query-task-kernel-barycenter-v1`  
**Schema:** `nal.tinyllm-final-query-task-kernel-barycenter.v1`  
**Preregistration:** [final-query task-kernel preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-final-query-task-kernel-preregistration.md)

## Verdict

The hypothesis is **not confirmed**. The frozen centered-answer-logit kernel is
causally inert enough to preserve the task in both regimes, and its complement
explains the full barycenter's output change. But the kernel contains only a
minority of the opposite-sheet query chord. Removing the kernel component
reduces pair distance by only `19.4%` in support and `15.3%` outside range,
against the preregistered `75%` contraction requirement.

The result rejects the proposed explanation:

> The full query barycenter does not fail because it accidentally removes one
> small task-active component from an otherwise task-null fiber chord. Most of
> that chord already lies in the 15-dimensional answer-sensitive row space.

| Primary item | Training support | Outside range | Gate |
| --- | ---: | ---: | --- |
| kernel-only task sufficiency | pass | pass | pass |
| remaining pair-distance ratio | `0.8061` | `0.8474` | **fail**; ceiling `0.25` |
| task-only arm causally active | yes | yes | pass |
| nonlinear effect attribution | pass | pass | pass |
| rank-matched random specificity | pass | pass | pass |
| replay / Jacobian / state validity | pass | pass | pass |
| complete hypothesis |  |  | **fail** |

The raw classification is `task_rowspace_contains_material_fiber_chord`.

## Frozen decomposition

The study reused the valid retained d8 cosine-interval seed-7 checkpoint and
the same `512` exact shared-nuisance fibers in each parent regime. It evaluated
only the final post-MLP query vector.

For pair barycenter `b`, row state `h`, and centered 16-answer-logit map `g`,

```text
v = b - h
J_b = Dg(b)
P_task = pinv(J_b) J_b

v_task   = P_task v
v_kernel = (I - P_task) v.
```

The actual frozen layer norm and answer head then evaluated:

```text
full:          h + v
task-only:     h + v_task
kernel-only:   h + v_kernel
random-kernel: h + (I - P_random) v.
```

`P_random` used a fixed Gaussian rank-15 row space. No state, carrier, probe,
decoder, model parameter, or threshold was fit.

## The kernel is task-preserving

The kernel-only arm passes the parent's simultaneous accuracy,
cross-entropy, and posterior-JS endpoint in both regimes:

| Regime | Baseline acc. | Kernel acc. | CE increase | Posterior JS | Sufficient |
| --- | ---: | ---: | ---: | ---: | --- |
| training support | `0.4385` | `0.4395` | `+0.00052` | `0.00042` | yes |
| outside range | `0.1523` | `0.1641` | `-0.11623` | `0.01919` | yes |

The extrapolation result is close to, but still below, the locked `0.02` JS
ceiling. It is not used to change the threshold.

This establishes a real local fact: the finite kernel-component intervention
is output-inert enough under both declared shifts. It does not establish that
the kernel is a global invariant subspace.

## But the kernel does not contain the fiber chord

| Regime | Original mean pair norm | Kernel-patched mean pair norm | Remaining ratio | Mean task-component norm fraction | Mean kernel-component norm fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| training support | `121.65` | `98.07` | `0.8061` | `0.7770` | `0.5844` |
| outside range | `318.01` | `269.47` | `0.8474` | `0.8341` | `0.5074` |

The task and kernel components are orthogonal per row; the displayed norm
fractions are averaged separately and therefore need not sum to one. Their
squared per-row norms reconstruct the full displacement.

Despite having only rank 15 in a 512-dimensional residual, the answer-sensitive
row space captures a large fraction of the phase-reflected chord. The local
kernel patch leaves most of the pair separation intact. This directly fails
the registered geometric prediction.

## The task component explains the full output change

The table compares actual nonlinear centered-logit changes, not Jacobian
predictions.

| Regime | Mean task/full residual | P95 residual | Median cosine | Mean kernel/full effect | P95 kernel/full effect | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| training support | `0.0297` | `0.0817` | `0.99981` | `0.0821` | `0.2039` | pass |
| outside range | `0.1318` | `0.6058` | `0.99797` | `0.2192` | `0.5615` | pass |

The task-only arm reproduces the parent full intervention's material posterior
change and fails task sufficiency in both regimes:

| Regime | Full JS | Task-only JS | Kernel-only JS |
| --- | ---: | ---: | ---: |
| training support | `0.02746` | `0.02721` | `0.00042` |
| outside range | `0.20593` | `0.18033` | `0.01919` |

Thus the decomposition is causally meaningful even though the complete
hypothesis fails: the task-visible component causes the output change, and the
kernel component is comparatively inert.

## Rank-matched specificity

The deterministic random rank-15 control behaves like the full patch rather
than the task kernel:

| Regime | Random-kernel JS | Task-kernel JS | JS disadvantage | Random sufficient |
| --- | ---: | ---: | ---: | --- |
| training support | `0.02579` | `0.00042` | `0.02538` | no |
| outside range | `0.19221` | `0.01919` | `0.17302` | no |

Both disadvantages exceed the locked `0.01` margin. Preservation is therefore
specific to the frozen answer-map kernel rather than a generic consequence of
removing a rank-15 component.

## Numerical and lifecycle integrity

| Contract | Training support | Outside range |
| --- | ---: | ---: |
| Jacobian rank | `15` in `512/512` fibers | `15` in `512/512` fibers |
| maximum finite-difference relative error | `7.89e-11` | `7.39e-11` |
| maximum decomposition relative error | `2.82e-17` | `2.73e-17` |
| maximum kernel leakage | `2.52e-14` | `8.37e-15` |
| parent baseline posterior replay | exactly `0.0` | exactly `0.0` |
| parent full posterior replay | exactly `0.0` | exactly `0.0` |

The checkpoint's model and system-state hashes are unchanged. The primary run
used Python `3.11.13`, PyTorch `2.5.1+cu121`, and an NVIDIA GeForce RTX 2060
SUPER on CUDA device `1`. It allocated at most `498,819,072` CUDA bytes and
completed in `18.20` seconds. A second invocation left the campaign and
diagnostics hashes byte-identical.

The earlier eight-fiber CUDA shakedown is labeled
`systems_lifecycle_only_not_quality_evidence`; it contributes no scientific
evidence.

## Interpretation

The final query representation has a genuine task-null complement, but the
exact same-cosine phase chord is not primarily stored there. Instead, the
network's answer-sensitive coordinates move substantially between the two
sheets even though their target distribution is identical. Averaging those
coordinates improves accuracy and cross-entropy, but it also changes the
posterior, which is why the parent barycenter is useful without being a
sufficient state.

This also separates two earlier observations:

```text
conditional branch probe at chance
    != phase-reflected chord lies in the frozen output kernel

task score improves under averaging
    != frozen computation is preserved.
```

Probe collapse says a registered decoder cannot recover branch identity. It
does not say that the paired chord is orthogonal to every answer-sensitive
direction.

## Scientific decision

Stop the simple Euclidean final-query tangent/kernel branch. Do not relax the
contraction gate, replace the strict posterior endpoint with accuracy alone,
scan earlier layers, fit a different carrier, or retrain a model under this
hypothesis.

The cheapest remaining distinction is conceptual rather than another
optimization run: the one-dimensional cosine target and the complete
15-dimensional answer-posterior computation are different quotients. Any
future experiment must declare which one it intends to preserve. A
target-coordinate-only intervention may improve semantic accuracy, but it
cannot be cited as autonomous frozen-computation closure under the present
posterior endpoint.

## Artifacts and reproduction

| Item | SHA-256 / value |
| --- | --- |
| campaign | `93d9e22d766aa56943f0bd0c41b31ed25dc592e97ae0667863f3963588c38cde` |
| diagnostics | `ad1798b960375503381cb59725dfb84db9d823f488ab8073fea178d0399e0974` |
| implementation | `2e6a8312d73c31be8dd0ea9df066dcd375a83c5bc3659d2436baa605554f2161` |
| runner | `52b59f96e53e76c1794e9d2d251760a76eb2300f8e64e70dbe165090a5899895` |
| preregistration | `6c99f8b4e6feb91fe5ca90a6846cc2b1472825874cd201d3fb5f0f51504347c3` |
| parent campaign | `b5d6b613683326024eeb00944a3d0aba4dd7a251dd22f5fa7e8561b5e9d6aae4` |
| meta-hypothesis record | `926fac34777d6d3fd97055a334bec8f9935821fcfa2ac4a69afeb9a82f9dff8d` |
| DVC data root | `6eb1667e3243bf4272fc861dcf7782e1.dir` |
| lakeFS commit | `0f5d1e6e68df035f38521e9ade7208fadd6047186c29fc4b022d24d4f2d4cd14` |

- primary campaign:
  `data/experiments/tinyllm_final_query_task_kernel/20260810_d8_seed7_preregistered/`
- systems-only shakedown:
  `data/experiments/tinyllm_final_query_task_kernel/20260810_shakedown_cuda/`

```bash
MPLCONFIGDIR=/tmp/matplotlib-final-query-kernel-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_final_query_task_kernel \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_final_query_task_kernel/20260810_d8_seed7_preregistered
```

## Boundaries

This result covers one retained d8 seed-7 cosine checkpoint, two exact
synthetic regimes, final post-MLP query activations, the centered native
16-answer-logit map, and Euclidean Jacobian geometry at each pair barycenter.
It does not establish population prevalence, a global kernel, an earlier-layer
mechanism, literal branch erasure, or behavior on natural-language tasks. The
negative result closes the registered small-task-component explanation, not
all possible nonlinear quotient coordinates.
