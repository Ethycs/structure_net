# TinyLLM local continuation tangent/kernel preregistration

**Status:** PREREGISTERED POST-OUTCOME DIAGNOSTIC — NO JACOBIAN OR PATCH OUTCOME INSPECTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome frozen-checkpoint diagnostic  
**Hypothesis:** `tinyllm-c2-local-continuation-tangent-kernel-v1`  
**Schema:** `nal.tinyllm-c2-local-continuation-tangent-kernel.v1`

## Question and directional prediction

The fixed-gauge writer ladder ended with exact rank-three carrier coordinates
passing every held-out causal cell while all small absolute writers failed at
least one cell per checkpoint. The shortest remaining ambiguity is whether the
order-four writer is wrong mainly in the local task metric of the frozen
continuation, or whether the continuation requires nonlinear interaction with
the nominally first-order-null component.

The directional hypothesis is:

> At the order-four predicted carrier state, the exact-minus-predicted
> coordinate residual is causally repaired by its local task-tangent component
> alone; its first-order task-kernel component is output-inert, and the
> normalized tangent metric is stable across composition and extrapolation.

This is a no-fit diagnostic. It trains no model, adapter, writer, probe,
readout, or observer.

## Locked predecessor and replication unit

The direct predecessor is immutable:

```text
data/experiments/tinyllm_frozen_writer_capacity/
    20260807_d6_preregistered_diagnostic/campaign_results.json
SHA-256 7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b
implementation d53edaedd49ae553af9f8393d92254664239e5100246ac0fd3a06cb420ca80ed
```

The replication unit is one frozen d6 checkpoint. The only selected stable
checkpoints are seeds `7`, `29`, and `53`, so the campaign is explicitly
`UNDERPOWERED`. Each checkpoint retains the predecessor's two held-out cohorts
under both composition and extrapolation, with 64 exact `C2` orbits per cell.
These cells have appeared in prior post-outcome diagnostics and do not provide
a fresh confirmatory split.

## Frozen intervention

For each held-out orbit, reconstruct the stored order-four Fourier prediction
`c_hat` in the checkpoint-local three-dimensional source basis and the exact
rank-three coordinate `c_star`. Define

```text
e = c_star - c_hat.
```

At `c_hat`, differentiate the frozen continuation's complex first circular
posterior moment

```text
m(c) = [sum_j p_j(c) cos(2 pi j / 16),
        sum_j p_j(c) sin(2 pi j / 16)]
```

with respect to the three carrier coordinates. This gives an orbit-local
Jacobian `J in R^(2 x 3)`. With an SVD pseudoinverse using relative tolerance
`1e-6` and absolute tolerance `1e-10`, define

```text
P_T = J^+ J
e_T = P_T e
e_K = (I - P_T) e.
```

The registered states are:

| State | Carrier coordinates | Purpose |
| --- | --- | --- |
| predicted | `c_hat` | failed order-four writer |
| tangent | `c_hat + e_T` | local task-metric intervention |
| kernel | `c_hat + e_K` | first-order-null intervention |
| full | `c_hat + e` | exact rank-three positive control |

For each component, generate eight deterministic isotropic Gaussian controls
in carrier-coordinate space, normalized per orbit to exactly match `||e_T||`
or `||e_K||`. Random streams are fixed by checkpoint, cohort, regime, and
control index. No outcome-dependent direction selection is allowed.

## Fixed controls

| Field | Locked value |
| --- | --- |
| checkpoints | predecessor d6 seeds `7,29,53` and stored SHA-256 identities |
| writer | stored source-fit order-four Fourier writer; no refit |
| carrier | predecessor checkpoint-local rank-three source basis |
| generator | predecessor exact-orbit generator, cohort seeds, and regimes |
| continuation | frozen target checkpoint and block-0 target cut |
| readout | predecessor circular posterior endpoint and scalar calibration |
| intervention count | predicted, tangent, kernel, full, 8 tangent-random, 8 kernel-random |
| training | none |
| exclusions/retries | only fingerprint, provenance, numerical, or infrastructure failure; scientific misses remain terminal |

## Primary endpoint and contracts

Every state retains the predecessor continuous endpoint:

- circular-alignment loss at most `0.005`;
- mean circular-moment shift at most `0.125` bins;
- p95 shift at most `0.50` bins;
- resolved sampling; and
- winding degree within `0.10` of degree two.

Before classification, each checkpoint must satisfy all of these contracts:

1. checkpoint, character source, readout, predecessor result, and campaign
   hashes match;
2. stored order-four coordinate and continuous metrics replay within `1e-6`;
3. the predicted state fails at least one of four cells while the full
   rank-three state passes all four;
4. all Jacobians are finite and have numerical rank at least one;
5. maximum central finite-difference relative error is at most `0.05`, using
   step `1e-3` along the normalized exact residual (or a fixed axis when that
   residual vanishes);
6. maximum relative decomposition error, kernel leakage
   `||J e_K||/(||J e||+1e-12)`, and tangent mismatch
   `||J e_T-J e||/(||J e||+1e-12)` are at most `1e-6`.

A checkpoint passes the full **stable local task-metric** gate only when:

1. tangent passes the continuous endpoint in all four held-out cells;
2. kernel is output-inert in every cell: mean moment movement from predicted
   is at most `0.05` bins and p95 movement is at most `0.20` bins;
3. at most one of eight norm-matched tangent-random policies passes all four
   cells, and tangent's aggregate mean error is at least `0.125` bins below
   the median random policy;
4. kernel's aggregate mean movement from predicted is at least `0.05` bins
   below the median norm-matched kernel-random policy; and
5. across the four cells, every pair of trace-normalized mean metrics
   `G = mean(J^T J)` has Frobenius distance at most `0.35`, and every pair of
   leading eigenvectors has absolute cosine at least `0.90`.

The campaign supports the directional hypothesis only if all three selected
checkpoints pass. With three post-selected checkpoints this remains
mechanistic evidence, not population prevalence.

## Fixed classifications

Classify each checkpoint by the first applicable rule:

1. `invalid` if any provenance, replay, target-control, Jacobian, finite-
   difference, or decomposition contract fails.
2. `stable_task_tangent_sufficient` if the full gate above passes.
3. `checkpoint_local_task_tangent_sufficient` if tangent and causal
   specificity pass but cross-cell metric stability fails.
4. `nominal_kernel_causally_active` if the kernel is not output-inert and
   improves aggregate mean error from exact by at least `0.125` bins or passes
   the continuous endpoint in any cell.
5. `nonlinear_tangent_kernel_interaction_required` if full passes all cells,
   tangent fails at least one, and kernel remains output-inert.
6. `mixed_local_continuation_geometry` otherwise.

The campaign conclusion is the common non-invalid classification only if all
three checkpoints agree; otherwise it is
`checkpoint_stratified_local_continuation_geometry`.

## Secondary measurements

Report, without allowing them to rescue the primary gate:

- per-cell Jacobian singular values and rank;
- tangent/kernel norm fractions and the linearized moment-error fraction;
- task-metric eigenspectra and all cross-cell similarities;
- exact-bin accuracy and Fisher-effect preservation for every named state;
- cellwise and aggregate closure fractions relative to the full correction;
- random-control distributions rather than only pass counts.

## Outcome meanings and stop rules

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| stable tangent sufficient | writer uses the wrong but stable local task metric | encode/fix that metric in a typed sidecar |
| checkpoint-local tangent sufficient | first-order geometry works but varies with state/support | model the declared metric equivariantly before training |
| kernel causally active | the first-order chart misses higher-order causal behavior | estimate continuation curvature/Hessian-vector effects |
| tangent fails, kernel inert, full passes | nonlinear tangent-kernel interaction or state conditioning is required | test Hessian cross-terms, not a larger observational writer |
| mixed/stratified | no single continuation mechanism spans checkpoints | retain a checkpoint-stratified atlas and stop universal sidecar claims |
| invalid | evidence contract failed | repair in a new root without interpreting scientific outcomes |

## Artifact and execution plan

- runner:
  `experiments/structure_net/tinyllm_local_continuation_tangent_kernel.py`
- tests:
  `tests/structure_net/test_tinyllm_local_continuation_tangent_kernel.py`
- systems-only root:
  `data/experiments/tinyllm_local_continuation_tangent_kernel/20260807_shakedown_cuda`
- primary root:
  `data/experiments/tinyllm_local_continuation_tangent_kernel/20260807_d6_preregistered_diagnostic`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-local-continuation-tangent-kernel.md`
- meta hypothesis:
  `tinyllm-c2-local-continuation-tangent-kernel-v1`

Focused unit/contract tests and a real CUDA shakedown must pass before the
primary root is launched. The shakedown may use one checkpoint, eight orbits,
and two random controls only when `--allow-underpowered` is explicit; it is
`systems_lifecycle_only_not_quality_evidence` and cannot be pooled.

## Method boundaries

The exact coordinate residual uses latent/exact held-out activations and is a
diagnostic intervention, not a deployable computation. The Jacobian and its
kernel are decoder-conditioned, local to an off-manifold predicted state, and
checkpoint-specific. The source-fitted carrier basis and order-four writer
have already been selected on earlier data. First-order nullness does not imply
finite-intervention causal nullness. Three post-selected checkpoints and reused
held-out cells are insufficient for a prevalence or general-architecture
claim.

## Amendment A — systems-only replay serialization

**Recorded:** 2026-08-07, after the first eight-orbit CUDA lifecycle failed
strict JSON and before any 64-orbit primary Jacobian or patch outcome existed.

The systems-only shakedown reached evaluation successfully, but its order-four
metrics have eight observations and are intentionally non-comparable with the
stored 64-orbit predecessor aggregate. The inherited generic comparator
returned positive infinity for that shape mismatch, which strict JSON
correctly rejected. The implementation now records a non-comparable replay as
`null` with a failed replay contract. The primary configuration remains 64
orbits and still requires a finite maximum replay difference at most `1e-6`.
The failed partial root remains preserved; the corrected lifecycle uses
`20260807_shakedown_cuda_v2`.

No checkpoint, primary cell, Jacobian definition, intervention, random stream,
threshold, classification, or scientific gate changed. No shakedown metric is
quality evidence.

## Amendment B — finite-difference corrective replication

**Recorded:** 2026-08-07, after the original 64-orbit campaign was classified
invalid and before interpreting any tangent/kernel patch outcome.

The original primary root is preserved at
`20260807_d6_preregistered_diagnostic`. Replay, target controls, Jacobian rank,
and algebraic decomposition passed in 3/3 checkpoints, but the preregistered
maximum finite-difference error exceeded `0.05` in all three (`0.061--0.094`).
Every Jacobian had rank two, the maximum decomposition error was below
`8.5e-13`, and cellwise mean finite-difference errors were only
`0.0011--0.0041`. This pattern identifies float32 subtraction cancellation at
the registered `1e-3` step, not a failed autograd derivative.

The corrective producer changes only the central-difference step from `1e-3`
to `1e-2`. The maximum-error threshold remains `0.05`; all scientific states,
random controls, causal endpoints, stability thresholds, classifications,
checkpoints, and cohorts remain fixed. The schema becomes
`nal.tinyllm-c2-local-continuation-tangent-kernel.v1.1`, the evidence role is
`post_outcome_corrective_replication_evidence`, and the new root is:

```text
data/experiments/tinyllm_local_continuation_tangent_kernel/
    20260807_d6_corrective_v2
```

The correction can reproduce and support the registered mechanism, but it is
not fresh confirmation. The invalid original remains reported.
