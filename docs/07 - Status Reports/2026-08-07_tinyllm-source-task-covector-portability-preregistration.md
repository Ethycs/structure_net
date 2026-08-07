# TinyLLM source task-covector portability preregistration

**Status:** PREREGISTERED FRESH-COHORT POST-OUTCOME DIAGNOSTIC — COHORT C OUTCOMES NOT INSPECTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-source-task-covector-portability-v1`  
**Schema:** `nal.tinyllm-c2-source-task-covector-portability.v1`

## Question and directional prediction

Three preceding protocols showed that an example-local, decoder-conditioned
task tangent can repair the failed order-4 carrier write. Across 36 protocol
cells the local tangent intervention passed, while kernel, shuffled, flipped,
or random controls were generally inert or harmful. Those studies computed the
correction from the same example's exact carrier residual or exact output
error. They therefore establish a local causal direction but not a portable
mechanistic law.

This study asks the shortest unresolved question:

```text
Can a task covector and signed correction magnitude learned only on prior
cohorts predict a successful causal carrier correction on a fresh cohort,
without using the fresh example's exact residual or derivative?
```

The directional prediction is that source-predicted task corrections will pass
the frozen continuous continuation endpoint in both fresh composition and
fresh extrapolation cells for all three checkpoints, while correspondence,
sign, direction, and constant-covector controls will not.

No TinyLLM checkpoint, probe, carrier basis, decoder, or order-4 writer is
trained in this study. Six small ridge maps are fit: one three-output covector
map and one scalar-error map per checkpoint.

## Locked sources and replication units

Reuse the three selected d6 degree-two checkpoints `7`, `29`, and `53`, the
source-selected rank-three block-0 attention defect bases, and the exact
order-4 writers from:

```text
data/experiments/tinyllm_fixed_gauge_writer_capacity/
    20260806_d6_preregistered_diagnostic/campaign_results.json
SHA-256 c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078
implementation 7c284e35b5afc225eea45309262ab83c5f6d276736a557ebf20675ed3ccbfe7b
```

The previously interpreted local-tangent campaign is provenance-linked but is
not replayed as primary evidence:

```text
data/experiments/tinyllm_local_task_tangent/
    20260807_d6_preregistered_diagnostic/campaign_results.json
SHA-256 824a655b5c6d74f3c77259b9b7cacce3b4b3ea868ba74f48ba63fd5a24395130
```

The three frozen checkpoints are the replication units. This selected
three-checkpoint study is underpowered and cannot establish population
prevalence. Orbits and regime cells are repeated measurements.

## Data and split contract

The existing exact-orbit generator and task configuration are unchanged. Each
cell contains 64 paired degree-two orbits.

| Role | Cohort | Composition seed | Extrapolation seed | Outcome use |
| --- | --- | ---: | ---: | --- |
| writer alignment only | `alignment_fit` | `130007` | `130008` | reconstruct frozen writer and coordinate scaling |
| source-map fit | `heldout_a` | `230007` | `230008` | fit covector and scalar maps |
| source-map fit | `heldout_b` | `330007` | `330008` | fit covector and scalar maps |
| fresh primary test | `heldout_c` | `430007` | `430008` | primary endpoints and controls |

The fresh cohort seeds were selected and searched before cohort-C outcomes were
generated. Source A/B outcomes were known from preceding studies; consequently
this is a fresh-cohort post-outcome diagnostic, not an independent replication
of the original tangent hypothesis.

## Source-only maps

For each source example, reconstruct the frozen order-4 coordinate `c_4`, the
direct rank-three coordinate `c_*`, and the alignment-fit coordinate scale `s`.
At `c_4`, estimate the circular output-angle covector `g` in output-bin units
per standardized carrier coordinate using centered finite differences at
`0.025` standard deviations; use `0.05` as the convergence control. Define the
signed exact output error

```text
y = wrap(theta(c_*) - theta(c_4))  [output bins].
```

Pool source cohorts A and B across composition and extrapolation. From the
declared quotient phase, construct the fixed nine-dimensional feature vector

```text
x(phi) = [cos(phi), sin(phi), ..., cos(4 phi), sin(4 phi), 1].
```

Fit two ridge-regularized linear maps with ridge `1e-6`:

```text
g_hat(phi) = x(phi) W_g       (three signed covector components)
y_hat(phi) = x(phi) W_y       (one signed output error).
```

The primary standardized correction is the frozen one-step task inverse

```text
delta_source = g_hat y_hat / (||g_hat||^2 + 1e-12).
```

It is converted to raw rank-three coordinates with `s` before insertion into
the frozen block-0 attention defect state. Neither `c_*`, the fresh exact
output error, nor a fresh derivative participates in the primary correction.

The Fourier input is an oracle quotient-phase feature already used by the
locked predecessor. This test asks whether a causal correction law is portable
conditional on that known semantic chart; it does not establish an observable
or deployable front end.

## Frozen fresh-cohort interventions

At each fresh cell evaluate:

| State | Correction | Role |
| --- | --- | --- |
| `zero` | no rank-three defect | required negative target control |
| `exact` | full actual defect | required positive target control |
| `direct_rank3` | exact rank-three coordinate | carrier positive control |
| `order4` | frozen order-4 prediction | failed baseline |
| `local_oracle` | fresh local `g` and fresh exact `y` | local mechanism replication |
| `source_covector_oracle_error` | source `g_hat`, fresh exact `y` | covector portability |
| `local_covector_source_error` | fresh local `g`, source `y_hat` | scalar-error portability |
| `source_predicted` | source `g_hat`, source `y_hat` | primary no-fresh-target intervention |
| `source_mean_covector` | source mean `g`, source `y_hat` | phase-dependent-law control |
| `source_shuffled_error` | source `g_hat`, within-cell permutation of `y_hat` | correspondence control |
| `source_flipped` | negative source-predicted correction | sign control |
| `source_random_direction` | norm-matched isotropic direction | direction control |

Permutation and random streams are fixed from checkpoint and fresh evaluation
seed before evaluation. Exact fresh coordinates and derivatives are used only
for positive controls, secondary diagnostics, and outcome classification; they
cannot select or alter the primary correction.

## Numerical and local validity gates

The campaign is valid per checkpoint only if:

1. predecessor campaign/result, implementation, checkpoint, basis, and writer
   provenance replay exactly within the existing `1e-6` tolerance;
2. all coordinates, fitted coefficients, derivatives, and interventions are
   finite and coordinate scales exceed `1e-8`;
3. the source and fresh fine/coarse derivative protocols each satisfy the
   existing local-linearization gate: cosine at least `0.98`, relative L2 at
   most `0.15`, zero-referenced signed-error R2 at least `0.50`, residual MAE
   fraction at most `0.50`, and sign agreement at least `0.75` above `0.01`
   bins; and
4. `zero` fails while `exact` and `direct_rank3` pass in both fresh cells.

A failed validity gate cannot be interpreted as evidence against portability.

## Primary endpoint and campaign gate

Use the frozen predecessor's joint continuous endpoint in every fresh cell:

- circular alignment loss from exact at most `0.005`;
- mean circular-moment shift at most `0.125` output bins;
- p95 shift at most `0.50` bins;
- winding degree within `0.10` of degree two; and
- resolved sampling.

A checkpoint passes the **source-covector portability gate** only if:

1. every validity and target-control gate passes;
2. `local_oracle`, `source_covector_oracle_error`,
   `local_covector_source_error`, and `source_predicted` each pass both fresh
   composition and fresh extrapolation;
3. each of `source_mean_covector`, `source_shuffled_error`, `source_flipped`,
   and `source_random_direction` fails at least one fresh cell; and
4. the aggregate source-predicted mean shift is at least `0.125` output bins
   lower than each of those four controls.

The campaign supports the hypothesis only if all three checkpoints pass this
joint gate. The `3/3` rule is deliberately strict but remains underpowered.
Secondary fit scores, averages, or a partial checkpoint count cannot rescue a
failed full gate.

## Secondary measurements

Report without using them to rescue the primary gate:

- source and fresh local-linearization diagnostics;
- source fit zero-referenced R2 and MAE for `g_hat` and `y_hat`;
- fresh signed covector cosine and relative L2 error;
- fresh signed scalar-error R2, MAE, and sign agreement;
- correction norms and all continuous endpoint components;
- component-oracle pass counts and per-regime failures.

## Fixed classifications

Apply the first matching row per checkpoint:

| Outcome | Classification |
| --- | --- |
| provenance, numerical, replay, local-linearization, or target controls fail | `invalid` |
| local oracle fails a fresh cell | `local_tangent_not_replicated` |
| full primary and specificity gates pass | `portable_source_covector_and_scalar` |
| source-predicted passes but any specificity control does not | `nonspecific_source_correction` |
| both component-oracle interventions pass but their source-only composition fails | `source_components_portable_joint_not` |
| source covector with fresh error passes, but local covector with source error fails | `source_covector_portable_scalar_not` |
| local covector with source error passes, but source covector with fresh error fails | `source_scalar_portable_covector_not` |
| neither source component passes while local oracle passes | `local_oracle_only` |

If component outcomes do not match a preceding row, classify
`mixed_source_portability`.

## Outcome meanings and next action

| Outcome | Interpretation | Next shortest action |
| --- | --- | --- |
| full gate passes `3/3` | the local task metric and correction amplitude form a source-predictable causal law on a fresh cohort | replace oracle phase with an observable carrier estimator and test another checkpoint cohort |
| covector only | the decoder-sensitive direction transports, but error amplitude remains example-local | learn only a scalar residual sensor |
| scalar only | error magnitude transports, but the causal metric field is local | test a constrained covector field or parallel transport |
| components pass but joint fails | first-order composition or calibration is wrong | measure the interaction; do not add writer capacity |
| local oracle only | the causal tangent is real but not predictable from the declared phase chart | stop portable sidecar claims; test richer observable context only if justified |
| local oracle fails | the earlier effect does not reproduce on a fresh cohort | audit support dependence before any new model training |
| nonspecific | correction helps, but the claimed phase-conditioned causal law is not identified | prefer the simpler passing control and narrow the claim |

## Artifacts and execution plan

- runner:
  `experiments/structure_net/tinyllm_source_task_covector_portability.py`
- tests:
  `tests/structure_net/test_tinyllm_source_task_covector_portability.py`
- primary root:
  `data/experiments/tinyllm_source_task_covector_portability/20260807_d6_preregistered_fresh_cohort`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-source-task-covector-portability.md`
- meta hypothesis:
  `tinyllm-c2-source-task-covector-portability-v1`

The runner records strict JSON, source and result hashes, producing-code
digest, scientific fingerprints, map coefficients, fresh seeds, control stream
digests, and immutable resume. A one-seed CUDA shakedown is systems-only and
uses a separate root.

## Exclusions, retries, and method boundaries

No scientifically completed checkpoint is retried for a threshold miss. An
infrastructure failure may be resumed only under an identical fingerprint and
implementation digest; all failures and retries remain visible. There are no
post-outcome exclusions.

This intervention is decoder-conditioned, off-manifold, first-order, and
conditional on an oracle quotient-phase chart. Fresh C is a new generator seed
within the already-declared composition and extrapolation regimes, not a new
shift family. Source A/B were selected after their outcomes were known. Three
selected checkpoints are underpowered, and even a full pass would establish a
portable frozen diagnostic law—not a naturally used circuit, population-wide
universality, or a deployable observable encoder.
