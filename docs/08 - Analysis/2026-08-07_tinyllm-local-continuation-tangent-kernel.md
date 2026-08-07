# TinyLLM local continuation tangent/kernel audit

**Status:** NOT CONFIRMED — LOCAL TANGENT REPAIRS ALL CELLS, BUT THE METRIC IS NOT CROSS-CELL STABLE  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED` WITH POST-OUTCOME NUMERICAL CORRECTION, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-local-continuation-tangent-kernel-v1`  
**Preregistration:** [local continuation tangent/kernel preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-local-continuation-tangent-kernel-preregistration.md)

## Verdict

The hypothesis of one stable local task metric is **not confirmed**. The valid
schema-v1.1 corrective campaign passed the full preregistered gate in `0/3`
checkpoints and classified all three as
`mixed_local_continuation_geometry`.

The narrower causal result is strong and consistent. At every one of the 12
held-out composition/extrapolation cells, projecting the exact-minus-order-4
writer residual into the orbit-local row space of the frozen continuation
Jacobian was sufficient to pass the continuous task endpoint. The complementary
first-order kernel moved the output by only `0.00005--0.00021` bins on average
per checkpoint. No one of eight norm-matched isotropic tangent-control policies
passed a complete checkpoint.

What failed was portability of that local metric. The four trace-normalized
mean metric tensors within a checkpoint were close in Frobenius norm, but their
leading directions had minimum absolute cosines of only `0.759--0.811`, below
the registered `0.90` floor. A second conservative specificity gate also
failed: kernel patches were much less disruptive than norm-matched random
patches, but their absolute advantage was `0.0188--0.0436` bins rather than the
registered `0.05`.

The supported mechanism is therefore:

```text
order-four coordinate error
  -> orbit-local rank-two continuation tangent carries the causal error
  -> orbit-local one-dimensional Jacobian kernel is inert at this scale
  -> tangent orientation changes with cohort/support inside a checkpoint.
```

This closes the generic writer-capacity branch. The missing object is not a
larger absolute writer or a Hessian correction on these cells; it is a
state/support-conditioned law for the local task metric.

## Campaign integrity

The campaign reused the exact three frozen d6 checkpoints, source-fitted
rank-three carriers, order-four writers, cohort seeds, exact orbit generator,
and readout rotations from the writer-capacity predecessor. It trained and fit
nothing.

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded / reused | 3 / 3 / 0 / 0 / 0 |
| held-out cells | 3 checkpoints x 2 cohorts x 2 shifts = 12 |
| exact `C2` orbits per cell | 64 |
| trained models / fitted writers / fitted observers | 0 / 0 / 0 |
| local Jacobians | 12 tensors, each `64 x 2 x 3` |
| registered states | predicted, tangent, kernel, full, 8 tangent-random, 8 kernel-random |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| peak allocated CUDA memory | 0.293 GiB |
| summed campaign analysis time | 14.97 seconds |
| implementation SHA-256 | `31ef191a31bbd5d509fb912cd5164385d846039a3b7c98aca04e6f5e29835c38` |
| campaign SHA-256 | `8b2f25dc61d1ffb93ba9dc66f1a85f36c07bda5a43ed3ed34a9627ca1485c12a` |
| final DVC data root | `7053c5bcd5433ee6822ec9825b782b53.dir` (`1,847` files, `39,814,283,869` bytes) |
| lakeFS snapshot | `fd8392ef275fda3a4e98fbe957208151061cfb8c0aeb48451b6455ecc326ed55` |

Fingerprint-matched resume returned the existing aggregate and preserved the
campaign SHA-256. All three predecessor result hashes and checkpoint hashes
were verified before intervention. Order-four coordinate and continuous
metrics replayed to at most `3.25e-7` absolute error.

The final DVC root was pushed to the configured
`lakefs://artifacts/main/structure-net/` remote and is contained in the cited
clean lakeFS commit.

The selected cohort remains `UNDERPOWERED`: these are the three previously
identified stable checkpoints, not a random five-seed prevalence sample.

## Numerical correction history

The original schema-v1 primary root is preserved and invalid. Its replay,
target-control, Jacobian-rank, and algebraic decomposition gates passed in
3/3, but central differencing a float32 continuation at step `1e-3` produced
isolated worst-orbit relative errors of `0.061--0.094`, above the registered
`0.05` maximum. Cellwise means were only `0.0011--0.0041`, every Jacobian had
rank two, and maximum tangent/kernel algebraic leakage was below `8.5e-13`.

Amendment B changed only the finite-difference step to `1e-2`, retained the
same maximum-error threshold and every scientific intervention, and created
schema v1.1 under a new root. The corrected maximum errors were
`0.0073--0.0261`; every validity gate then passed. Because the first producer
had already generated patch outcomes, the corrected result is explicitly
`post_outcome_corrective_replication_evidence`, not fresh confirmation.

A separate eight-orbit CUDA lifecycle was systems-only and was never pooled.
Its 64-orbit predecessor replay and target controls are intentionally invalid
at that reduced resolution.

## Preregistered gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| provenance and order-four replay | **3/3** | 3/3 | pass |
| predicted fails and full rank-three passes | **3/3** | 3/3 | pass |
| Jacobian, finite-difference, and decomposition contracts | **3/3** | 3/3 | pass |
| tangent passes all four held-out cells | **3/3** | 3/3 | pass |
| kernel output-inert in all four cells | **3/3** | 3/3 | pass |
| tangent norm-matched specificity | **3/3** | 3/3 | pass |
| kernel absolute random-control specificity | 0/3 | 3/3 | **fail** |
| cross-cell normalized metric stability | 0/3 | 3/3 | **fail** |
| complete stable local task-metric gate | **0/3** | 3/3 | **fail** |

The two failed gates cannot be rescued by the narrower positive findings.

## Causal tangent correction

The tangent patch nearly reproduced the full rank-three positive control while
using only the component visible to the local first derivative.

| Seed | Predicted mean shift | Tangent mean shift | Full rank-three mean shift | Tangent held-out pass |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.14181 | **0.02170** | 0.02093 | **4/4** |
| 29 | 0.22078 | **0.04582** | 0.04408 | **4/4** |
| 53 | 0.15723 | **0.01735** | 0.01593 | **4/4** |

The worst tangent p95 displacement was `0.1965` bins, comfortably below the
registered `0.50` ceiling. All winding, sampling, and alignment requirements
also passed.

The tangent component carried a mean `0.867--0.930` of residual norm. The
orthogonal kernel still carried `0.265--0.390` of residual norm, so the result
is not explained by a negligible coordinate remainder. Orthogonal component
fractions combine in squared norm, not by ordinary addition.

## Kernel and random controls

The kernel was first-order null by construction and remained causally inert at
the finite intervention scale:

| Seed | Kernel mean movement from predicted | Worst cell mean / p95 movement | Random-kernel median movement | Required absolute advantage |
| ---: | ---: | ---: | ---: | ---: |
| 7 | `0.000050` | `0.000087 / 0.000435` | 0.01881 | 0.05 |
| 29 | `0.000210` | `0.000379 / 0.001628` | 0.04383 | 0.05 |
| 53 | `0.000194` | `0.000567 / 0.000873` | 0.03360 | 0.05 |

The observed separations are large in relative terms but miss the locked
absolute `0.05`-bin specificity threshold, so the formal kernel-specificity
gate fails. This threshold miss does not turn the kernel into a causal
component; `kernel_causally_active` was false in all three checkpoints.

The tangent controls are decisive. The true tangent aggregate errors were
`0.0217`, `0.0458`, and `0.0173` bins, versus random-policy medians of
`0.1554`, `0.2399`, and `0.1755`. Zero of eight random tangent policies passed
all four held-out cells in every checkpoint.

## The metric is local, not stable

All 768 orbit-local Jacobians had numerical rank two. Their trace-normalized
cell means were moderately close:

| Seed | Maximum normalized Frobenius distance | Minimum leading-direction cosine | Registered gates |
| ---: | ---: | ---: | --- |
| 7 | 0.1831 | **0.8105** | `<=0.35`, `>=0.90` |
| 29 | 0.1311 | **0.7645** | `<=0.35`, `>=0.90` |
| 53 | 0.1441 | **0.7587** | `<=0.35`, `>=0.90` |

The leading normalized eigenvalue ranges from roughly `0.505` to `0.624`
across the campaign. With no overwhelming spectral gap, the leading vector can
rotate even while the full metric changes modestly. The preregistration
correctly required both conditions, and every checkpoint fails the directional
stability floor.

This explains why absolute quotient writers and one checkpoint-level metric
could look close yet fail causally: the relevant row space is an orbit-local
field over the recipient state, not one fixed coordinate plane.

## Interpretation and next action

This experiment makes a real interpretability advance despite the failed full
gate. It identifies where the remaining writer error matters:

- first-order task-tangent correction is causally sufficient under both shifts;
- a substantial coordinate kernel is causally inert at the tested scale;
- higher-order tangent/kernel interaction is unnecessary for these cells; and
- the task metric's orientation is support/state dependent.

Do not train a larger generic sidecar or add a Hessian module next. The shortest
remaining test is a no-training equivariance audit of the orbit-local metric
field: generate phase-matched nuisance orbits, compare their rank-two
projectors and one-dimensional kernel lines after the known `C2` action, and
test whether a declared group transport law makes the field stable. If it
does, that transport law is the architectural interface. If it does not, keep
the carrier as a checkpoint-local causal atlas and stop the universal-sidecar
claim.

## Artifacts and reproduction

- Authoritative corrective aggregate:
  `data/experiments/tinyllm_local_continuation_tangent_kernel/20260807_d6_corrective_v2/campaign_results.json`
- Per-seed evidence and compressed Jacobians:
  `data/experiments/tinyllm_local_continuation_tangent_kernel/20260807_d6_corrective_v2/runs/seed_*/`
- Preserved invalid schema-v1 primary:
  `data/experiments/tinyllm_local_continuation_tangent_kernel/20260807_d6_preregistered_diagnostic/`
- Corrected systems-only lifecycle:
  `data/experiments/tinyllm_local_continuation_tangent_kernel/20260807_shakedown_cuda_v3/`
- Runner:
  `experiments/structure_net/tinyllm_local_continuation_tangent_kernel.py`
- Tests:
  `tests/structure_net/test_tinyllm_local_continuation_tangent_kernel.py`
- Meta-hypothesis record:
  `data/meta_hypotheses/tinyllm-c2-local-continuation-tangent-kernel-v1.json`

The named hypothesis and all three experiment records passed authoritative
Chroma readback. The legacy Chroma transport emitted its known NumPy-2.0
consumer and telemetry warnings; the readback gate passed, and the strict JSON
record is the portable evidence ledger.

`dvc status` reports the refreshed local root and pipeline as up to date. The
exact root object is present in the cited clean lakeFS commit.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_local_continuation_tangent_kernel \
  --device cuda:0 \
  --post-outcome-corrective-replication \
  --output \
  data/experiments/tinyllm_local_continuation_tangent_kernel/20260807_d6_corrective_v2
```

## Method boundaries

The exact coordinate residual uses held-out exact activations and is not a
deployable signal. The source basis and order-four writer were selected in
predecessor studies. Jacobians are decoder-conditioned, local to an
off-manifold predicted state, and checkpoint-specific. Random controls test
isotropic coordinate directions, not every structured alternative. The three
checkpoints and reused cells do not establish population prevalence or a
universal architecture law.
