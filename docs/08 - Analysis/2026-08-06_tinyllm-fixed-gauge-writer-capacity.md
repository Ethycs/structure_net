# TinyLLM fixed-gauge writer capacity

**Status:** DIAGNOSTIC RESOLVED — SMALL DECLARED WRITER INTERFACES INSUFFICIENT  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome frozen-checkpoint diagnostic  
**Hypothesis:** `tinyllm-c2-fixed-gauge-writer-capacity-v1`  
**Preregistration:** [fixed-gauge writer-capacity preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-fixed-gauge-writer-capacity-preregistration.md)

## Verdict

Neither quotient curvature nor the declared calibration-conditioned write
rescued the fixed causal interface. All replay, oracle, context, and target
controls passed in all three retained checkpoints. Nevertheless, Fourier
orders 2, 4, and 40 and the parameter-matched order-4 calibration-context arm
each passed zero of three checkpoint gates. Every checkpoint received the
preregistered classification `unresolved_writer_limited`.

The low-order writers improved the failed order-1 writer descriptively, but
not enough to cross the causal endpoint. Orders 2 and 4 each passed only two of
12 held-out cells, both heldout-B composition cells. Neither passed an
extrapolation cell. Order 40 and the context arm passed zero of 12 cells.

The supported conclusion is narrow but decisive:

> At the block-0 attention defect cut, the required rank-three causal write is
> not a portable global function of quotient phase through order 40, nor of
> order-4 phase jointly with the full observed calibration packet, under these
> reused cohorts and three checkpoints.

This ends the fixed sidecar-writer branch. The next intervention should act on
the downstream continuation's local nonlinear state dependence.

## Preregistered decision

| Gate or classification | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| predecessor replay contract | **3/3** | 3/3 | pass |
| oracle and context contract | **3/3** | 3/3 | pass |
| continuous target controls | **3/3** | 3/3 | pass |
| specific order-2 quotient writer | **0/3** | a passing checkpoint | fail |
| specific order-4 quotient writer | **0/3** | a passing checkpoint | fail |
| specific order-40 quotient writer | **0/3** | a passing checkpoint | fail |
| specific order-4 calibration-context writer | **0/3** | a passing checkpoint | fail |
| final classification | **3/3 unresolved** | fixed table | resolved |

The campaign is a mechanistic decomposition, not a positive confirmation
claim. Checkpoints are the replication units; the 12 cells are repeated
held-out measurements.

## Campaign integrity

The campaign loaded frozen d6 C2 checkpoints 7, 29, and 53 and reused their
locked rank-three bases, four held-out cells, continuation cut, and readout
calibrations. It trained no model or observer. Twenty-seven ridge writers were
fit: five real arms and four shuffled controls per checkpoint.

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 checkpoints |
| trained models / predictive observers | 0 / 0 |
| fitted ridge writers | 27 |
| held-out cells | 12: 2 cohorts x 2 shifts x 3 checkpoints |
| exact C2 orbits per cell | 64 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| analysis time | 12.73 seconds |
| implementation SHA-256 | `7c284e35b5afc225eea45309262ab83c5f6d276736a557ebf20675ed3ccbfe7b` |
| campaign SHA-256 | `c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078` |
| final DVC data root | `f29e1f0e920aff74661e2a64d7ec56c1.dir` (`1,796` files, `39,812,097,258` bytes) |
| lakeFS snapshot | `71cda38c5b84bfa364c136a0741dd4ff6e77040395f4e24b5d50d8419c11a648` |

The order-1 mapping reproduces the predecessor to at most `8.88e-16`; held-out
continuous records replay to at most `3.57e-7`, below the `1e-6` contract. The
oracle phase error is at most `6.24e-16` output bins and the calibration packet
is exactly identical across the two sheets of every orbit. An immutable module
entry-point resume left the aggregate SHA unchanged. The final DVC root is
current locally, was pushed to the configured
`lakefs://artifacts/main/structure-net/` remote, and is contained in the cited
clean lakeFS commit.

## Causal result

| Arm | Width | Mean shift by seed, output bins | Passing cells | Passing checkpoints |
| --- | ---: | --- | ---: | ---: |
| quotient order 1 | 3 | 0.2410, 0.2317, 0.1611 | 1/12 | 0/3 |
| quotient order 2 | 5 | 0.1462, 0.2242, 0.1552 | 2/12 | 0/3 |
| quotient order 4 | 9 | 0.1418, 0.2208, 0.1572 | 2/12 | 0/3 |
| quotient order 40 | 81 | 0.4369, 0.5016, 0.4002 | 0/12 | 0/3 |
| order 4 x calibration context | 81 | 0.2227, 0.4405, 0.2775 | 0/12 | 0/3 |

The low-order result is a real but insufficient effect. Relative to order 1,
order 4 reduced mean circular shift by `0.0992` bins in seed 7, `0.0109` in seed
29, and `0.0038` in seed 53. It retained exact degree two in every cell. The
remaining failures were primarily mean-shift failures: the endpoint requires
at most `0.125` bins, while the order-4 four-cell means were `0.142--0.221`.

Only heldout-B composition passed for order 2 and order 4 in seeds 7 and 53.
All 12 extrapolation cells across those two low-order arms failed. This is not
merely an extrapolation failure, however: four of six composition cells also
failed for each arm, and seed 29 passed none.

## More capacity makes generalization worse

The order-40 arm raised alignment-fit coordinate variance explained to
`0.994--0.998`, yet its worst held-out value fell to `0.697--0.833`. Its causal
mean shifts rose to `0.400--0.502` bins and no cell passed. This is the expected
signature of a flexible phase chart fitting the alignment cohort without
recovering a stable causal interface.

The context arm was parameter matched to order 40 at 81 features. It also fit
alignment coordinates extremely well (`R2 = 0.993--0.998`) but generalized
poorly: worst held-out `R2` was `0.824--0.964`, and mean causal shifts were
`0.223--0.440` bins. Thus its failure cannot be attributed simply to having
fewer coefficients than the high-order phase control.

Every shuffled control behaved as required. The real-versus-shuffled mean-shift
margins ranged from `3.25` to `4.43` bins, so the failed writers still captured
genuine paired structure. Descriptive correspondence is therefore present but
is not sufficient for the frozen continuation.

## Mechanistic interpretation

The combined evidence removes three increasingly flexible explanations:

1. exact quotient phase does not make a global linear write sufficient;
2. small neutral Fourier curvature does not make it sufficient;
3. neither 81 phase harmonics nor an 81-wide phase-by-calibration write makes
   it sufficient out of cohort.

Direct rank-three defects continue to pass, so the target causal subspace is
adequate. What fails is the attempt to predict its required coordinates from a
small external chart. The likely missing variable is the propagated internal
state itself: the downstream continuation uses a local, state-dependent task
metric, not a checkpoint-global coordinate convention.

The calibration-context arm uses the complete observed eight-field packet. It
is not an architecturally equivariant context map, so this result does not
falsify every possible symmetry-constrained interface. It does show that adding
raw acquisition context and matched parameter capacity is not a rescue. Any
future symmetry experiment must specify an intertwiner or invariant context
port in advance and compare it with this failed broad control.

## Decision and next shortest test

Stop fitting portable quotient-to-rank-three sidecar writers at this cut. Use
the frozen artifacts to measure the continuation directly:

1. estimate the task-relevant continuation Jacobian at each propagated state;
2. compare the exact defect with its local linear and quadratic continuation
   effects;
3. Reynolds-average those effects over the C2 orbit and decompose them into
   neutral character couplings;
4. test matched random directions and shuffled orbit membership.

This asks whether the causal carrier is defined by a state-dependent
intertwiner. It is cheaper and more decisive than another model training run,
another global writer ladder, or a topology scan.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_fixed_gauge_writer_capacity/20260806_d6_preregistered_diagnostic/campaign_results.json`
- Per-checkpoint records:
  `data/experiments/tinyllm_fixed_gauge_writer_capacity/20260806_d6_preregistered_diagnostic/runs/seed_*/result.json`
- Systems-only lifecycle roots:
  `data/experiments/tinyllm_fixed_gauge_writer_capacity/2026080{6,7}_shakedown_cuda/`
- Runner:
  `experiments/structure_net/tinyllm_fixed_gauge_writer_capacity.py`
- Tests:
  `tests/structure_net/test_tinyllm_fixed_gauge_writer_capacity.py`

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_fixed_gauge_writer_capacity \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_fixed_gauge_writer_capacity/20260806_d6_preregistered_diagnostic
```

Direct-path execution is not the supported entry point because the runner uses
repository package imports. Use the module command above.

## Method boundaries

This is a post-outcome, three-checkpoint diagnostic on selected stable block-0
models and reused held-out cells. Its primary quotient features use latent phase
and are nondeployable. Writers have target-coordinate access on alignment-fit
orbits and patches are off manifold. Order 40 is a capacity control, not a
proposed architecture. The raw calibration tensor product is not an exact
group-equivariant network. The result does not establish population prevalence
or rule out a local state-conditioned continuation mechanism.
