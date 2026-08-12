# TinyLLM C3 temporal sensor function-class preflight

**Status:** FROZEN BEFORE EXECUTABLE PREFLIGHT

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `DESIGN / NO-TRAINING FUNCTION-CLASS PREFLIGHT`

**Hypothesis:** `tinyllm-c3-temporal-sensor-function-class-v1`

## Decision question

The observable-`C3` positive control established an exact invariant carrier, but
the trained TinyLLM continuation did not preserve its physical metric reliably
outside support. Before authorizing another learned system, ask the cheaper
structural question:

> Does the existing 184-parameter exact-`C3` sensor family contain the known
> analytic carrier, and can the true task reach its task-active parameters
> through the fixed temporal operator and fixed interval decoder?

This preflight performs no optimization, instantiates no TinyLLM, and loads no
checkpoint. A deterministic closed-form parameter assignment is a
function-class witness, not a trained result.

## Fixed source contracts

| Source | SHA-256 |
| --- | --- |
| existing learned `C3` sensor family | `dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6` |
| observable `C3` generator and analytic carrier | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| fixed interval likelihood | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |

Use the registered 4,096-example composition and extrapolation cohorts. The
sensor must have exactly 184 parameters and the generator must report no
quantizer saturation. Any source hash, parameter count, shape, or finiteness
failure makes the preflight invalid.

## Constructive witness

The current encoder applies a shared scalar MLP to every sensor channel, takes
the first `C3` character, normalizes it, and cubes it. Use the exact GELU
identity

```text
GELU(x) - GELU(-x) = x
```

to construct the following state without fitting:

1. zero every parameter;
2. assign the first two hidden weights to `+1` and `-1`, with zero bias;
3. subtract those two GELU outputs into character channel zero;
4. set the real mixer coefficient of channel zero to one;
5. leave every other parameter zero.

This makes the shared map's first output the corrected scalar, so its first
deck character equals the analytic complex carrier before normalization and
cubing.

The witness passes only if, on both shifts:

- scalar reconstruction maximum absolute error is at most `5e-7` on a fixed
  10,001-point grid over `[-4,4]`;
- complex carrier maximum absolute error is at most `2e-6`;
- maximum carrier change under either nonidentity deck action is at most
  `2e-6`;
- temporal target correlation is at least `.99` and RMSE at most `.08`;
- the fixed sixteen-bin task satisfies the predecessor's complete natural-task
  gates: posterior-mean correlation at least `.90`, composition/extrapolation
  accuracy at least `.50/.35`, cross-entropy at most `1.80/2.20`, and predicted
  coverage at least `14/12`;
- a fixed target roll by 137 has absolute temporal correlation at most `.10`
  and RMSE at least `.80`.

## Target-gradient route

Instantiate the same encoder at deterministic random seed `57031`. Generate
512 paired training examples at seed `97103`. Feed only the true soft target
loss through:

```text
learned exact-C3 sensor
  -> q7 * conjugate(q6) * q7
  -> real part
  -> fixed sixteen-bin interval posterior
  -> target cross-entropy.
```

The route passes only if:

- the loss and every gradient are finite;
- the total sensor gradient norm is at least `1e-6`;
- at least 90 percent of the 184 scalar parameters have nonzero numerical
  gradients;
- rolling every input by one deck element changes loss by at most `1e-6` and
  changes any parameter gradient by at most `2e-5`;
- one diagnostic normalized negative-gradient perturbation of radius `1e-3`
  lowers the same batch loss by at least `1e-4`, after which the exact initial
  state is restored.

The perturbation is a local derivative check, not an optimizer step or a
learned outcome. No claim about global optimization follows from this gate.

## Locked classification

| Outcome | Classification | Decision |
| --- | --- | --- |
| witness, fixed pipeline, controls, and gradient route all pass | `existing_c3_sensor_contains_analytic_carrier_and_task_gradient` | license one prospective sensor-only five-seed campaign with the temporal operator and interval decoder frozen |
| witness fails but sources are valid | `existing_c3_sensor_function_class_insufficient` | redesign the sensor before training |
| witness passes but task-gradient route fails | `analytic_witness_exists_but_task_gradient_blocked` | repair the fixed differentiable interface before training |
| source/hash/count/finiteness contract fails | `invalid_source_contract` | stop without scientific interpretation |

No threshold may be changed after reading the executable result. Passing this
preflight establishes representation capacity and a local task-gradient path;
it does not establish trainability, sample efficiency, or TinyLLM utility.

## Conditional successor

Only the first outcome licenses a new campaign. That campaign must retain the
same 184-parameter sensor, frozen analytic temporal operator, frozen interval
decoder, five seeds `(7, 17, 29, 41, 53)`, true and target-shuffled arms, and
the predecessor's simultaneous composition/extrapolation gates. It requires a
new dated preregistration before any optimizer step.

Expected preflight artifact:

```text
data/experiments/tinyllm_c3_temporal_sensor_function_class/
  20260811_preregistered/result.json
```
