# TinyLLM C3 temporal sensor-only result

**Status:** PREREGISTERED FIVE-SEED RESULT CONFIRMED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-temporal-sensor-only-v1`

**Classification:** `task_only_sensor_acquisition_supported`

**Preregistration:** [C3 temporal sensor-only campaign](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-sensor-only-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_temporal_sensor_only/20260811_preregistered/campaign_results.json`

## Verdict

Task-only training of the existing 184-parameter exact-`C3` sensor succeeds in
all five seeds when the temporal operator and physical interval decoder are
fixed. Every learned-true seed passes the simultaneous carrier, exact-action,
composition-task, extrapolation-task, checkpoint-reload, and exact-resume
gate. Every matched target-shuffled seed fails.

```text
learned_true:             5/5
learned_target_shuffled:  0/5
required:                >=4/5 and <=1/5
```

The positive result is stronger than task adequacy alone. One orthogonal
carrier gauge fitted on a disjoint composition reference cohort transfers to
both primary shifts with mean unit-vector dot product above `.99999` in every
seed. All five fitted gauges have determinant `+1`. The randomly initialized
sensors therefore converge to the analytic carrier itself up to a tiny global
rotation, rather than merely discovering an unrelated shortcut through the
fixed operator.

No TinyLLM model was instantiated. This localizes the predecessor's failure to
the learned continuation/readout interface, not to the exact-`C3` sensor
family or task-only acquisition of the invariant carrier.

## Primary per-seed result

| Seed | Comp acc | Comp corr | Comp CE | Extrap acc | Extrap corr | Extrap CE | Extrap carrier dot | Extrap carrier RMSE | True joint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | `.9199` | `.99945` | `1.27444` | `.9199` | `.99938` | `1.27832` | `.999996` | `.002016` | pass |
| 17 | `.9541` | `.99958` | `1.27173` | `.9580` | `.99954` | `1.27531` | `.999999` | `.001216` | pass |
| 29 | `.9453` | `.99959` | `1.27214` | `.9609` | `.99956` | `1.27577` | `.999999` | `.000747` | pass |
| 41 | `.9492` | `.99962` | `1.27191` | `.9551` | `.99958` | `1.27549` | `.999995` | `.002276` | pass |
| 53 | `.9482` | `.99955` | `1.27221` | `.9580` | `.99951` | `1.27580` | `.999999` | `.000819` | pass |

The maximum learned-true deck-action error over all seed/shift cells is
`1.789e-6`, below the locked `2e-6` threshold. All true cells cover all sixteen
answer bins and pass the fixed cross-entropy limits, not merely correlation or
argmax accuracy.

## Population measurements

Means over five seeds:

| Arm | Shift | Accuracy | Posterior-mean correlation | Cross-entropy | Carrier dot | Carrier RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| learned true | composition | `.94336` | `.999559` | `1.27248` | `.9999976` | `.001413` |
| learned true | extrapolation | `.95039` | `.999513` | `1.27614` | `.9999976` | `.001415` |
| target shuffled | composition | `.10137` | `-.12912` | `14.8340` | descriptive only | descriptive only |
| target shuffled | extrapolation | `.09590` | `-.17364` | `15.2876` | descriptive only | descriptive only |

The shuffled population passes no joint seed. One shuffled seed has a
descriptively positive extrapolation correlation, but its accuracy,
cross-entropy, carrier, and joint gates fail. No endpoint was relaxed.

## What the intervention establishes

### The small sensor is learnable, not merely expressive

The preceding no-training study supplied a five-nonzero-parameter analytic
witness and a local gradient. This experiment closes the remaining gap:
ordinary random initialization, AdamW, and target soft cross-entropy recover
the carrier in `5/5` independent seeds without carrier supervision or an
analytic warm start.

### Architectural symmetry pays rent

The sensor is exactly invariant for every parameter state. Training therefore
cannot spend capacity memorizing deck orientation. With the correct temporal
law and physical decoder fixed downstream, the task gradient selects the
analytic invariant coordinate and that coordinate extrapolates.

This is the constructive result the earlier penalty and unrestricted-interface
studies did not obtain:

```text
restricted invariant sensor
  + fixed symmetry-compatible operator
  + fixed metric decoder
  -> support-stable learned quotient carrier.
```

### The transformer continuation was the avoidable failure surface

The earlier analytic d6 TinyLLM population passed representation and causal
closure in `5/5` but natural task utility in `2/5`. Affine output repairs then
passed at most `1/5`. The present sensor-only system uses the same observation,
carrier family, temporal relation, and interval task, yet passes `5/5` with
outside-support accuracy around `.95`.

The evidence now separates the components:

- sensor function class: sufficient;
- task-only sensor optimization: reliable in this five-seed population;
- analytic temporal operator: sufficient;
- fixed physical decoder: sufficient;
- unrestricted learned TinyLLM continuation/readout: not support-reliable in
  the predecessor population.

## Scope boundary

This is not evidence that TinyLLM itself learned the new quotient. TinyLLM was
absent by design. It is evidence that the quotient theory can construct a
small, trainable, extrapolating system once symmetry, temporal composition, and
metric decoding are put into the function class.

The result also does not establish:

- language-model usefulness;
- robustness to sensor noise, missing calibration, or approximate group
  actions;
- transfer to a new group, target relation, or temporal law;
- successful reintegration with an independently trained transformer
  continuation;
- superiority to directly computing the known analytic carrier on this
  noiseless synthetic generator.

The analytic rule remains the cheaper deployed solution for the present task.
The scientific value is that an unsupervised carrier target is unnecessary:
the task objective alone recovers it when the surrounding architecture is
typed correctly.

## Program decision

Do not put an unrestricted TinyLLM continuation back immediately. The current
claim has been answered cleanly. The shortest decisive successor must change
scope while retaining this positive control, for example:

1. add declared observation noise or missing calibration so the analytic
   carrier becomes a genuine learned estimator;
2. replicate the same typed sensor/operator/decoder separation on a different
   identifiable group or task relation; or
3. test a prospective typed continuation whose allowed operations provably
   preserve the learned carrier's orientation, scale, and metric chart.

An artifact-only local geometry audit of the five learned sensors is optional,
but it must answer a new mechanistic question. Another optimizer, seed, loss,
or carrier-fit sweep is not licensed.

## Integrity and lifecycle

| Check | Result |
| --- | ---: |
| requested/completed/failed seeds | `5 / 5 / 0` |
| primary optimizer steps | `6,000` |
| exact-resume verification steps | `3,000` |
| trainable parameters per arm | `184` |
| TinyLLM instances | `0` |
| true checkpoint reload | `5/5` |
| shuffled checkpoint reload | `5/5` |
| true midpoint exact resume | `5/5` |
| shuffled midpoint exact resume | `5/5` |
| matched initial states | `5/5` |
| analytic positive controls | `5/5` |
| maximum CUDA allocation | `.06410 GiB` |
| aggregate cell wall time | `158.03 s` |

The managed execution wrapper returned between spawned scheduler waves. The
final invocation reused four complete fingerprint-matched cells and scheduled
seed 53. No source, configuration, threshold, data hash, or result was changed;
pre-terminal checkpoint fragments were overwritten only by the identical
fingerprint. The final campaign reports `reused=4`, `scheduled=1`, and every
retained cell independently passes checkpoint reload and byte-exact midpoint
resume. Re-running the complete command leaves the final campaign bytes
unchanged.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-sensor-only \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_sensor_only \
  --mode primary \
  --output \
  data/experiments/tinyllm_c3_temporal_sensor_only/20260811_preregistered
```

| Artifact | SHA-256 |
| --- | --- |
| campaign | `4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012` |
| five-result manifest | `a23d19892f112645a7d3b5401d1528a0eb612a5d6db26520fe3514246c4c6d1a` |
| twenty-checkpoint manifest | `e83832872f29d072d710859e022f4e17d1b6da6a9e16b63049f41d4ea2eb01a0` |
| scientific implementation | `f4e0755c228475a7edaf52f458e964c6df5f6beb737db81833a5dd7a692e3b88` |
| campaign fingerprint | `bdad912fedee9e3323ca6e3c29cd330131b80b5015a647eac89cd1cfdc4c07a7` |
| runner | `7f4a5990f2f9a56bcaad0032d7cf9eca20f74b599a0684337c48dcdf9593b3ed` |
| preregistration | `e0c16420ea79130b22d6940b6f4943b5dda6bd3e048418eff078c0190e69a2bb` |
| licensing function-class result | `6a01db25ebc2ed15d202884c39f16db685d5218647b0bb209e2e5a737696a383` |

The focused preflight, meta-ledger, and campaign suite passes `18/18` tests
before the primary launch. The source hashes, training tensors, minibatches,
target permutations, evaluation cohorts, per-seed fingerprints, checkpoints,
and manifests are recorded in the artifact tree.
