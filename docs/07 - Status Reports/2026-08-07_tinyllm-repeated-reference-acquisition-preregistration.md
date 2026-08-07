# TinyLLM repeated-reference acquisition preregistration

**Status:** SUPERSEDED FOR CONFIRMATION — CORRECTIVE EXPANDED CAMPAIGN REQUIRED

**Date:** 2026-08-07

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, existing-checkpoint,
outcome-directed acquisition intervention

**Hypothesis:** `tinyllm-repeated-reference-acquisition-v1`

**Schema:** `nal.tinyllm-repeated-reference-acquisition.v1`

**Post-outcome provenance correction (2026-08-07):** The original locked
study specified the analytic circular mean only. Its one-seed shakedown was
inspected and showed recovery at `m=64` and `m=256`. A later implementation
audit then expanded the runner and this document with the learned equivariant
aggregator, Noise2Noise contract, and rate-law gates. That expansion is
scientifically useful, but it occurred after the analytic shakedown outcome
was known and while an authoritative run using the earlier implementation was
in progress. The in-progress runner correctly aborted on its implementation-
digest guard after seven of ten seed-arm cells. Those partial artifacts are
quarantined from confirmatory use. Any campaign using the expanded design must
use a fresh result root and be reported as a **corrective, outcome-informed
study**, not as confirmation under this preregistration.

**Pre-outcome implementation clarification (2026-08-07):** No new task
outcomes had been generated or inspected. To make the registered `m=1` source
replay exact, the first replicate is the source campaign's locked standard-
normal error. The other `255` replicates are new independent draws from the
same declared distribution and fixed acquisition seed. Thus all prefix
conditions remain identically distributed, while `m=1` can be verified
against the source artifact byte-for-formula.

**Historical implementation-audit claim (superseded):** An intermediate edit
described the expanded runner as pre-outcome. The provenance reconstruction
above shows that claim was incorrect: an analytic shakedown had already been
inspected. The expansion implements the learned aggregator, exact-reference
oracle, Noise2Noise contract, group-action checks, controls, and rate gate, but
those additions belong to the corrective outcome-informed design only.

## Decision question

The orientation-noise titration showed that a single `0.175`-radian noisy
orientation reference preserves the declared quotient representation but
destroys frozen exact-bin utility in all ten retained systems. The subsequent
activation/readout decomposition showed that the frozen local task computation
can express the correct answer when given the exact semantic coordinate, while
the reference-derived coordinate is not precise enough.

This study asks:

> Does reducing orientation-reference standard error through repeated
> independent acquisition restore the unchanged TinyLLM task output, and can
> one label-free equivariant learned aggregator match the analytic circular-mean
> positive control?

The directional prediction is that nested independent observations reduce
angular RMSE approximately as `m^-1/2`, and that both analytic and learned
aggregation recover the frozen task gate by `m=256` in each structured model
arm.

## Locked sources

Use the orientation-noise campaign:

```text
campaign
data/experiments/tinyllm_calibration_orientation_noise/
    20260807_d8_preregistered/campaign_results.json

campaign SHA-256
876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f

implementation SHA-256
990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70

source noise-array SHA-256
b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7

calibrated-checkpoint campaign SHA-256
80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501
```

The motivating activation/readout corrective campaign is
`833392ad7956ddcf715a20211431586f371edee280f167bf5a4fa51437cdc6c6`.
It selects the acquisition intervention but supplies no fitted object or task
outcome to this experiment.

Hard-validate all ten orientation result hashes, model/front-end checkpoint
hashes, model/system state hashes, dataset hashes, and the source conclusion
that the representation gate passes while the task gate fails at `0.175`.
Abort before new outcome inspection on any mismatch.

## Replication units and frozen systems

The checkpoint seed is the replication unit. Use both retained conditions:

- `analytic_calibrated`;
- `learned_calibrated_equivariant`.

Use seeds `7`, `17`, `29`, `41`, and `53`. All TinyLLM weights, structured
front ends, scalar embeddings, layer norms, and answer-token rows remain
frozen. The study trains exactly one acquisition-only denoiser shared across
all ten systems. It trains no model, front end, task head, or representation
probe.

## Data and independent acquisition process

Replay the orientation campaign's exact task/generator configuration and split
seeds:

| Split | Seed | Regime | Examples | Role |
| --- | ---: | --- | ---: | --- |
| train | `19184` | interpolation | `2048` | source/data identity only |
| validation | `19294` | interpolation | `512` | source/data identity only |
| composition | `20399` | composition | `1024` | primary |
| extrapolation | `21408` | extrapolation | `1024` | primary |

For every unique exact-cosine fiber in every split, construct `256`
independent standard-normal angular errors. Use the source campaign's locked
base error as column zero and generate columns `1:256` from a locked acquisition
seed.
Both C2 sheets in one fiber receive bit-identical replicate vectors. The same
replicate tensor is reused across model arms and checkpoints. Replicate-count
conditions are nested prefixes:

```text
m in {1, 4, 16, 64, 256}
sigma = 0.175 radians per observation
theta_i = theta_true + sigma * epsilon_i
```

Only the two orientation fields of the eight-field calibration packet change.
Signed speed, amplitude, offset, and drift remain exact. Inputs and targets are
bit-identical across acquisition rules and counts.

Store the complete replicate arrays in NPZ and hash their fiber IDs, shapes,
dtypes, values, and prefix relation. Verify unit orientation norm, formula
replay, independent replicate columns, shared paired-sheet errors, and
cross-arm/checkpoint identity.

## Acquisition rules

### Single observation

Use the first noisy orientation vector. This must reproduce the source
`0.175` task metrics. It is a negative baseline, not an aggregation method.

### Analytic circular mean

For each prefix, normalize the vector sum:

```text
q_m = normalize(sum_i y_i).
```

This is the positive-control estimator under the declared homoscedastic wrapped
Gaussian observation model. It uses only observed references.

### Learned equivariant aggregator

Train one shared reliability-weighted set aggregator, independent of TinyLLM
condition and checkpoint. Given a replicate set, form its provisional circular
mean. For each observation, compute rotation-invariant agreement with that
mean, the set resultant length, and the declared normalized log replicate
count. A fixed-width MLP maps those invariants to softmax reliability weights;
the normalized weighted vector sum is the output.

This construction is permutation invariant and exactly `SO(2)` equivariant:

```text
D({R_alpha y_i}) = R_alpha D({y_i}).
```

Train on an acquisition-only synthetic stream disjoint from every task split:

- latent reference direction sampled uniformly on the circle;
- independent input and held-out reference observations at `sigma=0.175`;
- replicate counts sampled from `4, 16, 64, 256`;
- loss `1 - dot(D(input observations), held-out noisy observation)`;
- `32` hidden units, Adam, learning rate `3e-3`, batch size `512`, at most
  `1200` steps;
- validation every `50` steps with patience `6`;
- training seed `271828`, validation seed `271829`.

The clean latent direction is used only to synthesize conditionally independent
observations and for post-training diagnostics. It is not passed to the model
and does not enter the denoiser loss. Phase, cosine, task bins, model
activations, model logits, and checkpoint identity are absent from denoiser
training. Store the selected denoiser state and normalization-free architecture
metadata before checkpoint evaluation.

Hard gates require maximum permutation error and rotation-equivariance error
at most `1e-6`, maximum unit-norm error at most `1e-6`, and a byte/state-stable
denoiser throughout all ten checkpoint evaluations.

### Oracle and pairing controls

- **Exact reference oracle:** use the original clean orientation packet. It
  must replay clean source task metrics and is positive-control evidence only.
- **Fiber-shuffled aggregate:** deterministically permute the analytic
  `m=256` estimates across fibers while retaining paired-sheet sharing and the
  same orientation marginal. It must not recover the task gate.

## Measurements

At every count and acquisition rule, measure on composition and extrapolation:

- circular angular MAE, RMSE, signed bias, and `95%` absolute-error quantile
  against the clean observed calibration direction;
- empirical resultant length;
- exact-bin task accuracy;
- target cross-entropy;
- accuracy loss from the unchanged clean system.

No representation probe is refit. The source campaign's passing `0.175`
representation endpoint is a prerequisite, not a remeasured endpoint. This
study tests whether acquisition precision is sufficient for the actual frozen
computation.

## Primary gates

All gates are evaluated per checkpoint before aggregation.

### Validity and replay

Every source/configuration/artifact/state/data hash must match; all metrics must
be finite; `m=1` task metrics must reproduce the source `0.175` metrics to
`1e-5`; and the exact-reference oracle must reproduce source clean metrics to
`1e-5`. Failure makes the campaign `invalid`.

### Frozen task recovery

For an acquisition rule and count, a checkpoint passes when exact-bin accuracy
loss from its unchanged clean baseline is at most `0.03` on both composition
and extrapolation. Each model arm passes at `4/5` checkpoints. The primary
count is `m=256`.

Both analytic and learned aggregation must pass both model arms at `m=256` for
the full hypothesis. In addition, the learned aggregator must be no more than
`0.02` absolute accuracy below the analytic mean on either shift in at least
`4/5` checkpoints per arm.

### Standard-error law

For the analytic circular mean, on both primary splits and counts
`m in {16,64,256}`:

```text
0.80 <= angular_RMSE / (0.175 / sqrt(m)) <= 1.20.
```

The log-log RMSE slope fitted over those three counts must lie in
`[-0.60,-0.40]`. This gate is evaluated once per split because the acquisition
arrays are common across checkpoints.

### Controls

The exact-reference oracle must pass `5/5` checkpoints in both model arms. The
fiber-shuffled `m=256` analytic control may pass at most `1/5` checkpoints per
arm. Single-observation replay must retain the source task failure.

## Fixed classification

Apply the first matching label:

| Outcome | Classification | Next action |
| --- | --- | --- |
| Any validity, replay, state, finite, or oracle gate fails | `invalid` | repair only the digital contract under a new root |
| Both aggregators recover both arms, learned matches analytic, rate law and controls pass | `acquisition_variance_causally_sufficient` | replicate on a measured calibration-noise model |
| Analytic recovers both arms but learned or matching gate fails | `analytic_averaging_sufficient_learned_gap` | fix the acquisition learner; do not alter TinyLLM |
| Learned recovers where analytic does not | `learned_non_gaussian_advantage` | inspect weighting and replicate on fresh acquisition noise |
| Neither recovers by 256 while oracle and controls pass | `repeated_acquisition_insufficient` | test reference bias/model mismatch, not variance tuning |
| Different model arms give different decisions | `arm_stratified_acquisition_limit` | preserve the arm distinction |
| Otherwise | `mixed_acquisition_result` | retain per-seed evidence and narrow the next test |

Only `acquisition_variance_causally_sufficient` confirms the full hypothesis.
Secondary trends cannot rescue a failed primary gate.

## Shakedown and execution

Before the primary campaign:

1. pass focused CPU contract tests for nested noise, circular averaging,
   equivariance, permutation invariance, gates, fingerprints, and resume;
2. run one seed with counts `1,4` and shortened denoiser training under
   `--allow-underpowered`;
3. verify real CUDA forward/save/reload, source replay, finite metrics, and
   exact resume;
4. freeze the implementation digest; and
5. launch the five-seed primary root once.

The shakedown is `systems_lifecycle_only_not_quality_evidence` and cannot enter
the scientific aggregate.

## Artifacts

- runner:
  `experiments/structure_net/tinyllm_repeated_reference_acquisition.py`
- tests:
  `tests/structure_net/test_tinyllm_repeated_reference_acquisition.py`
- quarantined incomplete root:
  `data/experiments/tinyllm_repeated_reference_acquisition/20260807_d8_preregistered`
- superseded corrective root (completed under a runner changed immediately
  afterward; retained append-only, not promoted):
  `data/experiments/tinyllm_repeated_reference_acquisition/20260807_d8_corrective_v2`
- superseded same-protocol corrective replay (valid, but not promoted):
  `data/experiments/tinyllm_repeated_reference_acquisition/20260807_d8_corrective_v3`
- authoritative corrective expanded root:
  `data/experiments/tinyllm_repeated_reference_acquisition/20260807_d8_corrective_expanded_v8`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-repeated-reference-acquisition-corrective.md`
- meta hypothesis:
  `tinyllm-repeated-reference-acquisition-v1`

The campaign stores strict JSON per checkpoint, one aggregate, the common
replicate NPZ, and the selected denoiser checkpoint. It records requested,
completed, failed, excluded, retried, and reused cells; trained-model,
front-end, task-head, probe, and acquisition-denoiser counts; environment;
implementation and scientific fingerprints; and all artifact hashes. Resume
must verify every byte-linked artifact and leave complete outputs unchanged.

## Method boundaries

This is a synthetic homoscedastic angular-noise study on ten retained d8/N3
systems, not a real calibration-cost estimate or architecture-population
claim. The checkpoint systems were selected by prior successful experiments.
The learned denoiser is a new acquisition module even though TinyLLM remains
frozen. Its Noise2Noise-style objective establishes performance under the
declared conditional-independence model, not arbitrary sensor bias. The exact
reference oracle uses hidden clean calibration and is not deployable. Passing
the task gate would show that reference variance was sufficient to explain the
tested failure; it would not prove that every quotient or downstream task can
be stabilized through repeated acquisition.
