# TinyLLM calibrated architecture-family replication

**Status:** VALID PREREGISTERED NEGATIVE; CAUSAL SUBCLAIM REPLICATED  
**Date:** 2026-08-10  
**Evidence role:** `prospective_architecture_family_replication`  
**Primary artifact:** `data/experiments/tinyllm_calibrated_architecture_replication/20260810_d6_d10_preregistered/campaign_results.json`

## Verdict

The calibrated quotient mechanism generalizes across the fresh d6 and d10
architectures, but its natural task usefulness under the fixed 600-step
training protocol does not. Every analytic and learned-equivariant checkpoint
passed the representation gate and the four-cut frozen causal-closure gate on
composition and extrapolation. The preregistered architecture-family claim
still failed because same-seed d8-relative task adequacy reached only `3/5`
d10 analytic seeds and `1/5` seeds in each learned arm.

In operational terms, the quotient is paying rent as a mechanism: replacing a
task orbit by its exact barycenter preserves the actual frozen computation,
whereas semantic shuffling does not. It is not yet paying rent as a portable
engineering result: the same architecture-level constraint did not preserve
the trained system's natural task quality across the declared model family.

The locked classification is:

```text
structured_closure_not_architecture_stable
```

## Primary result

The checkpoint seed is the replication unit. A structured seed passes only if
representation, same-seed task adequacy, causal closure at all four cuts, and
all validity controls pass jointly on both held-out shifts.

| Preset | Arm | representation | task adequacy | causal all cuts | joint | arm/preset pass |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| d6 | raw | 0/5 | descriptive only | 3/5 | 0/5 | no; specificity comparator |
| d6 | analytic | 5/5 | 5/5 | 5/5 | 5/5 | yes |
| d6 | learned equivariant | 5/5 | 1/5 | 5/5 | 1/5 | no |
| d10 | raw | 0/5 | descriptive only | 4/5 | 0/5 | no; specificity comparator |
| d10 | analytic | 5/5 | 3/5 | 5/5 | 3/5 | no |
| d10 | learned equivariant | 5/5 | 1/5 | 5/5 | 1/5 | no |

The primary hypothesis required both structured arms to pass both fresh
presets at `4/5`. It is rejected. No execution failure, invalid cell, retry,
or post-outcome threshold change contributed to that result.

## What replicated

### Representation geometry

Each structured arm/preset contributed 20 held-out representation cells: five
seeds, two cuts, and two shifts. Every one passed the joint threshold
`corr >= 0.90`, conditional branch accuracy `<= 0.55`, and conditional
log-loss gain `<= 0.02`.

| Preset | Arm | minimum cosine correlation | maximum branch accuracy | maximum conditional log-loss gain |
| --- | --- | ---: | ---: | ---: |
| d6 | analytic | 0.957706 | 0.507812 | 0.000115 |
| d6 | learned equivariant | 0.923500 | 0.514648 | 0.000622 |
| d10 | analytic | 0.957706 | 0.508789 | 0.000115 |
| d10 | learned equivariant | 0.943676 | 0.534180 | 0.000488 |

The raw comparator passed no seed. Its minimum correlation was `-0.005557`,
maximum conditional branch accuracy was `0.999023`, and maximum conditional
log-loss gain was `0.653784`.

The trained learned front ends also retained their exact architectural
contract: maximum measured group-action error was `3.576e-7` for d6 and
`2.980e-7` for d10, both below the locked `1e-5` tolerance.

### Frozen causal closure

Every structured checkpoint passed exact task-orbit barycenter substitution at
all four registered cuts under both shifts:

| Preset/arm | pre-block | block-0 post-attention | block-0 post-MLP | full depth |
| --- | ---: | ---: | ---: | ---: |
| d6 analytic | 5/5 | 5/5 | 5/5 | 5/5 |
| d6 learned equivariant | 5/5 | 5/5 | 5/5 | 5/5 |
| d10 analytic | 5/5 | 5/5 | 5/5 | 5/5 |
| d10 learned equivariant | 5/5 | 5/5 | 5/5 | 5/5 |

All structured semantic-shuffle controls passed `0/5` at the pre-block cut.
Maximum direct replay and paired-state identity errors were exactly zero, and
all model and system state hashes remained unchanged. This establishes that
the causal result is neither a generic average-state effect nor accidental
continued training.

Raw closure was `3/5` d6 and `4/5` d10 at pre-block, becoming `5/5` at later
cuts. It remains descriptive because the raw natural task is not an adequate
positive control and its representation gate passes `0/5`.

## What failed

The only failed primary conjunct was natural task adequacy relative to the
matched d8 seed minus three percentage points.

| Arm | Preset | mean composition accuracy | mean extrapolation accuracy | passing seeds |
| --- | --- | ---: | ---: | ---: |
| analytic | retained d8 anchor | 0.744922 | 0.616406 | outcome-known reference |
| analytic | d6 | 0.752344 | 0.628906 | 5/5 |
| analytic | d10 | 0.728906 | 0.615820 | 3/5 |
| learned equivariant | retained d8 anchor | 0.716797 | 0.491602 | outcome-known reference |
| learned equivariant | d6 | 0.766602 | 0.473438 | 1/5 |
| learned equivariant | d10 | 0.692578 | 0.348047 | 1/5 |

For d10 analytic, seed 17 missed the composition floor by `0.0618` and the
extrapolation floor by `0.0208`; seed 29 missed only the composition floor by
`0.000273`. Its population means are therefore close to the d8 anchor, but the
strict seedwise `4/5` claim is still false.

The learned failures are more substantial. D6 passed only seed 29. D10 passed
only seed 17; the other seeds had extrapolation deficits from `0.1702` to
`0.2317`, and seed 29 also had a `0.3323` composition deficit.

This result does not establish that d10 is generally worse. The registered
presets jointly change depth, width, and head count, and every cell used the
same fixed optimizer and 600-step budget. It establishes only that the full
trained behavior is not architecture-family stable under that declared
protocol.

## Scientific accounting

### What the theory has earned

- Identifiability remains a necessary gate: adding an observed calibration
  reference made absolute cosine a well-defined target.
- The base/fiber distinction remains predictive: structured front ends retain
  cosine and contract tested conditional branch information across two new
  architectures.
- Causal quotient sufficiency is not merely probe geometry: exact orbit
  barycenters preserve the frozen computation at every registered cut, and
  task-changing shuffles fail.
- Architectural restriction is stronger than a sampled residual penalty: the
  declared group contract survives training to numerical tolerance.

### What the theory has not earned

- A quotient-sufficient representation does not guarantee accurate natural
  task behavior.
- Exact equivariance does not select a universally well-calibrated scalar
  coordinate or answer-bin interface.
- One d8 success does not license a model-family engineering claim.
- The study cannot separate depth, width, or head-count effects.

The result narrows the failure to the interface from a valid invariant
coordinate into the trained task continuation, rather than refuting the
quotient account itself.

## Shortest decisive follow-up

Do not retrain and do not add another representation loss. On the failed cells,
perform a frozen one-dimensional interface decomposition:

1. replay the natural learned scalar;
2. substitute exact latent cosine through the existing scalar embedding;
3. search the one-dimensional scalar input to determine whether the correct
   answer bin is reachable through the frozen continuation;
4. retain passing cells as positive controls and use shuffled or wrong-cosine
   substitutions as specificity controls.

If exact cosine repairs a cell, the failure is sensor calibration. If only the
one-dimensional oracle repairs it, the failure is coordinate or bin-boundary
calibration. If no scalar value repairs it, the frozen continuation or answer
row is the failing component. This is cheaper and more discriminating than a
new training sweep.

This follow-up is complete in the
[frozen scalar-interface decomposition](2026-08-11_tinyllm-frozen-scalar-interface-decomposition.md).
Exact cosine repairs `1/10` failures, the registered one-dimensional oracle
repairs `9/10`, and one d10 learned checkpoint has a four-bin hole on the
admissible scalar interval. No retraining was required.

## Execution and lifecycle

The primary campaign trained 30 fresh CUDA cells: three arms, two fresh
presets, and five seeds. It completed in `32.10` minutes of elapsed campaign
time and `53.32` aggregate GPU-minutes. The primary artifact tree contains 152
files and occupies `6,708,413,912` bytes.

Lifecycle shakedowns were kept outside the primary root. The first found a
reduced-probe versus full-cohort task-comparison error; the corrected runner
marks that comparison non-applicable only in lifecycle mode. A learned d10
shakedown then found CPU construction of group-action tensors against a CUDA
sensor; the corrected helper constructs them on the sensor device. Both failed
shakedowns remain preserved as systems evidence and contribute no scientific
cell.

The final d10 raw, analytic two-slot, and learned shakedowns passed replay,
state, artifact, and CUDA checks with `1.579`--`1.856` GB measured peak memory.
The 30-cell primary campaign then completed with zero failures and no observed
retry. Re-execution under exact resume left the campaign byte-identical.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_calibrated_architecture_replication \
  --gpus 0 --slots-per-gpu 2 --max-parallel 2 --retries 1 \
  --output data/experiments/tinyllm_calibrated_architecture_replication/20260810_d6_d10_preregistered
```

| Artifact | SHA-256 |
| --- | --- |
| campaign | `656d9814a032d1899810e81d398adf935cea3e1116712460e2062da188a0c9e2` |
| result manifest | `903b61792fd624c520f2e563caf7742a40717993d129fcf489f618975bf41052` |
| checkpoint/front-end/diagnostics manifest | `97550245f9f52a93c84dc6faf486d9837e1460832316ece51d23e23d15c37a4b` |
| campaign fingerprint | `b8dc7ce7c717fdf166e83468813fd6771fb729d4d6bc94e620e8b4e7b271ff5d` |
| scientific preflight | `ea8b12c006dc860d41715095c4a111bb1fcdd844a859c83427122ad1d02bbe6a` |
| primary runner | `661384de2eac23d95dbc550e0ecf49b14ebfeb01ef47fc1a8164e7e3b2b0ca90` |
| preregistration | `36c1a8c35823fda3076b6a73648facd7fc18513c3c969bfa297ca9c0b34c4c77` |
| combined implementation | `5ac391fd2d719a219cff9673c92004c4ae3e7401b242d8561844e26cd6bf21ab` |
| composition dataset | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation dataset | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |

The retained d8 cells are outcome-known anchors and are excluded from fresh
confirmation. The four valid lifecycle roots and two preserved failed
lifecycle roots are not pooled with the primary campaign.

## Data and evidence backup

The complete data tree is tracked by DVC root
`e02d4354a7463797d6e7881d571c298c.dir`: `48,960,266,515` logical bytes across
3,362 files. DVC reports the local cache and configured `lakefs` remote in
sync. The uploaded objects were sealed as lakeFS commit
`40140ec85a29a638924c56f62da14c4cdf4e9d4955000f91da2c6e64462e3660`,
and the branch has no uncommitted `structure-net` diff.

The immutable root object is available at
`lakefs://artifacts/40140ec85a29a638924c56f62da14c4cdf4e9d4955000f91da2c6e64462e3660/structure-net/files/md5/e0/2d4354a7463797d6e7881d571c298c.dir`.
Direct lakeFS metadata checks returned checksum
`478c40485c67f55f1d32460344c69572` for the primary campaign JSON and
`51dac684781ef55c958958ef164ee444` for the stored meta-hypothesis JSON at the
same commit.
