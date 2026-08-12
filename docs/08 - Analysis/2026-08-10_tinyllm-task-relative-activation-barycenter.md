# TinyLLM task-relative activation barycenter causality

**Status:** COMPLETE PREREGISTERED UNDERPOWERED CAUSAL NULL — d8 VALID,
d6 QUARANTINED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`,
frozen-checkpoint causal intervention  
**Hypothesis:** `tinyllm-task-relative-activation-barycenter-v1`  
**Schema:** `nal.tinyllm-task-relative-activation-barycenter.v1`  
**Preregistration:** [task-relative activation barycenter preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-task-relative-activation-barycenter-preregistration.md)

## Verdict

The hypothesis is **not confirmed**. The valid d8 checkpoint has no isolated
or mature cut at which the exact cosine-fiber barycenter preserves the frozen
continuation. The d6 arm cannot be used as direct evidence because its
outside-range cosine baseline accuracy is `0.1328`, below the locked `0.15`
validity floor.

This is the central causal result:

> The previously observed interval-like geometry and conditional branch-probe
> collapse do not make the complete residual barycenter an autonomously
> sufficient state for the raw TinyLLM continuation.

The intervention is nevertheless useful. In d8 it improves cosine exact-bin
accuracy and cross-entropy at the observational reference cut and full depth,
under both shifts. But it changes the frozen posterior more than the locked
Jensen--Shannon ceiling at every cut. A beneficial projection is not the same
as preservation of the model's computation.

| Lifecycle item | Outcome |
| --- | --- |
| retained checkpoints | `4`: d6/d8 phase-circle and cosine-interval, all seed 7 |
| exact fibers | `512` per regime, two shared-nuisance sheets each |
| regimes | training support and outside range |
| activation cuts | query embedding and every post-attention/post-MLP residual |
| models, probes, decoders, or alignments fit | `0` |
| d8 causal front | **none**; valid causal null |
| d6 causal front | not interpretable; baseline-invalid quarantine |
| primary hypothesis | **fail** |
| raw classification | `observational_task_geometry_not_causally_sufficient` |

## Registered intervention

For each shared-nuisance pair, the two phases were

```text
phi+ = arccos(u)
phi- = 2 pi - arccos(u),
```

so both rows had the same cosine target and different phase targets. Direction,
amplitude, orientation, offset, harmonic strength, angular speed, and the full
pre-quantization noise array were identical within each pair. The complete
residual sequence at each cut was averaged exactly and patched into both rows:

```text
b_i = (h_i,+ + h_i,-) / 2.
```

The actual frozen remainder and native answer head then produced the posterior.
No latent value entered a continuation. Exact replay at the original activation
was evaluated separately.

Sufficiency required all three conditions simultaneously relative to replay:

```text
accuracy loss <= 0.03
target cross-entropy increase <= 0.05
posterior JS <= 0.02.
```

A mature front also had to pass both regimes at that cut and every later cut.
The fixed observational references were d6 block 5 post-attention and d8 block
3 post-attention.

## Valid d8 causal null

The d8 baseline, replay, cohort, source, finite-value, and model-state contracts
all pass. Its outside-range cosine baseline accuracy is `0.1523`, narrowly above
the preregistered `0.15` floor. No correct-barycenter cut passes the simultaneous
endpoint in either regime, so there is neither an isolated passing cut nor a
mature front.

### Observational reference cut: block 3 post-attention

| Regime | Baseline acc. | Patched acc. | Accuracy gain | CE improvement | Posterior JS | JS gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| training support | `0.4385` | `0.5762` | `+0.1377` | `+0.1149` | `0.02763` | fail |
| outside range | `0.1523` | `0.1855` | `+0.0332` | `+0.9180` | `0.20906` | fail |

### Full depth: block 8 post-MLP

| Regime | Baseline acc. | Patched acc. | Accuracy gain | CE improvement | Posterior JS | JS gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| training support | `0.4385` | `0.5781` | `+0.1396` | `+0.1182` | `0.02746` | fail |
| outside range | `0.1523` | `0.1855` | `+0.0332` | `+1.0713` | `0.20593` | fail |

The in-support miss is not a rounding accident: `0.02746--0.02763` remains
above the frozen `0.02` ceiling. Outside range, the posterior divergence is an
order of magnitude above the ceiling. Accuracy and target likelihood improve
because averaging removes a harmful component, but the continuation does not
behave as though it received the original sufficient state.

Pairwise residual differences also remain large. At full depth the mean
opposite-sheet pair RMS is `8.6912` in support and `9.5663` outside range,
respectively `0.9141` and `1.0376` times the state RMS. This study does not
claim literal residual invariance.

## Controls

The controls show that the intervention is semantically meaningful even though
the primary closure claim fails.

At d8 full depth, applying the opposite-phase barycenter to the matched phase
task destroys native phase performance:

| Regime | Phase accuracy loss | CE increase | Posterior JS | Sufficient |
| --- | ---: | ---: | ---: | --- |
| training support | `0.5791` | `1.4038` | `0.28337` | no |
| outside range | `0.1270` | `1.3823` | `0.30833` | no |

The fixed half-list cosine semantic reassignment also fails strongly:

| Regime | CE increase | Posterior JS | Sufficient |
| --- | ---: | ---: | --- |
| training support | `7.7219` | `0.67496` | no |
| outside range | `3.3703` | `0.50740` | no |

Every fiber changes cosine by at least `0.9464` under that control, well above
the locked `0.50` minimum. These failures establish target specificity; they
cannot rescue the absence of a correct-barycenter front.

## d6 quarantine

All d6 source, replay, cohort, finite-value, and state-integrity checks pass,
and its exact replay error is also zero. Its outside-range cosine baseline is,
however, only `0.1328125`, below the registered `0.15` accuracy floor. The
complete d6 arm is therefore validity-only evidence.

For transparency, its descriptive pattern resembles d8: the correct
barycenter improves accuracy and cross-entropy but misses posterior
preservation. At full depth, posterior JS is `0.02682` in support and `0.20296`
outside range. Those candidate endpoints are reported but not promoted, and
they do not count as an architectural replication.

The campaign-level `valid: false` is caused by this preregistered d6 baseline
failure. It must not be read as invalidating the separately valid d8 causal
null.

## What the result changes

The old task atlas established a genuine task-relative observational contrast:
phase and cosine supervision produce different activation geometry, and the
cosine branch probe collapses early. This experiment supplies the missing
causal qualification:

```text
task-aligned geometry
    != complete-state quotient closure

useful fiber projection
    != frozen-computation preservation.
```

The result also sharpens the calibrated-front-end comparison. In the
structured calibrated models, exact task-fiber barycenters are sufficient from
before attention in `5/5` seeds per arm on both shifts. In the retained raw d8
model, the same class of full-state intervention never preserves the posterior.
Architectural quotient structure, not an observationally interval-like cloud,
is what produced the earlier causal closure.

## Exact query-only corollary

The atlas records only the final-query representation, whereas the registered
causal intervention averaged the complete residual sequence. That scope
difference initially suggests a cheaper query-only follow-up. The final cut
rules it out without another model run.

At the final post-MLP cut, the frozen continuation is exactly

```text
posterior = softmax(answer_head(final_layer_norm(state[:, -1, :])))
```

and never reads `state[:, :-1, :]`. A full-sequence barycenter and a query-only
barycenter have the same `state[:, -1, :]`, so their final posteriors are
identical for every input. Conversely, a context-only barycenter is exactly
inert at that cut.

The mature-front definition requires the selected cut and every later cut,
including final post-MLP, to pass. The valid d8 full barycenter fails there in
both regimes. Therefore:

```text
full final barycenter fails
    + full final posterior == query-only final posterior
    => no mature query-only barycenter front.
```

This is a deductive consequence of the frozen continuation and the already
registered endpoint, not a new experiment or an extra evidence count. An
executable regression constructs full, query-only, and context-only patches on
a TinyLLM and verifies both exact equalities.

## Scientific decision

Close the causal interpretation of the old raw task-geometry atlas. Do not run
a query-versus-context barycenter campaign: exact final-cut equivalence has
already falsified that rescue. Also do not retrain the four models, relax the
posterior ceiling, fit another probe, or add another topology scan in this
branch.

If the raw-model question is reopened, the shortest admissible diagnostic is a
new preregistered **within-query carrier/kernel** intervention: preserve the
context and complementary query residual, then patch only a task-effective
query subspace declared from the frozen continuation, with its orthogonal
complement as a matched control. That would test whether a small
quotient-sufficient component coexists with branch-bearing query state. It must
not reuse this whole-residual null as permission for post-hoc subspace
selection.

## Campaign integrity

| Item | Value |
| --- | --- |
| primary analysis date | 2026-08-10 |
| environment | Python `3.11.13`, PyTorch `2.5.1+cu121` |
| accelerator | NVIDIA GeForce RTX 2060 SUPER, CUDA device `1` |
| peak CUDA allocation | `459,044,864` bytes |
| primary wall time | `89.74` seconds |
| d6 / d8 analysis time | `28.58` / `57.58` seconds |
| exact replay maximum error | `0.0` |
| exact resume | second invocation left primary artifact bytes unchanged |
| trained/fitted parameters | `0` |

The primary campaign validates all four checkpoint hashes, both upstream result
hashes, the preregistration and frozen runner digests, exact pair construction,
native continuation replay, and unchanged model/system-state digests. An early
systems-only shakedown exposed batch-dependent CUDA replay rounding and was not
used scientifically; the corrected shakedown replayed exactly before the
primary campaign. Neither shakedown contributes evidence.

## Artifacts and reproduction

| Item | SHA-256 / value |
| --- | --- |
| campaign | `b5d6b613683326024eeb00944a3d0aba4dd7a251dd22f5fa7e8561b5e9d6aae4` |
| implementation | `ebdc1e9f376a0e6a274f34a172949e1d12f2e6f6674c71eafb24d9773145dc3c` |
| runner | `67e0854162cfd5ef4ff7dbc18f2cd2b7e9c6e88e10543e30bf0baf6b87b1c8ee` |
| result manifest | `0403600e910911415b76bdf61b139f9fa6ac403158fc8f0805015e7693008990` |
| d6 result / diagnostics | `8eeb07a703912dc1f090f8337097baaa84856984c2c447c394773192999f9e73` / `45ec9f8dba69ad852e5eb39bdcbd08e28fd79a013ef854384953ea11805cff45` |
| d8 result / diagnostics | `79119a4fa6df16a2fe2d65c34ec1aca30072a0b53d3d8ea312c92b29f86d1476` / `ef86c31d418c04b7ffd4bec3b66135057c4ca825b3e4da8263a920cbce0de1dd` |
| preregistration | `15eeea68dde87d101f587ea258c9d928b55b9d9a55a9d897ccd027446203926e` |
| meta-hypothesis record | `fcdd9dcd03185ecc679fe44cdf47566bb44a74ebb4675b3fc9a99a12a27e5f52` |
| DVC data root | `868c354fc681068d67d6df2ac4768d97.dir` |
| lakeFS commit | `23e16a9d1e9d248eb250e14f105a1778d9b28e239d90ff43bb0cdb1c1fbaec7b` |

- primary campaign:
  `data/experiments/tinyllm_task_relative_activation_barycenter/20260810_seed7_preregistered/`
- aggregate meta record:
  `data/meta_hypotheses/tinyllm-task-relative-activation-barycenter-v1.json`

```bash
MPLCONFIGDIR=/tmp/matplotlib-task-barycenter-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_task_relative_activation_barycenter \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_task_relative_activation_barycenter/20260810_seed7_preregistered
```

Store and verify the meta-hypothesis record with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-task-barycenter-meta \
pixi run python -m \
  experiments.neural_architecture_lab.store_task_relative_activation_barycenter_meta_hypothesis
```

## Boundaries

This study covers four retained seed-7 raw TinyLLMs, one exact two-sheet
phase-fiber construction, two depths, two tasks, two nuisance regimes, the
complete residual sequence, and native frozen continuations. The depths are
mechanistic replications, not independent training seeds. The d8 null does not
establish population prevalence, and the d6 arm is quarantined. Off-manifold
barycenter failure does not prove that no smaller causal quotient carrier
exists; it proves only that the registered complete-state barycenter is not a
sufficient frozen state in the valid d8 scope. The exact final-cut corollary
also excludes token-locus restriction alone as a mature-front rescue, but does
not test a within-query task-tangent or kernel decomposition.
