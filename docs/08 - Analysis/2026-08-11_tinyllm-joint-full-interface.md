# TinyLLM Joint Full-Interface Physical Typing

**Status:** VALID PREREGISTERED NEGATIVE — FLEXIBLE FULL-INTERFACE TYPING INSUFFICIENT

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED CONDITIONAL EXTENSION`

**Hypothesis:** `tinyllm-joint-full-interface-physical-typing-v1`

**Preregistration:** [joint full-interface physical typing](../07%20-%20Status%20Reports/2026-08-11_tinyllm-joint-full-interface-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_joint_full_interface/20260811_d6_d10_preregistered/campaign_results.json`

## Verdict

Allowing the complete TinyLLM continuation to adapt does not make the learned
sensor adopt the declared physical cosine chart. The physical-true arm passes
`0/5` seeds in d6 and `0/5` in d10; pair-shuffled controls also pass `0/5` in
both families. The valid locked classification is:

```text
flexible_full_interface_physical_typing_insufficient
```

The intervention was complete and material. Every learned encoder, scalar
embedding, token/LM-head embedding, position embedding, attention, MLP,
layer-normalization, and final-scalar parameter was trainable. All twenty arms
changed their TinyLLM continuation, retained the exact parameter topology and
tied token/LM-head alias, and reloaded from a full checkpoint exactly.

The negative is therefore not evidence that the frozen continuation alone
prevented physical typing. Under the registered equal-weight, 600-update
objective, a flexible end-to-end path still coadapts around a private sensor
chart instead of selecting the declared sign and scale.

## Campaign integrity

| Check | Result |
| --- | ---: |
| source cells requested / completed / failed | `10 / 10 / 0` |
| matched full-interface fits requested / completed | `20 / 20` |
| valid cells | `10/10` |
| exact source model and Stage A zero-head replay | `20/20` arms |
| changed TinyLLM continuations | `20/20` arms |
| all declared parameters trainable | `20/20` arms |
| model topology and tied-weight alias unchanged | `20/20` arms |
| full checkpoint and diagnostics reload | `10/10` cells |
| physical-true passes | d6 `0/5`; d10 `0/5` |
| pair-shuffled passes | d6 `0/5`; d10 `0/5` |
| exact campaign resume | byte-stable |

The d6 fit exposed `29,967,858` trainable parameters per arm. The d10 fit
exposed `81,413,746`. Peak allocated CUDA memory was `0.627 GB` for d6 and
`1.586 GB` for d10. The ten cells completed in `12.68` elapsed minutes and
`25.41` aggregate cell-minutes. The primary artifact root contains 51 files
and `4.2 GB`, dominated by twenty full model/interface checkpoints.

## Primary endpoints

The table reports five-seed physical-true means. `Pass` is descriptive at one
cut and shift; the registered seed gate requires all four cut/shift endpoints
simultaneously.

| preset | cut | shift | corr | RMSE | slope | branch acc | log-loss gain | exact acc | Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d6 | front | composition | `.1991` | `.3280` | `.3940` | `.5068` | `.000103` | `.2693` | `0/5` |
| d6 | front | extrapolation | `.1917` | `.3506` | `.3892` | `.5072` | `.000199` | `.1850` | `0/5` |
| d6 | full | composition | `.9989` | `.0326` | `.9804` | `.5020` | `.000043` | `.8059` | `5/5` |
| d6 | full | extrapolation | `.9788` | `.1089` | `.9718` | `.5004` | `.000152` | `.4500` | `2/5` |
| d10 | front | composition | `-.9988` | `.6186` | `-.1480` | `.5021` | `.000003` | `.0859` | `0/5` |
| d10 | front | extrapolation | `-.9806` | `.6218` | `-.1469` | `.4998` | `.000016` | `.0736` | `0/5` |
| d10 | full | composition | `.9986` | `.0687` | `1.0128` | `.5053` | `.000124` | `.5549` | `2/5` |
| d10 | full | extrapolation | `.9803` | `.1245` | `1.0017` | `.5059` | `.000131` | `.4020` | `2/5` |

Branch contraction is not the failure. Every physical arm passes both branch
gates at both cuts and shifts. Full-depth cosine retention also passes in all
ten seeds. The failures are the front-end physical chart and, in many seeds,
the inherited exact-bin task floor.

### D6 retains seed-dependent gauge

Three d6 front ends are positively oriented with composition slopes from
`.669` to `.808`; two remain sign-reversed with slopes `-.122` and `-.161`.
The correlation gate passes the three positive seeds, but the fixed physical
interval task floor passes no front-end seed. Thus no d6 seed passes the front
endpoint even though the mean sensor MSE improves relative to Stage A.

### D10 retains one stable wrong gauge

All five d10 front ends remain sign-reversed. Their composition correlations
range from `-.998` to `-.999`, with slopes from `-.122` to `-.174`. Full-depth
outputs reverse that private chart back into the correct physical orientation:
all five have composition correlations at least `.998` and slopes near one.

This is direct evidence of coadaptation. The model can use the learned sensor,
but the end-to-end objective does not make the sensor itself physically typed.

## Comparison with the frozen-continuation interventions

| preset | measure | Stage A global clip | block clip | full interface |
| --- | --- | ---: | ---: | ---: |
| d6 | final logged sensor MSE | `.2027` | `.2146` | `.1738` |
| d10 | final logged sensor MSE | `.3560` | `.3767` | `.3909` |
| d6 | final logged final MSE | `.00129` | `.00108` | `.00154` |
| d10 | final logged final MSE | `.00135` | `.00100` | `.00281` |
| d6 | full composition accuracy | `.8088` | `.8191` | `.8059` |
| d10 | full composition accuracy | `.7324` | `.8102` | `.5549` |
| d6 | full extrapolation accuracy | `.4172` | `.4426` | `.4500` |
| d10 | full extrapolation accuracy | `.4088` | `.3941` | `.4020` |

Unfreezing does not produce a consistent task benefit. D6 remains close to
the frozen comparators. D10 composition accuracy falls substantially, while
extrapolation remains approximately unchanged. The sensor convention is also
essentially unchanged: d6 stays mixed and d10 stays sign-reversed.

The full continuation therefore adds capacity to compensate for the private
chart, not a force that selects the declared chart.

## Scientific accounting

### What this result rejects

- The frozen TinyLLM continuation is not the sole obstruction to learned
  physical typing under this protocol.
- Full end-to-end flexibility plus direct sensor MSE does not reliably select
  an absolute sign, scale, and interval convention.
- The failure cannot be rescued by high absolute correlation, full-depth
  success, or post-hoc affine calibration; those were excluded by the joint
  endpoint before training.
- Another loss-weight, clipping, warm-start, or seed sweep is not licensed.

### What remains supported

- The learned sensor carries a highly ordered nuisance-invariant coordinate.
- Branch contraction remains stable under composition and extrapolation.
- A flexible continuation can translate a private sensor gauge into a nearly
  physical full-depth scalar.
- Analytic fixed-chart controls remain the positive evidence that these model
  families can use the declared coordinate.

The distinction is now sharp:

```text
invariant coordinate available       supported
absolute physical chart selected     contradicted for this construction
task-calibrated continuation          seed- and architecture-dependent
```

## Next action

Close flexible joint supervision as a physical-interface construction method.
The successor architecture must type sign, scale, and chart by construction—for
example, an analytic orientation anchor with only an orientation-preserving
residual correction, or a monotone fixed-endpoint sensor head whose output
cannot be sign-reversed or rescaled arbitrarily.

Before another training campaign, use the saved checkpoints for one no-training
gauge audit: affinely canonicalize each learned front scalar on the sealed
training cohort and counter-transform the linear scalar embedding so the full
frozen function is algebraically unchanged. Measure the residual non-affinity
and recompute the registered endpoints descriptively. This cannot rescue the
preregistered negative. It cheaply distinguishes a pure coordinate gauge from
nonlinear sensor error and specifies how restrictive the fixed-chart successor
must be.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_joint_full_interface \
  --gpus auto --max-parallel 3 --slots-per-gpu 1 \
  --output data/experiments/tinyllm_joint_full_interface/20260811_d6_d10_preregistered
```

| Artifact | SHA-256 |
| --- | --- |
| campaign | `cf8f27e088f9022b78f36d285f2ddb49920bd6bc740d71e6efac7b04ab877cc1` |
| result manifest | `658b47f2311035139df5cf038e612f6d7c6c2d495fc1267094dce1617df78d2c` |
| diagnostics manifest | `4489e7dba9e6db55646268b023ab3d143b37bdb02c1f63db0525d89319638e9a` |
| full-interface manifest | `4dba8670d5622b65b1eeb1df54eff8d2b7a5137cda614211de4d62ec3a96fd22` |
| implementation | `e719cfd42c3e14be5a37700d55272dd0f6172698fd302b8b5f9be136c1ea11f9` |
| producing runner | `b4509a24436e5767afa96a42672f6ab7d0cc857230e62fc205be5214049bd65a` |
| preregistration | `1921afdeb0fca28c80ff9fb151c767dd6b8baa8aac39967b3ae614bef5df9329` |
| campaign fingerprint | `dbefa30cfd6ba1cda963ff771fdbf2fa364815265bd43a7b06bffab6854464e8` |

The valid systems-only d6 shakedown is preserved under a separate root and is
excluded from all population claims.

## Data and evidence backup

The complete repository data tree is tracked by DVC root
`cf0cdde08dc6bb7c6ad463545848c649.dir` (`53,849,913,271` logical bytes,
`3,944` files). DVC pushed 70 new objects and reports the cache and `lakefs`
remote in sync.

lakeFS commit
`af24f6b57ba10d83dd08d86c144c3177c522741cca1370d47761359b4388c996`
seals the object graph on `artifacts/main`, with parent
`e1653d5f4de8c341358666d68ff8c6071b9706322212af9728dc301248700836`.
The branch diff is empty after commit. Direct lakeFS object checks recover the
DVC root checksum `cf0cdde08dc6bb7c6ad463545848c649`, campaign MD5
`165d3b082765311836b6bcf6eb45c2cc`, and meta-record MD5
`84b51589b6cb12a6b836c14a9680b22e`.
