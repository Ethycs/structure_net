# TinyLLM observable C3 d6 campaign shakedown

**Status:** CAMPAIGN EXECUTION PATH VALIDATED — PRIMARY d6 AUTHORIZED

**Date:** 2026-08-11

**Evidence role:** `systems_lifecycle_only_not_quality_evidence`

**Hypothesis:** `tinyllm-c3-temporal-quotient-training-v1`

**Preregistration:** [d6 C3 temporal quotient](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-quotient-d6-preregistration.md)

## Verdict

The separate confirmatory runner passes its focused contracts and a real
two-step analytic d6 CUDA cell. It exercises the prospective training,
fresh-split probes, four-cut causal continuation, checkpoint reload, diagnostic
reload, strict JSON, scheduler, and artifact paths without executing a primary
seed.

The first shakedown correctly found and preserved one systems defect. After the
registered repair, identity replay is exact at all four cuts on both shifts.
The d6 analytic population may now run under the preregistered stop rule.

No task, representation, or specificity value in this report is scientific
evidence.

## Unit and contract coverage

The combined C3 suite passes `21/21` focused tests. It covers:

- exact observable token group laws and the no-training carrier preflight;
- paired training support and real held-out nuisance composition;
- learned invariance at initial, perturbed, and optimized parameter states;
- matched TinyLLM and structured-injection initialization;
- tensor-exact optimizer/checkpoint continuation;
- disjoint probe-fit and final-test latent fingerprints;
- target-bin derangement without fixed bins;
- semantic and conditional-deck endpoint schemas;
- structured orbit action at every residual cut;
- causal identity and barycenter continuation;
- every raw attention/MLP Reynolds-defect cut;
- numeric task, representation, causal, relative-utility, population, and
  positive-control stop gates; and
- the full fifteen-cell scientific fingerprint grid.

## First shakedown: preserved lifecycle failure

The initial two-step analytic d6 cell completed training and artifact reload,
and exact structured barycenter preservation passed. Identity replay nevertheless
reached `2.861e-6` on extrapolation, above the frozen `2e-6` ceiling.

Natural activation extraction used batches of 24 individual sheets, while
identity continuation flattened batches of 24 complete orbits into 72 sheets.
CUDA selected numerically different batch-shape kernels. This made the replay
gate measure kernel scheduling as well as graph identity.

The repair changes no scientific endpoint, threshold, model, example, or
optimizer. Causal extraction now treats its batch size as a latent-orbit count
and uses the same flattened three-sheet shape for natural and replay paths.
Because no primary outcome existed, the corrected implementation received a new
source hash and the shakedown was rerun from initialization.

## Corrected end-to-end cell

| Contract | Result |
| --- | ---: |
| preset / arm / seed | d6 / analytic / 7 |
| optimizer steps | `2` |
| primary optimizer steps | `0` |
| TinyLLM parameters | `29,950,080` |
| sequence-injection parameters | `1,152` |
| checkpoint reload | pass |
| diagnostic reload | pass |
| exact action | pass |
| causal barycenter, every cut and shift | pass |
| identity replay, every cut and shift | exactly `0.0` |
| result validity | pass |

Natural task, representation, and target-changing specificity gates fail after
two steps, as expected. The output classification
`c3_positive_control_task_failure` is the underpowered quality classifier and
has no evidentiary role. The shakedown establishes only that the exact registered
measurements execute and serialize correctly.

## Source and artifact identity

| Artifact | SHA-256 |
| --- | --- |
| d6 preregistration | `33537d32c4e8361fb325a1792a3779a2cb76a0248ef2945d727ac78dcb17d71a` |
| sealed Stage-0 runner | `dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6` |
| corrected analysis source | `89dacc60d02707678e689c6ce1e8f9c963889af352565a227bb90ed8e367e6a3` |
| campaign runner | `9b2cd0e3ce3752b7eea80d5859c11880a9d3732fb48b58306e34eab4f080d5ec` |
| shakedown campaign JSON | `278e41f43146c6e5936d847f5ec0ac8720851fa6cb7ad3bf7f50b0bc612f76c7` |
| shakedown result JSON | `b78159d0fff423e6d35aae55414f2cf11038e4625a605c81fea477c375fa5da7` |
| model checkpoint | `922e008a2b6109fdd22886e9f00125bfe5da472725a9913d94f9b6f6d27bd427` |
| front-end checkpoint | `48a19551b9ed28fd12821aed124f6928477c4975ab2bfa4afdfa4ae2b7b40428` |
| diagnostics | `1e0761e168940dd296b94e65c920bee4ee5645f44a7aa1085d9b6103dfc47354` |

The disposable shakedown root is
`/tmp/tinyllm-c3-temporal-quotient-campaign-shakedown-v2`. It is not part of the
DVC scientific evidence root.

## Decision

Execute the five analytic d6 seeds first. If fewer than four pass their complete
natural-task, representation, exact-action, causal, replay, and population
specificity gates, stop without training raw or learned d6 cells. If the
analytic population passes, execute the matched raw and learned populations.
D10 remains unauthorized in either case.

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl-c3-final \
pixi run pytest -q \
  tests/structure_net/test_tinyllm_c3_temporal_quotient_preflight.py \
  tests/structure_net/test_tinyllm_c3_temporal_quotient_training.py \
  tests/structure_net/test_tinyllm_c3_temporal_quotient_analysis.py \
  tests/structure_net/test_tinyllm_c3_temporal_quotient_campaign.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl-c3-campaign-v2 \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_quotient_campaign \
  --shakedown --gpus 0 --no-resume \
  --output /tmp/tinyllm-c3-temporal-quotient-campaign-shakedown-v2
```
