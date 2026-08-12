# TinyLLM posterior-coordinate transport rank ladder

**Status:** COMPACT-CHART PREDICTION REJECTED — FULL ANSWER-LOGIT CHART TRANSPORTS `5/5`; `high_rank_answer_chart`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, outcome-directed frozen-checkpoint mechanistic diagnostic with one-checkpoint shakedown outcome exposure  
**Hypothesis:** `tinyllm-posterior-coordinate-transport-v1`  
**Schema:** `nal.tinyllm-posterior-coordinate-transport.v1`  
**Preregistration:** [posterior-coordinate transport preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-posterior-coordinate-transport-preregistration.md)

## Verdict

The scalar path-moment rollout failed because it discards answer-relevant
posterior shape, **not** because answer-output coordinates cannot integrate the
trained residual path. Transporting the complete centered answer-logit
coordinate schedule along the frozen model's own reference path passes the
registered task gate in **5/5 analytic and 5/5 learned** checkpoints, at both
`K=4` and `K=16`. Every fiber-block-shuffled schedule passes `0/5` at every
rank in both arms.

The strong compact-chart prediction nevertheless fails. No rank at or below
four reaches the four-of-five population gate in either arm, and rank eight
falls one checkpoint short in the analytic arm. The locked classification is:

```text
high_rank_answer_chart
```

Posterior shape is a sufficient transport target, but it is not compact. The
`answer_coordinates_nonintegrable` terminator is rejected: the residual-writer
branch does not end in nonintegrability. It ends in a dimensionality
statement — the frozen continuation can be steered along its own path by its
answer coordinates only when essentially the whole fifteen-dimensional centered
logit chart is supplied. This also explains, in one measurement, why every
preceding scalar and low-rank writer attempt failed: those interfaces were
structurally undersized, not merely misfit.

No model, front end, answer head, probe, observer, or transport parameter was
trained or fit.

## Primary gates

One checkpoint passes when its exact-bin accuracy loss from its unchanged
exact-reference clean baseline is at most `0.03` on both composition and
extrapolation. A rank passes when the `K=16` actual-coordinate rollout reaches
at least `4/5` in each arm while its shuffled control stays at or below `1/5`.

| Gate | Analytic | Learned equivariant | Required | Result |
| --- | ---: | ---: | ---: | --- |
| actual `m=64` reference | **5/5** | **5/5** | 5/5 | pass |
| exact endpoint residual | **5/5** | **5/5** | 5/5 | pass |
| rank 1, `K=16` | 0/5 | 0/5 | >=4/5 | fail |
| rank 2, `K=16` | 0/5 | 0/5 | >=4/5 | fail |
| rank 4, `K=16` | 3/5 | 1/5 | >=4/5 | fail |
| rank 8, `K=16` | 3/5 | 4/5 | >=4/5 | fail |
| rank full (15), `K=16` | **5/5** | **5/5** | >=4/5 | **pass** |
| shuffled schedule, all ranks, `K=16` | 0/5 | 0/5 | <=1/5 | pass |
| inherited scalar comparator, `K=16` | 0/5 | 0/5 | comparator | replayed |

The dose response is graded and monotone at the population level. `K=4`
counts are `0/0`, `0/0`, `3/2`, `3/3`, and `5/5` across the ladder, so the
full-rank chart does not even need the fine sixteen-step schedule.

Per-checkpoint ladder at `K=16` (`P` = pass, ranks `1/2/4/8/full`):

| Seed | Analytic | Learned equivariant |
| ---: | --- | --- |
| 7 | `. . P P P` | `. . P P P` |
| 17 | `. . . . P` | `. . . P P` |
| 29 | `. . P P P` | `. . . P P` |
| 41 | `. . P P P` | `. . . P P` |
| 53 | `. . . . P` | `. . . P P` |

Seeds 17 and 53 are the hard checkpoints: analytic seeds 17 and 53 pass only
at full rank, and learned seed 53 likewise. No lower-rank miss is hidden by a
higher-rank pass; the ladder is nested by construction.

## Mechanism

Aggregates below are means over five checkpoints and both shifts at `K=16`
(actual-coordinate schedule). Residual error is normalized by the actual
`m=1` to `m=64` endpoint chord; JS is the divergence between the rollout's
final answer posterior and the actual endpoint posterior.

| Arm | Rank | Mean accuracy loss | Max loss | Residual error | Posterior JS | Max condition | Min effective rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| analytic | 1 | 0.2427 | 0.3203 | 0.804 | 0.0537 | 1.0 | 1 |
| analytic | 2 | 0.0985 | 0.1357 | 0.559 | 0.0153 | 4.5 | 2 |
| analytic | 4 | 0.0215 | 0.0479 | 0.327 | 0.0008 | 5.7 | 4 |
| analytic | 8 | 0.0212 | 0.0381 | 0.316 | 0.0004 | 5.9 | 8 |
| analytic | full | **0.0155** | **0.0273** | 0.310 | **0.0000** | 6.0 | 15 |
| learned | 1 | 0.2034 | 0.3223 | 0.845 | 0.0541 | 1.0 | 1 |
| learned | 2 | 0.0905 | 0.1562 | 0.637 | 0.0181 | 5.6 | 2 |
| learned | 4 | 0.0243 | 0.0557 | 0.426 | 0.0014 | 6.7 | 4 |
| learned | 8 | 0.0217 | 0.0527 | 0.409 | 0.0009 | 7.0 | 8 |
| learned | full | **0.0125** | **0.0273** | 0.400 | **0.0000** | 8.1 | 15 |

Three observations carry the mechanistic content:

1. **Task recovery does not require reaching the trained residual state.** The
   full-rank rollout stops `31--40%` of the chord away from the actual
   endpoint residual, far off the trained path, yet reproduces the endpoint
   answer posterior to JS near machine precision and passes every task gate.
   The scalar comparator, by contrast, ends `85--87%` away with JS
   `0.042--0.046` and passes nothing. The residual directions the full chart
   omits are answer-null at the final query.
2. **The compact-rank failures are chart insufficiency, not numerics.** Every
   coordinate Jacobian retained full row rank at every step of every rollout
   (minimum effective rank equals the nominal rank), and the largest condition
   number across all `1,200` rollout cells is `8.1`. The pseudoinverse never
   truncated a direction.
3. **Large vector-valued displacements succeed where the large scalar step
   failed.** Maximum step norms reach `47` residual units — the same scale as
   the predecessor's failed one-step writes (`41--65`) — but distributed over a
   posterior-shape schedule they transport the task instead of destroying it.

## Evidence pedigree

The preregistration was locked before any posterior-transport outcome existed.
One systems-only CUDA shakedown (seed 7, both arms, separate root) then
exposed seed-7 outcomes before the population campaign launched. No
hypothesis, metric, threshold, control, schedule, or implementation constant
changed after that exposure; the campaign is therefore sequential confirmation
on seeds 17, 29, 41, and 53 and outcome-exposed on seed 7. The shakedown root
is retained and never pooled.

The design is additionally outcome-directed at the campaign level: it was
selected after, and consumes, the valid corrective reference-path transport
result. A positive result here identifies a minimal vector chart; it is not an
independent confirmation of the wider quotient program.

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed | 10 / 10 / 0 checkpoints |
| examples | 1,024 composition + 1,024 extrapolation per checkpoint |
| frozen systems | two d8/N3 arms × seeds 7, 17, 29, 41, 53 |
| TinyLLM parameters | 50,965,504 per checkpoint |
| coordinate ladder | DCT ranks 1, 2, 4, 8, full(15) over 16 ordered answer bins |
| transport grid | 17-point stored path; nested `K=4,16`; `rcond=1e-6` |
| trained/fitted objects | 0 |
| maximum source-metric replay error | **0.0** (all ten checkpoints) |
| DCT orthonormality / constant leak | `2.02e-15` / `9.71e-16` against `1e-10` |
| device | NVIDIA GeForce RTX 2060 SUPER, `cuda:1` |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| peak allocated CUDA memory | 343,060,992 bytes |
| analysis time | 4,851.7 seconds |
| implementation SHA-256 | `a91cae656663ebb6ab68b9fdacf82806bf417c5bffb5cc7b70cab32ebfaf5fe2` |
| campaign SHA-256 | `0733c3069d2c012f760a112307cfcb1c743c8ac19c64f288d19db6f97a08c57d` |
| result-manifest SHA-256 | `317cc0840c8808dc82632f19acb45c6a255853d3c13d1a2ef225b356b2ce33ef` |
| meta-hypothesis JSON SHA-256 | `d408ac5b41c6e2b92d8e831820de4121557c55f493ef552f7329de6f37b894dc` |
| final DVC data root | `b76d24966f6e969b1998cd825edeb15c.dir` (`2,712` files; `40,062,661,789` bytes) |
| lakeFS commit | `1f7ec52afae01257084c6cf85106d01126f61d061b0cf11415ffeb861b4313a0` |
| DVC / lakeFS | post-ledger snapshot pushed; local cache and `lakefs` remote in sync |

Every source campaign, result, array, checkpoint, and system-state hash of the
reference-path transport predecessor validated before intervention, and the
predecessor's own upstream acquisition chain revalidated through its locked
loader. Clean metrics, all seventeen path-step metrics, the exact endpoint
replay, and the recomputed `K=4/16` scalar path-moment comparator replayed the
stored source values with maximum error exactly `0.0`. All ten system-state
digests were unchanged after intervention. An exact resume returned
`campaign already complete` and preserved the campaign SHA byte for byte. The
focused runner and meta-hypothesis suites completed with **17 passed**; the
ChromaDB readback verified the hypothesis and all ten experiment records (the
18 legacy telemetry warnings are the repository's known Chroma/NumPy noise).
The campaign and meta-ledger blobs were independently verified at the lakeFS
commit above, and the branch has no uncommitted diff. The immutable DVC root is
`lakefs://artifacts/1f7ec52afae01257084c6cf85106d01126f61d061b0cf11415ffeb861b4313a0/structure-net/files/md5/b7/6d24966f6e969b1998cd825edeb15c.dir`.

## Mechanistic decision

Per the locked decision table, the outcome is recorded as: posterior shape is
sufficient but not compact. The residual-writer branch closes with the full
centered answer-logit chart as an **explanatory mechanism**, not a deployable
interface — the schedule it transports is read from the frozen model along the
stored reference path and is unavailable without that path.

The preregistration designates this the last no-fit output-coordinate writer
test in the current branch, and its outcome licenses no model retraining,
representation penalty, observer fitting, topology scan, or link-cobordism
analysis. Direct projection to the observed residual curve — the fallback
reserved for the nonintegrable outcome — is not activated, because
nonintegrability was rejected.

The quantitative boundary is itself the finding for the program: any future
deployable transport interface for these checkpoints must carry a
high-dimensional posterior-shape signal (between nine and fifteen centered
logit coordinates on this evidence), which is why every rank-one, rank-two,
rank-three, and scalar sidecar attempt in the preceding campaign sequence was
structurally incapable of portability.

## Artifacts and reproduction

- primary aggregate:
  `data/experiments/tinyllm_posterior_coordinate_transport/20260807_d8_preregistered/campaign_results.json`
- per-checkpoint records:
  `data/experiments/tinyllm_posterior_coordinate_transport/20260807_d8_preregistered/runs/*/seed_*/result.json`
- per-sample diagnostics:
  `data/experiments/tinyllm_posterior_coordinate_transport/20260807_d8_preregistered/runs/*/seed_*/coordinate_transport_diagnostics.npz`
- systems-only shakedown (outcome-exposed, never pooled):
  `data/experiments/tinyllm_posterior_coordinate_transport/20260807_shakedown_cuda/`
- source campaign:
  `data/experiments/tinyllm_reference_path_residual_transport/20260807_d8_corrected_v4/campaign_results.json`
- runner and tests:
  `experiments/structure_net/tinyllm_posterior_coordinate_transport.py`,
  `tests/structure_net/test_tinyllm_posterior_coordinate_transport.py`
- meta-hypothesis record, builder, and storage command:
  `data/meta_hypotheses/tinyllm-posterior-coordinate-transport-v1.json`,
  `src/neural_architecture_lab/posterior_coordinate_transport_meta_hypothesis.py`,
  `experiments/neural_architecture_lab/store_posterior_coordinate_transport_meta_hypothesis.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-posterior-coordinate-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_posterior_coordinate_transport \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_posterior_coordinate_transport/20260807_d8_preregistered
```

## Scope boundary

The coordinate schedules read frozen model outputs along one stored synthetic
reference path and are mechanistic oracles, not deployable sensors. The DCT
basis is example-free, but the chart remains conditioned on the frozen answer
decoder and the final query position; earlier block geometry is not measured.
The five checkpoints are retained replication units, not an architecture
population, and the result does not establish natural-language behavior or the
minimal sufficient rank between eight and fifteen.
