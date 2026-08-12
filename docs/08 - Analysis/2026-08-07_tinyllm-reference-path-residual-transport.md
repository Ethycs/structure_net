# TinyLLM reference-path residual transport

**Status:** VALID CORRECTIVE RESULT — FINITE-RADIUS HYPOTHESIS REJECTED; SEMANTIC SCHEDULE ONLY  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED` with pre-primary outcome exposure and post-primary producer correction  
**Hypothesis:** `tinyllm-reference-path-residual-transport-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-07_tinyllm-reference-path-residual-transport-preregistration.md`

## Verdict

Repeated local relinearization does repair the earlier one-step true-cosine
write, but it does **not** repair transport along the frozen model's own
reference-path moment schedule. At the preregistered `K=16` endpoint:

- the label-using true-cosine rollout passes `5/5` analytic and `4/5` learned
  checkpoints;
- the model-derived path-moment rollout passes `0/5` in both arms; and
- its fiber-block-shuffled control also passes `0/5` in both arms.

The registered finite-radius hypothesis required both structured schedules to
pass at least four of five checkpoints per arm. It therefore fails. The locked
classification is `semantic_schedule_only`, not finite-radius repair.

The result separates two claims that the earlier one-step intervention could
not distinguish:

> Local task-gradient integration can follow an externally supplied semantic
> coordinate, but the frozen model's one-dimensional ordered-moment path is
> not a sufficient coordinate for transporting its computation.

No model, front end, answer head, probe, observer, or transport parameter was
trained or fit.

## Primary gates

One checkpoint passes when its exact-bin accuracy loss from its unchanged
clean exact-reference baseline is at most `0.03` on both composition and
extrapolation.

| Gate | Analytic | Learned equivariant | Required | Result |
| --- | ---: | ---: | ---: | --- |
| actual `m=64` reference | **5/5** | **5/5** | 5/5 | pass |
| exact endpoint residual | **5/5** | **5/5** | 5/5 | pass |
| `K=16` true-cosine rollout | **5/5** | **4/5** | >=4/5 | pass |
| `K=16` path-moment rollout | 0/5 | 0/5 | >=4/5 | **fail** |
| `K=16` shuffled path moment | 0/5 | 0/5 | <=1/5 | pass |
| complete finite-radius hypothesis | — | — | every row | **rejected** |

The dose response is not simple numerical convergence:

| Steps | True cosine, analytic | True cosine, learned | Path moment, analytic | Path moment, learned | Shuffled, either arm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0/5 | 2/5 | 0/5 | 0/5 | 0/5 |
| 2 | 5/5 | 3/5 | 0/5 | 0/5 | 0/5 |
| 4 | 5/5 | 5/5 | 0/5 | 1/5 | 0/5 |
| 8 | 5/5 | 4/5 | 0/5 | 1/5 | 0/5 |
| 16 | 5/5 | 4/5 | 0/5 | 0/5 | 0/5 |

Every analytic checkpoint first passes the true-cosine gate at `K=2`. Learned
seeds 7 and 41 first pass at `K=1`, seed 29 at `K=2`, and seeds 17 and 53 at
`K=4`. Only learned seed 7 at `K=4` and seed 41 at `K=8` ever pass the
path-moment gate, and neither remains passing at `K=16`.

## What the actual path says

The stored shortest circular reference path was evaluated at 17 nested points
for every example. Its final-query residual curve is close to a chord, and its
local scalar first-order moment prediction is accurate:

| Arm / shift | Arc/chord | Max relative chord deviation | Parallel increment | Orthogonal increment | Relative first-order moment error | Local posterior JS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| analytic / composition | 1.0116 | 0.0638 | 0.6215 | 0.7764 | 0.00743 | 0.000132 |
| analytic / extrapolation | 1.0127 | 0.0686 | 0.6192 | 0.7778 | 0.00733 | 0.000138 |
| learned / composition | 1.0297 | 0.0846 | 0.5827 | 0.8041 | 0.01045 | 0.000156 |
| learned / extrapolation | 1.0417 | 0.0953 | 0.5813 | 0.8051 | 0.01059 | 0.000164 |

Thus failure is not explained by a wildly curved observed path or inaccurate
local scalar differentials. Approximately `78--81%` of each actual increment
nevertheless lies outside the one-dimensional task-gradient direction. The
minimum-norm scalar rollout omits those directions at every step, so small
locally valid updates can accumulate a large off-manifold displacement.

## Terminal-coordinate separation

The `K=16` aggregates make the distinction sharper. Values below are means
over five checkpoints; residual error is normalized by the actual `m=1` to
`m=64` endpoint chord.

| Arm / shift / schedule | Endpoint-moment error | True-cosine error | Residual error | Posterior JS | Mean accuracy loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| analytic / composition / true cosine | 0.0360 | **0.0031** | 1.730 | 0.0565 | -0.0164 |
| analytic / composition / path moment | **0.0029** | 0.0361 | **0.851** | **0.0404** | 0.0568 |
| analytic / extrapolation / true cosine | 0.0575 | **0.0048** | 2.847 | 0.0869 | -0.1283 |
| analytic / extrapolation / path moment | **0.0037** | 0.0576 | **0.855** | **0.0431** | 0.0504 |
| learned / composition / true cosine | 0.0414 | **0.0038** | 2.555 | 0.0655 | -0.0111 |
| learned / composition / path moment | **0.0035** | 0.0413 | **0.872** | **0.0451** | 0.0529 |
| learned / extrapolation / true cosine | 0.0721 | **0.0061** | 4.093 | 0.1095 | -0.2262 |
| learned / extrapolation / path moment | **0.0037** | 0.0718 | **0.875** | **0.0468** | 0.0117 |

The path-moment rollout does what it was asked to do: it reaches the frozen
path's ordered posterior moment to within `0.0029--0.0037` and remains much
closer to the actual endpoint residual and posterior than the true-cosine
rollout. It still misses the task because the ordered moment itself remains
`0.036--0.072` away from the semantic cosine and does not specify posterior
shape. Conversely, the true-cosine rollout succeeds despite ending much
farther from the actual residual manifold because it is explicitly supplied
the missing semantic coordinate.

This falsifies the simple explanation that the earlier failure was only one
step exceeding a local radius. The decisive limitation is the scalar chart:
the frozen path's ordered moment is neither a complete answer-state coordinate
nor a deployable estimate of the semantic target.

## Evidence pedigree and correction lineage

This is valid **corrective, outcome-exposed** evidence, not a fresh
confirmation. The initial one-seed systems shakedown exposed seed 7 `K=1` and
`K=16` outcomes before the population campaign. It also showed that a finite
small task gradient is a scientific observation, not a systems-validity
failure; schema `v1.1` removed the misplaced magnitude gate without changing
any intervention or task endpoint.

The completed `v1.1` primary root is preserved as invalid. Its only failed
contract was one analytic seed-17 extrapolation replay scalar differing from
the source by `3.814697e-6`, above the locked `2e-6` tolerance. All scientific
outputs, positive controls, shuffled controls, and state hashes otherwise
passed. Three auditable producer checks followed under fresh roots:

1. exact `q_1/q_64` endpoint overwrite (`v1.2`) did not remove the mismatch;
2. matching the source norm operator (`v1.3`) did not remove it; and
3. assigning the stored terminal `target_cosine` tensor directly (`v1.4`)
   restored exact source replay.

The corrected ten-checkpoint campaign has maximum source-metric replay error
`0.0`, all systems and scientific controls valid, and the same substantive
causal outcome as the invalid predecessor. Failed validation roots remain
available and are not pooled as evidence.

## Mechanistic decision

Stop the scalar-gradient writer branch. The result does not license another
representation loss, observer fit, residual penalty, or retraining run.

One final [no-fit posterior-coordinate rank ladder](../07%20-%20Status%20Reports/2026-08-07_tinyllm-posterior-coordinate-transport-preregistration.md)
is independently justified if a more precise failure localization is needed:
transport the complete centered answer-logit
or answer-simplex coordinate along the already stored reference path. This
tests whether the missing directions are answer-relevant posterior-shape
directions or lie in a continuation-relevant residual nullspace. It must use
the same checkpoints, paths, nested step counts, task gate, and shuffled
fiber-block control. If the multi-coordinate schedule fails despite accurate
local replay, terminate residual transport and retain the actual reference
path only as an explanatory oracle.

Link cobordism is not activated by this outcome. No canonical codimension-two
singular locus was measured; accumulated off-manifold drift is not by itself
a topological defect. Link-cobordism work remains restricted to the separately
localized degree/branch-locus program.

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed | 10 / 10 / 0 checkpoints |
| examples | 1,024 composition + 1,024 extrapolation per checkpoint |
| frozen systems | two d8/N3 arms × seeds 7, 17, 29, 41, 53 |
| TinyLLM parameters | 50,965,504 per checkpoint |
| fine reference path | 17 points; nested `K=1,2,4,8,16` |
| trained/fitted objects | 0 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| peak allocated CUDA memory | 0.306 GiB |
| analysis time | 235.56 seconds |
| implementation SHA-256 | `5cb1944e57db0515dbc7ad5e3956a328be733cdda7f5dda34631d21e8cf3a81c` |
| result-manifest SHA-256 | `5d765445082346512092dfa0acb00e3e39db369bd71829cbe4d96c8871593b4b` |
| campaign SHA-256 | `6b232f523cd570f10ebfcc07c47abae6a724b568d4cfc2852e4b91ccff01321f` |
| meta-hypothesis JSON SHA-256 | `3b2bcc1f11f8143aeabde44af6d32f0cfcec58af686956a438ae92f80c0778c5` |
| DVC data root | `b76d24966f6e969b1998cd825edeb15c.dir` (`2,712` files; `40,062,661,789` bytes) |
| lakeFS commit | `1f7ec52afae01257084c6cf85106d01126f61d061b0cf11415ffeb861b4313a0` |

An exact resume returned `campaign already complete` and preserved the campaign
SHA byte-for-byte. The focused transport, source-acquisition, and source-ledger
verification completed with **28 passed**. The hypothesis and ten direct
experiment records were read back successfully from the persistent ledger.

The configured DVC remote reports `Everything is up to date`. The immutable
DVC directory object, campaign blob, and meta-hypothesis blob were verified at
the lakeFS commit above; the branch has no uncommitted diff. The directory
object is
`lakefs://artifacts/1f7ec52afae01257084c6cf85106d01126f61d061b0cf11415ffeb861b4313a0/structure-net/files/md5/b7/6d24966f6e969b1998cd825edeb15c.dir`.

## Artifacts and reproduction

- primary aggregate:
  `data/experiments/tinyllm_reference_path_residual_transport/20260807_d8_corrected_v4/campaign_results.json`
- per-checkpoint records:
  `data/experiments/tinyllm_reference_path_residual_transport/20260807_d8_corrected_v4/runs/*/seed_*/result.json`
- per-sample diagnostics:
  `data/experiments/tinyllm_reference_path_residual_transport/20260807_d8_corrected_v4/runs/*/seed_*/transport_diagnostics.npz`
- source campaign:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_d8_preregistered/campaign_results.json`

```bash
MPLCONFIGDIR=/tmp/matplotlib-reference-path-transport \
pixi run python -m \
  experiments.structure_net.tinyllm_reference_path_residual_transport \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_reference_path_residual_transport/20260807_d8_corrected_v4
```

The path comes from one stored unbiased synthetic Gaussian acquisition array.
The true-cosine and path-moment schedules are mechanistic oracles. The result
concerns final-query residual transport in this synthetic d8 population; it
does not establish natural-language behavior or architecture-wide prevalence.
