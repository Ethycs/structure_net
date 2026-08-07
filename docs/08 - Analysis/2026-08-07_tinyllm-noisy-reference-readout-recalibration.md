# TinyLLM noisy-reference readout recalibration

**Status:** LEARNED ARM SUPPORTED; UNIVERSAL CLAIM REJECTED — `arm_stratified_readout_repair`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED` WITH MIXED EVIDENCE PEDIGREE, frozen-system readout intervention  
**Hypothesis:** `tinyllm-noisy-reference-readout-recalibration-v1`  
**Schema:** `nal.tinyllm-noisy-reference-readout-recalibration.v1`  
**Preregistration:** [noisy-reference readout recalibration preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-noisy-reference-readout-recalibration-preregistration.md)

## Verdict

Readout-only recalibration completely repairs the learned calibrated-
equivariant arm at the first registered orientation error. A noisy-fitted
`16 × 384` answer head passes noisy composition, noisy extrapolation, clean
composition, and clean extrapolation simultaneously in **5/5** checkpoints.
The low-capacity affine interval calibrator passes **4/5**. The untouched
learned head passed only **2/5** at the same `sigma=0.035` radians.

The analytic canonicalizer does not show a population-level repair. Its
linear head and affine scalar calibrator each pass **3/5**, below the registered
four-of-five requirement. The locked campaign classification is therefore:

```text
arm_stratified_readout_repair
```

This is strong evidence that the learned encoder's two-degree failure was at
the task interface rather than in quotient formation. It is not evidence that
every clean quotient can be recovered by a refitted linear head. In the
analytic arm, nonlinear probe decodability and linear task-interface
sufficiency remain distinct.

## Evidence pedigree

The learned-arm outcome is preregistered and was unseen before the
authoritative campaign. The analytic arm is corrective replication: an
initial command-window interruption wrote all five analytic records without a
campaign aggregate, and their gate booleans were inspected while diagnosing
the lifecycle. The clean persistent relaunch reproduced those analytic gates
exactly.

A subsequent `cuda:2` attempt failed before loading a checkpoint because host
physical GPU indices and PyTorch logical ordinals are remapped. The
authoritative run used PyTorch logical `cuda:1`, the free RTX 2060 SUPER, in a
persistent terminal. No scientific parameter, endpoint, or classification
changed during either lifecycle repair.

The campaign records the arm roles explicitly:

| Arm | Evidence role |
| --- | --- |
| analytic calibrated | `post_outcome_corrective_replication_evidence` |
| learned calibrated-equivariant | `preregistered_unseen_arm_evidence` |

## Primary gates

A seed passes only when a readout fitted on noisy interpolation examples stays
within three absolute accuracy points of that seed's untouched clean baseline
on both clean and noisy composition/extrapolation. The inherited two-degree
representation gate must also pass.

| Arm | Untouched noisy | Noisy-fitted linear | Noisy-fitted affine scalar | Required |
| --- | ---: | ---: | ---: | ---: |
| analytic calibrated | `0/5` | **`3/5`** | **`3/5`** | `4/5` |
| learned calibrated-equivariant | `2/5` | **`5/5`** | **`4/5`** | `4/5` |

Every clean-fitted linear positive control passes (`5/5` per arm), every
noisy-fitted linear head preserves clean utility (`5/5` per arm), and every
target-shuffled negative control remains below `0.20` accuracy (`5/5` per
arm). Thus the learned repair is neither a failed fitting protocol nor a
generic high-capacity classifier artifact.

## Accuracy effects

Values below are mean absolute accuracy loss from the untouched clean head;
negative values are improvements over that baseline.

| Arm | Readout | Noisy composition | Noisy extrapolation | Clean composition | Clean extrapolation |
| --- | --- | ---: | ---: | ---: | ---: |
| analytic | untouched frozen | `+3.77` points | `+3.42` | `0` | `0` |
| analytic | noisy-fitted linear | `+1.93` | `+1.78` | `-3.52` | `-2.56` |
| analytic | noisy-fitted scalar | `+1.88` | `+1.46` | descriptive | descriptive |
| learned | untouched frozen | `+4.39` | `+0.70` | `0` | `0` |
| learned | noisy-fitted linear | **`-2.48`** | **`-2.03`** | **`-9.20`** | **`-4.06`** |
| learned | noisy-fitted scalar | **`-1.31`** | **`-1.99`** | descriptive | descriptive |

The learned linear head does more than recover the lost points: on average it
exceeds the untouched clean head under both noisy shifts and retains that
advantage on clean inputs. Mean target cross-entropy also falls from
`1.4506` to `1.4222` on learned composition and from `1.6023` to `1.5926` on
learned extrapolation.

The analytic failures are checkpoint-specific composition misses. Linear
seeds `7` and `29` exceed the three-point ceiling; scalar seeds `7` and `17`
do. No post-outcome union of those readout families is permitted.

## What the controls reveal

The affine scalar fits are close to identity. Across all checkpoints their
slopes range from about `0.962` to `1.048`, and intercepts from about `-0.035`
to `+0.020`. Four learned checkpoints therefore need only a small movement of
ordered cosine-bin boundaries. This is the cleanest evidence for literal
calibration failure.

The full linear heads are not small perturbations: their Frobenius movement
from the frozen answer rows is `5.52--7.34` times the norm of those rows. They
match the existing answer-head parameter count and pass the shuffled control,
but should be described as **readout replacements**, not infinitesimal
calibration. The scalar result prevents the learned conclusion from resting
only on that large reweighting.

Expected-cosine correlations remain high after repair. The worst shifted cell
is `0.9762` for the learned linear head and `0.9761` for the learned scalar
calibrator; analytic minima exceed `0.9904`. The target-shuffled readouts reach
at most `0.0772` exact-bin accuracy.

## Mechanistic interpretation

The learned system now supports this causal decomposition:

```text
noisy gauge reference
  -> learned equivariant front end
  -> base-preserving, branch-contracted residual
  -> miscalibrated frozen answer interface
  -> refitted interval boundary/head restores task utility.
```

The intervention changes no residual state, so the recovered performance
cannot be attributed to a new quotient representation. It demonstrates that
the required task information was already present and usable by an
existing-capacity linear interface. Full TinyLLM or front-end retraining is not
justified for the learned arm at two degrees.

The analytic result places a useful limit on the claim. Its representation
passes the same nonlinear quotient probes, yet neither registered readout
family reaches four-of-five. A probe can establish that cosine is decodable
without establishing that one declared task interface can recover the clean
decision margins. The next analytic-arm test, if needed, should be one bounded
nonlinear readout ceiling; it should not trigger transformer retraining.

## Validity and campaign integrity

- all ten source orientation-result, model, front-end, system-state, dataset,
  and corruption hashes replay;
- frozen exact-bin accuracy and cross-entropy replay within `1.91e-7`;
- the two-degree representation gate is inherited as passing in every system;
- TinyLLM, front-end, scalar embedding, layer norm, and residual states remain
  byte/state unchanged;
- all twenty clean/noisy cross-condition matrices are finite;
- `30` linear heads and `20` affine scalar calibrators are saved;
- no model, front end, or representation probe is trained; and
- exact resume preserves campaign, result, and readout-weight bytes.

| Item | Value |
| --- | --- |
| frozen systems requested/completed | `10/10` |
| TinyLLM/front ends trained | `0/0` |
| new representation probes | `0` |
| fitted linear/scalar readouts | `30/20` |
| linear answer-head parameters | `6,144` (`16 × 384`, no bias) |
| source orientation sigma | `0.035` radians |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA allocation | `292,023,808` bytes |
| analysis time | `99.53` seconds |
| implementation SHA-256 | `05a6eda8d84535fdeda3f232069b64b079724fc1b5a5a549078c6cfd03b53e94` |
| campaign SHA-256 | `4f068487264e541fe65da955b7ef820047aab63455045b0631189dcf6da0331f` |
| combined result SHA-256 | `ecb23253d19b5d5560bc5e407483c754c2a70f40993e8b2dedbf3fe8b9300cd6` |
| combined readout SHA-256 | `4102afda8c807d30a486dd4c51481d559f9c24bc6154e250727c88d416e5f6a0` |
| exact-resume tree-manifest SHA-256 | `22fbced0e4657c0eed36fa72e44ba2a99f963ccfa272791d0c21868e4c78b364` |
| typed meta-evidence SHA-256 | `06d896f58b68de9e456f171873370cf5f906713558e314344e2a5de741e884fc` |

The superseding complete data tree is tracked by DVC as
`9faf6cff337d28f563fff273fa45edf4.dir` (`2,297` files,
`39,907,857,134` logical bytes). DVC reports the local cache and `lakefs`
remote in sync. lakeFS commit
`decdbb9de45f710cfc604b76488cb1a51d2e6dc01efc6c87a828885b71a2938b`
records that object set, and the branch has no uncommitted diff. The exact DVC
directory object is addressable at
`lakefs://artifacts/decdbb9de45f710cfc604b76488cb1a51d2e6dc01efc6c87a828885b71a2938b/structure-net/files/md5/9f/af6cff337d28f563fff273fa45edf4.dir`.

The typed meta-hypothesis and all ten checkpoint records passed ChromaDB
readback. The focused runner and meta-ledger gate completed with **12 passed**;
the 18 warnings were the repository's known legacy Chroma/NumPy telemetry
noise and did not prevent persistence or readback. The complete five-campaign
calibration/readout branch completed with **85 passed** and the same 18 known
warnings. A locked exact-resume invocation returned the existing aggregate and
preserved the tree-manifest hash byte for byte.
| meta-hypothesis SHA-256 | `06d896f58b68de9e456f171873370cf5f906713558e314344e2a5de741e884fc` |
| DVC data root | `9faf6cff337d28f563fff273fa45edf4.dir` (`2,297` files; `39,907,857,134` bytes) |
| lakeFS backup commit | `decdbb9de45f710cfc604b76488cb1a51d2e6dc01efc6c87a828885b71a2938b` |

## Artifacts and reproduction

- authoritative campaign:
  `data/experiments/tinyllm_noisy_reference_readout_recalibration/20260807_d8_preregistered_cuda1_persistent_mixed_pedigree/campaign_results.json`
- per-system metrics and saved readouts:
  `data/experiments/tinyllm_noisy_reference_readout_recalibration/20260807_d8_preregistered_cuda1_persistent_mixed_pedigree/runs/*/seed_*/`
- systems-only lifecycle:
  `data/experiments/tinyllm_noisy_reference_readout_recalibration/20260807_shakedown_cuda/`
- preserved partial analytic root:
  `data/experiments/tinyllm_noisy_reference_readout_recalibration/20260807_d8_preregistered/`
- failed pre-result logical-`cuda:2` root:
  `data/experiments/tinyllm_noisy_reference_readout_recalibration/20260807_d8_preregistered_cuda2_mixed_pedigree/`
- runner:
  `experiments/structure_net/tinyllm_noisy_reference_readout_recalibration.py`
- tests:
  `tests/structure_net/test_tinyllm_noisy_reference_readout_recalibration.py`
- typed meta hypothesis:
  `data/meta_hypotheses/tinyllm-noisy-reference-readout-recalibration-v1.json`
- meta builder and storage command:
  `src/neural_architecture_lab/noisy_reference_readout_recalibration_meta_hypothesis.py`
  and
  `experiments/neural_architecture_lab/store_noisy_reference_readout_recalibration_meta_hypothesis.py`
- strict meta-hypothesis aggregate:
  `data/meta_hypotheses/tinyllm-noisy-reference-readout-recalibration-v1.json`
- meta-hypothesis adapter and tests:
  `src/neural_architecture_lab/noisy_reference_readout_recalibration_meta_hypothesis.py`,
  `tests/neural_architecture_lab/test_noisy_reference_readout_recalibration_meta_hypothesis.py`

The persistent meta store read back the hypothesis and all ten checkpoint-level
experiment records. `dvc status` and `dvc status --cloud` are clean, the DVC
root object is present on lakeFS, and the lakeFS branch has no uncommitted diff.

```bash
MPLCONFIGDIR=/tmp/matplotlib-noisy-readout-primary-persistent \
pixi run python -m \
  experiments.structure_net.tinyllm_noisy_reference_readout_recalibration \
  --device cuda:1 \
  --partial-outcome-relaunch \
  --output \
  data/experiments/tinyllm_noisy_reference_readout_recalibration/20260807_d8_preregistered_cuda1_persistent_mixed_pedigree
```

## Scope boundary

This establishes held-out repair for one synthetic two-degree orientation-
error distribution and selected d8/N3 checkpoints. It does not locate the
sub-two-degree clean-head boundary, test a real calibration process, or show
that one readout transfers between checkpoints. The analytic outcome is
corrective replication; only the learned-arm result retains preregistered
unseen-outcome status.
