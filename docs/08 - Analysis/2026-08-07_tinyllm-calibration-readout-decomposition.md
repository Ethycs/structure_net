# TinyLLM calibration readout decomposition

**Status:** COMPLETED — VALID POST-OUTCOME CORRECTIVE REPLAY  
**Date:** 2026-08-07  
**Experiment:** `tinyllm-calibration-readout-decomposition-v1`  
**Classification:** `reference_coordinate_precision_limited`

## Verdict

The calibration-noise failure is not repaired by changing argmax to posterior
mean, by reading the structured front end directly, or by decoding cosine from
the frozen full-depth residual. It is also not repaired by writing that decoded
coordinate through the model's local task gradient.

The label-using positive control is different: writing the true cosine through
the same local task gradient passes all five seeds in both arms at
`sigma=0.20`. Flipped, shuffled, and gradient-orthogonal controls pass none.
The registered classification is therefore
`reference_coordinate_precision_limited`.

This is a useful mechanistic localization. The frozen task computation remains
locally capable of expressing the answer when supplied the right semantic
coordinate. The perturbed calibration reference does not determine that
coordinate precisely enough. Another representation loss or full TinyLLM
retraining is not the next justified move.

This result is a locked post-outcome corrective replay, not fresh confirmation.
The first complete root was quarantined after its producing runner changed.
The corrective root reproduced the classification under a stable digest and
passed exact-resume verification.

## Design

The campaign reused the same five retained `d8/N3` checkpoints in each of two
structured arms:

- `analytic_calibrated`;
- `learned_calibrated_equivariant`.

It replayed the predecessor's exact train, validation, composition, and
extrapolation cohorts and common calibration perturbations at
`sigma = 0.00, 0.05, 0.10, 0.20`. No TinyLLM model, front end, or task head was
trained or fine-tuned. One nonlinear full-depth cosine observer per retained
system was reconstructed on the predecessor's clean splits and then frozen.

Four fixed readouts were compared:

1. unchanged model argmax;
2. unchanged model posterior mean followed by nearest interval center;
3. structured front-end scalar followed by nearest interval center;
4. frozen full-depth cosine observer followed by nearest interval center.

The causal intervention used the unchanged posterior mean `m(h)` and its task
gradient `g = grad_h m(h)`:

```text
h' = h + ((u_hat - m(h)) / (||g||^2 + 1e-8)) g
```

It compared the clean observer coordinate with a true-cosine positive control,
a flipped coefficient, a shuffled coordinate, and an equal-norm direction in
the local gradient kernel. The primary gate required clean adequacy and at most
a three-point loss on both shifts in at least four of five seeds.

## Primary result

Pass counts at `sigma=0.20` are shown as analytic / learned:

| Diagnostic | Passing seeds |
| --- | ---: |
| Model argmax | `0/5` / `0/5` |
| Model posterior mean | `0/5` / `0/5` |
| Front-end coordinate | `0/5` / `0/5` |
| Frozen full-depth observer | `0/5` / `0/5` |
| Observer-coordinate gradient write | `0/5` / `0/5` |
| True-coordinate gradient write | `5/5` / `5/5` |
| Flipped control | `0/5` / `0/5` |
| Shuffled control | `0/5` / `0/5` |
| Gradient-kernel control | `0/5` / `0/5` |

The lower-level order is also informative:

| Arm | Readout | `0.00` | `0.05` | `0.10` | `0.20` |
| --- | --- | ---: | ---: | ---: | ---: |
| Analytic | argmax | 5 | 1 | 0 | 0 |
| Analytic | posterior mean | 5 | 2 | 0 | 0 |
| Analytic | observer | 5 | 2 | 0 | 0 |
| Learned | argmax | 5 | 4 | 0 | 0 |
| Learned | posterior mean | 5 | 4 | 0 | 0 |
| Learned | observer | 5 | 0 | 0 | 0 |
| Analytic | true-coordinate write | 5 | 5 | 5 | 5 |
| Learned | true-coordinate write | 5 | 5 | 5 | 5 |

Posterior shape is not the main defect: posterior-mean binning adds only small
accuracy changes and never reaches the primary gate. Nor does the diagnostic
observer expose a hidden robust coordinate that the model head simply ignores.

## Continuous coordinate versus exact bins

At `sigma=0.20`, the full-depth observer still has high mean correlation, but
its absolute error is large relative to the width of the 16 target intervals:

| Arm | Shift | Observer correlation | Observer RMSE | Observer exact accuracy |
| --- | --- | ---: | ---: | ---: |
| Analytic | composition | 0.9843 | 0.0958 | 0.5264 |
| Analytic | extrapolation | 0.9672 | 0.1407 | 0.4348 |
| Learned | composition | 0.9842 | 0.0958 | 0.5076 |
| Learned | extrapolation | 0.9675 | 0.1385 | 0.3977 |

This resolves the apparent conflict in the predecessor. Correlation correctly
reported that a quotient-like coordinate remained ordered. It did not imply
that the coordinate was calibrated precisely enough to select a narrow target
bin. Interpretability must keep rank/order, absolute coordinate precision, and
task use as separate claims.

## Causal localization

The observer-coordinate write makes the posterior mean agree with the observer
to first order, but does not restore exact-bin utility. By contrast, the true
coordinate write reaches mean exact-bin accuracies of:

| Arm | Composition | Extrapolation |
| --- | ---: | ---: |
| Analytic | 0.7955 | 0.7199 |
| Learned | 0.7836 | 0.6701 |

All ten true-coordinate cells pass the registered utility gate. All three
specificity controls fail in all ten systems. This supports a bounded causal
claim: the existing local task covector has enough capacity to express the
correct task when supplied a sufficiently precise semantic scalar. It does not
prove that the decoder is globally linear, and the true coordinate is not a
deployable input.

## Validity and reproducibility

- all ten retained systems passed provenance, source state, input/target
  identity, clean replay, and finite gates;
- maximum clean observer/task replay error was `1.06e-7` against `1e-5`;
- the corrected implementation stayed fixed throughout the full run;
- exact resume preserved all result, observer, and aggregate bytes;
- corrective campaign SHA-256:
  `833392ad7956ddcf715a20211431586f371edee280f167bf5a4fa51437cdc6c6`;
- implementation SHA-256:
  `0649a40c17384266f360daf33b68519e8f6d068aeec03893cfe9229f4f8d222d`;
- exact-resume tree-manifest SHA-256:
  `ec2a7338655b3cb3e02ab3808665d3b0947318abf06b0a17e0fbedab6798cc1c`;
- quarantined first-root campaign SHA-256:
  `1dd86dca5ef02bf2f3d5fd206ee972eaac8663e786f29ea8dd57a49a20038e39`.

The corrective run used Python `3.11.13`, PyTorch `2.5.1+cu121`, and an NVIDIA
GeForce RTX 2060 SUPER through `cuda:1`. It took `105.66` seconds and peaked at
`333,281,280` allocated CUDA bytes.

The superseding complete data tree is tracked by DVC as
`9faf6cff337d28f563fff273fa45edf4.dir` (`2,297` files,
`39,907,857,134` logical bytes). DVC reports the local cache and `lakefs`
remote in sync. lakeFS commit
`decdbb9de45f710cfc604b76488cb1a51d2e6dc01efc6c87a828885b71a2938b`
records that object set, and the branch has no uncommitted diff. The exact DVC
directory object is addressable at
`lakefs://artifacts/decdbb9de45f710cfc604b76488cb1a51d2e6dc01efc6c87a828885b71a2938b/structure-net/files/md5/9f/af6cff337d28f563fff273fa45edf4.dir`.

Command:

```bash
pixi run python -m experiments.structure_net.tinyllm_calibration_readout_decomposition \
  --device cuda:1 \
  --output data/experiments/tinyllm_calibration_readout_decomposition/20260807_d8_corrective_v2
```

Artifacts:

- promoted corrective campaign:
  `data/experiments/tinyllm_calibration_readout_decomposition/20260807_d8_corrective_v2/campaign_results.json`;
- quarantined lifecycle root:
  `data/experiments/tinyllm_calibration_readout_decomposition/20260807_invalid_prelock_runner/`;
- unpromoted same-protocol operational replay:
  `data/experiments/tinyllm_calibration_readout_decomposition/20260807_d8_existing_checkpoints/`;
- per-system results and observer states under the corrective root's `runs/`;
- meta hypothesis:
  `data/meta_hypotheses/tinyllm-calibration-readout-decomposition-v1.json`;
- ChromaDB readback verified one hypothesis and ten corrective experiment
  records.

The typed meta-evidence JSON has SHA-256
`c771018f3cabbe91d57bb47bed9dcf77a929c5580882ee95b76de0d53e580bf0`.

## Interpretation boundary

This is an outcome-directed mechanistic follow-up using already inspected
checkpoints, followed by a post-outcome corrective replay. It is not an
independent replication. The nonlinear observer is a diagnostic estimator,
not proof of information absence. The true-coordinate write uses labels and is
a positive-control ceiling only. The perturbations are synthetic stress
coordinates rather than a measured instrument-noise model.

Within those boundaries, the evidence changes the engineering direction:

```text
quotient geometry survives in rank/order
        -> absolute semantic-coordinate precision degrades
        -> exact-bin utility fails
        -> oracle coordinate restores the frozen local computation
```

## Next shortest experiment

Do not train TinyLLM again. Make the reference acquisition itself causal:

1. generate repeated independent calibration observations with a declared
   angular noise model;
2. combine them with the analytic circular mean and a matched learned denoiser;
3. keep all ten front ends, transformers, and task heads frozen;
4. predict recovery as a function of measured angular standard error;
5. retain the same true-coordinate write as the positive-control ceiling.

This directly tests the newly localized mechanism. If denoising approaches the
oracle trend, measurement precision explains the failure within this synthetic
system. If it does not, the remaining gap is reference-model bias or local
task-covector nonlinearity—not quotient formation.
