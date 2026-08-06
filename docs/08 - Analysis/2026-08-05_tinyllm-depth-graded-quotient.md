# TinyLLM Depth-Graded Task Quotients

**Status:** NUMERICAL DEPTH FAMILY SUPPORTED; SINGLE-SEED CLAIM NOT CONFIRMED  
**Date:** 2026-08-05  
**Applies to:** `experiments/structure_net/tinyllm_depth_graded_quotient.py`  
**Hypothesis:** `tinyllm-depth-graded-task-fronts-v1`  
**Depends on:** `../03 - Architecture/depth-graded-transformer.md`

## Measured verdict

A single d8 TinyLLM can be trained and evaluated as a compatible real-depth residual-gate family with one shared head. Both joint-depth arms substantially improve shallow task performance, and both move the cosine internal-quotient proxy from depth **1.85** to no later than **0.005** of block 1.

Continuous gating does not dominate discrete multi-exit training. Discrete supervision produces the earliest phase front; continuous supervision preserves the best full-depth phase accuracy.

| Task / arm | Refined front | Depth-1 accuracy | Full-depth accuracy |
| --- | ---: | ---: | ---: |
| Phase, ordinary | 0.045 | 36.33% | 71.29% |
| Phase, discrete multi-exit | ≤0.005 | **66.80%** | 68.75% |
| Phase, continuous gate | 0.020 | 65.23% | **71.88%** |
| Cosine, ordinary | 1.85 | 47.07% | 51.76% |
| Cosine, discrete multi-exit | ≤0.005 | **60.35%** | 60.16% |
| Cosine, continuous gate | ≤0.005 | 59.77% | 60.16% |

The front is a threshold crossing on the continuous-input slice, while accuracy is measured on held-out hard-token examples. A near-immediate degree-one phase map therefore does not imply high exact-bin classification accuracy.

All trained phase/depth cylinders obeyed the numerical identity

\[
\deg c_{8,\tau}-\deg c_{0,\tau}=\sum_z \operatorname{index}(z)=+1.
\]

The result is recorded as `numerical_depth_family_supported: true` and `confirmed: false`. It is one gated d8 seed with operational branch/geometry proxies—not a neural-ODE limit, Reeb cosheaf, Whitney stratification, or multi-seed result.

## Protocol

| Field | Value |
| --- | --- |
| Model | d8, 8 blocks, 50,964,992 parameters |
| Seed | 7 |
| Tasks | phase circle and cosine interval |
| Arms | ordinary final-depth, discrete multi-exit, continuous gate |
| Shared decoder | tied TinyLLM LM head at every depth |
| Training | 600 AdamW updates, batch 64, learning rate `3e-4`, weight decay `0.01` |
| Checkpoints | 0, 25, 50, 100, 200, 400, 600 |
| Main depth grid | `0,0.25,…,8` |
| Final refinement | `Δs=0.05` globally; `Δs=0.005` on the first tenth of block 1 |
| Input topology | fixed-nuisance continuous adjacent-token-embedding lift |
| Final H1 | posterior Fisher–Rao persistence at every integer depth |
| Hardware | NVIDIA GeForce RTX 2060 SUPER |

The two ordinary arms exactly reproduced the retained task-quotient source state hashes. Every one of the six final checkpoints and six compressed depth/training field archives was retained.

## Training-front trajectories

At the main `0.25` grid:

| Arm / task | step 0 | 25 | 50 | 100 | 200 | 400 | 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary phase | – | – | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| discrete phase | – | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| continuous phase | – | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| ordinary cosine | – | – | 0.25 | 1.75 | 2.00 | 2.50 | 2.00 |
| discrete cosine | – | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| continuous cosine | – | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |

Joint-depth supervision establishes both fronts by step 25 and keeps them shallow. Ordinary phase reaches the coarse front by step 50. Ordinary cosine is nonmonotone: a provisional shallow proxy passes at step 50, then the stricter mature quotient front moves deeper and ends at 1.85 under final refinement. Threshold-front movement should therefore not be interpreted as a material point being transported monotonically through layers.

## Final topology by integer depth

The output posterior already distinguishes the three arms at depth 1:

| Task / arm | Depth-1 map metric | Depth-1 H1 | Full-depth map metric | Full-depth H1 |
| --- | ---: | ---: | ---: | ---: |
| Phase, ordinary | alignment 0.9700 | 0.6165 | alignment 0.9972 | 0.9366 |
| Phase, discrete | alignment 0.9964 | 0.9114 | alignment 0.9952 | 0.9524 |
| Phase, continuous | alignment 0.9959 | **0.9390** | alignment 0.9970 | 0.9246 |
| Cosine, ordinary | Pearson 0.9321 | 0.4351 | Pearson 0.9974 | 0.0044 |
| Cosine, discrete | Pearson 0.9990 | 0.0240 | Pearson 0.9985 | 0.0077 |
| Cosine, continuous | Pearson 0.9987 | 0.0239 | Pearson 0.9986 | 0.0092 |

Joint-depth training produces a mature interval-like posterior by depth 1, whereas the ordinary cosine model still has substantial posterior H1 there. This agrees with the refined quotient-front comparison.

## Depth defects

For every trained checkpoint and every phase arm:

- depth zero had winding degree zero;
- full depth had winding degree `+1`;
- total indexed depth-defect charge was `+1`;
- the charge identity held on a common resolved phase grid.

The discrete and continuous arms each had one charged grid cell at every trained checkpoint. The ordinary arm had net `+1` but transient cancelling pairs at steps 25, 50, and 400, with cell counts 3, 5, and 3. This suggests that joint-depth supervision simplifies the sampled defect decomposition, but cell counts are grid-dependent and are descriptive rather than certified root counts.

The initialized model was much closer to zero. Five under-resolved depth rows were adaptively checked:

| Depth | Resolved degree | Samples | Minimum sampled `|m|` |
| ---: | ---: | ---: | ---: |
| 3.75 | 0 | 2,048 | 0.001918 |
| 4.00 | +1 | 8,192 | 0.000539 |
| 4.50 | 0 | 2,048 | 0.001236 |
| 6.00 | +2 | 8,192 | 0.000120 |
| 6.25 | 0 | 8,192 | 0.000216 |

No degree value changed under refinement. The original initialization charge-cell decomposition used a shared 1024-point grid and remains explicitly exploratory; it is excluded from the trained charge claim.

## What is supported

**Observed directly:**

- real-depth evaluation is compatible with exact integer transformer prefixes;
- ordinary retraining reproduces the source experiments exactly;
- joint-depth supervision dramatically improves shallow accuracy in both tasks;
- both joint-depth arms move mature cosine geometry into partial block 1;
- phase topology appears extremely early in all arms, before strong hard-token accuracy;
- trained phase depth changes have the predicted net indexed defect charge;
- discrete and continuous supervision realize different shallow/full-depth tradeoffs.

**Not established:**

- conditional branch mutual information from a held-out probe at every real depth;
- a Reeb graph/cosheaf component merger;
- independent persistent-cohomology coordinates at every depth/checkpoint;
- a continuously interpolated optimization-time axis or actual defect curves in `(φ,s,τ)`;
- linked task-defect curves;
- convergence under depth-grid refinement or a neural-ODE continuum;
- robustness across seeds, widths, maximum depths, or nuisance slices.

## Next experiment

Freeze these six checkpoints and train strong held-out conditional branch probes on a targeted depth grid around the refined cosine fronts. Then replay selected optimization intervals with weight interpolation, producing a resolved `(phase, depth, training-path)` moment field. That would test whether the isolated depth defects continue into curves and whether the proxy quotient front coincides with actual loss of branch information.

The neural-ODE arm should remain a separate null/control experiment with matched `N`/`2N` refinement losses. Its primary question is whether an invertible finite-depth flow retains branch information until an explicit projection.

## Artifacts and reproduction

| Path | Contents |
| --- | --- |
| `data/experiments/tinyllm_depth_graded_quotient/20260805_d8_seed7/results.json` | six runs, checkpoint diagrams, criteria, and resolution audit |
| `…/initial_resolution.json` | adaptive checks for the five near-zero initialized rows |
| `…/front_refinement.json` | final global `Δs=0.05` sweep |
| `…/front_refinement_early.json` | final first-block `Δs=0.005` sweep |
| `…/checkpoints/` | all six final model checkpoints |
| `…/fields/` | all six compressed depth/training posterior fields |
| `data/meta_hypotheses/tinyllm-depth-graded-task-fronts-v1.json` | conservative aggregate and six linked experiment records |

```bash
pixi run python experiments/structure_net/tinyllm_depth_graded_quotient.py \
  --device cuda:auto \
  --preset d8 \
  --seed 7 \
  --training-arms standard_final,discrete_multi_exit,continuous_gate \
  --quotients phase_circle,cosine_interval \
  --steps 600 \
  --checkpoints 0,25,50,100,200,400,600 \
  --depth-step 0.25 \
  --output data/experiments/tinyllm_depth_graded_quotient/20260805_d8_seed7

pixi run python \
  experiments/neural_architecture_lab/store_depth_graded_quotient_meta_hypothesis.py
```

The strict-JSON campaign result and partial envelopes are byte-identical with SHA-256 `5094e9ce26baee4908c2f1880b5add1c26e6f972408a908d94abd3055bcb7056`. The initialization, global-front, and early-front supplements have hashes `dcbbea533a97aa1274fdb90e1a52b4157a0b28748471c455aae7c32859bf0d77`, `8a5e6d319cd1441b4c93979bb56c5e7198cefdf9c3327bf6b5789d41afcbc2fc`, and `b6eb23f531286585d6ab3d45144408a18359abb75ce739213da00a110119731f`. The meta-hypothesis and all six direct experiment records were read back from ChromaDB. The focused model, runner, and ledger gate completed with **34 passed, 0 failed**. The full repository gate completed with **368 passed, 1 skipped, 0 failed** and 23 warnings in 387.68 seconds.
