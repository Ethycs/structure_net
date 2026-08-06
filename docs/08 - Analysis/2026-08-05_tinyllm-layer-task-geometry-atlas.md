# TinyLLM Layer-by-Task Geometry Atlas

**Status:** ALL FROZEN PROXY CRITERIA PASSED; CLAIM NOT CONFIRMED  
**Date:** 2026-08-05  
**Applies to:** `experiments/structure_net/tinyllm_task_geometry_atlas.py`  
**Hypothesis:** `tinyllm-layer-task-carrier-and-quotient-atlas-v1`  
**Depends on:** `../03 - Architecture/task-geometry-atlas.md`

## Measured verdict

The atlas localized task-specific geometry to individual transformer sublayers in both retained model classes.

- The phase carrier first satisfied the joint decoder/cohomology criterion at **d6 block 2 post-attention** and **d8 block 1 post-attention**.
- Conditional phase-branch information in both cosine models fell to chance after **block 1's MLP**.
- The stricter internal cosine-quotient criterion, which also requires contracted fibers and matched interval geometry, appeared later: **d6 block 5 post-attention** and **d8 block 3 post-attention**.
- Every localized carrier or quotient survived through the final block, and the held-out branch criterion passed on the disjoint nuisance family.

This provides an operational layer-by-task atlas and separates three events: task decodability, loss of an equivalence-class distinction, and mature task-relative metric geometry. It does not compute a chain-level induced map, prove a homotopy retract, or construct a Reeb cosheaf. The claim remains unconfirmed because only seed-7 checkpoints were retained and the strongest mathematical objects are still proxies.

## Protocol

| Field | Value |
| --- | --- |
| Models | frozen d6/d8 phase and cosine seed-7 checkpoints |
| Stages | query embedding, then post-attention and post-MLP for every block |
| d6/d8 atlas cells | 13 / 17 stages per task |
| Reference spaces | phase chord metric on `S1`; absolute-distance cosine interval |
| Correspondence | the same 64 phase-grid examples in reference and residual space |
| Paired geometry | distance Spearman, scale-normalized stress, 8-neighbor recall |
| Local decoders | nonlinear phase, cosine, and conditional-branch probes |
| Fibers | exact equal-cosine phase pairs |
| Independent topology | longest persistent cocycle coordinate from residual Euclidean distance |
| Held-out family | third harmonic, temporal drift, Laplace noise |
| Training/evaluation | 4,096 probe train; 1,024 validation; 1,024 ID and disjoint test |
| Hardware | NVIDIA GeForce RTX 2060 SUPER |

The source model is never updated. The query embedding is constant at the designated prediction token and is retained only as a negative baseline.

## Localization summary

| Model | Phase carrier | Branch collapse in cosine model | Full cosine quotient |
| --- | --- | --- | --- |
| d6 | block 2 post-attention | block 1 post-MLP | block 5 post-attention |
| d8 | block 1 post-attention | block 1 post-MLP | block 3 post-attention |

“Branch collapse” means the strong held-out conditional probe falls below the frozen 60% ceiling. “Full quotient” additionally requires cosine Pearson at least 0.9, fiber ratio at most 0.5, and paired interval-distance Spearman at least 0.6.

## Phase carrier

| Model/stage | Phase probe | Paired distance Spearman | H1 | PH coordinate alignment | Degree | Carrier proxy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| d6 block 1 attention | 0.979 | 0.336 | 0.086 | 0.244 | +1 | no |
| d6 block 2 attention | **0.988** | **0.742** | **0.478** | **0.865** | -1 | **yes** |
| d6 final MLP | 0.994 | 0.909 | 0.700 | 0.983 | -1 | yes |
| d8 block 1 attention | **0.983** | **0.681** | **0.364** | **0.890** | -1 | **yes** |
| d8 final MLP | 0.992 | 0.929 | 0.842 | 0.932 | -1 | yes |

d6 illustrates why decodability alone is weaker than a carrier criterion. Phase is already easy to probe after block 1 attention, but the residual has weak H1 and its independent coordinate is not aligned. Block 2 attention is the first stage where decoder agreement, robust H1, recovered-coordinate alignment, and degree agree.

d8 forms the joint carrier one attention stage earlier. In both models, later stages improve paired geometry and persistence without changing the recovered generator's degree.

## Cosine quotient formation

| Model/stage | Branch | Cosine Pearson | Fiber ratio | Interval Spearman | Stress | Full quotient |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| d6 block 1 attention | 83.9% | 0.787 | 0.767 | 0.181 | 0.632 | no |
| d6 block 1 MLP | **55.4%** | 0.411 | 0.520 | 0.166 | 0.682 | no |
| d6 block 5 attention | 52.5% | **0.986** | **0.435** | **0.652** | 0.447 | **yes** |
| d6 final MLP | 50.0% | 0.987 | 0.386 | 0.822 | 0.361 | yes |
| d8 block 1 attention | 61.6% | 0.985 | 0.492 | 0.387 | 0.561 | no |
| d8 block 1 MLP | **52.2%** | 0.983 | **0.368** | 0.383 | 0.568 | no |
| d8 block 3 attention | 52.4% | **0.986** | **0.326** | **0.730** | 0.411 | **yes** |
| d8 final MLP | 53.3% | 0.985 | 0.316 | 0.709 | 0.405 | yes |

Both first-block MLPs remove probe-accessible branch information. In d8, cosine remains decodable and fibers contract immediately, but the residual distance structure does not yet resemble the interval. In d6, the same MLP temporarily weakens cosine decodability as it removes the branch. Later attention stages organize the surviving task coordinate into a progressively better interval metric.

This is the central developmental result:

```text
branch distinction removed by MLP
            ↓
cosine task coordinate recovered/refined
            ↓
paired interval geometry matures under later attention
```

It would have been invisible in a final-posterior or final-layer-only analysis.

## What the atlas supports

**Observed in both retained model classes:**

- phase task-carrier candidates localize to attention outputs;
- cosine branch collapse localizes to the first MLP;
- mature cosine quotient geometry localizes to later attention;
- paired geometry, local decoding, persistent coordinates, and fiber tests agree at the final layer;
- the task-specific branch asymmetry survives the disjoint nuisance family.

**Not established:**

- a chain map between reference and witness/Rips complexes;
- actual induced-map rank on homology;
- a homotopy retract or direct-summand theorem for these residuals;
- one-versus-two cosine-fiber components from a Reeb graph/cosheaf;
- repeatability across training seeds.

The implemented `target_h1_carrier_rank_proxy`, `approximate_retract_probe_score`, and `fiber_branch_component_proxy` are named and documented as operational proxies. Their values must not be cited as proofs of the stronger objects.

## Next mathematical increment

Use the existing same-example correspondence to construct a validated witness-complex simplicial map and measure the persistent image rank of the phase generator. Separately build a Reeb graph or cosheaf over decoded cosine and test whether typical fibers change from two connected components to one across block 1's MLP. These should be new analyzer contracts with synthetic correctness tests, not relabelings of the current proxies.

## Artifacts and reproduction

| Path | Contents |
| --- | --- |
| `data/experiments/tinyllm_task_geometry_atlas/20260805_d6_d8_seed7/results.json` | four frozen sublayer atlases and aggregate localization |
| `data/meta_hypotheses/tinyllm-layer-task-carrier-and-quotient-atlas-v1.json` | conservative atlas meta-hypothesis and four linked records |
| `data/chroma_db` | searchable hypothesis and experiment records |

The result and partial envelope are strict-JSON, byte-identical, and have SHA-256 `da6dad8c00738e5fc364de45b970245fa0667731aeedbcb462b85aa94bf1c3d6`.

```bash
env MPLCONFIGDIR=/tmp/structure-net-matplotlib \
  pixi run python experiments/structure_net/tinyllm_task_geometry_atlas.py \
    --device cuda:auto \
    --probe-steps 300 \
    --train-samples 4096 \
    --validation-samples 1024 \
    --test-samples 1024 \
    --output data/experiments/tinyllm_task_geometry_atlas/20260805_d6_d8_seed7

pixi run python \
  experiments/neural_architecture_lab/store_task_geometry_atlas_meta_hypothesis.py
```

The focused analyzer, atlas, probe, and ledger gate completed with **17 passed, 0 failed**. The full repository gate completed with **350 passed, 1 skipped, 0 failed** and 23 warnings in 386.19 seconds.
