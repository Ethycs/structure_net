# TinyLLM Degree–Defect Training Cobordism

**Status:** NUMERICAL CHARGE IDENTITY SUPPORTED; FORMAL CLAIM NOT CONFIRMED  
**Date:** 2026-08-05  
**Applies to:** `experiments/structure_net/tinyllm_degree_defect_cobordism.py`  
**Hypothesis:** `tinyllm-degree-change-equals-indexed-defect-charge-v1`  
**Depends on:** `../03 - Architecture/degree-defect-cobordism.md`

## Measured verdict
Awinding degree zero and ended with degree `+1`. Deterministic replay localized the only degree-changing optimizer interval at step **15 for d6** and step **17 for d8**. On a 1024-phase × 66-path straight-line interpolation between each pair of parameter states, each cylinder contained one indexed cell with charge `+1`:

| Model | Degree-changing step | Endpoint Δdegree | Defect cells | Total charge | Identity |
| --- | ---: | ---: | ---: | ---: | --- |
| d6 | 15 | +1 | 1 | +1 | pass |
| d8 | 17 | +1 | 1 | +1 | pass |

The final-state SHA-256 digest of each 600-step replay exactly matched its source task-quotient campaign. Every adaptive winding trace resolved under the `π/2` adjacent-angle criterion. This is direct numerical evidence for the degree-change/indexed-defect-charge identity on the declared continuous input lift.

It is not a formal polynomial cobordism certificate. The finite grid does not isolate an exact root, prove transversality, establish a Whitney stratification, or make the trained hard tokenizer continuous. Accordingly, the meta-hypothesis remains `confirmed: false` and records the narrower status `numerical_charge_identity_supported_not_formally_certified`.

## Protocol

| Field | Value |
| --- | --- |
| Models | d6 (29,956,224 parameters), d8 (50,964,992 parameters) |
| Task | 16-bin circular phase posterior |
| Seed | 7 |
| Training | 600 AdamW steps, batch 64, learning rate `3e-4`, weight decay `0.01` |
| Reproduction | same initialization, dataset, conditioned targets, minibatches, and clipping as source campaign |
| Nuisance slice | amplitude 1, orientation 0, offsets 0, harmonic 0.25, angular speed 0.35, direction +1, no noise |
| Continuous input | piecewise-linear interpolation between adjacent token embeddings |
| Trace | 128 phase samples, adaptively doubled through at most 16,384 |
| Transition cylinder | 1024 phase samples × 66 straight-line weight-path samples |
| Hardware | NVIDIA GeForce RTX 2060 SUPER |

The adjacent-embedding lift is exact at hard-token bin centers. Between centers it is an explicit continuous extension, not the original quantizer.

## Localized events

| Model | Approximate phase φ | Weight-path fraction | Minimum sampled field magnitude | Charge |
| --- | ---: | ---: | ---: | ---: |
| d6 step 14→15 | 4.323 | 0.208 | 0.00141 | +1 |
| d8 step 16→17 | 4.114 | 0.731 | 0.00727 | +1 |

The values identify charged grid cells, not exact posterior-moment roots. The d6 cell spans phase `[4.3197, 4.3258]` and path fraction `[0.2000, 0.2154]`; the d8 cell spans phase `[4.1111, 4.1172]` and fraction `[0.7231, 0.7385]`.

At the full training endpoints, the continuous-lift moments were comfortably nonzero and aligned:

| Model | Initial degree | Initial min `|m|` | Final degree | Final min `|m|` | Final phase alignment |
| --- | ---: | ---: | ---: | ---: | ---: |
| d6 | 0 | 0.0863 | +1 | 0.9041 | 0.9986 |
| d8 | 0 | 0.0536 | +1 | 0.8889 | 0.9969 |

## Resolution finding

d8 temporarily formed a very sharp, near-zero degree-one map after the main transition. A fixed 128-point mesh under-resolved steps 17–26. Adaptive refinement resolved step 18 at 2048 samples and the hardest state, step 20, at 16,384 samples:

- step 20 minimum sampled moment magnitude: `0.002151`;
- maximum adjacent angular increment at 16,384 points: `0.9076` radians;
- resolved degree: `+1`.

The degree did not change during this excursion. The observation is still scientifically useful: later same-degree intervals could contain cancelling defect pairs, which this campaign did not enumerate because it only interpolated intervals whose endpoint degree differed.

## What is supported

**Observed in both retained model classes:**

- the continuous-lift circular map changed from degree zero to degree `+1`;
- the change localized to a single optimizer interval;
- the indexed charge on that interval equaled the endpoint degree change;
- the charged set was nonempty, as the degree argument predicts;
- exact final-state hashes tied the analysis to the original training trajectory.

**Not established:**

- a continuous cobordism for the hard token IDs;
- exact root coordinates, regularity, or uniqueness inside a charged cell;
- a polynomial or Chebyshev surrogate with bounded value/derivative error;
- Whitney strata, Pontryagin–Thom framing, or interval-certified local indices;
- the full defect curve or sheet over nuisance space;
- repeatability across seeds, optimizers, or nuisance slices;
- absence of cancelling `+1/-1` defects on same-degree optimizer intervals.

## Next mathematical increment

Re-evaluate the two localized cylinders with an interval-bounded low-dimensional surrogate for `Re(m), Im(m)`. Isolate every zero, certify a nonsingular Jacobian, and compare the exact local-index sum with the grid charge. Then sweep a small nuisance coordinate to continue each point into a defect curve. This would upgrade localization without pretending to polynomialize the full transformer.

## Artifacts and reproduction

| Path | Contents |
| --- | --- |
| `data/experiments/tinyllm_degree_defect_cobordism/20260805_d6_d8_seed7/results.json` | complete traces, localized cylinders, criteria, and provenance |
| `data/meta_hypotheses/tinyllm-degree-change-equals-indexed-defect-charge-v1.json` | conservative aggregate and two linked experiment records |
| `data/chroma_db` | searchable hypothesis and experiment records |

```bash
pixi run python experiments/structure_net/tinyllm_degree_defect_cobordism.py \
  --device cuda:auto \
  --presets d6,d8 \
  --seed 7 \
  --steps 600 \
  --trace-phase-points 128 \
  --trace-refinement-limit 128 \
  --interpolation-phase-points 1024 \
  --interpolation-path-points 66 \
  --output data/experiments/tinyllm_degree_defect_cobordism/20260805_d6_d8_seed7

pixi run python \
  experiments/neural_architecture_lab/store_degree_defect_cobordism_meta_hypothesis.py
```

The strict-JSON result and partial envelopes are byte-identical with SHA-256 `04f974f56b40a4b8bd76b009bb8d779379c70f79d86fd4ba1942720b596b22c6`. The meta-hypothesis and both direct experiment records were read back from ChromaDB. The focused analyzer, runner, and ledger gate completed with **16 passed, 0 failed**. The full repository gate completed with **358 passed, 1 skipped, 0 failed** and 23 warnings in 391.52 seconds.
