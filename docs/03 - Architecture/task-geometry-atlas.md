# Task Geometry Atlas

**Status:** EXPERIMENTAL AS-BUILT  
**Date:** 2026-08-05  
**Applies to:** `semantic_quotient_analyzer.py`, `tinyllm_task_geometry_atlas.py`

## Purpose

The task geometry atlas asks whether a designated neural representation carries a task-derived reference space and whether task-equivalent input fibers collapse there. It compares the same examples in task and representation space; it does not ask whether the entire hidden state is homeomorphic to the raw input manifold.

```text
examples x_i ──task map──> reference q_T(x_i)
     │                         │
     └──frozen model──> stage representation z_i^s
                               │
          paired metrics + probes + persistent coordinate
                               │
                     per-task, per-stage atlas cell
```

The current TinyLLM extractor records the final-query representation at the constant query embedding and after every attention and MLP residual update. It rejects feedback/refinement graphs because a manual feed-forward sublayer trace would not represent their recurrent execution.

## Atlas cell

Each task/stage cell contains:

- paired reference/representation distance Spearman correlation;
- optimal-scale normalized stress and local-neighborhood recall;
- nonlinear held-out phase, cosine, and conditional-branch decoders;
- exact cosine-fiber within/between distance ratio;
- Euclidean residual H1 and a circular coordinate derived from its longest persistent cocycle;
- in-distribution and disjoint-nuisance evaluations;
- operational carrier, quotient, and fiber-component proxies.

`paired_geometry_alignment` is reusable for any two distance matrices whose rows refer to the same examples. `persistent_cohomology_circle_coordinate` derives a finite-field cocycle coordinate without semantic labels; labels enter only afterward when an experiment evaluates alignment.

## Operational criteria

For the synthetic phase task, a stage is a phase-carrier candidate when:

- a held-out phase decoder has alignment at least 0.9;
- normalized residual H1 is at least 0.3;
- the independently recovered coordinate aligns at least 0.8;
- its ordered-grid winding degree has magnitude near one.

For the cosine task, a stage is an internal-quotient candidate when:

- held-out cosine Pearson correlation is at least 0.9;
- conditional branch accuracy is at most 0.6 on exactly matched cosine pairs;
- cosine-fiber distance ratio is at most 0.5;
- paired interval-distance Spearman correlation is at least 0.6.

These thresholds are experiment-level defaults, not universal standards.

## Mathematical boundary

The current implementation deliberately uses conservative names:

| Field | What it measures | What it does not prove |
| --- | --- | --- |
| `target_h1_carrier_rank_proxy` | robust degree-one coordinate aligned to the reference generator | chain-level induced-map rank |
| `approximate_retract_probe_score` | held-out task reconstruction from the stage | a homotopy retraction |
| `fiber_branch_component_proxy` | nonlinear branch decodability conditioned on exact cosine | a Reeb graph or Reeb cosheaf |

A validated witness-complex chain map, Reeb/cosheaf construction, or homotopy certificate should be added as a distinct analyzer with its own tests and schema. These stronger terms must not be backfilled onto the current proxy results.

## Extension path

New tasks provide:

1. a reference representation or distance matrix derived from soft targets, temporal relations, physical latents, interventions, or judgments;
2. a same-example stage extractor;
3. task-specific fibers and held-out nuisance families;
4. a declared criterion for carrier and quotient candidates.

Hard labels alone usually define only a finite discrete reference. Rich topology requires relational task information.

## Verification

```bash
pixi run pytest -q \
  tests/structure_net/test_semantic_quotient_analyzer.py \
  tests/structure_net/test_tinyllm_task_geometry_atlas.py
```

The measured TinyLLM localization is recorded in `../08 - Analysis/2026-08-05_tinyllm-layer-task-geometry-atlas.md`.
