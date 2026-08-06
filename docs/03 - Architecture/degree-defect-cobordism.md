# Degree–Defect Cobordism Analyzer

**Status:** EXPERIMENTAL AS-BUILT  
**Date:** 2026-08-05  
**Applies to:** `semantic_quotient_analyzer.py`, `tinyllm_degree_defect_cobordism.py`

## Purpose

This analyzer turns a before/after winding observation into a localized numerical training event. For a complex circular-posterior moment

\[
m(\phi,t)=\sum_k p_t(k\mid\phi)e^{i\theta_k},
\]

the normalized map `m / |m|` has an integer winding degree whenever the moment is nonzero. On a phase/path cylinder, indexed cells enclosing zeros of `m` carry signed defect charge. The analyzer checks the discrete boundary identity

\[
\deg m(\cdot,1)-\deg m(\cdot,0)=\sum_z \operatorname{index}(z).
\]

```text
saved optimizer endpoints
          │
          ├── straight-line path in weights
          │
phase × path posterior moment grid
          │
          ├── endpoint winding degrees
          └── signed boundary winding per grid cell
                         │
                  charge-identity record
```

## Analyzer contract

`complex_defect_charge` accepts a complex field with shape `(path, phase)`, an ordered periodic phase grid, and an ordered path grid. It returns:

- start and end winding degrees;
- their integer difference;
- every cell with nonzero boundary winding;
- the orientation-corrected integer charge of each cell;
- total charge and whether it equals the endpoint degree change;
- the minimum sampled field magnitude.

The phase edge is periodic; the path edge is not. For the stored array orientation, the reported defect index is the negative of the counterclockwise cell-boundary winding. Synthetic tests cover a degree-zero-to-one homotopy with one `+1` defect and a constant degree-one path with no defects.

## TinyLLM continuous-input boundary

The trained TinyLLM sensor tokenizer rounds each scalar to a token bin. As a function of phase, that map is discontinuous, so it cannot directly define the smooth training cobordism in the mathematical proposal.

The experiment therefore declares an adjacent-embedding lift. A sensor value is mapped piecewise linearly between the two neighboring token embeddings; every token-bin center maps exactly to its original hard-token embedding. The lift is continuous but not unique and is not claimed to be identical to hard-token inference between bin centers.

Training itself is unchanged. The model initialization, data, minibatches, optimizer, clipping, and 600 updates are deterministically replayed, and the final state digest must match the retained source campaign. The continuous lift is used only to evaluate topology.

## Resolution protocol

The training trace begins with 128 evenly spaced phase samples. If any adjacent posterior-moment angle changes by at least `π/2`, sampling doubles until resolved or until the configured refinement limit is reached. Only rounded-degree changes nominate optimizer intervals.

For each nominated interval, the two exact parameter states are replayed and joined by a straight-line weight path. A higher-resolution phase/path field is evaluated and passed to `complex_defect_charge`. This isolates a grid cell containing net charge; it does not solve for an exact root.

## Claim boundary

| Result | Supported interpretation | Not established |
| --- | --- | --- |
| Endpoint degree changes by `+1` | circular posterior map changes homotopy class on the declared lift | topology of the discontinuous hard tokenizer |
| Indexed cell charge sums to `+1` | numerical degree/defect identity on the sampled cylinder | interval-certified root existence or uniqueness |
| Replayed state hash matches | the analyzed optimization trajectory reproduces the source run | robustness to other seeds or optimizers |
| One fixed nuisance slice passes | a direct slice-level training event was localized | a full defect sheet over nuisance space |

This is a finite-grid numerical cobordism certificate. It is not a polynomial surrogate, Whitney stratification, transversality proof, Pontryagin–Thom certificate, or certified root-isolation result. Those upgrades require separate analyzer contracts.

## Verification

```bash
pixi run pytest -q \
  tests/structure_net/test_semantic_quotient_analyzer.py \
  tests/structure_net/test_tinyllm_degree_defect_cobordism.py
```

The measured d6/d8 result is recorded in `../08 - Analysis/2026-08-05_tinyllm-degree-defect-cobordism.md`.
