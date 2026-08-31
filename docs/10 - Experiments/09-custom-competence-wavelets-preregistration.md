# Experiment 09 — Custom Competence Wavelets Preregistration

**Status:** LOCKED BEFORE OUTCOMES  
**Shared construction:** [`08-09-strata-wavelets-preregistration.md`](08-09-strata-wavelets-preregistration.md)

Use 512 deterministic stratum-proportional landmarks, the 12-neighbor carrier
graph, Bernoulli Fisher competence penalty, seam floor, lazy diffusion
operator, dyadic spectral bands, and within-band orthogonal varimax rotation
specified in the shared construction. This defines the custom task-native
wavelets before any test outcome.

The canonical NAL outputs are `wavelet_basis.npz` and `wavelets.json`.
Connectedness and `<=1e-5` orthogonality error are hard gates; reconstruction,
boundary recall, and competence compressibility are reported without post-hoc
basis selection.
