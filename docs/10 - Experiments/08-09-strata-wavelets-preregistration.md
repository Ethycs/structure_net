# Experiments 08–09 — Operational Strata and Custom Wavelets Preregistration

**Status:** LOCKED BEFORE CONSTRUCTION OUTCOMES  
**Date:** 2026-08-16

## Operational Whitney construction

Fit the continuous carrier on development partitions 1–2 only: 16 PCA prompt
coordinates, five frozen lexical/word-order features, and logits of three local
competence probabilities estimated from the partition-1 atlas. Define strata
by constant human label, the recoverable generation family
`paws_wiki_unknown`, and exact A/B/C success signature. Empty strata remain
listed. Partition 3 audits but does not select the construction.

Audit the frontier incidence graph, 90%-variance tangent rank under bootstrap,
and the orthogonal residual of cross-stratum secants to the incident stratum
tangent. These are empirical Whitney-style diagnostics. A finite sample is
zero-dimensional and these results are explicitly not a proof of a Whitney
stratification of language or of an underlying tame set.

## Custom basis

Choose 512 partition-1/2 landmarks by deterministic proportional allocation
across observed strata. Construct a symmetric 12-neighbor task graph in the
carrier. Weight edges by carrier distance and a discrete Fisher penalty from
the three Bernoulli competence distributions; retain a seam floor of 0.02 so
known frontier incidence is not disconnected.

Form the lazy symmetric diffusion operator
`T = (I + D^-1/2 W D^-1/2) / 2`. Partition its eigenspaces by the first dyadic
diffusion scale at which `|λ|^(2^j) < 0.1`, rotate each detail band with an
orthogonal varimax localization, and retain the persistent space as scaling
functions. This fixes a task-native orthonormal diffusion-wavelet basis; no
generic Euclidean mother wavelet is selected post hoc.

Report orthogonality error, normalized-Laplacian spectrum, 128-coordinate
competence reconstruction error, coefficients needed for 95% competence
energy, boundary-signal reconstruction, connected components, and all basis,
graph, landmark, stratum, and source hashes. Failure of graph connectivity or
orthogonality error above `1e-5` blocks Experiment 10.
