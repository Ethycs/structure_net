# Fisher-Wavelet Competence Routing

**Status:** PROPOSED AND SYNTHETICALLY TESTABLE  
**Date:** 2026-08-16  
**Source:** `../000 - Ingest/Model_Routing_Based_on_Embeddings_6a7fa16e_2026-08-16T22-29-31-399Z.md`

## Purpose

Route each request to the cheapest model predicted to succeed. The router must
learn separate competence functions for models A, B, and C; parameter count or
distillation does not guarantee nested success sets.

For a prompt `x`, model costs `c_A < c_B < c_C`, and a required success
probability `tau`, the policy is

\[
r(x)=\arg\min_m c_m\quad\text{subject to}\quad \widehat p_m(x)\geq\tau.
\]

If the prompt is outside calibrated support, or no cheaper model qualifies, the
policy falls back to C.

## Representation

The construction deliberately separates task state from prompt form:

\[
x\mapsto (s(x), e_{\mathrm{fiber}}(x)).
\]

The finite state graph supplies topology. Teacher probability vectors supply
behavioral geometry. A cheap embedding supplies only within-state nuisance
coordinates. This avoids requiring a general-purpose embedding model to
discover the task topology.

For adjacent task states `i,j`, use categorical Fisher-Rao distance

\[
d_{FR}(p_i,p_j)=2\arccos\sum_k\sqrt{p_{ik}p_{jk}}
\]

and the topology-preserving weight

\[
W_{ij}=A_{ij}\left[\epsilon+(1-\epsilon)
\exp\left(-d_{FR}(p_i,p_j)^2/(2\sigma^2)\right)\right].
\]

The normalized graph Laplacian is eigendecomposed. Low-frequency eigenvectors,
scaled by a heat kernel, are the finite spectral diffusion-wavelet coordinates.
The present implementation uses this finite spectral realization rather than a
multilevel compressed diffusion-wavelet tree.

## Competence atlas

A held-out routing set records

\[
(x_i,s_{iA},s_{iB},s_{iC}),\qquad s_{im}\in\{0,1\}.
\]

Distance-weighted nearest neighbors estimate each model's success probability
independently. This supports non-nested signatures such as `101` and `110`.
The routing labels must be based on objective correctness, a verifier, or a
declared fidelity target; training-set performance is not a valid calibration
set.

## Safety boundary

The router measures empirical adequacy under the calibration distribution. It
does not certify correctness for an individual prompt. A support-radius gate
must escalate points far from calibration data. Production use additionally
requires calibrated lower confidence bounds, deployment-like labels, shift
tests, and measured inference costs.

## First experiment

The first executable study is intentionally synthetic. It asks whether exact
task topology plus Fisher-weighted wavelet coordinates can recover a
non-nested competence atlas more efficiently than nuisance-only coordinates.
It does not train or benchmark language models.

