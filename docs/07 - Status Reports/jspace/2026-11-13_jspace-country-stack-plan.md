# JSPACE-COUNTRY-STACK-GROEBNER-01 — stacked country readouts and compression

Conformance: EXPLORATORY, one frozen Qwen2.5-3B checkpoint, confirmed=false.
New root: `data/experiments/jspace_country_stack/20261113_v1`. Parent:
`jspace_attention_transport/20261113_v1`. Parent evaluation outcomes are already
known. This is a derived study on reused worlds, with a new dense-swap readout
capture; no fresh-independent-accuracy or parent-gate rescue claim.

## Question

Can the four country readouts at block outputs 27–35 be stacked and represented
by a smaller polynomial codec, with an exact Gröbner certificate for the fitted
surrogate and measured fidelity to continuous neural scores? Also construct and
validate an exact finite country-label quotient. These are separate objects.

Layers and anchor 27 are fixed from the parent's observed readability. Use the
parent's frozen linear four-country probes on its two-pooled-position eight-row
J banks. Scores are continuous ridge class scores, not calibrated probabilities.
Their affine sum-to-one identity must hold numerically to 1e-8. No equations such
as score^2=score are imposed on continuous neural coordinates.

## Continuous stack

Training uses the parent's 384 worlds (768 paired observations); development uses
24 worlds (48 observations). The parent's country/attribute query holdouts and
split-disjoint name pools remain unchanged. Fit all later score vectors jointly
from the anchor's first three country scores and known binary requested attribute.
The fourth score is one minus their sum. Compare degree-one and degree-two ridge
families with the parent penalty grid 1e-4,.001,.01,.1,1,10. Feature columns are
scaled by training standard deviation, floor 1e-8. Development MSE chooses each
penalty; development score MSE standardized by training output std chooses the
family, breaking ties toward lower degree. Anchor carry is an additional baseline.

This is a joint anchor-to-layer decoder, not composition of transformer blocks.
All fits freeze before the new dense trace and final reuse analysis. Quantize
coefficients to rational multiples of 2^-24, require development prediction RMS
change <=1e-5, and use the same rationalized coefficients for algebra and metrics.

For each of nine layers and four country scores, form a graph polynomial
`s_layer,country - f_layer,country(x_China,x_France,x_Germany,u)` plus `u^2-u`.
Compute lexicographic Gröbner basis over QQ, output variables before roots. Eliminate
the score variables and save their polynomial normal forms as decoders. Verify
every generator has zero remainder and normal forms agree exactly with the numeric
coefficient codec. The retained ring has three continuous roots plus a binary
attribute: it is not a finite-dimensional country quotient. Boolean reduction
merges u^2 with u; sum-to-one removes the fourth stored score polynomial.

Report both symbolic canonicalization and approximate neural reconstruction;
an exact surrogate certificate does not make noisy neural scores exact. Bound
each symbolic family to 30 seconds; timeout is unresolved, never a negative proof.

## Finite semantic quotient

Declare country codes China=(0,0), France=(0,1), Germany=(1,0), Russia=(1,1).
Construct `G={a^2-a,b^2-b}` over QQ, standard monomials `{1,a,b,ab}`. Indicators
are `(1-a)(1-b),(1-a)b,a(1-b),ab`. Stack these four readouts at all nine layers,
eliminate all 36 indicator variables, and certify exact decoding on the four states.
Calculate multiplication matrices and all six country-swap maps; exhaustively check
the 24 input/swap pairs. This is a constructed semantic encoding, not equations
learned for continuous activations or a proof of neural causal swaps.

Apply it to actual probe argmax trajectories. A repeated country takes two data
bits; use one per-world framing bit, and encode nonconstant trajectories verbatim
with nine two-bit labels. A ten-byte format/shape header is counted on both raw
and compressed baselines. Require exact roundtrip, including exceptions. Compare
against raw two-bit labels and zlib of that same baseline. Ordinary repeated-label
coding can achieve the same gain; categorical encoding is not a Gröbner-specific
compression discovery. Never compare categorical bits to continuous float bytes
as though the discarded confidence scores were losslessly preserved.

## Evaluation and endpoints

Reuse all 128 evaluation worlds for natural sources, donors and the existing compact
J intervention. Add the full donor attention swap at layer 26, capturing downstream
J coordinates. Use 128 donor plus 128 dense forwards, identical to parent generation
settings; donor scores and generated IDs must replay exactly, and dense generated
IDs must match the parent's full-swap results. All arms use identical fixed probes.
No semantic model training, new directions or strength changes.

For each cohort report: per-layer country accuracy, agreement with original neural
probe labels, fraction of worlds with all nine labels preserved, and continuous
score NRMSE. Score NRMSE uses training per-coordinate mean/std (floor 1e-6): norm
of standardized reconstruction error divided by norm of the standardized original
scores. World is the unit, not nine independent layer examples. Source and donor
are paired. Report the 32 held-out query combinations separately.

Joint exploratory continuous-quality gate: selected source and dense cohorts each
preserve all nine labels in >=95% worlds and score NRMSE<=.25; exact symbolic
certificate; numerical readout payload at least 2x smaller. Failure does not block
reporting the independently exact categorical codec. No change after observing
new metrics. Always confirmed=false due reuse and single checkpoint.

Count the actual basis, probe, normalization and decoder tensors and serialized
files. Export a standalone readout codec containing the anchor bank/head plus
later-layer polynomial coefficients. It must reproduce the selected surrogate
from saved anchor J scores within 1e-10 after reload. The complete Qwen backbone
is not replaced; no model-weight compression or tokens/second claim.

## Runtime and provenance

Freeze source, tests, this protocol, parent producer dependencies, Pixi lock and
all parent input hashes before fitting. New append-only artifacts and strict JSON;
coefficient model, full and reduced bases, codecs, stage manifests, raw predictions,
report and conservative exact-ID registry readback. Preserve unrelated dirty work.

CPU fitting/algebra/codec cells are explicitly CPU-only, CUDA_VISIBLE_DEVICES empty,
four BLAS threads, root locked Pixi. GPU capture uses `pixi run --locked -e ncm`,
inference-ctl lease run 3060=10240 MiB for at most 30 minutes, guard before CUDA,
periodic checks and release. The unchanged parent runtime already passed actual-model
identity, VJP, persistence and memory tests; new exact replay is the lifecycle check.
No new services. Focused tests include algebra on known nonlinear maps, exact
normal-form agreement, coefficient persistence, finite country action, and exhaustive
mixed-label codec roundtrip/corruption checks.

Polynomial operations use [SymPy's documented Gröbner and normal-form API](https://docs.sympy.org/latest/modules/polys/reference.html#sympy.polys.polytools.groebner).
