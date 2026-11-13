# JSPACE-GROEBNER-CLASSIFICATION-01 — protocol

Status: written before execution of this study. CPU only, through the root locked
Pixi environment with CUDA hidden. Parent neural results have already been seen;
their reuse is an exploratory audit, not a fresh confirmation. Completed parent
artifacts and producer code remain unchanged.

## Question and scope

Classify the distinct ways polynomial ideals can contribute to representation
interpretability and compression. Separate discovery of approximate relations,
exact statements about a specified algebraic surrogate, causal neural evidence,
coordinate storage, and actual model/runtime reduction.

The earlier propagation protocol intentionally required at least one accepted
source relation and one mixed source/destination relation. This tests transport
of an existing constraint but excludes constraints created by the transition
graph itself. The new graph-only analysis uses the zero source ideal and the
previously frozen mixed relations. It does not silently change the previous
study's eligibility rule or its negative findings.

## Constructed controls

All eleven cases are specified by the producer before execution:

1. An unconstrained scalar maps to `(u, v)=(x, x²)`. Eliminate `x` and recover
   `v-u²` without any source relation.
2. Reuse the exact Boolean AND/OR to XOR/AND transport control and its altered
   transition. Check the exhaustive state sets and normal forms.
3. On `y=x²`, retain `x` and reconstruct `y`. Actually pack and bitwise round-trip
   all 65 dyadic inputs `i/32`, `i=-32,…,32`, using float64 payloads.
4. Retain only `y`; exhibit opposite-sign collisions. Retain an additional packed
   sign bit and verify the repaired round trip. Fixed decoder/framing excluded
   from payload counts; no neural parameter compression implied.
5. On `xy=1`, examine absolute reconstruction sensitivity as `x` approaches zero.
   Report that relative conditioning is benign, so the limitation concerns
   absolute errors and fixed-step quantization.
6. On `y=x²`, test the closed update `(x+1,(x+1)²)` by normal-form reduction and
   65 trajectories of eight steps. Detect the non-preserving update `(x+1,y+1)`.
7. Add free `u` and update `(x+u,(x+u)²,u)`. Exhibit two states with identical
   retained `x` and different next `x`. A valid relation does not establish
   autonomous dynamics of the chosen projection.
8. Eliminate `x` from `y=x²`. Show that the algebraic image closure does not encode
   the real inequality `y>=0`.
9. Two individually tiny residuals, `x` and `x-10^-6` at `x=0.5*10^-6`, generate
   the unit ideal when treated as exact equations.
10. Scale a polynomial by `10^-12`; compare raw residual and gradient-normalized
    residual at a regular point. This is a normalization control, not a paper
    replication or a general distance-to-manifold certificate.
11. On `y=x^8`, compare reading stored `y` with reconstructing it from `x` using
    three multiplications. Local storage reduction can increase output work.

Pass means the positive and negative cases are classified correctly. The pass
fraction must never be logged as language-model accuracy. Exact statements apply
to these systems and domains only.

## Frozen neural graph audit

Parent: `data/experiments/jspace_relation_propagation/20261113_neural_v1`.
Reuse its frozen J, PCA and random banks, 12 sampled blocks, six edges, continuous
four-dimensional scores, 512 training rows and 128 development/evaluation rows.
No coefficients, search degrees, thresholds, cases, or directions are refitted.

For all 18 bank/edge cells, take at most four previously development-accepted
mixed relations, ordered by development RMS. With none, report
`no_accepted_graph_relations`. Otherwise rationalize the same standardized
coefficients with denominator <=1000. Require RMS perturbation <=0.01 and new
RMS <=max(1.5*old RMS, old RMS+0.01). Eliminate the four source variables over QQ
in lexicographic order, using the existing killable five-second worker and its
degree/size caps. Record timeouts, unit ideals, rejected rationalizations and
empty elimination separately. Never interpret a timeout as an absent relation.

Report linear source-block rank when all selected relations are linear; use this
only as a diagnostic. A nonzero target consequence is not evidence that a source
constraint propagated: there are no source-ideal generators in this audit.

Freeze all graph results and development normalizers before reading the reused
evaluation arrays in this process. Evaluate target-only consequences using the
parent's RMS<=0.10 and <=one-third of 32 coordinate shuffles. Record both
development and evaluation acceptance. These are exploratory descriptive gates,
not familywise significance claims. An exact Gröbner calculation of rationalized
approximate relations cannot certify the neural network.

## Lifecycle and decision rules

Archive source, this protocol, root `pyproject.toml`/`pixi.lock`, dependency versions and
hashes of all consumed parent artifacts before running. Use an exclusive dated
artifact root. Verify hashes before execution and after analysis; write strict
JSON result and aggregate with explicit scopes, null unmeasured neural accuracy,
compression and throughput, and conservative registry readback.

If the algebra controls fail, stop the neural audit. If graph elimination is
empty, the available mixed relations do not supply coordinate-removal equations
under these bounds. If it times out, the symbolic computation is unresolved. If
it yields held-out consequences, inspect uniqueness, conditioning and causal
closure before proposing any model change. In every case, the next neural study
must use the now-validated active token positions and distinguish country
concepts from already-computed answer colours.
