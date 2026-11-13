# Research programme: Gröbner methods for interpretation and compression

Date: 2026-11-13. Current research objective, as requested by the user:
**validate and explore how Gröbner bases can help interpretability and
compression, using task-Jacobian space to propose semantic coordinates.**

This programme guides the current work. Individual studies retain their own
frozen protocols, measured endpoints and completion status. The programme is
exploratory and remains open; completing a control study does not complete it.

## The claim we are trying to establish

Find a small set of semantic observables that the model actually uses; describe
their relationships with low-complexity polynomials; determine whether those
relationships can explain a computation, reduce the state needed to perform it,
or support a cheaper task-specific replacement.

The working chain is:

`task Jacobians → candidate CAVs → causal validation → observable transport and relations → bounded symbolic reasoning → measured replacement`

Each arrow is a separate empirical question. A token-associated Jacobian row is
a candidate direction, not automatically a validated concept. A low residual is
evidence about sampled data, not an exact equation of the transformer. A compact
symbolic expression does not automatically use fewer bytes or run faster.

## Reassessment after the user's request for perspective

This is a correction to the research framing, made after the completed studies.
It changes neither their frozen questions nor their measured results. The
original conversation already distinguishes the action on observables from
preserving their defining equations. Our neural pilots did not adequately test
that broader construction.

Four mathematical objects must remain separate:

1. A within-layer vanishing relation, `g(z_l)=0`.
2. A mixed-layer graph relation, `p(z_l,z_next)=0`.
3. Transport of observable values, such as
   `Phi_next(F_l(H)) = M_l Phi_l(H)` on a specified domain.
4. Operator/module relations among transported readouts, as in the fixed-linear
   CAV/Krylov construction in the original discussion.

Our neural searches primarily addressed the first two, and the polynomial heads
addressed classification. Neither is a complete test of the third or fourth.
The original requirement for a nonempty source ideal was legitimate for its
particular ideal-transport certificate, but must not gate all observable
transport research. For `x_next=2x`, `x_next²=4x²` transports a polynomial
observable exactly even though unrestricted source `x` has zero vanishing ideal.
Likewise, projecting `(t,t²)` to `t` leaves no destination-only constraint, but
the second source coordinate remains exactly recoverable as `t²`. An empty
destination ideal does not establish loss of recoverability or failure of
compression relative to a redundant source representation.

The central interpretation question is whether semantically anchored functions
of activations have a compact, predictable action through actual model layers.
The polynomial's coefficients and its coordinate chart may change with depth.
The relevant layer pullback is `U_l f = f composed with F_l`. Different
transformer layers have different maps, so ordinary powers of one fixed matrix
are not a faithful model without an additional justified construction.
Coordinate polynomials commute under pointwise multiplication; arbitrary layer
operators need not commute. The fixed-operator module prototype cannot be
transferred to multiple layer operators by assuming those operators commute.

Interpretive value can precede autonomous state compression. Recovering a
concept with a transported nonlinear readout, identifying where two concepts
become indistinguishable, or predicting an interaction under intervention would
each be informative. A final-token closure failure establishes omitted context
for that chosen state/domain; it does not invalidate a contextual mechanism or
every possible concept algebra. Conversely, merely finding a polynomial that
fits a finite dataset is not sufficient evidence for any of these claims.

The current positive signal is seven held-out J cross-layer relations, compared
with thirteen PCA and zero random relations. Their counts are not independent
mechanisms or a measure of algebraic dimension. Residual carryover can explain
some such relationships, and the tests have not isolated the semantic content
or a multi-layer action. The current negative signal is that a small, selected
J dictionary and weak country CAVs fail the declared intervention gate. That
limits this implementation, not the full-vocabulary J-space hypothesis. The
linear exit is an engineering baseline, not evidence for the concept algebra.

The possible contribution is a causally meaningful algebra of transformer
observables with a measured, composable action and a characterized error/domain.
Novelty and practical compression are separate claims, both still unestablished.

## Classification of useful outcomes

| Use | What the algebra contributes | Evidence required | Present status |
| --- | --- | --- | --- |
| Describe representation structure | Identify relationships among continuous observables, remove redundant polynomial descriptions, expose interactions | New-world and new-format residuals; stable coefficients/support; shuffled, linear, PCA and random controls; semantic anchoring | Some held-out mixed-layer relations; no demonstrated J advantage |
| Explain how concept functions transform | Recover an action on a family of observables, including nonlinear downstream readouts and operator/module dependencies | Contiguous-layer and composed predictions; held-out concept combinations; predicted intervention effects; matched carryover and learned linear controls | Central construction remains under-tested; separate adjacent-pair equations and output classifiers are partial evidence |
| Explain how relations propagate | Eliminate intermediate variables; check a target relation's pullback by normal-form reduction; distinguish created, preserved and broken relations | Valid source conditions and transition model; intervention agreement; explicit domain and approximation errors | Exact constructed controls work; neural transport remains unestablished |
| Compress stored coordinates | Solve for omitted coordinates, identify necessary branch information and numerical sensitivity | Unique/stable decoder, actual byte counts, round-trip errors and downstream task quality | Finite toy codec works; no neural state compression validated |
| Replace a task-specific output computation | Use discriminative vanishing-polynomial features with a small trained head, potentially exiting the backbone earlier | Fresh task accuracy against the full model and linear, polynomial and MLP heads with full cost accounting; total parameters and measured latency | Linear early exit scored 62/64; polynomial 61/64, MLP 62/64. No polynomial advantage; costs were reported, not exactly matched |
| Replace a model segment with reduced dynamics | Simplify a polynomial transition modulo validated relations and retain only sufficient state | Closure under controlled changes to omitted information; multi-step error; causal interventions; task quality | Eight exact-copy counterexamples for selected middle-layer projections on a hybrid-intervention domain; natural-manifold compression remains open |
| Reduce deployed model cost | Compile and execute the successful replacement, actually removing weights or computation | Resident/serialized bytes, prefill/decode speed, batch and context matched comparisons, accuracy and confidence intervals | Research four-colour classifier removes 15% of parameters and achieves 1.196× matched single-label throughput; no general LLM compression or deployed service |

Interpretability can succeed even if compression fails. A task head can succeed
without reconstructing every hidden state. Conversely, exact reconstruction of
a selected coordinate does not prove that discarded information is irrelevant
to the next layer.

## Algebraic questions to keep distinct

Let `H_l` denote the relevant sequence state, including context needed by
attention, and let `z_l=E_l(H_l)` be retained observables. Let `I_l` describe
validated polynomial relations and `K_l(z_l,z_next)` describe a transition
surrogate. The two elimination questions are:

1. **Created constraints:** eliminate `z_l` from `K_l` alone.
2. **Transport with source constraints:** eliminate `z_l` from `I_l + K_l`.

The first can yield a nontrivial answer even when `I_l` is zero. For a known
polynomial transition `T`, reducing `g(T(z))` by a Gröbner basis of `I_l` to zero
certifies ideal membership. A nonzero remainder is not generally a complete
test of vanishing on the real domain: radical ideals, inequalities and branch
conditions matter. These certificates concern the specified polynomial system,
not Qwen's nonpolynomial operations.

For computation in the reduced state, the central condition is
`E_next(F(H)) = T(E(H))`. A counterexample consists of two admissible states with
the same retained coordinates and different next retained coordinates. Attention
requires the appropriate sequence/context state in this test; a final-token
vector alone need not be sufficient.

Do not count Gröbner generators as compressed dimensions. At regular points,
independent constraint gradients can diagnose local dimension. The quotient
`R[x,y]/<y-x²>` has one free coordinate but is infinite-dimensional as a vector
space. A finite set of ideal generators is not a finite linear activation basis.
Likewise, fitting a finite point cloud exactly is not evidence of generalization.

Approximate equations do not inherit exact ideal properties automatically.
Rationalizing noisy dense coefficients can introduce inconsistency and enormous
expressions. Numerical approximate-ideal discovery, sparse interpretable support
and bounded exact verification should therefore be separate stages.

## Evidence already available

- [Relation study](../../08%20-%20Analysis/jspace/2026-11-13_jspace-relation-propagation.md):
  seven J and thirteen PCA mixed-layer relations passed its exploratory held-out
  checks. No within-layer source relations passed, so its source-constraint
  transport gate did not reach symbolic elimination. No neural compression was
  measured.
- [Residual expansion](../../08%20-%20Analysis/jspace/2026-11-13_jspace-residual-expansion.md):
  expanding a fixed J dictionary did not capture most activation energy. This
  does not decide causal importance, and retaining a full residual is not a
  storage reduction.
- [Country probe](../../08%20-%20Analysis/jspace/2026-11-13_jspace-country-probe.md):
  the chosen J representation did not outperform full-state or PCA readouts.
  Probe accuracy alone is not intervention success.
- [Donor localization](../../08%20-%20Analysis/jspace/2026-11-13_jspace-donor-localization.md):
  at block 24, replacing question states except the final token redirected
  63/63 eligible answers; replacing only the final token redirected 0/63. At
  block 29 these results reversed. This supplies a dense positive control at
  appropriate positions, not evidence that the learned CAVs work. Blocks 25–28
  remain unmeasured.
- [Classification audit](../../08%20-%20Analysis/jspace/2026-11-13_jspace-groebner-classification.md):
  eleven constructed controls passed. The new graph-only neural audit produced
  no validated target-only relation; six symbolic cells were unresolved under
  the declared time/size limits. This is a bounded result, not a universal
  absence claim.
- [Active-position CAVs and task heads](../../08%20-%20Analysis/jspace/2026-11-13_jspace-active-cav-heads.md):
  dense query-donor transfer works at the chosen sites, but full country CAVs
  redirect only 6/59 and 11/59 answers; J components redirect 0/59. The stronger
  predeclared causal gate fails. The selected linear task head scores 62/64
  versus teacher 61/64 and polynomial head 61/64. A 30-block prefix removes
  462 million parameters; matched single-label latency improves 77.04→64.39 ms.
  The original 2.505× comparison also avoided a teacher generation step and is
  not the isolated compression gain. Post-hoc exact-copy witnesses show omitted
  context changes next states and answers despite identical retained patterns.

## Relevant literature and its limits

[Extended DMD](https://arxiv.org/abs/1408.4408) approximates the evolution of
observable functions from paired states and an observable dictionary. This is
the closest dynamical framing for the transport question; it does not require
discovering a nonzero within-layer vanishing ideal first. Finite useful closure
remains something to establish, not a consequence of invoking Koopman theory.

[Representing Neural Network Layers as Linear Operations via Koopman Operator
Theory](https://arxiv.org/abs/2409.01308) applies observable embeddings and DMD to
replace layers in pretrained MLPs on Yin-Yang and MNIST. This is direct neural
layer precedent, not a demonstration of semantic polynomial transport in a
large language model.

[Neiger and Schost, Computing syzygies in finite dimension using fast linear
algebra](https://arxiv.org/abs/1912.01848) computes Gröbner bases of relations
among module elements given a finite-dimensional module action. It grounds the
operator/module part of the fixed-linear CAV construction. The known action and
finite-dimensional setting are real assumptions. The new empirical burden is
finding a compact, useful action and semantic generators in the trained model.

[Vanishing Component Analysis](https://proceedings.mlr.press/v28/livni13.html)
provides a numerical route to approximate vanishing polynomials and compact
classifiers. It motivates relation discovery; it is not a transformer compression
result.

[Gradient Boosts the Approximate Vanishing Ideal](https://arxiv.org/abs/1911.04174)
uses gradient-based normalization to address spurious vanishing and redundancy.
This motivates a comparison with our coefficient-normalized SVD search. Merely
scaling polynomial coefficients down must never count as a better relation.

[VI-Net, ICML 2025](https://proceedings.mlr.press/v267/pelleriti25a.html) truncates
pretrained image classifiers, constructs class-specific approximate vanishing
polynomials, and retrains polynomial coefficients and a classifier. Its ResNet18
medium variant reports 4.35M parameters and 92.92% CIFAR-10 accuracy versus
11.24M and 92.89% for the baseline; reported throughput is 71,851 versus 66,533
images/s at batch 256. This supports testing a task-specific polynomial head,
not assuming lossless compression or transferring that throughput to Qwen.
[The authors' code](https://github.com/ZIB-IOL/approximating-neural-network-manifolds)
is available; the README notes that an internal dataset cache is not included.

## Next experiments, in decision order

1. **Revise the causal coordinates after the active-position result.** That
   experiment completed with weak full-CAV effects and no measured J advantage.
   Its 94-token dictionary omitted the actual colour output tokens. A fresh
   study should include task-output directions and source-fact/context
   observables, keep country and answer changes independent, and retain dense
   positive and norm-matched random controls. Report target redirection and
   original-answer retention separately; answer retention is not specificity
   evidence when the corresponding dense control also cannot change the answer.
2. **Test the induced action on observables through contiguous layers.** Start
   with degree one/two and a fixed training-only complexity budget. Track actual
   attention and MLP boundaries, allow readout coordinates to change with depth,
   and include necessary source/context observables or explicitly condition on
   them. Fit action maps jointly enough to preserve observable identity, then
   compare composed predictions with direct measurements several layers later.
   A nonempty within-layer ideal is not an entry gate. Compare identity/residual
   carryover, learned linear maps, ordinary polynomial regression, PCA and J/CAV
   coordinates. Freeze before fresh evaluation; hold out concept combinations,
   not only new names or random rows. Report transport error and operator cost.
3. **Test whether transported functions explain semantic interventions.** Use
   the fitted action to predict changes in later observables and task output
   before interventions run. Distinguish preserved, mixed, newly created and
   unrecoverable concept functions; compare nonlinear recovery with linear
   probes. Measure derivative agreement when making tangent/rank claims.
   Partial context-dependent explanations can be useful before full autonomous
   closure, but their conditioning variables and costs must remain explicit.
4. **Test closure before replacing attention or layers.** Hold retained
   observables fixed while varying admissible residual/context information.
   Evaluate the next observables and task output, including interventions and
   multiple steps. Eight counterexamples already establish this problem for
   the tested middle-layer projections on the intervention-augmented domain.
   Expand retained source/context information or narrow the domain before
   fitting reduced dynamics; do not extrapolate hybrid-state failures into a
   universal impossibility theorem on natural activations.
5. **Use algebra to remove demonstrated redundancies.** Discover stable
   approximate vanishing or operator/module relations where the measured action
   supports them. Count independent information and actual representation cost,
   not generators. Compare composition and generalization with the same feature
   library without algebraic reduction. Exact normal-form certificates apply to
   a justified polynomial surrogate/domain, not automatically to Qwen.
6. **Benchmark only a quality-qualified replacement.** Actually remove the
   replaced computation, include encoder/decoder and branch costs, and measure
   memory and end-to-end latency on the intended token range. Match output
   interfaces: 2.505× became 1.196× when both models returned one class label.
   Parameter counts, theoretical FLOPs and toy payload savings are separate from
   measured tok/s. The current result is a one-run batch-one synthetic benchmark.

The existing linear task-head shortcut is a separate engineering branch. Any
extension needs new worlds/formats and more than 64 evaluation cases. Its
classification accuracy must not substitute for testing semantic transport.

Each step receives a separate prospective protocol with numerical gates before
execution. Do not keep increasing symbolic timeouts or search degree simply to
obtain a nonempty basis. Stop or reclassify a route when controls explain its
gain, interventions fail, reconstruction is ambiguous, or costs exceed savings.
GPU stages must acquire and release an experiment reservation; CPU analyses use
root locked Pixi with CUDA hidden. Existing services remain offline unless the
user requests a deployment.

## Compact attention transport pilot, 2026-11-13

[JSPACE-ATTENTION-TRANSPORT-01](../../08%20-%20Analysis/jspace/2026-11-13_jspace-attention-transport.md)
tests actual attention writes and composed nonzero-observable maps without a
source-ideal gate. The eight selected country-token J rows do not constitute a
validated causal concept: a layer-26 compact edit changes country identity in
0/128 held-out worlds, versus 127/128 for dense and complementary donor edits.
Country identity is readable from the compact coordinates in 128/128 examples
from layer 27 onward. Exact task answer accuracy is only 65/128 because country
names usually replace requested nationality adjectives; the identity endpoint
was declared separately before evaluation and must not replace that task score.

Quadratic composed change error at layer 29 is 8.181 versus linear 5.799 and
zero-change baseline 1. A posthoc full-state audit shows the perturbation is not
simply erased: median final L2 is 19.75× injection, but only 0.49% of aggregate
energy is inside the selected J chart. All 128 worlds show downstream growth,
and replayed generations match exactly. This is numerical propagation outside
the narrow measured span, not a transported semantic or polynomial certificate.

The concrete next target is the successful layer-26 donor route and the layer-27
attention transition: learn cross-attribute country contrasts in a broader J
dictionary, validate causal edits against full/PCA controls, then test contextual
transport of their readouts. The study narrows the coordinate-construction issue;
it does not invalidate the broader operator/module or Gröbner programme.

## Country stack and a library of reduced operators, 2026-11-13

[JSPACE-COUNTRY-STACK-GROEBNER-01](../../08%20-%20Analysis/jspace/2026-11-13_jspace-country-stack.md)
produced a finite four-country quotient, exact symbolic country actions, and a
lossless recorded-label codec. It reduced 128 natural nine-layer label traces
from 298 to 58 bytes, with explicit exceptions for nonconstant trajectories.
The selected continuous readout file is 7.97× smaller, but its neural fidelity
failed: source label agreement is 96/128 and dense-swap agreement 97/128. The
linear comparator preserves 128/128 and 126/128 labels, respectively, but also
fails the predeclared continuous-score error threshold. None of these results
establishes a smaller full transformer.

The exact training-support algebra identifies two unconstrained country/task
cells. The selected quadratic decoder gets every seen source combination right
and every held-out source combination wrong. This supports using algebra to audit
semantic coverage before treating an empirical relation as a global task law.

A tiny actual Qwen control also verifies exact fusion of a linear attention
readout into V/O projections (maximum output difference 2.602e-17, input-gradient
difference zero). This is an operator simplification with Q/K and softmax retained,
not a closure result or a runtime benchmark.

The user's proposed library is a research extension, not a completed experiment.
Each entry should contain an encoder into task-relevant continuous coordinates,
normalized approximate relations or exact surrogate certificates, a reduced
update operator, an output map, a validity domain, and measured error/cost. Learn
small local representations for broad task families rather than running a single
symbolic calculation over the full hidden dimension.

Compare entries after alignment on shared calibration examples. Use subspace
overlap for linear coordinates, held-out relation residuals, similarity of induced
updates under the alignment, and causal transfer. Raw Gröbner coefficient distance
is not a sufficient similarity measure: coordinates and monomial order matter.
Even the same four-state label algebra can describe unrelated tasks. Finite
quotient multiplication matrices can help compare algebraic structure, but their
existence must not be assumed for our infinite-dimensional continuous quotient.
For relations meant to hold across a union of families, test them on that union;
simply adding all generators can impose mutually incompatible restrictions.

Compile shared operations plus small family-specific corrections offline. At
inference, route to one or a few entries using a cheap router and retain a dense
fallback where validation requires it. Include router, correction, cache and
fallback costs. If routing changes KV representations during a sequence, provide
a compatible cache or account for migration/recomputation. Do not perform symbolic
Gröbner computations for each generated token.

Start with country-capital and country-nationality plus a structurally matched
non-geographic entity-attribute task. Cover all semantic cells in training, then
hold out names, prompt formats, whole families and compositions separately. Compare
original attention, rank-matched SVD/low-rank attention, independent family modules,
a shared library without algebraic reduction, and the proposed algebraic library.
Measure task accuracy, intervention agreement, composed-update error, payload,
peak memory and end-to-end tokens/s. The key evidence is transfer to an unseen
family or composition with lower total execution cost at matched task quality.
Freeze a numerical protocol before any such new training or GPU experiment.

Related primary work supports individual components, not this exact proposal:
[Eigen Attention](https://arxiv.org/abs/2408.05646) computes attention in a low-rank
space; [MoH](https://arxiv.org/abs/2410.11842) routes tokens to selected heads;
[gradient-normalized approximate vanishing ideals](https://arxiv.org/abs/1911.04174)
address spurious vanishing and coordinate scaling. Their reported results do not
establish our library's transfer, accuracy, or speed.
