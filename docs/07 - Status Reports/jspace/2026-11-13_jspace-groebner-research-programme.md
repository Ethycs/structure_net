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

`task Jacobians → candidate CAVs → causal validation → approximate relations → bounded symbolic reasoning → measured replacement`

Each arrow is a separate empirical question. A token-associated Jacobian row is
a candidate direction, not automatically a validated concept. A low residual is
evidence about sampled data, not an exact equation of the transformer. A compact
symbolic expression does not automatically use fewer bytes or run faster.

## Classification of useful outcomes

| Use | What the algebra contributes | Evidence required | Present status |
| --- | --- | --- | --- |
| Describe representation structure | Identify relationships among continuous observables, remove redundant polynomial descriptions, expose interactions | New-world and new-format residuals; stable coefficients/support; shuffled, linear, PCA and random controls; semantic anchoring | Some held-out mixed-layer relations; no demonstrated J advantage |
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
2. **Compare relation-discovery methods on the resulting coordinates.** Start
   with degree one/two, gradient-normalized approximate vanishing relations and
   sparse support. Compare J/CAV, PCA and random coordinates, pooled and
   class-conditional relations. Include identity/residual-stream baselines:
   adjacent layers can correlate because they mostly copy their inputs. Freeze
   on training/development before fresh evaluation. Measure conditional
   reconstruction, coefficient stability and constraint-gradient rank, not just
   the number of small residuals.
3. **Replicate the measured task-head shortcut before enlarging it.** The linear
   baseline matched or exceeded the tested polynomial heads and was cheaper.
   Test new worlds and formats, more than 64 evaluation cases, and then a task
   with practical relevance. Keep total encoder/head costs and a strong linear
   baseline. Any further polynomial work needs an advantage that survives these
   controls; the existing gain is task specialization.
4. **Test closure before replacing attention or layers.** Hold retained
   observables fixed while varying admissible residual/context information.
   Evaluate the next observables and task output, including interventions and
   multiple steps. Eight counterexamples already establish this problem for
   the tested middle-layer projections on the intervention-augmented domain.
   Expand retained source/context information or narrow the domain before
   fitting reduced dynamics; do not extrapolate hybrid-state failures into a
   universal impossibility theorem on natural activations.
5. **Benchmark only a quality-qualified replacement.** Actually remove the
   replaced computation, include encoder/decoder and branch costs, and measure
   memory and end-to-end latency on the intended token range. Match output
   interfaces: 2.505× became 1.196× when both models returned one class label.
   Parameter counts, theoretical FLOPs and toy payload savings are separate from
   measured tok/s. The current result is a one-run batch-one synthetic benchmark.

Each step receives a separate prospective protocol with numerical gates before
execution. Do not keep increasing symbolic timeouts or search degree simply to
obtain a nonempty basis. Stop or reclassify a route when controls explain its
gain, interventions fail, reconstruction is ambiguous, or costs exceed savings.
GPU stages must acquire and release an experiment reservation; CPU analyses use
root locked Pixi with CUDA hidden. Existing services remain offline unless the
user requests a deployment.
