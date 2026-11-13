# Gröbner classification: exact controls work; neural compression remains unvalidated

Date: 2026-11-13. Study: `JSPACE-GROEBNER-CLASSIFICATION-01`.
Status: completed CPU exploratory study; confirmed=false.
Classification: `no_validated_graph_consequences_with_unresolved_cells`.

All eleven constructed algebra controls passed. They demonstrate both useful
compression mechanisms and specific reasons a valid relation can fail to
support compression. A separate audit of the previously frozen Qwen relations
found no validated destination-only constraint. Six symbolic cells remain
unresolved under the declared limits; they are not evidence that no relation
exists. No Qwen task accuracy, parameter reduction or speedup was measured in
this study.

- [Research programme and classification](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-groebner-research-programme.md).
- [Frozen protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-groebner-classification-plan.md).
- [Aggregate](../../../data/experiments/jspace_groebner_classification/20261113_v1/campaign_results.json).
- [Direct result](../../../data/experiments/jspace_groebner_classification/20261113_v1/result.json).
- [Frozen graph calculations](../../../data/experiments/jspace_groebner_classification/20261113_v1/frozen_graphs.json).
- Fingerprint: `93a2acdf16f79c91d9da3ecf38a4870ded7007a166b0783028c369ceb4dc56cd`.
- Result SHA-256: `21bd5fd41c9e53354c3765acbddda9345e9f887df941804d0112dae8ac7a98c0`.
- Aggregate SHA-256: `6e413c34edb71df3788eb92a00a882b78a8be68f2dd15882eb4392f196cf9ce2`.

## What the exact controls establish

| Question | Measured or symbolic result | Consequence for the neural research |
| --- | --- | --- |
| Can a transition create a constraint without a source constraint? | Eliminating `x` from `u=x, v=x²` gives `v-u²` | Test graph-created constraints separately from transported source constraints |
| Can the implementation distinguish preserved and broken relations? | Exhaustive Boolean AND/OR→XOR/AND control passes; altered transition is detected | Exact algebra machinery has a positive and a negative control |
| Can a relation support actual storage savings? | For 65 dyadic `(x,x²)` pairs, stored float64 payload falls from 1,040 to 520 bytes, with bitwise round trip | A unique decoder can compress coordinates on a declared domain |
| Is the direction of compression interchangeable? | Retaining `y=x²` alone leaves 32 ambiguous fibres out of 33; packed sign bits repair all cases at 529 bytes | Branch information and domain restrictions are real costs |
| Is uniqueness enough for numerical safety? | `y=1/x` has absolute derivative magnitude `10^12` at `x=10^-6` | Measure absolute error amplification; relative condition number here is only one |
| Can reduced dynamics be exact? | `(x+1,(x+1)²)` preserves `y=x²`; 520 sequential transitions have zero mismatch | Closure is possible for a specified polynomial system |
| Does a valid relation imply closure of any chosen projection? | States `(1,1,0)` and `(1,1,1)` share retained `x=1`, but update `x+u` gives 1 and 2 | Omitted variables can control future computation |
| Does elimination capture every real-domain restriction? | Projection of `y=x²` has zero elimination ideal, although `y=-1` has no real preimage | Algebraic closure must not be mistaken for the exact real image |
| Can tiny approximate residuals be treated as exact equations? | `x` and `x-10^-6`, each with residual magnitude `5*10^-7` at the midpoint, generate `[1]` | Rationalized noisy equations can be exactly inconsistent |
| Can coefficient scaling fake a good fit? | Scaling `y-x²` by `10^-12` shrinks raw residuals but preserves the gradient-normalized residual | Use a defensible normalization; the control is not a full algorithm replication |
| Does fewer coordinates mean less computation? | Reading stored `y` costs no multiplication; reconstructing `x^8` from `x` takes three by repeated squaring | Count reconstruction work before claiming a speedup |

The 50% payload reduction belongs to a constructed finite dyadic codec. Fixed
decoder code and framing are excluded from both payload counts. It is not
compression of Qwen weights, a floating-point guarantee for arbitrary inputs,
or a neural throughput result. The eleven passing controls are not an 11/11
language-model accuracy score.

## What changed in the neural question

The parent study required both an accepted source relation and an accepted
mixed-layer relation before symbolic elimination. No source relations passed,
so none of its 18 cells reached Gröbner computation. That remains the correct
record for its source-constraint transport protocol.

This study asks a broader, separate question: do the mixed relations alone
create a destination constraint? It uses a zero source ideal, the same frozen
development-selected mixed relations, and the same rationalization and bounded
symbolic worker. This avoids discarding a valid graph-only possibility without
retroactively changing the original experiment.

The parent supplies Qwen2.5-3B-Instruct captures, three four-dimensional banks
and six layer edges. This process used 128 development and 128 already-used
evaluation rows; it did not refit on the 512 training rows. New graph results
were frozen before loading the evaluation tensors, but the audit remains
post-hoc because the parent outcomes were already known. No model was loaded
on a GPU.

## Frozen graph audit

| Bank | No accepted mixed relations | Rationalization rejected | Completed, empty destination ideal | Time limit | Expression-size limit | Validated destination relations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J | 4 | 0 | 1 | 1 | 0 | 0 |
| PCA | 0 | 1 | 0 | 1 | 4 | 0 |
| Random | 6 | 0 | 0 | 0 | 0 | 0 |

Seven of the 18 cells reached symbolic computation. The **J block 34→35** case
completed: its four rationalized linear equations have source coefficient rank
four, and its elimination ideal contains no destination-only polynomial. For
this surrogate, the four destination variables are not constrained by a
coordinate-removal equation. This is an example of a relationship across layers
without a destination redundancy. One of these four development-selected
relations had failed the parent's evaluation check; it was retained here by
the frozen selection rule, not silently filtered using evaluation.

The **J block 29→30** and **PCA block 5→6** cells hit the five-second wall limit,
including process startup and serialization. The **PCA block 17→18** cell failed
rationalization checks: two RMS changes, approximately 0.01259 and 0.01235,
exceeded 0.01. This is a rejected surrogate, not a symbolic absence result.

The other four PCA cells returned the worker status `degree_explosion`. In this
worker that status also covers expression length. Their recorded maximum
degree was four and basis size was four or six, below the degree-12 and
32-polynomial limits; therefore the triggered cap was the combined 50,000
character expression limit. Dense rational coefficients can make even low-degree
bases unwieldy. No unit ideal was returned in these neural cells.

The six timeout/size-limited cells remain unresolved. No destination polynomials
were returned for held-out evaluation, so the count of validated consequences
is zero. This does not establish a zero-dimensional phenomenon, a universal
absence of relations, or failure of the broader research direction.

## Interpretation and next decision

The most useful classification is currently:

- **Interpretation of a specified polynomial system:** working, with explicit
  positive and negative exact controls.
- **Approximate neural representation structure:** some evidence in the parent
  study, without a J-specific advantage or validated causal interpretation.
- **Coordinate compression:** demonstrated only in a constructed codec.
- **Neural closure, model compression and speed:** unestablished.

The next neural priority is validating continuous CAV coordinates at positions
where dense donor interventions actually work, then comparing sparse,
gradient-normalized relation discovery with linear and PCA controls. A separate
polynomial task-head experiment can test practical compression without first
requiring exact hidden-state closure. The [programme](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-groebner-research-programme.md)
gives these routes separate success criteria. Increasing the exact-rational
timeout alone would not resolve the missing causal and compression evidence.

## Reproduction and integrity

Root locked Pixi, Python 3.11.13, NumPy 2.3.1, SymPy 1.14.0 and Torch 2.5.1+cu121;
CUDA explicitly hidden. Study execution took 26.29 seconds, excluding setup and
tests. Ten focused tests passed, including graph-only discovery, an altered
held-out relation, rationalization rejection, timeout classification, byte-exact
payload restoration and strict JSON serialization. Source, protocol, dependency
lock and consumed input hashes are archived in the new artifact root. Parent
artifacts were verified and remain unchanged.

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m pytest \
  --confcutdir=tests/structure_net \
  tests/structure_net/test_jspace_groebner_classification.py -q
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_groebner_classification \
  prepare --output data/experiments/jspace_groebner_classification/NEW_ROOT
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_groebner_classification \
  run --output data/experiments/jspace_groebner_classification/NEW_ROOT
```

No GPU reservation was needed for this CPU-only study. No service was restarted
and no Qwen deployment configuration was changed.
