# Stacked country Gröbner representations and measured compression

Date: 2026-11-13. Study: `JSPACE-COUNTRY-STACK-GROEBNER-01`.
Status: completed exploratory study on reused parent worlds; confirmed=false.
Classification: `exact_country_codec_continuous_stack_not_quality_qualified`.

We produced an exact stacked four-country algebra, its compressed Gröbner basis,
country decoders, multiplication matrices and all six symbolic country swaps.
The lossless codec for observed nine-layer country-label trajectories takes
**58 bytes instead of 298 bytes** on 128 natural examples (5.14× smaller), and
**62 bytes** on the full-attention-swap trajectories (4.81× smaller). It preserves
every label, including the two nonconstant swapped trajectories via fallback.

We also produced exact Gröbner certificates for linear and quadratic continuous
score surrogates. The development-selected quadratic readout file is **7.97×
smaller**, but failed the neural-fidelity gate: it preserves all nine original
labels on 96/128 natural worlds and 97/128 full-swap worlds. The predeclared linear
baseline preserves 128/128 and 126/128, respectively, but also misses the continuous
score reconstruction threshold. The exact symbolic certificate applies to the
fitted surrogate, not to Qwen's nonlinear computation.

[Download artifact bundle](../../../data/experiments/jspace_country_stack/20261113_v1/country_stack_bundle.zip).
[Prospective exploratory protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-country-stack-plan.md).
[Aggregate](../../../data/experiments/jspace_country_stack/20261113_v1/campaign_results.json).
[Registry record](../../../data/hypothesis_registry/jspace_country_stack_20261113.json).

## The exact country algebra

Encode China=(0,0), France=(0,1), Germany=(1,0), Russia=(1,1). With coordinates
`a,b`, the reduced Gröbner basis over QQ is

```text
G = { a² - a, b² - b }
```

The quotient has dimension four and standard monomials `{1,a,b,ab}`. Its country
indicators decode as follows:

| Country | Polynomial |
| --- | --- |
| China | (1-a)(1-b) |
| France | (1-a)b |
| Germany | a(1-b) |
| Russia | ab |

For a constant country trajectory, stack these four indicators at each of block
outputs 27–35: 36 indicator variables. The full graph Gröbner basis has 38
polynomials. Eliminating all 36 dependent indicators leaves the two polynomials
above, with exact normal-form decoders for every layer. This is a constructed
finite semantic encoding, not Boolean equations inferred for continuous neural
activations. All four states and all 24 input/swap pairs were checked exactly.

For example, swapping France and Germany exchanges `a` and `b`. Swapping China
and France is `a_next=a`, `b_next=1-a-b+2ab`. The complete artifact contains all
six swaps. Multiplication by `a` and `b` is represented by two commuting idempotent
4×4 matrices; this does not assume that arbitrary country-swap operators commute.

[Full basis, compressed basis, decoders and actions](../../../data/experiments/jspace_country_stack/20261113_v1/country_quotient.json).

## What the categorical compression actually saves

The baseline already encodes each country using two bits, so nine layers cost
18 bits/world. The compact stream uses one framing bit plus a two-bit country
when all nine labels agree; a nonconstant trajectory uses the framing bit plus
all nine two-bit labels. Both formats include a ten-byte header and final padding.

| Cohort, 128 worlds each | Constant trajectories | Baseline bytes | Compact bytes | zlib of baseline |
| --- | ---: | ---: | ---: | ---: |
| Natural source | 128 | 298 | 58 | 176 |
| Natural donor | 128 | 298 | 58 | 169 |
| Compact J edit | 128 | 298 | 58 | 176 |
| Full attention swap | 126 | 298 | 62 | 169 |

All decoded label arrays match bit for bit. The two mixed full-swap trajectories
are explicitly retained rather than forced into the constant-country algebra.
An ordinary repeated-label codec can achieve the same savings. Gröbner methods
provide the algebraic representation and action rules here; the byte gain itself
comes from repeated country identity, not a newly discovered coding advantage.

This exact codec observes the original trajectory before encoding exceptions.
It compresses recorded country labels. It does not predict those exceptions from
two bits, preserve confidence scores, reconstruct hidden activations, or remove
the later transformer layers.

## The continuous score stack

The frozen parent supplies four-country linear readouts on eight J directions,
pooled separately over the question and final token. At nine post-MLP boundaries,
the stack contains 36 continuous ridge class scores. These are not calibrated
probabilities, although their affine sum is one to numerical tolerance.

Retain three scores from layer 27 and the known binary task attribute `u`.
Fit all later scores jointly from these four inputs, using linear or degree-two
ridge polynomials. Training uses 384 independent worlds / 768 paired observations;
development uses 24 worlds / 48 observations. Both exclude China/nationality and
Russia/capital. The selected quadratic family had lower development standardized
MSE, 0.05769 versus linear 0.07414, and both preserved every development label.
No transformer weights were trained.

Coefficients were rationalized on a 2^-24 grid before symbolic work. Development
prediction changes were only 3.91e-8 RMS for the quadratic family. For each score,
construct `s_layer,country - f_layer,country(x_China,x_France,x_Germany,u)` and add
`u²-u`. Each full graph basis contains 37 polynomials. Every generator reduces
to zero, and all output normal forms match the rational numerical decoder exactly.
Eliminating outputs leaves `{u²-u}` with three continuous free coordinates: this
quotient is infinite-dimensional and must not be confused with the four-state
country algebra above.

Boolean normal form reduces the quadratic feature library from 15 to 14 terms;
deriving the fourth score from the other three removes redundant score polynomials.
The selected stack has 487 nonzero coefficients in its unreduced explicit form
and 336 stored later-layer coefficients after reduction; anchor and fourth-score
decoders are supplied by shared formulas. Normalization, the anchor J bank and
probe, known attribute input, and numerical precision remain part of the codec.

| Readout artifact | Nine original readouts | Selected compact readout | Ratio |
| --- | ---: | ---: | ---: |
| Numerical tensor bytes | 598,176 | 69,152 | 8.65× |
| Actual serialized bytes | 611,940 | 76,740 | 7.97× |

Most of this storage reduction comes from retaining one J bank/head instead of
nine. It is not all attributable to Gröbner canonicalization. The unchanged Qwen
backbone is excluded from both numbers. Serialized bytes include container/storage
overhead; logical tensor byte counts do not count unused backing storage in views.

[Quadratic graph certificate](../../../data/experiments/jspace_country_stack/20261113_v1/continuous_stack_degree2.json),
[linear graph certificate](../../../data/experiments/jspace_country_stack/20261113_v1/continuous_stack_degree1.json),
[selected readout payload](../../../data/experiments/jspace_country_stack/20261113_v1/compressed_readout.pt),
[linear reference payload](../../../data/experiments/jspace_country_stack/20261113_v1/linear_readout_reference.pt).
The linear reference was exported after evaluation without changing the primary
selection or its failed gate.

## Neural validation

Evaluate all 128 parent worlds in four cohorts: source, donor, the existing compact
J edit, and a newly captured full attention-output swap at layer 26. The additional
GPU job made 256 forwards. Every donor score and donor/dense generated token sequence
matched the parent record exactly. The portable decoder reproduced the selected
surrogate from anchor J scores within the 1e-10 reload tolerance.

The following is **agreement with all nine original country-probe labels in a
world**, not general language-model accuracy or nine independent observations.

| Predictor | Natural source | Natural donor | Compact J edit | Full swap |
| --- | ---: | ---: | ---: | ---: |
| Copy anchor scores | 128/128 | 128/128 | 128/128 | 126/128 |
| Linear stack | 128/128 | 128/128 | 128/128 | 126/128 |
| Selected quadratic stack | 96/128 | 98/128 | 96/128 | 97/128 |

Continuous-score normalized reconstruction errors are:

| Predictor | Natural source | Full swap |
| --- | ---: | ---: |
| Copy anchor scores | 0.380 | 0.398 |
| Linear | 0.382 | 0.398 |
| Quadratic | 0.596 | 0.617 |

The joint gate required >=95% whole-world label agreement and <=0.25 normalized
score error in both source and dense cohorts, plus exact certification and at
least 2× numerical payload savings. Certification and payload pass; continuous
quality fails. Even the linear comparator misses the continuous-error threshold.
No replacement accuracy or end-to-end speedup is claimed.

![Continuous country reconstruction and readout file size](../../../data/experiments/jspace_country_stack/20261113_v1/country_stack.png)

## A useful algebraic diagnosis of the failure

On source worlds the quadratic decoder preserves all nine labels in 96/96 seen
combinations and **0/32 held-out combinations**. Development had no examples from
those two missing cells. A posthoc finite-grid audit makes the ambiguity explicit.

With country bits `a,b` and `u=0` for capital / `u=1` for nationality, functions
on all eight semantic states have dimension eight. Training covers six states,
so its evaluation matrix has rank six and a two-dimensional nullspace. Two
explicit unconstrained functions are

```text
g_China,nationality = (1-a)(1-b)u
g_Russia,capital   = ab(1-u)
```

Both vanish on every training/development semantic state. Each equals one at
its own missing evaluation state and zero elsewhere. Arbitrary multiples can
therefore change the missing outputs without changing any observed semantic-grid
training value. The training-support Gröbner basis is

```text
{ a²-a, ab-au-bu+u, b²-b, u²-u }
```

Its quotient dimension is six. The extra relation `ab-au-bu+u=0` is **false on
the omitted combinations**. Treating a training-support relation as a global
task law would incorrectly exclude valid countries/tasks.

This is an exact diagnosis of the semantic coverage gap. It is not proof that
the learned neural decoder implements that exact polynomial. On continuous anchor
scores, the corresponding missing-cell witnesses have training RMS 0.040/0.065,
development RMS 0.042/0.063, and held-out RMS 0.502/1.001. They are small rather
than identically zero on observed neural data. The enlarged response on missing
cells is consistent with the extrapolation failure. The published cohort subgroup
fields are grouped by the original source query; donor/dense targets can belong
to different semantic cells. The 0/32 claim above uses source worlds only.

[Identifiability audit and exact witnesses](../../../data/experiments/jspace_country_stack/20261113_v1/identifiability_audit.json).
This shows a concrete interpretability use for Gröbner analysis: distinguish the
algebra of observed training support from the algebra of the full intended task.

## Pulling observables back through attention

For a linear readout `C` of the attention output, the value and output projections
can be fused exactly. With column vectors, the per-head value projection becomes
`C W_O,h W_V,h`, with the corresponding projected biases. Keep the original Q/K,
RoPE, mask and softmax weights, apply those weights to the smaller values, and
sum the heads. This follows by reassociating the linear maps in
[standard multi-head attention](https://arxiv.org/abs/1706.03762).
Any polynomial of the retained readouts can then be evaluated after this step.

A CPU control using an actual randomly initialized tiny Qwen attention module
(four query heads, two KV heads, head width eight, readout rank three) passed:
maximum readout difference **2.602e-17**, maximum input-gradient difference **0**
for `z0² + z1*z2 + 0.3*z0`. Parameters and inputs were double precision; the
Transformers implementation retains its own internal precision conversions.
Both input gradients were nonzero, with maximum magnitude 0.025788. This verifies
the readout identity on the implemented attention module. It does not validate
a trained 3B replacement, approximate Q/K, or end-to-end speed.

For the actual 3B configuration, rank eight would reduce the counted V/O matrix
multiplication work from 4,718,592 to 262,144 multiply-accumulates per token, an
18× ratio for those suboperations only. Q/K, softmax, normalization, later layers,
readout encoding and implementation overhead are excluded. No runtime speedup
was measured. The fused values become specific to each query head; any loss of
GQA sharing is included in the 16 × 8 value-channel count.

A complete reduced mechanism requires `pi(F(H)) = F_small(pi(H), context)` on
the intended domain, where all retained context and its cost are explicit.
Pulling back a polynomial `p` means composing `p` with the original operator;
this can increase complexity, and softmax is not polynomial. Gröbner reduction
can simplify a justified polynomial representation, but a basis of country labels
does not itself establish that the required attention state is retained. The
linear identity above does not require a Gröbner basis.

The relevant next step is to find continuous country and context observables whose
updates close, then compile their reduced computation. Exact constrained model
reduction such as [CLUE](https://arxiv.org/abs/2004.11961) provides a related
observable-preserving principle for polynomial differential equations; it is not
a ready-made transformer reduction algorithm.

[Attention pullback audit](../../../data/experiments/jspace_country_stack/20261113_v1/attention_pullback_audit.json).

## Reproduction, provenance and next step

Main fingerprint:
`018f1d1bbdc5fffa50a2dae956e369a20bdda0798e4427fdea679cd0649b2e4f`.
All scientific producers, parent inputs, tests and protocol were frozen before fit.
Both symbolic families completed; no timeouts, retries, exclusions or failed jobs.
Fourteen focused tests passed. CPU fit plus algebra took 5.35 seconds. The reserved
RTX 3060 capture took 67.55 seconds after loading began, with peak reserved PyTorch
memory 6,469,713,920 bytes under its 10,240 MiB lease. Lease `4a540bb8b7dc` was
released; no bookings or leases remained. Timing is lifecycle accounting, not an
inference benchmark. No services changed.

Use root locked Pixi with CUDA hidden for prepare/fit/analyze. GPU capture requires
`inference-ctl lease run` and `pixi run --locked -e ncm`. The source entry point is
`experiments/structure_net/jspace_country_stack.py`; stage order is prepare, fit,
capture, analyze, with a new `--output` root for a rerun. Presentation and posthoc
diagnostic entry points remain alongside it. Tests:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 pixi run --locked python -m pytest \
  --confcutdir=tests/structure_net tests/structure_net/test_jspace_country_stack.py \
  tests/structure_net/test_jspace_country_stack_registry.py -q
```

Gröbner and normal-form computations use the [documented SymPy API](https://docs.sympy.org/latest/modules/polys/reference.html#sympy.polys.polytools.groebner).

The next neural experiment should either cover all eight semantic combinations in
training or explicitly require country identity to be independent of requested
attribute, then evaluate on fresh worlds. A broader decoder would also need to
retain within-country confidence variation. The present exact country codec can
already serve as a compact trace format; the selected continuous decoder should
not replace the original readouts where fidelity matters.
