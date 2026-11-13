# Compact attention concepts in J-space and their travel through Qwen

Date: 2026-11-13. Study: `JSPACE-ATTENTION-TRANSPORT-01`.
Main classification: `attention_concept_origin_not_causally_established`.
Single-checkpoint exploratory pilot; `confirmed=false`.

An eight-direction country J bank gives a perfectly readable country signal from
layer 27 onward on this task, but editing that bank at attention layer 26 changes
the output country in **0/128** held-out worlds. Replacing the full attention
write changes it in **127/128**, and replacing only the donor difference outside
the bank also changes it in **127/128**. Thus there is a working causal route at
this attention site, but the selected country-token J span does not capture it.

The composed quadratic transport model also fails to predict the intervention's
downstream effect. At the predefined layer-29 endpoint its normalized error is
**8.181**, versus **5.799** for linear transport and **1** for predicting no change.
This is a result about these coordinates, calibration, interventions and fitted
maps. It is not a test of all J-space CAVs or a refutation of polynomial/operator
methods. No Gröbner basis, compressed attention replacement or speedup was claimed
or measured in this experiment.

A full-state follow-up makes an important distinction: the perturbation **does
propagate and grow**. By the final block its median L2 magnitude is 19.75 times its
injected magnitude, while only 0.49% of aggregate perturbation energy remains in
the selected layer-specific J span. The compact coordinates lose track of most
of it. This is evidence of numerical perturbation propagation, not evidence that
the intended counterfactual country concept travels successfully.

![Attention edits, readable country signal, and composed transport error](../../../data/experiments/jspace_attention_transport/20261113_v1/attention_transport.png)

The left panel uses the original exact-answer endpoint (64 jointly correct pairs).
The middle and right panels use all 128 worlds. The middle-panel drop in edited
target readability occurs at layer 27's post-attention residual. Accuracy of a
separately trained probe at each node is not itself proof of causal transport.

## Frozen construction

- [Prospective protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-attention-transport-plan.md).
- [Country-identity amendment](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-attention-identity-amendment.md), declared after development and **before evaluation**.
- [Full-state energy audit](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-attention-full-trace-audit.md), declared after main evaluation; reused worlds, not another replication.
- [Frozen plan](../../../data/experiments/jspace_attention_transport/20261113_v1/plan.json), [main result](../../../data/experiments/jspace_attention_transport/20261113_v1/result.json), [trace](../../../data/experiments/jspace_attention_transport/20261113_v1/trace.json), [identity audit](../../../data/experiments/jspace_attention_transport/20261113_v1/identity_audit.json).

Main fingerprint:
`d6120f9e77dcdb22c6410123798d6b6e2b91b50aa487dc3126352745377dfc25`.
Result SHA256:
`659a5b61a5d111625859b734fc8d54c6cfc58adcd16c9576415da9736fdb6f56`.

The checkpoint is Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, 36 blocks, width 2,048,
3,085,938,688 unique parameters, BF16, eager attention. All weights stay frozen.
All layer numbers are zero based. Read actual attention outputs after `o_proj`,
post-attention residuals before the MLP, and block outputs before the next norm.

The J bank contains the bare and leading-space tokens for China, France, Germany
and Russia. Its VJPs average 64 independent generic WikiText contexts, 128 tokens
each, with the final pre-normalization block output as target. These are eight
selected semantic-token readouts with global positional averaging, not a full
vocabulary sparse nonnegative J-space or a supervised country CAV decomposition.
Every tested attention bank has numerical rank eight. At layer 26 its condition
number is approximately 16.45; the failure is not an empty or rank-deficient bank.

The eight-dimensional span is **per token**. The edit preserves the complementary
attention state. It does not replace a 2,048-dimensional attention computation with
eight numbers. One FP32 basis costs 65,536 bytes; one token's eight FP32 coordinates
cost 32 bytes. The transport model uses 16 pooled observations: question mean
excluding the last token, and the last token. All model weights, other coordinates,
token positions and subsequent layers remain present during intervention.

## Data and answer-form issue

Fresh assignment worlds map four names to the four countries. The model is asked
for either the country's capital or nationality adjective. There are 384 training,
24 development, 128 evaluation and four separate systems-shakedown worlds, with
disjoint name pools and unique named assignments. A donor changes the queried
person while retaining the assignment prefix and attribute. Source and donor
token lengths and question positions must align before outcomes are observed.
Donors are paired observations, not extra independent worlds.

Training/development omit China/nationality and Russia/capital queries on both
sides. Evaluation has 16 examples per country/attribute cell: 96 seen combinations
and 32 held-out combinations. These facts were already in the pretrained model;
the holdout tests the fitted coordinate models, not new world knowledge.

On development, every nationality source returned the correct country name,
such as **France**, instead of the expected adjective **French**. The frozen exact
score was retained. Before evaluation, a secondary finite dictionary was declared
that maps each country's name, capital and adjective to its country identity.
It deliberately ignores whether the requested attribute was executed correctly.
No LLM judge or retrospective synonym additions were used.

| Held-out natural endpoint | Source | Donor |
| --- | ---: | ---: |
| Original exact task answer | 65/128 (50.8%) | 64/128 (50.0%) |
| Secondary country identity | 128/128 (100%) | 128/128 (100%) |

All 64 capital sources were exact-correct; only one of 64 nationality sources used
the expected adjective. Joint exact correctness was 64/128; joint country-identity
correctness was 128/128. The original >=90% task-competence gate therefore fails.
The identity result's Wilson 95% interval is 97.1–100%; it is not full task accuracy.

## Origin selection and causal results

Development compared attention layers 11,17,23,24,25,26,27,28,29. Only layer 26
passed the >=50% dense replacement gate: 12/12 eligible exact-answer pairs redirected.
Layers 17 and 27 each redirected 3/12, the others 0/12. Both compact methods had
0/12 at every candidate. The frozen tie rule selected layer 26 projected donor.
The alternative swaps the coefficients for the source/target countries' lexical
rows. Random controls signed/permuted the J update while preserving its norm/rank.

| Held-out intervention | Exact target /64 eligible | Country target /128 eligible |
| --- | ---: | ---: |
| Full donor attention write | 63 (98.4%) | 127 (99.2%) |
| Donor difference projected into J span | 0 | 0 |
| Matched random J update | 0 | 0 |
| Donor difference outside J span | 63 (98.4%) | 127 (99.2%) |
| Country lexical coefficient swap | 0 | 0 |

The remainder complements the projected donor update, not the coefficient swap.
J-minus-random is zero with paired exact p=1. The country-target rate of 0/128 has
Wilson interval 0–2.91%; the dense rate's interval is 95.71–99.86%. These intervals
describe this synthetic sample, not a general model capability guarantee.

An example: source France answers **Paris**, donor Germany answers **Berlin**.
The J edit and coefficient swap still answer **Paris**; the dense and remainder
edits answer **Berlin**. The same pattern occurs for Russia→France and China→Germany
among the first three eligible examples in frozen evaluation order.

The median applied J update is 0.939% of the original whole-prefill attention
output L2 norm. Its median relative BF16 reconstruction error is 6.89%; median
outside-span L2 fraction is 6.81%, about 0.46% of squared update energy. These are
measured small BF16 deviations, not lossless rank-eight edits. They do not explain
away the much stronger full/remainder interventions, but strength or basis choice
could matter; no strength sweep was conducted.
The bank captures only **0.224% of donor-difference squared norm** at the median
world (0.232% pooled). The remainder therefore retains almost all of that update's
energy. Its high success is not a comparison at matched intervention strength.

## How the measured signal travels

Fit country probes separately at each boundary using training worlds only. At the
selected attention write the probe is weak on evaluation: 54/128 (42.2%) correct.
Therefore the initial eight rows are only candidate country directions, not an
already validated semantic bottleneck. The country becomes perfectly readable in
these coordinates after the next attention update and remains so to the output.

| Boundary | Natural country probe /128 | Edited target-country probe /128 | Standardized edit RMS |
| --- | ---: | ---: | ---: |
| Layer 26 attention write | 54 | 56 | — |
| Layer 26 post-attention residual | 65 | 46 | 0.72279 |
| Layer 26 post-MLP residual | 48 | 51 | 0.60191 |
| Layer 27 post-attention residual | 128 | 0 | 0.07708 |
| Layer 29 post-MLP residual | 128 | 0 | 0.06218 |
| Layer 35 post-MLP residual | 128 | 0 | 0.03643 |

At layer 26, compositional holdout is especially poor: the post-attention probe
gets 1/32 held-out combinations versus 64/96 seen combinations; after its MLP,
0/32 versus 48/96. From layer 27 onward both groups are perfect (32/32 and 96/96).
This sharp transition gives a concrete place to investigate concept construction
and contextual input. It does not show that an initial autonomous country code
has been transported intact. An untouched residual/context path remains available.

The standardized measured perturbation falls about 87% across layer 27's attention
update (0.60191→0.07708). This is a decrease in layer-specific pooled coordinates,
not yet a claim that the full activation perturbation disappears; see the full-state
audit below.

## Composed linear and quadratic transport

The starting state is the selected **post-attention residual**, including its skip
contribution. Fit maps for all 19 successive edges through 20 residual boundaries.
Inputs are 16 standardized coordinates and the known requested-attribute bit.
There are 768 training observations from 384 worlds and 48 development observations
from 24 worlds. Ridge penalties are selected on development only. Model weight
training is zero; CPU map/probe fitting took 0.265 seconds.

Clean and edited rollouts start from their respective observed layer-26 residual
coordinates, then receive no later actual activations. The main endpoint predicts
the **difference between edited and clean trajectories**, rather than receiving
credit for predicting a large unchanged background state.

| Primary endpoint: layer 29 block output | Linear | Quadratic | Carry initial change |
| --- | ---: | ---: | ---: |
| Intervention-change normalized error | 5.799 | 8.181 | 10.852 |
| Natural-state normalized error | 0.597 | 0.617 | — |
| Country decoded from predicted clean state /128 | 57 | 61 | — |
| Target country decoded from predicted edited state /128 | 48 | 46 | — |

Actual clean and edited states at this destination both decode the original
country in 128/128 cases. Predicted edited target counts are therefore erroneous
predictions, not successful interventions. Predicting a zero intervention change
has normalized error exactly 1 by definition; both learned rollouts are worse.
Neither reaches the predeclared <0.75 threshold or a quadratic advantage over
linear transport. No trajectories hit the declared nonfinite/1e4 divergence limit.
The first local MLP edge is easier (linear 0.504, quadratic 0.514); the major failure
appears at the following attention update. Polynomial transport is being tested
here without requiring a nonempty vanishing ideal, but this fitted degree-two
action does not establish closure of the selected pooled state.

## Full-state perturbation audit

The additional audit reused every evaluation world, with 384 generation replays
(source, donor, compact edit). All generated token IDs matched the original runs
exactly. Full prefill activations were processed one world at a time. The audit
was frozen after main analysis, changed no fitting/selection, and is a diagnostic
reuse rather than additional independent evidence.

| Boundary | Aggregate full perturbation L2 | Energy in that node's J span | Median per-world L2 / injected L2 |
| --- | ---: | ---: | ---: |
| Layer 26 attention injection | 16.641 | 99.585% | 1.000 |
| Layer 26 post-MLP | 21.178 | 49.347% | 1.299 |
| Layer 27 post-attention | 28.437 | 15.424% | 1.778 |
| Layer 29 post-MLP | 61.875 | 2.125% | 3.820 |
| Layer 30 post-MLP | 647.246 | 0.103% | 22.156 |
| Layer 35 post-MLP | 324.305 | 0.491% | 19.750 |

The full perturbation is larger than the injected one at every listed downstream
boundary in **all 128 worlds**. All boundaries before layer 26 have exactly zero
paired difference. Final-state relative L2 difference is 1.21% of the unmodified
whole-prefill state; amplification does not imply a large relative state change.
The sharp layer-30 growth is not just one outlier: the median growth is 22.16×,
although the largest world contributes 17.4% of aggregate squared difference there.

Thus the reduced-coordinate attenuation cannot be called full-state erasure.
Much of the response develops outside the selected eight-dimensional charts.
Nevertheless no output country changes. This does not prove that a semantic
country code, a polynomial relation, or useful information is preserved in those
remaining directions. BF16 arithmetic and nonlinear layers both contribute to
the measured dynamics. A transported readout, rather than fixed narrow charts or
full-state norm alone, remains necessary to establish semantic travel.

[Per-world energy records](../../../data/experiments/jspace_attention_transport/20261113_v1/full_trace/records.json),
[aggregate full trace](../../../data/experiments/jspace_attention_transport/20261113_v1/full_trace/result.json),
[lease/stage manifest](../../../data/experiments/jspace_attention_transport/20261113_v1/full_trace_manifest.json).

## Interpretation and next experiment

This provides useful localization: layer 26 has a strong causal donor route, and
layer 27's attention update is where our small pooled state stops predicting the
edit and the original country becomes reliably readable. The country-token bank
is an insufficient starting concept at the tested strength and construction.
The full-state audit also shows why following only its projected norm would give
the wrong account of what travels: the perturbation expands outside that chart.

The next discriminating test is to learn country contrasts from the successful
layer-26 attention differences, represent them sparsely in a broader J dictionary,
and compare matched-rank J, PCA and full-CAV interventions. Country directions
should transfer between requested attributes. Then fit contextual transport using
the demonstrated causal concept and test whether additional retained context fixes
the layer-27 failure. That tests the user's J-space→CAV→transport construction more
directly than assuming eight token-associated rows already constitute the CAV.

The main limitations are one 3B checkpoint, four countries, one prompt family,
selected-token/global-average Jacobians, a final-block target, late selected origin,
small perturbations, pooled context, no PCA comparator and no compression benchmark.
Neither readability nor a failed finite-degree fit decides whether a different
coordinate algebra can support interpretation or compression.

## Validation and reproduction

Fifteen focused tests passed: actual-Qwen attention VJP finite differences, exact
attention/addition gradient identity, near-machine-precision linear `o_proj`
covector pullback, projection/remainder identities, hook cleanup, known quadratic
action recovery, divergence accounting, holdout leakage rejection, and registry
scope checks. Large-Qwen shakedown also passed exact identity-patch token/score
readback, exact attention/addition VJP equality, and dictionary persistence.

All numerical stages use the root locked Pixi environment. PyArrow 19.0.1 was
added to read the cached calibration data before freezing the plan. The initial
missing-dependency prepare attempt created no result root or model observations.
The model/Jacobian stages use `-e ncm`; CPU stages hide CUDA. Every GPU job uses a
live inference-ctl lease and guards before CUDA initialization. No services were
restarted. Main GPU stage elapsed times: discovery 323.2 s, training capture 192.0 s,
evaluation 224.0 s, excluding checkpoint hashing/startup. Peak reserved PyTorch
memory was 7,241,465,856 bytes, below the 10,240 MiB reservation.
The full-state audit added 110.4 s. All four leases were released; the controller
reported no bookings or leases afterward.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 pixi run --locked python -m pytest \
  --confcutdir=tests/structure_net tests/structure_net/test_jspace_attention_transport.py \
  tests/structure_net/test_jspace_attention_registry.py -q

# Use a new append-only --output root for another collection.
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_attention_study prepare --output NEW_ROOT
inference-ctl lease run --allocation 3060=10240 --minutes 60 --reason 'attention transport discovery' -- \
  pixi run --locked -e ncm python -m experiments.structure_net.jspace_attention_study discovery --output NEW_ROOT
# Then training, CPU fit, reserved evaluation, CPU analyze, as specified in the protocol.
```

Raw artifacts, archived producers, strict JSON, stage manifests, per-forward events,
and generation records remain under
`data/experiments/jspace_attention_transport/20261113_v1`.
The registry records unchanged-teacher exact task accuracy separately from the
selected J edit's target-answer rate, with one study of evidence and confirmed=false.
[Registry record](../../../data/hypothesis_registry/jspace_attention_transport_20261113.json).
