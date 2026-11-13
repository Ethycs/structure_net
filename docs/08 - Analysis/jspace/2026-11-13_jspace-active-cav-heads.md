# Active-position CAVs, task heads, and limits of reduced state

Date: 2026-11-13. Main study: `JSPACE-ACTIVE-CAV-HEADS-01`.
Status: completed; `confirmed=false` for the causal hypothesis.
Classification: `country_CAV_causality_not_established`.

A 30-block Qwen prefix with a linear classifier scored **62/64 (96.9%)** on
fresh four-colour lookup worlds, versus **61/64 (95.3%)** for the 36-block
teacher. Actually removing six blocks reduced unique backbone parameters by
**15.0%**. A follow-up with matched single-label outputs measured **77.04 ms →
64.39 ms**, or **1.196× throughput** and **16.4% lower latency**. This is a
specialized synthetic-task result. It does not establish general language-model
compression or a population accuracy of at least 95%.

The class-conditional polynomial head scored 61/64, one case below the linear
baseline and MLP. No polynomial-specific gain was observed. Full country-CAV
edits had a weak effect, but their J components redirected 0/59 answers at both
tested sites. An exact-copy follow-up found eight concrete failures of reduced
state closure on a natural-versus-hybrid intervention domain.

These results separate three routes: task-specific early exit has a measured
benefit; the tested CAV/J construction has not passed its causal gate; the tested
middle-layer coordinates are insufficient for autonomous dynamics under the
audited context interventions. No new exact Gröbner calculation is claimed here.

## Protocols and evidence

- [Main frozen protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-active-cav-heads-plan.md).
- [Prospective specificity amendment](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-active-cav-heads-amendment-01.md).
- [Closure audit protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-active-closure-audit-plan.md), designed after the main results.
- [Matched latency audit protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-active-latency-audit-plan.md), designed after the original benchmark.
- [Main amended aggregate](../../../data/experiments/jspace_active_cav_heads/20261113_v1/campaign_results_amended.json); original result and aggregate remain alongside it.
- [Closure aggregate](../../../data/experiments/jspace_active_closure_audit/20261113_v1/campaign_results.json).
- [Matched latency aggregate](../../../data/experiments/jspace_active_latency_audit/20261113_v1/campaign_results.json).

Main base fingerprint:
`0358bf60a0efdff52fce29641c07c4857cd917f3b7a1bfbba9a48d138bda816b`.
Amended fingerprint:
`7c1ea29d6feaff0b47a9f54465bbde8fda195a5fe3708f122e6cfdd2b7ac09c2`.
Amended result SHA-256:
`3f1ea6a6ee5c7973244f724335279054bff6aa9363cc98dc20c495958ee22f9d`.
Matched latency result SHA-256:
`76139a955de846f626b0c6ece8022c1f9e4fb25bdd30a085695aec7d72533e8d`.
Each plan archives its own producer, protocol, input and model identities.

## Data, model and controls

The frozen checkpoint is Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`: 36 decoder blocks, width 2048,
3,085,938,688 unique parameters, BF16 backbone, eager attention. Block indices
are zero based. Only the small heads and concept directions were fitted; no
transformer weights were trained.

There are 512 training worlds, 64 development worlds, 64 evaluation worlds and
four separate shakedown worlds. Country×answer cells are exactly balanced;
uniform chance is 25%. Name pools are disjoint, prior study worlds are excluded,
and paired prompts must have aligned 21-token question suffixes. This filters
the task to a single-token-name subset. Anonymous mapping permutations can
recur. Heads train on the 512 source examples; paired source/donor differences
train the concept directions. Development selects hyperparameters and the one
benchmark candidate before evaluation. All 64 source worlds enter head accuracy.

The two sites were chosen using the preceding independent donor-localization
study: block 24's 20 question positions excluding the final token, and block
29's final token. For the main donor, facts stay fixed and the queried person
changes. For the answer-only donor, the queried person and country stay fixed
while two country-to-colour values swap. The latter changes the context itself.

Joint least squares estimates country and colour effects from paired differences
with design rank six; both centered concept banks have rank three. A new
position-aware task-Jacobian dictionary is calculated on 32 balanced training
sources. It uses 94 vocabulary directions, preserving positions, with the
terminal pre-normalization state as target. Sparse nonnegative pursuit uses
budget 25. This small dictionary includes country tokens but none of the four
colour-answer tokens. It is not the paper's full-vocabulary/global estimator.

The main intervention projects an actual donor-minus-source pattern onto the
country-CAV span, then applies shared coefficients to full, J and remainder
components. This is donor-assisted CAV transfer, not a label-only concept swap.
The J/full-CAV squared-norm ratios range from 0.20–1.43% at block 24 and
1.00–2.14% at block 29. These nonorthogonal components are not additive explained
variance fractions. A zero result at their fitted scales does not rule out every
dictionary, direction or intervention strength.

## Causal results and gates

Clean source accuracy was 61/64; query-donor accuracy was 62/64. Both were correct
on **59/64** worlds, which defines the main counterfactual denominator. The
answer-only donor was correct on 64/64, yielding **61** jointly correct controls.
An eligible pair means both unmodified answers were correct; low target success
after editing is a failed redirection, not a desired counterfactual outcome.

| Intervention | Block 24 target redirects /59 | Block 29 target redirects /59 |
| --- | ---: | ---: |
| Dense donor pattern | 55 (93.2%) | 58 (98.3%) |
| Full country CAV | 6 (10.2%) | 11 (18.6%) |
| J component | 0 | 0 |
| Remainder | 6 (10.2%) | 10 (16.9%) |
| Randomized full, same norm | 0 | 0 |
| Randomized J, same norm | 0 | 0 |
| Remainder scaled to J norm | 0 | 0 |
| Colour CAV | 4 (6.8%) | 0 |

Full country edits outperform their random controls in the paired exact tests:
block 24 has six discordant wins, no losses, p=0.03125; block 29 has eleven wins,
no losses, p=0.0009765625. These are weak real effects under the tested transfer,
but neither reaches the predefined 20% success threshold. Block 24 is primary;
block 29 is explanatory. J comparisons have zero discordant successes and p=1.

For every answer-only arm—dense, country and colour—at both sites, the result was
**0/61 donor-target answers and 61/61 original answers retained**. Retention alone
cannot demonstrate country specificity when even the dense answer-only control
does not work. Amendment 01 added this stricter control before evaluation began,
at 20:48:17 UTC; it did not alter fitting, interventions or head selection.

| Declared decision | Result |
| --- | --- |
| Each clean accuracy ≥90%; joint correctness ≥80% | Pass: 95.3%, 96.9%, 92.2% |
| Dense query-donor success ≥80% at both sites | Pass: 93.2%, 98.3% |
| Primary full-country success ≥20%, plus positive random-control advantage | Fail on strength: 10.2%; paired advantage alone passes |
| Original-answer retention ≥80% on answer-only worlds | Numerically passes: 100%; specificity is uninterpretable without the dense control |
| Amendment: answer-only dense success ≥80% | Fail: 0% |
| Conditional J success ≥20%, ≥10 pp advantage over both controls, p≤0.05 | Not supported: all relevant successes are zero; prerequisite CAV gate also fails |
| Selected head ≥95% and at most one extra error versus teacher | Observed pilot gate passes: 62/64 versus 61/64 |

No failed cell was retuned or removed. All 1,408 scheduled intervention forwards
completed. The remainder preserves most of the weak full-CAV effect; this is
evidence about this decomposition, not a theorem that all useful computation
lies outside J-space.

## Task-head comparison

All compact heads share an eight-coordinate supervised encoder: the joint
country/colour span followed by two training PCA residual directions. The
additional full-pattern linear baseline uses all observed coordinates. The
polynomial head uses 16 class-conditional, gradient-normalized low-residual
quadratics; their coefficients are frozen before fitting a readout. It is
inspired by approximate vanishing-ideal classifiers, not a full VI-Net
replication or an exact Gröbner basis.

| Head | Block 24 development /64 | Block 24 evaluation /64 | Block 29 development /64 | Block 29 evaluation /64 | Block 29 numerical bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-pattern linear | 21 | 12 | 63 | **62** | **40,976** |
| Eight-coordinate linear | 18 | 17 | 63 | 61 | 73,936 |
| Ordinary quadratic | 18 | 16 | 62 | 61 | 74,676 |
| Class-conditional polynomial | 18 | 18 | 62 | 61 | 76,976 |
| MLP 8→64→4 | 20 | 15 | 63 | **62** | 77,104 |

Byte counts include all numerical tensors, including normalization, encoders and
polynomial coefficients. The common encoder dominates the compact-head cost;
these are not exactly equal-compute architectures. The full-pattern linear head
is cheaper because it maps directly to four classes. At block 24 its 40,960-input
pattern costs 819,216 numerical bytes; the compact alternatives cost about
1.475–1.478 MB including their encoder.

The frozen selection rule chose block-29 full linear from development ties by
its smaller numerical payload. The primary polynomial-minus-MLP contrast is
**−1/64 = −1.5625 percentage points**; the exploratory paired bootstrap interval
is [−4.6875, 0] pp (2,000 resamples, seed 149). No polynomial advantage is observed.
The selected head's Wilson 95% interval is **89.3–99.1%**. Sixty-four worlds are
insufficient to prove population accuracy ≥95%, superiority or noninferiority.

An explicitly post-hoc audit scored unchanged heads on the two existing donor
variants. The selected linear head scored 62/64 on sources, 64/64 on query
donors and 62/64 on answer-only donors. Teacher scores were 61/64, 62/64 and
64/64. These are correlated variants of 64 worlds, not 192 independent tests;
they did not select a different head. Full per-head results and confusions are
in `jspace_active_closure_audit/20261113_v1/head_audit.json`.

## Actual removal and matched timing

The original prospective benchmark removes blocks 30–35, final norm and the
vocabulary head. Tied input embeddings remain and are counted once. The result
has **2,623,474,688 backbone parameters**, removing **462,464,000**. The linear
head adds 40,976 numerical bytes; its serialized research artifact is 42,580
bytes. This is an executable research recipe, not a deployed general LLM.

The original measurement was **160.01 → 63.87 ms (2.505×)**. However, every
teacher evaluation answer used two generated tokens: a colour and EOS. The
prefix directly returns a label. This ratio combines layer removal, readout
changes and avoiding generation completion; it must not be reported as a pure
layer-compression or generic tok/s gain.

The separate post-hoc audit disabled KV caching and continuation in every arm.
Each returns a class index after one prefill. Both teacher readouts reproduced
all 64 original teacher decisions; the shortened model reproduced all 64 frozen
head predictions. The four-row teacher uses the existing colour vocabulary
weights, providing a stronger comparison than projecting the whole vocabulary.

| Single-label implementation | Median latency | Speed relative to prefix |
| --- | ---: | ---: |
| 36 blocks, full vocabulary argmax | 79.01 ms | 1.227× slower |
| 36 blocks, four existing colour logits | **77.04 ms** | **1.196× slower** |
| 30 blocks, trained linear four-label head | **64.39 ms** | 1.000× |

Each arm has 160 timed calls: first 32 fixed evaluation prompts, batch one,
five repeats, four warmups, synchronized CUDA wall time including tokenization
and host readout. Teacher arms were interleaved in seeded random order; the
prefix followed after physical removal. All timed predictions matched their
frozen references. Peak reserved memory in this audit fell from 6,465,519,616
to 5,496,635,392 bytes, approximately **15.0%**.

Limitations include one RTX 3060, one timing run, warm resident weights,
teacher-then-prefix order, reused short prompts and a finite synthetic output
space. The task can also be solved by symbolic dictionary lookup. This measures
a neural early-exit mechanism, not a competitive production lookup algorithm,
general summarization quality, batch scaling or sustained text generation.

## Exact-copy closure witnesses

The answer-only control motivated a separately frozen, post-hoc replay of the
first four jointly correct evaluation worlds. At each site, direct assignment
made retained source coordinates **bitwise equal** to the natural donor while
leaving every outside position unchanged. Source-boundary replay was also exact.

| Copied coordinates | Next retained state differs from natural donor | First-token decision differs from natural donor |
| --- | ---: | ---: |
| Block 24, question except final token | 4/4 | 4/4 |
| Block 29, final token | 4/4 | 4/4 |
| Block 35, final token, terminal control | Not applicable | 0/4; entire first-logit vector matches exactly |

For example, the natural source answers **yellow** and its changed-facts donor
answers **green**. Copying the donor's block-29 final state into the source
still produces **yellow**; at block 30 the final state differs from the donor.
Copying at block 35 instead reproduces every donor first logit and answers
**green**. Raw natural and patched states/logits are archived for all cases.

For these middle-layer projections, two states share `E(H)` but differ in
`E_next(F(H))`. Thus no deterministic function of only the retained coordinates
can reproduce the observed next coordinates across this intervention-augmented
domain. Surrounding context matters. Hybrid interventions may leave the natural
activation manifold; this is not a proof against approximate compression on
natural inputs, against the paper's full J-space, or against the successful
task head. Decoding an answer and faithfully reproducing the model's internal
computation are different requirements.

## Research decision

Keep the linear early exit as a concrete baseline. The current polynomial head
does not justify extra complexity, and the current J dictionary does not pass
the causal intervention test. Increasing Gröbner degree or timeouts would not
address either limitation.

For interpretability, the next bounded experiment should include actual task
output tokens in the Jacobian dictionary and test concept interventions with
both query and fact changes. For reduced dynamics, include selected source-fact
or attention-context observables and rerun closure before fitting symbolic
transitions. A larger retained state may help; its encoder and attention costs
must count. For practical compression, first replicate the linear exit on new
worlds and prompt formats before attributing value to polynomial structure.

The [research programme](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-groebner-research-programme.md)
now distinguishes measured specialization from unvalidated algebraic compression.
The overall research objective remains open.

## Execution, reproducibility and conformance

Root locked Pixi throughout. GPU stages used the `ncm` environment, Torch
2.4.1+cu121, Transformers 4.53.3, CUDA 12.1, BF16. CPU fitting used the default
environment, Torch 2.5.1+cu121 and NumPy 2.3.1, with CUDA hidden and four threads.
All ten requested heads and all main stages completed; no evaluation retry or
test-based tuning occurred. The main quality split is fresh relative to prior
studies; the two follow-ups explicitly reuse it.

| Stage | Measured work time | Reservation |
| --- | ---: | --- |
| Training/development capture, shakedown and positional derivatives | 331.31 s | `988fc77a5523` |
| CPU CAV/head fitting | 7.45 s | CPU, CUDA hidden |
| Evaluation: 192 natural and 1,408 intervention forwards | 315.01 s | `d1d99bf22e58` |
| Original prefix benchmark | 42.57 s | `d3adf64eef70` |
| Closure replay | 10.08 s | `8bdf3339be22` |
| Matched single-label benchmark | 56.29 s | `9e0e151a0389` |

GPU stages acquired 10,240 MiB through `inference-ctl`, verified the live lease
before CUDA/model access and stayed within allocation. The wrapper released
every lease on exit; the controller subsequently reported no bookings or leases.
No service was restored. These stage times exclude preparation, input hashing,
tests, protocol/report writing and some model identity verification.

Twenty-five focused tests passed across the five test files: numerical/actual
Qwen derivative and prefix controls, CAV decomposition and known-mechanism
recovery, head persistence, specificity interpretation, exact-copy witnesses,
matched teacher logits, and registry scope/provenance rejection. Tests are not
task accuracy observations.

Conformance: main study is a **PREREGISTERED pilot** with a prospective stricter
amendment; closure and matched timing are **EXPLORATORY follow-ups**. No full
paper replication, independent model-seed confirmation, statistical
noninferiority or exact neural polynomial certificate is claimed. The registry
keeps one direct study, `confirmed=false`, and separates selected-head accuracy
from the primary polynomial-minus-MLP contrast and the causal classification.
The follow-ups are linked as reused-evaluation audits, not independent accuracy
replications. [Registry record](../../../data/hypothesis_registry/jspace_active_cav_heads_20261113.json).

Run focused tests with:

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m pytest \
  --confcutdir=tests/structure_net \
  tests/structure_net/test_jspace_active_cav_heads.py \
  tests/structure_net/test_jspace_active_specificity.py \
  tests/structure_net/test_jspace_active_closure_audit.py \
  tests/structure_net/test_jspace_active_latency_audit.py \
  tests/structure_net/test_jspace_active_registry.py -q
```

For an independent main rerun, use an unused output root. The following fixes
the exact stage order; the optional benchmark requires the declared quality gate.
Do not run these against the completed root.

```bash
task_root=data/experiments/jspace_active_cav_heads/NEW_ROOT
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_active_study prepare --output "$task_root"
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_active_specificity freeze --output "$task_root"
inference-ctl lease run --allocation 3060=10240 --minutes 60 --reason 'active CAV training' -- \
  pixi run --locked -e ncm python -m experiments.structure_net.jspace_active_study training --output "$task_root"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  pixi run --locked python -m experiments.structure_net.jspace_active_study fit --output "$task_root"
inference-ctl lease run --allocation 3060=10240 --minutes 60 --reason 'active CAV evaluation' -- \
  pixi run --locked -e ncm python -m experiments.structure_net.jspace_active_study evaluation --output "$task_root"
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_active_study analyze --output "$task_root"
inference-ctl lease run --allocation 3060=10240 --minutes 15 --reason 'qualified task prefix benchmark' -- \
  pixi run --locked -e ncm python -m experiments.structure_net.jspace_active_study benchmark --output "$task_root"
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_active_study finalize --output "$task_root"
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_active_specificity apply --output "$task_root"
```

The two audit modules intentionally reference the original completed main root.
To replay those fixed observations, pass an unused `--output` to their `prepare`
and `capture` stages; also run `head_audit` between them for the closure module.
Their GPU `capture` stages require the same reservation wrapper. Their archived
plans retain the exact case selection and parent-result hashes.
