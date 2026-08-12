# TinyLLM C3 posterior-estimator training-license audit

**Status:** NO-GO — UNCERTAINTY INTERFACE READY; LEARNABLE OBSERVATION LAW ABSENT

**Date:** 2026-08-11

**Evidence role:** repository/frontier audit; not a measured experiment

## Decision

Do not launch posterior-estimator or TinyLLM training.

The exact typed output is now known: a future estimator would emit the complex
C3 posterior-holonomy moment `m(O)`. The repository does not currently contain
an observation stream from which a nontrivial per-example `m(O)` can be learned
and evaluated.

The available connection studies contain:

- exact observed edge connections;
- no-connection, shuffled-connection, wrong-sign, and erased-edge controls;
- analytically declared independent symmetric error;
- deterministic posterior-simplex values used to verify the soft interface.

They do not contain repeated noisy measurements of the same edge, observed
reliability or quality evidence, an unknown connection-error mechanism with
train/test instances, or real calibration outcomes. Creating an arbitrary
feature-to-noise mapping now would test a mapping chosen to make a model useful,
not a hypothesis supplied by the task or data.

## Training-license requirements

| Requirement | Current evidence | Status |
| --- | --- | --- |
| exact covariant output type | complex moment `m=sum q_e exp(-2pi i e/3)` | satisfied |
| frozen consumer path | hard posterior average equals soft-carrier injection in `10/10` cells | satisfied |
| target identifiability under uncertainty | erasure is impossible; known symmetric noise is probabilistic | only declared cases closed |
| observed evidence for per-example posterior variation | none in current connection datasets | missing |
| unknown but learnable error law | no generator or real dataset declares one | missing |
| oracle posterior-moment arm | algebra available once a law exists | design-ready |
| fixed/global/adaptive controls | specified, but cannot be instantiated without observations | design-ready |
| Bayes-relative success gate | no uncertainty law or Bayes risk distribution exists | missing |
| disjoint train/composition/extrapolation streams | absent for an estimator task | missing |

The missing items are outcome-relevant task definitions, not implementation
details. They cannot be filled by choosing optimizer settings or by replaying
the existing checkpoints again.

## Minimum acceptable new data contract

A future proposal must provide, before any training outcome is inspected:

1. **Observed uncertainty evidence `O`.** For example, repeated edge reads,
   sensor quality packets, redundant paths, or a temporal constraint visible at
   inference. Latent clean connection values alone do not count.
2. **A declared error mechanism.** State what is fixed, what varies per example,
   and what is unknown to the estimator. Separate physical error from label or
   target corruption.
3. **Identifiability evidence.** Show analytically or by exact collision search
   that `P(E|O)` is not constant and that the desired posterior moment descends
   through the observation relation.
4. **An oracle ceiling.** Compute the true `m(O)` from generator-only latent
   variables and evaluate the achievable Bayes-relative task gate.
5. **Economic controls.** Include the fixed known-law moment, one global learned
   posterior, and the simplest observation-adaptive estimator justified by the
   data before a sequence model.
6. **Specificity controls.** Shuffle only the reliability evidence, preserve
   marginal error rates, and include a target-changing control.
7. **Shift definitions.** Declare which part of the error law changes under
   composition and extrapolation; clean-data thresholds must not be reused when
   the oracle Bayes ceiling cannot meet them.

## Preferred evidence sources

In order of scientific value:

1. real repeated sensor/connection measurements with calibration outcomes;
2. an externally motivated corruption mechanism from the intended deployment;
3. a synthetic mechanism derived from a physical sensor model and fixed before
   estimator design;
4. only as a methods benchmark, a deliberately synthetic heteroscedastic law
   with its artificial status explicit.

For repeated categorical edge reads, a fixed Bayesian or empirical-Bayes
aggregator is the first baseline. For scalar quality packets, monotone or
low-order calibration is the first baseline. TinyLLM is considered only if a
sequence-dependent residual remains after those controls and if patching the
estimated `m(O)` causally improves a frozen continuation.

## Repository audit

The audit searched experiment code, source modules, design/architecture/
roadmap documentation, and experiment JSON for connection reliability,
confidence, repeated edge observations, noisy/missing/unknown connection laws,
and holonomy data. The only connection/holonomy experiment families present
were the exact relational preflight, function-class and acquisition campaigns,
the artifact readout audit, the observation-identifiability audit, and the
posterior-interface audit.

Reproduction:

```bash
rg -n -i \
  "reliab|quality score|edge observation|connection.*noise|noisy.*connection|connection.*error|unknown.*connection|repeated.*edge|edge.*repeat|holonomy" \
  experiments src docs data/experiments

find data/experiments -maxdepth 2 -type d \
  \( -iname '*connection*' -o -iname '*holonomy*' \) | sort
```

## External candidate audit

The nearby `New Graph Architecture` workspace and the archived StructureNet
notes were also inspected because they contain fields named `confidence`,
`uncertainty`, `posterior_summary`, and `edge_traversed`. They do not satisfy
the connection-observation contract.

| Candidate field | Actual provenance | Why it is not C3 reliability evidence |
| --- | --- | --- |
| `confidence` | probability assigned by the same learned FSM-state classifier | endogenous model output; not an observed sensor-quality variable |
| `energy_breakdown.uncertainty` | a deterministic transform of classifier probability or prototype distance | inherits the prediction being evaluated and has no latent connection-error label |
| `posterior_summary` | aggregate Beta parameters for an FSM legality mask learned from labeled state-transition counts | global structural mask; not a per-example posterior over C3 edge error |
| `edge_traversed` | transition between consecutive quantized predicted output tuples in the product graph | records model trajectory, not a repeated measurement of a physical or latent connection |
| Dyck corruption indicator | deliberately mismatched close-token construction used to test FSM masking | token/label stressor with no C3 gauge action or holonomy target |
| E38 calibrated controller probability | belief-controller probability calibrated on separate trajectories | useful POMDP calibration evidence, but its latent variable is protocol state rather than connection error |
| E39 uncertain support | a perfect rule verifier labels one selected unknown FSM edge | deterministic support discovery with no noisy repeated read or posterior-holonomy estimation problem |
| E49 observation drift | a preregistered, globally known interpolation of POMDP emission matrices | real misspecification test, but no per-example quality variable and no C3 connection action |

The source makes the dependency explicit: the JSON experiment defines
`confidence = p_masked[argmax_state]`, constructs uncertainty from the same
prediction distribution or chosen-prototype distance, and creates
`edge_traversed` only after adding consecutive predicted tuples to the product
graph. Its Beta arrays model whether FSM state-to-state transitions are legal.
The accompanying research log also characterizes sigma as a structural FSM
property whose prediction-side value overlaps the trained classifier margin.

The later uncertainty experiments do not change the disposition. E38 isolates
calibration of a protocol-state belief. E39 resolves unknown support by calling
a perfect finite-rule verifier, so the relevant edge label becomes known on
the first queried verdict. E49 provides hidden-state evaluation labels and a
declared observation-noise sweep, but the drift severity is fixed globally per
evaluation cell and the adapted bound is explicitly handed the true drifted
distribution. These are legitimate experiments in their own state/action
spaces; none provides observations `O` paired with latent C3 connection error
`E` from which a nonconstant `q(E|O)` could be estimated.

The archived notes contain architecture-growth proposals and retrospective
MNIST claims, but no repeated connection observations, reliability packets,
clean/error pairs, or C3-compatible uncertainty labels. Reusing either source
would require inventing a map from graph-classifier confidence to holonomy
error. That would be a new synthetic task selected after the desired estimator
was known, not independent evidence for it.

This conclusion does not say the graph traces are scientifically useless. A
separate study could ask whether a posterior over categorical FSM transitions
improves graph control. It would need its own state/action identifiability and
calibration contract; it cannot serve as a surrogate C3 connection dataset.

External audit reproduction:

```bash
rg -n -i -C 4 \
  "uncertainty|confidence|edge_traversed|corrupt|posterior|reliab|noise" \
  "/home/rabbit/semantic_machine/New Graph Architecture/src/nga/exp/e9_dyck.py" \
  "/home/rabbit/semantic_machine/New Graph Architecture/src/nga/exp/e13_listops.py" \
  "/home/rabbit/semantic_machine/New Graph Architecture/src/nga/exp/e19_json.py" \
  "/home/rabbit/semantic_machine/New Graph Architecture/src/nga/exp/e38_calibrated_selective_control.py" \
  "/home/rabbit/semantic_machine/New Graph Architecture/src/nga/exp/e39_uncertain_support.py" \
  "/home/rabbit/semantic_machine/New Graph Architecture/src/nga/exp/e49_robustness_frontier.py" \
  "/home/rabbit/semantic_machine/New Graph Architecture/research_log.md"

rg -n -i \
  "uncertainty|confidence|noise|corrupt|reliab|posterior|connection|gauge|holonomy" \
  archive/old_research/notes/*.md
```

## Resume point

Resume only when an observation law or dataset satisfying the contract above is
available. Start with an identifiability/oracle/control preflight. Do not start
with model selection, and do not reinterpret the exact posterior interface as
evidence that an estimator will pay rent.
