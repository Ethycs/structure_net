# TinyLLM independent-analysis packet

**Status:** READY FOR INDEPENDENT AGENT REVIEW

**Date:** 2026-08-11

**Evidence role:** curated lab-report and review handoff; no new experimental
observation

**Evidence commit:** `f0024c5958a34b1502f49ebbf0caf0de8f9a43b2`

**DVC root:** `125447de251dda33ab40e06d680cbb17.dir`

## Review mandate

This packet asks an independent agent to decide what the TinyLLM experiments
actually establish. It is intentionally narrower than the full research
archive and includes positive, negative, partial, algebraic, and underpowered
results.

The reviewer should not assume that the program's synthesis is correct. In
particular, the reviewer must independently test these candidate claims:

1. fixing the observation gauge is necessary but not sufficient for a stable
   task quotient;
2. structured front ends expose a task-sufficient quotient before attention;
3. raw TinyLLMs synthesize a quotient through charged, distributed nonlinear
   interactions in the stable C2 scope, without one universal C3 or Morse
   normal form;
4. exact equivariance restricts a function class but does not guarantee
   population-stable acquisition;
5. fixed group operators often remove the proposed learned-model job;
6. total C3 holonomy and one posterior character moment define the information
   boundary for the relational connection task.

The requested output is an adversarial scientific review, not a summary.

## Evidence dependency map

```text
observation gauge
  -> calibrated identifiability
  -> structured front-end causal closure

raw branch-bearing cover
  -> exact Reynolds synthesis localization
  -> charged-character necessity
  -> head and subspace distribution
  -> Morse normal-form falsifier

C3 temporal redundancy
  -> fixed-operator ceiling
  -> no same-law TinyLLM job

relational connection task
  -> exact typed function class
  -> failed population acquisition
  -> total-holonomy / erasure / known-noise boundary
  -> exact soft posterior interface
```

Arrows indicate logical dependency, not independent replication. Several
studies reuse checkpoint lineages, so the reviewer must not count every row as
an independent model-population result.

## Two-pass review protocol

### Pass 1 — reconstruct the evidence without our interpretation

For each core study:

1. read the preregistration;
2. inspect the runner's gate implementation and control construction;
3. inspect the raw aggregate/result JSON;
4. recompute the seed-level and campaign-level verdict;
5. record deviations, reuse, exclusions, invalid controls, and shared
   checkpoint ancestry;
6. only then read the measured report.

Do not use the meta-hypothesis `confirmed` flag as evidence. It is a stored
disposition that must agree with the raw result, not an independent vote.

### Pass 2 — audit interpretation and novelty

After reconstructing the gates:

1. compare the measured report with the raw verdict;
2. determine whether secondary results improperly rescue a failed primary;
3. separate exact finite-group algebra from learned-model evidence;
4. distinguish task-relative causal sufficiency from full-state quotienting;
5. identify which claims are restricted to C2, C3, three stable checkpoints,
   five seeds, synthetic data, or outcome-directed analyses;
6. read the program synthesis only as an object to challenge:
   [publication-claim audit](2026-08-11_tinyllm-publication-claim-audit.md).

No new training is needed for this review. If a claim cannot be resolved from
the stored artifacts, report the missing evidence rather than generating a new
outcome.

## Core experiment set

### Experiment 1 — calibrated identifiability

**Question:** Does an observed orientation reference repair the gauge
obstruction, and does a symmetry-respecting front end use that information
more reliably than raw TinyLLM?

**Design:** Three matched d8/N3 arms, five seeds, composition and outside-range
extrapolation, joint cosine-retention and conditional-branch gates.

**Recorded outcome:** Analytic and learned structured front ends pass the full
front-end and full-depth gate in `5/5` seeds on both shifts. The raw calibrated
arm passes `0/5`.

**Reviewer challenge:** Verify that the calibration reference distinguishes
the relevant gauge orbit without leaking phase or the target. Decide whether
the raw-arm failure supports an architectural claim or only an optimization
claim.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-calibrated-identifiability-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_calibrated_frontend_causal.py)
- Raw: `data/experiments/tinyllm_calibrated_frontend_causal/20260806_d8_preregistered/campaign_results.json`
- Raw SHA-256: `80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501`
- [Measured report](2026-08-06_tinyllm-calibrated-identifiability-causal.md)
- Meta: `data/meta_hypotheses/tinyllm-calibrated-reference-stable-cosine-quotient-v1.json`

### Experiment 2 — structured front-end causal closure

**Question:** Can the exact task-fiber barycenter replace the natural
activation before attention while the frozen continuation preserves the task?

**Design:** Ten retained structured systems, two shifts, four residual cuts,
exact orbit-average intervention, semantic-barycenter shuffle control.

**Recorded outcome:** Both structured arms pass `5/5` at the front-end cut and
all later cuts on both shifts. Semantic shuffles pass `0/5`. Natural sheet
differences remain nonzero.

**Reviewer challenge:** Confirm that paired sheets truly share the task target,
that the shuffle preserves the intended marginal, and that the task gate—not a
generic cross-entropy improvement—drives the result. Decide whether “quotient”
is justified only task-relatively.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-calibrated-frontend-causal-closure-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_calibrated_frontend_causal_closure.py)
- Raw: `data/experiments/tinyllm_calibrated_frontend_causal_closure/20260810_d15_preregistered/campaign_results.json`
- Raw SHA-256: `1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14`
- [Measured report](2026-08-10_tinyllm-calibrated-frontend-causal-closure.md)
- Meta: `data/meta_hypotheses/tinyllm-calibrated-frontend-causal-closure-v1.json`

### Experiment 3 — Reynolds character-coupling synthesis

**Question:** Does the first sublayer whose actual next barycenter is
task-sufficient while the propagated barycenter fails identify synthesis of an
invariant from charged cover modes?

**Design:** Retained d6 degree-two and degree-three ladders, five seeds, 64
exact orbits, twelve sublayers, both shifts, propagated-versus-actual
barycenter patches, matched controls, quadratic approximation.

**Recorded outcome:** Exact localization and the full causal regime are stable
in `5/5` C2 seeds. C3 localization reaches only `3/5`, shift-stable regimes
`2/5`, and quadratic sufficiency `0/5`. The full hypothesis is not confirmed.

**Reviewer challenge:** Reconstruct the four-regime classification and verify
that adding the exact Reynolds defect, rather than generic displacement,
causes the C2 transition. Reject any prose that generalizes the C2 result to a
universal C3 or low-order Taylor law.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-reynolds-character-coupling-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_reynolds_character_coupling.py)
- Raw: `data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered/campaign_results.json`
- Raw SHA-256: `a7ccda0d8a36a5c96de96045a32400deaf3cdbdb0856d969164df8d6a455495b`
- [Measured report](2026-08-06_tinyllm-reynolds-character-coupling.md)
- Meta: `data/meta_hypotheses/tinyllm-reynolds-character-coupling-synthesis-v1.json`

### Experiment 4 — fixed-operator rent control

**Question:** Does the noiseless C3 temporal task leave any useful work for a
learned continuation after a fixed all-frame group statistic is supplied?

**Design:** Five fresh 4,096-example replicates, two shifts, last-increment
baseline versus one fixed unweighted circular mean, fixed interval decoder, no
training.

**Recorded outcome:** The fixed all-frame statistic wins in every seed and
shift, roughly halves temporal RMSE, and reaches mean exact-bin accuracies
`.9778/.9804`.

**Reviewer challenge:** Verify that the operator is target-free, fixed before
the result, and matched to information actually observable at inference.
Determine whether it closes only this generator or supports a broader claim
about model value.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-fixed-operator-ceiling-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_c3_temporal_fixed_operator_ceiling.py)
- Raw: `data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/20260811_preregistered/result.json`
- Raw SHA-256: `9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a`
- [Measured report](2026-08-11_tinyllm-c3-temporal-fixed-operator-ceiling.md)
- Meta: `data/meta_hypotheses/tinyllm-c3-temporal-fixed-operator-ceiling-v1.json`

### Experiment 5 — exact relational class versus acquisition

**Question:** Does random-initialized gradient training reliably acquire a
known connection-conditioned relation when the architecture contains the exact
solution at every allowed symmetry state?

**Design:** A 187-parameter typed C3 module, five fresh seeds, 2,400 matched
AdamW steps, analytic, true, no-connection, connection-shuffled, and
target-shuffled arms, two shifts.

**Recorded outcome:** The analytic solution passes `5/5`; learned true passes
`1/5`; every learned control passes `0/5`. A post-outcome frozen affine readout
reaches `4/5`, while one seed remains in a wrong winding sector. The corrective
readout does not rescue the primary.

**Reviewer challenge:** Verify the exact function-class witness, gradient and
checkpoint lifecycle, shared data/minibatches, and joint gates. Decide whether
failure is best described as optimization, winding-sector acquisition,
calibration, or a mixture—and keep the post-hoc readout evidentially separate.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-relational-connection-acquisition-preregistration.md)
- [Primary runner](../../experiments/structure_net/tinyllm_c3_relational_connection_acquisition.py)
- [Corrective runner](../../experiments/structure_net/tinyllm_c3_relational_connection_readout_audit.py)
- Raw: `data/experiments/tinyllm_c3_relational_connection_acquisition/20260811_preregistered/campaign_results.json`
- Raw SHA-256: `b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a`
- Corrective raw: `data/experiments/tinyllm_c3_relational_connection_readout_audit/20260811_artifact_audit/result.json`
- [Measured report](2026-08-11_tinyllm-c3-relational-connection-acquisition.md)
- Meta: `data/meta_hypotheses/tinyllm-c3-relational-connection-acquisition-v1.json`

### Experiment 6 — connection observation boundary

**Question:** Can missing, partial, or known-noisy connection observations
create legitimate learned value under the current independent-phase generator?

**Design:** Exact canonicalization to total C3 holonomy, all seven single-edge
erasures, exhaustive `3^7` known-noise enumeration, analytic Bayes ceiling,
five frozen source modules.

**Recorded outcome:** Total holonomy changes predictions by exactly zero. Each
single-edge erasure admits a same-observation target collision separated by
`1.5`. Known symmetric noise matches
`lambda(p)=(1-3p/2)^7`; the current joint scalar gate tolerates only about
`9.525e-6` per-edge error.

**Reviewer challenge:** Independently construct at least one erasure collision,
check the general seven-edge construction, and derive the noise attenuation.
Decide which conclusions are theorem-level and which depend on the inherited
task threshold.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-connection-observation-identifiability-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_c3_connection_observation_identifiability.py)
- Raw: `data/experiments/tinyllm_c3_connection_observation_identifiability/20260811_preregistered/result.json`
- Raw SHA-256: `23a8989e820d73d1b72c8abaf3f5b4fde0664b854fb03a17ff6df3c5e2d24c7c`
- [Measured report](2026-08-11_tinyllm-c3-connection-observation-identifiability.md)
- Meta: `data/meta_hypotheses/tinyllm-c3-connection-observation-identifiability-v1.json`

## Supporting mechanism and falsification set

These studies should be used to bound the core claims, not counted as five
additional independent confirmations.

### A. Charged-character necessity

Removing nontrivial deck characters destroys sufficiency and restoring the
exact carrier repairs it in `5/5` C2 and C3 seeds under both shifts, but the
universal finite-C3 phase phenotype fails its population gate.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-irrep-fusion-ablation-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_irrep_fusion_ablation.py)
- Raw: `data/experiments/tinyllm_irrep_fusion_ablation/20260806_d6_preregistered/campaign_results.json`
- Raw SHA-256: `bc1b60ea2fbda513a042c533d33921ced13a64005958ebd416ca7690bd890065`
- [Measured report](2026-08-06_tinyllm-irrep-fusion-ablation.md)

### B. Distributed attention-head synthesis

Exhaustive enumeration of 64 subsets selects four, four, and five heads in the
three cross-cohort-stable C2 checkpoints. The sparse one/two-head claim fails;
only two of three selected subsets pass the strict held-out causal endpoint.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-c2-attention-head-decomposition-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_c2_attention_head_decomposition.py)
- Raw: `data/experiments/tinyllm_c2_attention_head_decomposition/20260806_d6_preregistered/campaign_results.json`
- Raw SHA-256: `8d44328af6b4b44d3753305c4544e05d7024ad3685ea8c8a0c27c2f7adff2401`
- [Measured report](2026-08-06_tinyllm-c2-attention-head-decomposition.md)

### C. Vector-valued defect rank

Source-fitted rank one fails all three stable C2 checkpoints. The smallest
passing preregistered dyadic ranks are `2`, `8`, and `4`; the experiment is
explicitly underpowered and checkpoint-local.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-defect-subspace-rank-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_defect_subspace_rank.py)
- Raw: `data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered/campaign_results.json`
- Raw SHA-256: `b6586abd878a70819b8c7c921126c9cb86319f414886f7fa322d93535a05a324`
- [Measured report](2026-08-06_tinyllm-defect-subspace-rank.md)

### D. Morse–Cerf universal-form falsifier

Near-front landscape events are frequent, but mature basins, controls, and
event ordering do not define one stable degree-independent quotient normal
form. The reviewer should verify that the null is not merely a resolution
failure and should inspect the structurally invalid autonomous-closure Morse
gate discussed in the report.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-morse-cerf-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_morse_cerf.py)
- Raw: `data/experiments/tinyllm_morse_cerf/20260806_d6_preregistered/campaign_results.json`
- Raw SHA-256: `2f4987e7c57d76dd5cb9c7f84ab38df1ec7e3735e7fc2ac0e862ec9e1a760185`
- [Measured report](2026-08-06_tinyllm-morse-cerf.md)

### E. Exact soft posterior interface

For arbitrary uncertainty over total C3 connection error, one complex
character moment reconstructs the three-state posterior and gives the Bayes
cosine and risk. The implementation checks 2,080 simplex points, 4,259,840
posterior/phase cells, three coordinate shifts, and ten frozen replays; maximum
replay error is `2.38e-7`. No estimator or uncertain observation law is tested.

- [Preregistration](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-posterior-holonomy-interface-preregistration.md)
- [Runner](../../experiments/structure_net/tinyllm_c3_posterior_holonomy_interface.py)
- Raw: `data/experiments/tinyllm_c3_posterior_holonomy_interface/20260811_preregistered/result.json`
- Raw SHA-256: `102125d3c465a30be64a51b6a3b3a59ebb8c350dfb92a562f72684431b4601fc`
- [Measured report](2026-08-11_tinyllm-c3-posterior-holonomy-interface.md)

## Cross-experiment questions the reviewer must answer

1. **Identifiability:** Is the gauge obstruction stated over the actual
   observation map, and does the calibration reference repair it without
   target leakage?
2. **Causality:** Do the patching controls establish task-relative sufficiency,
   or could the endpoints be explained by denoising, Jensen convexity, or
   decoder insensitivity?
3. **Mechanism:** Is exact C2 Reynolds synthesis replicated strongly enough to
   call it a mechanism, given C3 instability and the three-checkpoint head/rank
   subset?
4. **Acquisition:** Does analytic `5/5` versus learned `1/5` isolate acquisition,
   or are optimization budget, parameterization, and public readout calibration
   still inseparable?
5. **Economic value:** Are fixed operators fair positive controls, and do they
   actually eliminate the learned job rather than merely solve a narrower
   oracle task?
6. **Novelty:** Which claims are standard consequences of Reynolds projection,
   character theory, conditional expectation, or finite cyclic Fourier
   analysis? Which empirical causal separations remain nontrivial?
7. **Generality:** What, if anything, transfers beyond the synthetic C2/C3
   generators, retained TinyLLM presets, and exact orbit access?
8. **Publication:** Is there one coherent paper, several bounded notes, or only
   an internal negative-results record?

## Required reviewer output

The reviewing agent should return:

1. an executive verdict of at most 300 words;
2. a claim table with `supported`, `partially supported`, `falsified`,
   `algebra-only`, or `not identifiable from artifacts`;
3. a per-experiment gate-reconstruction table showing its own pass counts;
4. a dependency audit distinguishing fresh seeds, reused checkpoints,
   post-outcome diagnostics, and underpowered subsets;
5. any report/raw/meta inconsistencies with exact file and JSON-key references;
6. a novelty audit separating standard mathematics from empirical findings;
7. the narrowest publishable claim and the claims that must be removed;
8. one shortest decisive external validation, if any.

## Copy-paste prompt for the reviewing agent

```text
Act as an independent, adversarial scientific reviewer of the TinyLLM causal
quotient program. Start with:

docs/08 - Analysis/2026-08-11_tinyllm-independent-analysis-packet.md

Use evidence commit f0024c5958a34b1502f49ebbf0caf0de8f9a43b2 and DVC root
125447de251dda33ab40e06d680cbb17.dir. Follow the packet's two-pass protocol:
inspect preregistrations, runner gate logic, and raw JSON before reading each
measured report. Recompute pass counts rather than trusting prose or
meta-hypothesis flags. Treat shared checkpoint lineages as dependent evidence;
do not let secondary or post-outcome results rescue failed primary gates.

Produce the eight required outputs in the packet. Separate exact group-theory
identities from learned-model evidence, task-relative causal sufficiency from
full-state quotienting, and C2 results from C3/general claims. Do not run new
training. If evidence is missing or a hash does not match, stop that claim and
report the discrepancy.
```

## Integrity and local verification

All eleven primary raw files listed above exist in the current DVC checkout and
their SHA-256 values were recomputed when this packet was written. The evidence
commit contains producing runners, preregistrations, reports, meta-hypothesis
code, and tests. `data.dvc` pins the large artifact tree.

Before analysis:

```bash
git rev-parse HEAD

DVC_SITE_CACHE_DIR=/home/rabbit/structure_net/.dvc/site-cache \
  pixi run dvc status

sha256sum \
  data/experiments/tinyllm_calibrated_frontend_causal/20260806_d8_preregistered/campaign_results.json \
  data/experiments/tinyllm_calibrated_frontend_causal_closure/20260810_d15_preregistered/campaign_results.json \
  data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered/campaign_results.json \
  data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/20260811_preregistered/result.json \
  data/experiments/tinyllm_c3_relational_connection_acquisition/20260811_preregistered/campaign_results.json \
  data/experiments/tinyllm_c3_connection_observation_identifiability/20260811_preregistered/result.json
```

The broader program integrity check most recently completed as:

```text
1,795 passed, 1 skipped, 4 deselected, 23 warnings
```

