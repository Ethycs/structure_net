# NAL Research Experiment and Report Standard

**Standard ID:** NAL-STD-EXPERIMENT v0-draft

**Status:** DRAFT — normative candidate, not frozen

**Date:** 2026-08-06

**Applies to:** new empirical studies executed through the Neural Architecture Lab (NAL), including training campaigns, frozen-model interventions, replications, benchmarks, and their reports

**Depends on:** `../../Agentic Technique Master.md`; `../03 - Architecture/nal-local-gpu-scheduler.md`; `../08 - Analysis/2026-08-03_runner-experiment-data-modernization.md`

This draft establishes the minimum evidence contract for a result to enter the active research record. It does not retroactively certify archived studies.

## 0. Scope

This standard governs:

- experimental questions, controls, interventions, seeds, splits, and endpoints;
- implementation on the canonical NAL runner;
- per-run and aggregate artifacts;
- preregistrations, amendments, analysis reports, and meta-hypothesis records;
- claims made from probes, geometry, benchmarks, and shakedowns.

It does not prescribe a model family, task, loss, statistical test, or universal sample size. Those choices remain study-specific and MUST be justified before the corresponding outcome is inspected.

## 1. Normative goals

1. An experiment MUST distinguish a scientific question from a systems check.
2. A confirmatory campaign MUST declare its primary endpoints and decision rule before primary outcomes are interpreted.
3. Compared conditions MUST be matched on every factor not named as an intervention, or the mismatch MUST be reported as a confound.
4. Every reported result MUST be traceable to executable code, a complete configuration, raw per-run evidence, and an aggregate artifact.
5. Failed gates, null results, amendments, retries, exclusions, and implementation defects MUST remain visible in the record.
6. Resume and reuse MUST be fingerprint-matched; an experiment ID alone MUST NOT authorize reuse.
7. Reports MUST state what the measurements establish and the important claims they do not establish.
8. Outcome-relevant implementation code MUST be frozen or fingerprinted during a campaign. Cells produced by materially different implementations MUST NOT be pooled as one condition without disclosure and justification.

## 2. Vocabulary

| Term | Meaning |
| --- | --- |
| hypothesis | A falsifiable claim with a stable ID, prediction, and decision rule |
| campaign | The complete collection of conditions and replicates used to test one hypothesis |
| condition or arm | One intervention level or control within a campaign |
| cell | One fully specified condition/configuration combination |
| replicate | One independently initialized run; normally identified by seed |
| primary endpoint | A measurement that participates in the preregistered decision rule |
| secondary measurement | A planned explanatory measurement that cannot rescue a failed primary gate |
| exploratory analysis | An analysis selected after primary outcomes were available |
| shakedown | A deliberately underpowered execution check; never model-quality evidence |
| preregistration | A dated, immutable statement of design and decision rules written before primary outcome interpretation |
| amendment | A dated change to a preregistration that states timing, reason, and affected claims |
| scientific fingerprint | A digest of the condition, seed, configuration, task, and other outcome-relevant inputs |

## 3. Study specification

### 3.1 Hypothesis identity

Every campaign MUST declare:

- a stable `hypothesis_id`;
- one question answerable by the design;
- a directional prediction or a declared two-sided/null comparison;
- the unit of replication;
- the conditions being compared;
- primary endpoints, thresholds, and aggregation rule;
- explicit interpretations for success, partial success, and failure.

A report MUST NOT silently replace the preregistered question with an easier claim after observing results.

### 3.2 Controls and interventions

The preregistration MUST contain a control table with at least:

| Field | Required content |
| --- | --- |
| intervention | the factors intentionally changed |
| fixed controls | architecture, data volume, examples or generator, minibatches, updates, optimizer, and evaluation protocol as applicable |
| stochastic controls | initialization, data, minibatch, augmentation, and probe seed policy |
| evaluation families | in-distribution and each declared shift or nuisance family |
| exclusions | conditions under which a run may be excluded or retried |

When a baseline is reused, the campaign MUST verify its configuration and checkpoint digest and MUST show that data and minibatch schedules match the new arm where matching is claimed.

Temporary training heads, adapters, or losses are interventions even if removed at inference. They MUST be named and their parameters, initialization, optimization, and checkpoint policy recorded.

### 3.3 Seeds and power

Confirmatory neural-network comparisons SHOULD use at least five independent seeds. A smaller campaign MUST be marked `UNDERPOWERED` and MUST NOT pass a gate that requires a larger replicate count.

Seed roles MUST be explicit. Model initialization, training data, minibatches, probes, and evaluation data SHOULD use independent, deterministic seed streams. The same model seed MUST NOT imply accidental reuse of evaluation samples.

Thresholds MUST be evaluated per seed before aggregation when the scientific claim requires joint behavior within individual models. Passing four separate marginal gates on different seeds is not equivalent to four seeds passing the joint gate.

### 3.4 Data and split integrity

Training, validation, probe training, probe validation, and final test data MUST be disjoint unless reuse is intrinsic to the declared question. Reuse MUST be disclosed.

Generated data MUST record generator version or code identity, complete parameters, sample counts, and seeds. Shifted families MUST be operationally defined: “composition” or “extrapolation” without a generator-level definition is insufficient.

If exact matching is scientifically material, such as predicting a nuisance branch conditional on cosine, the generator or sampler MUST construct that matching directly or document the matching tolerance.

Class-prior, majority-class, or other nonuniform baselines MUST be reported when nominal uniform chance is not the empirical chance baseline.

### 3.5 Endpoints and measurement boundaries

Primary endpoints MUST specify:

- the representation cut or output on which they are measured;
- the estimator or probe family;
- train/validation/test protocol for learned estimators;
- direction and numeric threshold;
- the seed-level and campaign-level gate.

A decoder or probe measurement establishes tested decodability under that estimator and data family. It MUST NOT be described as proof that information is absent, as mutual information, or as an intrinsic property of the representation without independent evidence.

Aggregate Euclidean distance, task-posterior Fisher geometry, persistent homology, and local derivatives measure different objects. One MUST NOT substitute for another solely because both are described as “geometry.” Decoder-conditioned measurements MUST be labeled decoder-conditioned.

Secondary or exploratory measurements MUST NOT rescue a failed primary endpoint. They MAY explain the failure and motivate a new hypothesis.

## 4. Implementation contract

### 4.1 Canonical runtime types

One independently schedulable replicate SHOULD be represented by one `Experiment`. Workers MUST implement:

```python
ExperimentWorker = (Experiment, device_id) -> ExperimentResult
```

The `device_id` is a logical CUDA ordinal after `CUDA_VISIBLE_DEVICES`, or `-1` for CPU. Experiment code MUST NOT silently rewrite device visibility after process start.

Each `ExperimentResult` MUST contain direct experiment and hypothesis IDs, flat headline metrics, a primary metric, architecture/parameter provenance, elapsed training or analysis time, checkpoint path when weights are material, observations, terminal status, and errors.

Study-specific detail MAY live in a versioned JSON record linked from `observations`. Flexible runtime dataclasses MUST NOT be treated as a durable wire schema by themselves.

### 4.2 Determinism and fingerprints

Each study MUST define a `SCHEMA_VERSION` and compute a scientific fingerprint or equivalent provenance check from all outcome-relevant inputs. At minimum this includes:

- condition and seed;
- complete study and task configuration;
- training/evaluation protocol version;
- data and minibatch schedule identities when exact matching is claimed.

The campaign MUST also record the producing implementation identity. A clean
source-control commit is preferred. Otherwise the experiment entry point and
all outcome-relevant uncommitted dependencies MUST be content-hashed and the
corresponding source state MUST be preserved. Changing producing code after
workers launch creates a new implementation version. Results from different
versions MUST be separated, or the report MUST identify and justify why the
change is scientifically inert.

Completed-result reuse MUST require a successful terminal status, matching schema, matching scientific fingerprint, and existence of required artifacts. Atomic writes SHOULD use a temporary file followed by rename.

### 4.3 Resource execution

GPU slot count MUST be selected from a representative memory pilot. A scheduler slot is admission control, not memory isolation.

Shared-run wall time MUST NOT be presented as a model benchmark. Timing claims require an isolated timing protocol with device, software stack, warm-up, synchronization, and repetition stated.

Retries MAY recover infrastructure failures. A scientific failure or threshold miss MUST NOT be retried merely to obtain a passing seed. All terminal failures and retry counts MUST remain in the campaign summary.

### 4.4 Shakedown ladder

Before the full campaign, an implementation SHOULD pass:

1. focused unit/contract tests on CPU;
2. a tiny CPU or CUDA path that performs a real forward, backward, save, and reload;
3. a representative CUDA memory pilot;
4. a concurrency pilot at the intended slot count;
5. a result-schema and resume round trip.

A shakedown MUST be labeled `systems_lifecycle_only_not_quality_evidence` or equivalent. Its metrics MUST NOT be merged into the confirmatory campaign.

## 5. Required artifacts

The canonical campaign root is:

```text
data/experiments/<study>/<dated-run>/
├── campaign_results.json
└── runs/
    └── <condition>/
        └── seed_<seed>/
            ├── result.json
            ├── model.pt                 # when weights are material
            └── <declared supplements>   # arrays, figures, temporary heads
```

Every completed campaign MUST retain:

| Artifact | Requirement |
| --- | --- |
| `result.json` | one strict-JSON, schema-versioned record per run |
| `campaign_results.json` | full config, environment, scheduler record, completion counts, aggregates, gates, artifact links, and method boundaries |
| checkpoint | required when later frozen-model analysis, reproducibility, or causal verification depends on the weights |
| producing code | committed experiment and analysis entry points |
| preregistration | required for confirmatory claims |
| analysis report | required before a result enters the active docs index |

JSON MUST reject NaN and infinity. Missing measurements MUST be represented explicitly with status/reason fields, not non-standard floating-point values.

Large arrays SHOULD be stored in NPZ/HDF5 with shape, dtype, semantics, and a link from the JSON. A report MUST identify the raw artifact rather than containing the only copy of a result.

## 6. Preregistration and amendments

A confirmatory preregistration belongs in `docs/07 - Status Reports/` and MUST be dated. Its minimum sections are:

1. question and prediction;
2. design and fixed controls;
3. data and split definitions;
4. training protocol;
5. primary endpoints and joint seed gate;
6. secondary/mechanistic measurements;
7. outcome interpretation table;
8. artifact and execution plan;
9. known method boundaries.

After primary results are inspected, the preregistration MUST NOT be edited except for a clearly labeled amendment. An amendment MUST state when it occurred, whether relevant outcomes had been seen, why it was necessary, which endpoints changed, and whether the original analysis is still reported.

## 7. Analysis report contract

The final report belongs in `docs/08 - Analysis/`, is dated, and MUST lead with the measured verdict. It MUST include:

- status, date, hypothesis ID, and preregistration link;
- one unambiguous verdict against every preregistered gate;
- campaign integrity: requested/completed/failed/reused runs and deviations;
- fixed configuration and environment sufficient to reproduce the run;
- per-condition aggregate results and important seed ranges;
- joint per-seed gate counts where required;
- task-performance or safety controls;
- mechanistic/secondary results separated from primary evidence;
- anomalies, nulls, failed endpoints, and method boundaries;
- raw artifact paths and an exact reproduction command.

Reports MUST use conservative verbs:

| Evidence | Permitted wording |
| --- | --- |
| primary gate passes | “supports” or “confirms under the preregistered operational definition” |
| gate fails | “does not confirm”; preserve narrower supported findings separately |
| frozen probe near chance | “not recoverable by the tested probe on this family” |
| decoder collapses outputs | “the learned output/posterior map groups the fibers” |
| shakedown passes | “execution path validated,” not “model works” |
| five-seed association | “repeatable across these seeds,” not a universal law |

## 8. Meta-hypothesis registration

After the campaign artifact and report are final, material results SHOULD be stored in the meta-hypothesis system. The record MUST:

- link the raw artifact and report;
- retain the original hypothesis ID and schema versions;
- encode the conservative verdict, including `confirmed: false` for a failed full gate;
- separate direct experimental records from derived interpretation;
- be read back and covered by a focused storage test.

Meta-hypothesis storage MUST NOT upgrade an exploratory or underpowered finding into confirmatory evidence.

Storage is complete only after read-back verifies the stable hypothesis ID,
verdict, direct experiment IDs, and expected evidence count. A returned hash or
successful CLI exit without read-back is insufficient conformance evidence.

## 9. Preservation and correction

Completed per-run and campaign artifacts are evidence records and MUST be
treated as append-only. A corrected aggregator, probe, threshold, exclusion,
or interpretation MUST NOT silently overwrite the original scientific record.

A correction MUST:

- preserve or reference the original artifact;
- state whether training, evaluation, aggregation, or prose changed;
- receive a new schema revision, artifact root, or explicit revision field;
- recompute fingerprints for affected cells;
- update the active report and meta-hypothesis record without erasing the prior
  verdict.

Producing code, tests, preregistration, and analysis SHOULD be committed
together. Large checkpoints MAY live outside Git when their durable location,
availability, and digest are recorded.

## 10. Conformance profiles

| Profile | Required evidence |
| --- | --- |
| `SHAKEDOWN` | implementation tests, real lifecycle artifact, explicit non-quality boundary |
| `EXPLORATORY` | complete config/artifacts/report; post-hoc choices labeled |
| `PREREGISTERED` | exploratory requirements plus prior preregistration, fixed gates, amendments |
| `REPLICATION` | preregistered requirements plus source-claim audit and compatibility evidence |
| `BENCHMARK` | preregistered requirements plus isolated timing/resource protocol |

Every report MUST name its conformance profile and any unmet clause.

## 11. Verification

```bash
rg -n "SCHEMA_VERSION|HYPOTHESIS_ID|scientific_fingerprint|ExperimentResult" \
  experiments/structure_net src/neural_architecture_lab

find data/experiments -name campaign_results.json -o -name result.json | sort
```

Before promoting this draft to frozen v0, audit at least one completed campaign against every MUST clause and add an executable conformance validator or schema test.
