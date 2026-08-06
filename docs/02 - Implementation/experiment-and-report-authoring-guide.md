# Writing Experiments and Reports on the NAL Platform

**Status:** CURRENT operational guide

**Date:** 2026-08-06

**Applies to:** experiment authors, analysis authors, and agents working in `experiments/structure_net`, `src/neural_architecture_lab`, `data/experiments`, and the research docs lanes

**Depends on:** `../05 - Standards/NAL-STD-EXPERIMENT-v0.md`; `../03 - Architecture/nal-local-gpu-scheduler.md`

The platform’s research unit is not a training script. It is a traceable chain:

```text
question
  -> preregistration
  -> tested implementation
  -> shakedown
  -> fingerprinted per-seed runs
  -> aggregate decision
  -> measured report
  -> conservative meta-hypothesis record
```

Use the draft standard for requirements. This guide gives the working sequence and reusable shapes.

## Quick path

For a new confirmatory study:

1. write the question, intervention, primary gate, seed rule, and outcome meanings;
2. save the dated preregistration in `docs/07 - Status Reports/`;
3. implement one immutable configuration and one independently retryable `Experiment` per replicate;
4. test the generator, objective, representation cuts, fingerprint, and joint gate;
5. pass the tiny lifecycle, representative CUDA, and intended-concurrency shakedowns;
6. freeze the producing implementation, record its identity, and launch into a new dated artifact root;
7. aggregate only complete fingerprint-matched cells;
8. write the measured report in `docs/08 - Analysis/` from `campaign_results.json`;
9. register material evidence in the meta-hypothesis system and read it back.

For an exploratory study, follow the same artifact and reporting steps but
label every post-outcome choice and do not present the result as preregistered.

## 1. Put each artifact in the right lane

| Artifact | Location | Role |
| --- | --- | --- |
| mathematical motivation | `docs/00 - Theory/` | definitions and predictions |
| proposed intervention | `docs/01 - Design/` | rationale before implementation |
| authoring/run instructions | `docs/02 - Implementation/` | executable procedure |
| runner/model contracts | `docs/03 - Architecture/` | as-built system map |
| normative requirements | `docs/05 - Standards/` | MUST/MUST NOT contract |
| preregistration or amendment | `docs/07 - Status Reports/` | dated pre-outcome snapshot |
| measured result | `docs/08 - Analysis/` | verdict and evidence |
| raw runs | `data/experiments/<study>/<run>/` | machine-readable evidence |
| durable synthesis | `data/meta_hypotheses/` plus storage backend | cross-study evidence record |

Do not use an analysis report as a preregistration, and do not hide the only copy of a metric in prose.

## 2. Start with a decision, not a method

Write the question so the planned measurements can answer it. Then write the possible outcomes before choosing implementation details.

```markdown
## Question

Does <intervention> change <primary property> relative to <control>
under <declared families>, without degrading <safety/task control>?

## Decision rule

Success requires <endpoint A> and <endpoint B> jointly in at least
<k> of <n> seeds. Task performance may fall by at most <delta>.

## Outcome meanings

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| A and B pass | target mechanism supported | replicate or scale |
| A only | partial mechanism | strengthen B-specific intervention |
| B only | compression/confound | preserve A explicitly |
| neither | intervention ineffective | test a different mechanism |
```

This prevents a large collection of measurements from turning into an unfalsifiable story.

## 3. Write the preregistration

Create `docs/07 - Status Reports/YYYY-MM-DD_<study>-preregistration.md` before reading primary outcomes. Include:

1. stable hypothesis ID and question;
2. control and intervention arms;
3. architecture, examples, updates, optimizer, and minibatch policy;
4. seed list and joint seed rule;
5. generator-level definitions of all splits and shifts;
6. primary endpoints with exact cuts, estimators, and thresholds;
7. planned secondary diagnostics;
8. task/safety degradation ceiling;
9. interpretation table and stop conditions;
10. expected artifact root and command.

Choose loss weights, probe strength, early stopping, thresholds, and seed count here. If a pilot is needed to choose them, label the pilot exploratory and use disjoint seeds/data for the confirmatory campaign.

## 4. Build one immutable configuration

Put all outcome-relevant choices in a validated dataclass. Avoid scientific defaults scattered across functions.

```python
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class StudyConfig:
    preset: str = "d8"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    training_steps: int = 600
    train_samples: int = 4096
    batch_size: int = 64
    learning_rate: float = 3e-4
    device_ids: tuple[int, ...] = (0,)
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory campaign requires five seeds")
```

Serialize `asdict(config)` into the campaign result and every worker’s parameters. An `--allow-underpowered` switch is useful for tests, but the resulting artifact must remain visibly non-confirmatory.

## 5. Make one seed one schedulable experiment

The canonical worker shape is:

```python
def worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    config = StudyConfig(**experiment.parameters["configuration"])
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    # Construct, train or load, evaluate, and atomically persist detail.
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=experiment.hypothesis_id,
        metrics={"primary_endpoint": value},
        primary_metric=value,
        model_architecture=architecture,
        model_parameters=parameter_count,
        training_time=elapsed,
        model_checkpoint=str(checkpoint),
        observations=[f"detail={detail_path}"],
    )
```

Use one `Experiment` per independently retryable replicate. If multiple matched arms must share initialization inside a seed, keeping those arms sequential inside one seed experiment is acceptable; document why that is the correct replication unit.

Keep `ExperimentResult.metrics` small and numeric so the runner and search layer can summarize it. Put nested probe results, arrays, gate details, and provenance in versioned `result.json`.

## 6. Preserve the scientific fingerprint

A resumption check should look like this:

```python
def scientific_fingerprint(parameters: Mapping[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

On reuse, verify:

- schema and terminal status;
- condition and seed;
- scientific fingerprint;
- checkpoint/supplement existence;
- checkpoint state digest when weights are a source of evidence.

Record the identity of the producing implementation too. Prefer a Git commit
when the tree is clean. If a campaign must launch from an uncommitted tree,
hash the experiment module and every outcome-relevant local dependency, and
preserve the corresponding diff before interpreting results. Do not edit
producing code while workers from that campaign are running. Cells produced by
materially different implementations are different conditions and cannot be
silently pooled.

For matched comparisons, also hash the generated training tensors and minibatch schedule. A matching seed name is not evidence that examples or updates were identical.

Write JSON atomically and strictly:

```python
temporary.write_text(json.dumps(record, allow_nan=False, ...))
temporary.replace(destination)
```

## 7. Separate training-time heads from held-out measurements

An auxiliary training head is part of the intervention. A held-out probe is an estimator used to evaluate the frozen model. Do not reuse one as the other.

For a frozen probe:

1. freeze the trained network;
2. generate disjoint probe train, validation, and test sets;
3. fit the probe only on its train split;
4. select/early-stop only on validation;
5. evaluate once on every declared test family;
6. compare against relevant null/priors;
7. report probe width, steps, seeds, and failures.

When testing conditional information, condition the probe explicitly and match the relevant semantic variable in the data. Report “conditional branch accuracy under the tested nonlinear probe,” not “the representation contains/does not contain the bit” without qualification.

## 8. Implement gates per seed before averaging

Suppose success requires cosine correlation at least `0.90`, conditional branch accuracy at most `0.55`, and conditional log-loss gain at most `0.02` at two cuts under two shifts. Compute:

```python
def endpoint_pass(m: Mapping[str, float]) -> bool:
    return (
        m["cosine_pearson"] >= 0.90
        and m["balanced_accuracy"] <= 0.55
        and m["conditional_log_loss_gain"] <= 0.02
    )

seed_pass = all(
    endpoint_pass(result[seed][cut][shift])
    for cut in ("post_mlp", "full")
    for shift in ("composition", "extrapolation")
)
campaign_pass = sum(seed_passes) >= 4
```

Do not separately count four passing seeds for each endpoint if they are different seeds. Preserve both the joint pass count and the per-cell table so the failure mode is inspectable.

## 9. Use the shakedown ladder

Run cheap checks before expensive evidence:

### 9.1 Focused CPU tests

Test configuration validation, objective signs, gradient routing, generator pairing, cut extraction, threshold inclusivity, fingerprints, and aggregation logic.

### 9.2 Tiny lifecycle run

Use a deliberately tiny preset and one or two steps. It must perform a real backward pass, save weights, reload them, write strict JSON, and run the aggregation path. Label it systems-only.

### 9.3 Representative CUDA pilot

Use the real architecture with the smallest scientifically disposable run that exposes peak allocation. Confirm CUDA ordinal mapping and record `max_memory_allocated`/`reserved`.

### 9.4 Concurrency pilot

Run the intended number of simultaneous workers. If two workers fit only after changing the computation, prove the new computation is algebraically/scientifically equivalent and document the amendment.

### 9.5 Full campaign

Only the fixed full configuration enters the primary aggregate. Do not mix shakedown or tuning seeds into it.

## 10. Launch and resume safely

Select physical GPUs outside Python and refer to logical ordinals inside it:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python -m experiments.structure_net.<study> \
  --seeds 7,17,29,41,53 \
  --gpus 0 \
  --slots-per-gpu 2 --max-parallel 2 \
  --resume \
  --output data/experiments/<study>/<dated-run>
```

The runner resumes completed fingerprint-matched experiments, not partial optimizer state. Inspect failure records before retrying. A code or configuration change should generate a different fingerprint and normally a new dated run directory.

Prefer module execution so package imports resolve consistently in the parent
process and spawned CUDA workers. A direct file entry point is acceptable only
when that script explicitly supports direct execution.

In this example, physical device `2` becomes logical `cuda:0` after visibility
filtering. Confirm the mapping with `torch.cuda.get_device_name()`; do not
assume PyTorch ordinals and `nvidia-smi` display indices are interchangeable.

## 11. Write `result.json` and `campaign_results.json`

Recommended per-run envelope:

```json
{
  "schema_version": "nal.<study>.v1",
  "hypothesis_id": "<stable-id>",
  "experiment_id": "<condition>-seed7",
  "status": "completed",
  "condition": "<condition>",
  "seed": 7,
  "scientific_fingerprint": "<sha256>",
  "configuration": {},
  "training": {},
  "analysis": {},
  "task_metrics": {},
  "artifacts": {},
  "method_boundaries": []
}
```

Recommended aggregate envelope:

```json
{
  "schema_version": "nal.<study>.v1",
  "hypothesis_id": "<stable-id>",
  "status": "completed",
  "configuration": {},
  "task_config": {},
  "environment": {},
  "scheduler": {},
  "summary": {
    "requested": 10,
    "reused": 0,
    "scheduled": 10,
    "completed": 10,
    "failed": 0
  },
  "aggregates": {
    "preregistered_gate": {},
    "conclusion": "<controlled vocabulary>"
  },
  "results": [],
  "method_boundaries": []
}
```

Store detailed per-seed values even when the report presents means. Include hashes for scientifically material supplements and checkpoints when practical.

## 12. Write the report from the raw aggregate

Create `docs/08 - Analysis/YYYY-MM-DD_<study>.md`. Use this order:

```markdown
# <Study>

**Status:** <measured verdict>
**Date:** YYYY-MM-DD
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED` | ...
**Hypothesis:** `<id>`
**Preregistration:** `<relative link>`

## Verdict
<Answer the question in the first paragraph.>

## Campaign integrity
<requested/completed/failed/reused, deviations, fixed controls>

## Primary endpoints
<table with conditions, shifts, cuts, means, ranges, and pass counts>

## Preregistered gates
<one row per gate; passed/failed without reinterpretation>

## Task and safety controls

## Mechanistic and secondary measurements

## Interpretation and boundaries

## Artifacts and reproduction
<paths, hashes where useful, exact command>
```

Lead with the result, not the chronology of implementation. A failed full hypothesis can coexist with a useful narrower finding; state both without upgrading the latter into confirmation.

## 13. Register the result in the meta-hypothesis system

After the raw aggregate and report stabilize:

1. build a typed, conservative record under `src/neural_architecture_lab/`;
2. add a small storage CLI under `experiments/neural_architecture_lab/`;
3. link direct experiment records and source artifacts;
4. store the JSON under `data/meta_hypotheses/` and the configured search backend;
5. read it back in a focused test;
6. keep `confirmed: false` when the full preregistered gate failed.

Use an existing recent module such as `nuisance_support_scaling_meta_hypothesis.py` as the structural example. Do not copy its study-specific claims.

After storage, verify both representations of the record: read the aggregate
JSON and query the configured hypothesis and experiment collections by stable
hypothesis ID. Backend telemetry warnings are not evidence of failure or
success; the read-back result is authoritative.

## 14. Worked example: block-1 horizontal/vertical control

The current causal quotient experiment is a compact example of the standard:

| Requirement | Concrete design |
| --- | --- |
| question | can semantic-base preservation plus fiber contraction create a stable internal quotient? |
| control | ordinary d8/N3 training |
| intervention | block-1 post-MLP cosine auxiliary head plus cosine-conditioned branch adversary through gradient reversal |
| matched factors | architecture, examples, paired minibatches, steps, optimizer hyperparameters, and five model seeds |
| shifts | in-distribution, held-out nuisance composition, outside-range extrapolation |
| primary cuts | post-MLP block 1 and full depth |
| joint endpoint | cosine correlation `>= .90`, conditional branch accuracy `<= .55`, log-loss gain `<= .02` |
| campaign gate | the joint endpoint passes in at least four of five seeds on composition and extrapolation; full task accuracy drops no more than three points |
| mechanistic cuts | query, post-attention, post-MLP |
| secondary geometry | nuisance-whitened finite-difference base sensitivity divided by matched opposite-branch sensitivity |

The minimized implementation objective is written with an explicit sign convention:

```text
L = L_task + lambda_base * MSE(a(r1), cos(phi))
           + lambda_inv  * BCE(D(GRL(r1), cos(phi)), b)
```

The gradient-reversal layer is identity forward, sends the adversary a normal minimizing gradient, and negates the transformer-side branch gradient. This avoids the ambiguity created by writing a negative cross-entropy whose own definition already contains a negative sign.

The ordinary checkpoints may be reused only after configuration, state digest, training-data hash, and minibatch-schedule hash checks. Temporary auxiliary heads are stored for provenance but excluded from inference. Held-out nonlinear probes, rather than the training adversary, decide the primary branch endpoint.

## 15. Publish, revise, and preserve

Treat completed raw artifacts as append-only evidence. Do not overwrite a
campaign merely to make a changed aggregator, probe, threshold, or report agree
with a desired conclusion. Instead:

1. preserve the original campaign and report;
2. write a dated amendment or correction describing what changed and when;
3. use a new schema revision or dated artifact root when machine-readable
   interpretation changes;
4. rerun only affected cells, with their new fingerprints visible;
5. update the active analysis index and meta-hypothesis record without deleting
   prior evidence.

Source, tests, preregistration, analysis, and lightweight metadata belong in
the repository commit. Large checkpoints and arrays may remain in the declared
data store, but the report must state whether they are locally available,
externally archived, or missing.

## 16. Author’s release checklist

- [ ] Stable hypothesis ID and question
- [ ] Dated preregistration written before outcome interpretation
- [ ] Intervention and all fixed controls listed
- [ ] Data families and split independence defined operationally
- [ ] Primary endpoints and joint per-seed rule encoded in code and prose
- [ ] Configuration validation and scientific fingerprint tested
- [ ] Producing implementation frozen and identified by commit or content digest
- [ ] Objective signs and gradient routing tested
- [ ] Tiny lifecycle and representative CUDA/concurrency pilots passed
- [ ] Strict per-run and campaign JSON written atomically
- [ ] Required weights and supplements retained
- [ ] Completed/failed/retried/reused counts reported
- [ ] Report verdict matches the machine-readable gate
- [ ] Secondary findings do not rescue failed primary gates
- [ ] Reproduction command and raw artifact paths included
- [ ] Meta-hypothesis record stored and read back when material
- [ ] Focused tests and an appropriate repository gate recorded

## Verification

```bash
pixi run pytest -q \
  tests/neural_architecture_lab/test_runner_lifecycle.py \
  tests/neural_architecture_lab/test_data_factory_integration.py

rg -n "HYPOTHESIS_ID|SCHEMA_VERSION|campaign_results.json|preregistered_gate" \
  experiments/structure_net docs/07\ -\ Status\ Reports docs/08\ -\ Analysis
```

Then compare the intended campaign against `NAL-STD-EXPERIMENT v0-draft` and record any unmet clause in the preregistration before launching expensive runs.
