# Experimental System Master Guide

**Status:** CURRENT session entry point  
**Date:** 2026-09-09  
**Applies to:** confirmatory and exploratory experiments in `structure_net`,
especially the Neural Architecture Lab (NAL), TinyLLM campaigns, GPU execution,
artifact handling, analysis, and hypothesis registration

This is the one-file orientation and handoff guide for a new working session.
It consolidates the operating rules; it does not replace the detailed standard
or a study's preregistration.

## 1. Start a new session here

Give a new session this prompt:

~~~text
Work in /home/rabbit/structure_net.

Read docs/02 - Implementation/experimental-system-master-guide.md completely.
Then inspect git status, the latest commit, available GPUs, and the
preregistration plus analysis report for the experiment in scope. Preserve all
existing worktree changes. Treat completed experiment artifacts as append-only,
use a new dated artifact root for changed code or configuration, and do not
launch an expensive job until you have reported the current state and verified
the relevant scientific fingerprint and stop rules. Use CUDA for model training
and decomposition campaigns unless the selected preregistration explicitly
defines a CPU-only cell.
~~~

The new session should begin with read-only inspection:

~~~bash
cd /home/rabbit/structure_net
git status --short
git log -5 --oneline
nvidia-smi
pixi run test-cuda
~~~

Before acting, it must identify:

1. the exact hypothesis ID and preregistration;
2. whether the study is unregistered, registered, running, stopped, or completed;
3. the producing source state and any uncommitted user work;
4. the expected artifact root and whether it already contains evidence;
5. the next gate in the preregistered decision order;
6. whether the requested action is a systems check, confirmatory run, or
   exploratory follow-up.

Do not infer that a campaign should be rerun merely because its code is
available.

## 2. Authority and source-of-truth order

Use these records together:

1. The dated preregistration fixes the question, controls, endpoints, seed rule,
   stop rules, and outcome meanings.
2. The producing implementation and scientific fingerprint identify what
   actually ran.
3. Per-run `result.json` files contain direct cell or replicate evidence.
4. `campaign_results.json` is the machine-readable aggregate and decision
   record.
5. The dated analysis report explains the aggregate without changing its gate.
6. The meta-hypothesis registry is the conservative searchable index of the
   stabilized result.

If these disagree, stop and report the mismatch. Preserve the original
artifacts; do not edit one layer until it agrees with another.

The governing documents are:

- [NAL experiment standard](../05%20-%20Standards/NAL-STD-EXPERIMENT-v0.md)
- [Experiment and report authoring guide](experiment-and-report-authoring-guide.md)
- [Local GPU scheduler](../03%20-%20Architecture/nal-local-gpu-scheduler.md)
- [Developer guide](developer-guide.md)

The standard is normative. This guide is the practical entry point. The code is
authoritative about the behavior of a concrete entry point, but code does not
silently amend a preregistration.

## 3. The experimental unit

The research chain is:

~~~text
question
  -> dated preregistration
  -> tested implementation
  -> CPU/lifecycle/CUDA/concurrency shakedowns
  -> frozen or content-hashed producer
  -> fingerprinted per-seed runs
  -> aggregate decision
  -> measured analysis report
  -> conservative NAL registration and read-back
~~~

Important vocabulary:

| Term | Meaning |
| --- | --- |
| hypothesis | One falsifiable claim with a stable ID and fixed decision rule |
| campaign | All conditions and replicates used to test that hypothesis |
| arm | One control or intervention |
| cell | One completely specified configuration/condition |
| replicate | One independently initialized run, normally identified by seed |
| evaluation seed | Sometimes a deterministic input-prefix replicate rather than a separately trained model; the study must say which |
| shakedown | Underpowered systems validation that is never quality evidence |
| scientific fingerprint | Digest of outcome-relevant configuration, seed, task, protocol, and data identities |
| classification | The result obtained by applying the preregistered decision order |

### What “all seeds completed” means

It means every scheduled seed reached a successful terminal execution state and
wrote the required fingerprint-matched artifacts. It does **not** mean:

- the hypothesis passed;
- every seed passed the scientific gate;
- the seeds are independent trained-model replicates;
- a downstream conditional stage is authorized;
- failed or excluded cells may be ignored.

Always inspect the per-seed joint gates and the campaign classification after
checking completion counts.

## 4. Repository map

| Location | Purpose |
| --- | --- |
| `docs/07 - Status Reports/<project>/<date>_<study>-preregistration.md` | Immutable confirmatory design and dated amendments |
| `experiments/structure_net/<study>.py` | Training, intervention, measurement, and campaign entry points |
| `tests/structure_net/` | Study behavior, numerics, artifact, and CLI tests |
| `src/neural_architecture_lab/core.py` | Canonical `Experiment`, `ExperimentResult`, `LabConfig`, and worker contracts |
| `src/neural_architecture_lab/runners.py` | Canonical local runner lifecycle |
| `src/neural_architecture_lab/local_scheduler.py` | Logical GPU slots and completed-result ledger |
| `data/experiments/<study>/<dated-run>/` | Raw per-run and aggregate evidence; ignored by Git |
| `docs/08 - Analysis/<project>/<date>_<study>.md` | Human-readable measured result |
| `src/neural_architecture_lab/*_hypothesis.py` | Typed conservative hypothesis/result records |
| `experiments/neural_architecture_lab/register_*.py` | Prospective registry writers |
| `experiments/neural_architecture_lab/finalize_*.py` | Hash-checked finalization and read-back |
| `data/hypothesis_registry/` and `data/chroma_db/` | Local machine-readable/searchable registry; ignored by Git |

Large checkpoints and raw arrays normally remain in `data/`. Source, tests,
preregistrations, reports, and lightweight provenance belong in Git.

`docs/10 - Experiments/` is a particular A/B/C semantic-routing program, not the
general NAL operator manual.

## 5. End-to-end workflow

### 5.1 Define the decision before the method

Write one question that the proposed comparison can answer. Fix:

- hypothesis ID and directional prediction;
- unit of replication and seed roles;
- intervention and matched controls;
- data families and disjoint splits;
- primary measurements, representation cuts, and estimators;
- per-seed joint gate and campaign aggregation rule;
- validity checks, exclusions, retries, and stop rules;
- meanings of success, partial success, failure, and invalidity;
- intended artifact root and reproduction command.

Avoid turning an available implementation into the question. The claim should
determine the experiment.

### 5.2 Preregister

Put confirmatory designs in:

~~~text
docs/07 - Status Reports/<project>/YYYY-MM-DD_<study>-preregistration.md
~~~

Once primary outcomes are inspected, do not rewrite that document. Add a dated,
explicit amendment stating what was known, why the change is needed, what it
changes, and whether the original analysis remains reported.

Exploratory work follows the same artifact discipline but must be labeled
exploratory and cannot retroactively pass the parent hypothesis.

### 5.3 Implement one immutable configuration

Prefer a frozen dataclass or equivalent immutable object. Serialize the complete
configuration into per-run and campaign artifacts. Define a `SCHEMA_VERSION` and
stable `HYPOTHESIS_ID`.

One independently retryable replicate should normally be represented by one
`Experiment`:

~~~python
ExperimentWorker = (Experiment, device_id) -> ExperimentResult
~~~

The result must directly identify the experiment, hypothesis, seed, primary
metric, status, timing, model/checkpoint provenance, observations, and errors.

Study-specific direct scripts are acceptable, including the A5--A7 scripts, if
they preserve the same configuration, fingerprint, strict-JSON, gate, and
artifact contracts.

### 5.4 Test before spending GPU time

Test at least:

- configuration validation and threshold inclusivity;
- objective signs and gradient routing;
- deterministic seed stream separation;
- data/prefix pairing and split integrity;
- representation extraction and shape contracts;
- numerical controls and finiteness;
- scientific fingerprint sensitivity;
- strict JSON serialization;
- aggregation and stop-rule classification;
- checkpoint save/reload and completed-result reuse.

Use focused suites, because unrestricted historical test discovery can still
enter archived code:

~~~bash
env PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache \
    UV_CACHE_DIR=/tmp/structure-net-uv-cache \
    pixi run pytest tests/structure_net tests/neural_architecture_lab -q
~~~

### 5.5 Pass the shakedown ladder

Run in this order:

1. Focused CPU unit and contract tests.
2. Tiny lifecycle run with a real forward/backward where training applies,
   checkpoint save/reload, strict JSON, and aggregation.
3. Representative CUDA memory pilot using the real architecture.
4. Intended-concurrency pilot.
5. Full fixed campaign.

Every shakedown artifact must be visibly labeled
`systems_lifecycle_only_not_quality_evidence` or equivalent and use a separate
artifact root.

For scripts that expose `--shakedown`:

~~~bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
pixi run python -m experiments.structure_net.<study> \
  --device cuda:0 --shakedown \
  --output data/experiments/<study>/<date>_shakedown
~~~

Run `--help` first. Not every study exposes the same scheduler or resume flags.

### 5.6 Freeze the producer

Before a confirmatory launch:

- prefer a clean source-control commit;
- otherwise content-hash every outcome-relevant uncommitted dependency and
  preserve that source state;
- record the complete configuration, checkpoint and data hashes, environment,
  and command;
- use a new dated artifact root.

Changing outcome-relevant code after workers start creates a new producing
version. Do not pool the two versions without an explicit, defensible amendment.

### 5.7 Select GPUs correctly

Select physical GPUs before Python starts:

~~~bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python -m experiments.structure_net.<study> --device cuda:0
~~~

Physical GPU 2 becomes logical `cuda:0` inside the process. NAL device IDs and
study `--device` arguments refer to logical ordinals.

The canonical scheduler parallelizes independent experiments, usually seeds. It
does not distribute one model with DDP. Multiple seeds may share a GPU only
after a representative peak-memory and concurrency pilot.

SVD or decomposition calls can sometimes be batched or scheduled across
independent cells, but “many SVDs” does not by itself make many full models fit.
Model activations, dense operators, workspaces, precision, and retained outputs
still determine VRAM. Preserve the preregistered algorithm and numerical
precision when optimizing kernels.

Check immediately before launch:

~~~bash
nvidia-smi
pixi run test-cuda
~~~

Do not silently fall back to CPU for an intended GPU campaign. Do not change
`CUDA_VISIBLE_DEVICES` after Python imports CUDA.

### 5.8 Launch, monitor, and resume

Prefer module execution from the repository root. A typical NAL-style launch is:

~~~bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python -m experiments.structure_net.<study> \
  --seeds 7,17,29,41,53 \
  --gpus 0 --slots-per-gpu 2 --max-parallel 2 \
  --resume \
  --output data/experiments/<study>/<dated-run>
~~~

This is a pattern, not a universal CLI. Confirm the selected study's arguments.

During a long run, inspect without mutating:

~~~bash
nvidia-smi
find data/experiments/<study>/<dated-run> -name result.json -print
jq '{status, summary, classification: .aggregates.classification}' \
  data/experiments/<study>/<dated-run>/campaign_results.json
~~~

NAL resume reuses only successful, fingerprint-matched completed experiments.
It does not generally restore a model halfway through training. A training
script may separately implement optimizer-state checkpoints; verify that
contract before relying on it.

After an interrupted shell or session, first check for a live producer and
inspect its artifact directory. Never start a duplicate job based only on the
absence of terminal output.

### 5.9 Aggregate and apply gates

A canonical root is:

~~~text
data/experiments/<study>/<dated-run>/
├── campaign_results.json
└── runs/
    └── <condition>/
        └── seed_<seed>/
            ├── result.json
            ├── model.pt
            └── <declared supplements>
~~~

The exact study layout may vary, but the aggregate must record:

- full configuration and environment;
- schema and producing implementation identity;
- requested, completed, failed, retried, excluded, and reused counts;
- flat primary metrics and per-seed joint gates;
- campaign-level gate and classification;
- artifact links and hashes;
- declared limitations and unexecuted conditional stages.

Reject NaN and infinity in JSON. Missing measurements need an explicit
status/reason. Apply joint gates per seed before aggregation; marginal endpoints
passing on different seeds do not create one passing seed.

Respect decision order. If a competence, validity, or positive-control gate
stops the campaign, do not execute a conditional downstream stage as though it
were confirmatory.

### 5.10 Report from the aggregate

Put the measured report in:

~~~text
docs/08 - Analysis/<project>/YYYY-MM-DD_<study>.md
~~~

Lead with status, hypothesis ID, classification, artifact path, and artifact
hash. Then report campaign integrity, primary endpoints, gates, controls,
mechanistic measurements, interpretation, boundaries, and reproduction.

A useful secondary result cannot rescue a failed primary gate. Say both.

### 5.11 Register and verify read-back

After the aggregate and report stabilize:

1. create or update the typed record under `src/neural_architecture_lab/`;
2. use a registration/finalization CLI under
   `experiments/neural_architecture_lab/`;
3. hash-check the campaign artifact;
4. store the JSON registry and searchable record;
5. query it by the exact hypothesis ID;
6. verify tested status and classification on read-back.

Keep `confirmed: false`, or the equivalent conservative state, whenever the full
preregistered gate failed. Backend telemetry is not verification; read-back is.

### 5.12 Commit the durable record

Before committing:

~~~bash
git status --short
git diff --check
git diff --stat
~~~

Run the focused tests appropriate to the changed files. Commit source, tests,
preregistration, report, and lightweight provenance. Do not force-add ignored
checkpoints or raw campaigns unless repository policy explicitly changes.

Never discard an unfamiliar dirty-worktree change. Completed raw evidence is
append-only; changed interpretations get a dated amendment or new schema/root.

## 6. A5--A7 concrete map

The menu numbering is:

| Menu | Hypothesis | Purpose | Entry point |
| --- | --- | --- | --- |
| A5 | `tinyllm-ttno-cut-localization-parity-v1` | Localize the rank cliff and test parity/topology artifacts | `experiments/structure_net/tinyllm_ttno_cut_localization_parity.py` |
| A6 | `tinyllm-hss-shared-basis-nesting-v1` | Test shared boundary bases and parent-child nesting | `experiments/structure_net/tinyllm_hss_shared_basis_nesting.py` |
| A7 | `tinyllm-causal-h2-attention-v1` | Construct one normalization-faithful causal H² approximation | `experiments/structure_net/tinyllm_causal_h2_attention.py` |

Their design and result pairs are:

- [A5 preregistration](../07%20-%20Status%20Reports/tinyllm/2026-08-30_tinyllm-ttno-cut-localization-parity-preregistration.md)
  and [A5 result](../08%20-%20Analysis/tinyllm/2026-08-30_tinyllm-ttno-cut-localization-parity.md)
- [A6 preregistration](../07%20-%20Status%20Reports/tinyllm/2026-08-30_tinyllm-hss-shared-basis-nesting-preregistration.md)
  and [A6 result](../08%20-%20Analysis/tinyllm/2026-08-30_tinyllm-hss-shared-basis-nesting.md)
- [A7 preregistration](../07%20-%20Status%20Reports/tinyllm/2026-08-30_tinyllm-causal-h2-attention-preregistration.md)
  and [A7 result](../08%20-%20Analysis/tinyllm/2026-08-30_tinyllm-causal-h2-attention.md)

The prospective records were written by
`experiments/neural_architecture_lab/register_dynamic_ttno_followups.py` and
were finalized, hash-checked, and read back by
`experiments/neural_architecture_lab/finalize_dynamic_ttno_followups.py`.

### Completed findings

| Study | Completion | Registered classification | Key boundary |
| --- | --- | --- | --- |
| A5 | 5/5 evaluation seeds, 120/120 frozen cells | `intrinsic_operator_rank_growth` | One checkpoint; evaluation seeds are input prefixes, not model replicates |
| A6 | 5/5 evaluation seeds, all declared cells | `shared_and_nested_hierarchy_supported` | Necessary compatibility, not a simultaneous H² construction |
| A7 | 5/5 evaluation seeds, 480/480 numerical records | `h2_representation_pass_no_finite_size_compression` | Representation passed; storage/work at length 256 remained worse than dense |

The completed registered roots are immutable evidence:

~~~text
data/experiments/tinyllm_ttno_cut_localization_parity/20260830_registered/
data/experiments/tinyllm_hss_shared_basis_nesting/20260830_registered/
data/experiments/tinyllm_causal_h2_attention/20260830_registered/
~~~

Do not rerun a modified implementation into those directories.

## 7. Native-2048 crossover handoff

The long-context follow-up is:

- [A7R/native-2048 preregistration](../07%20-%20Status%20Reports/tinyllm/2026-09-07_tinyllm-native-2048-h2-crossover-preregistration.md)
- [A7R/native-2048 result](../08%20-%20Analysis/tinyllm/2026-09-07_tinyllm-native-2048-h2-crossover.md)
- training entry point:
  `experiments/structure_net/tinyllm_native_2048_pretrain.py`
- crossover entry point:
  `experiments/structure_net/tinyllm_native_2048_h2_crossover.py`
- typed record:
  `src/neural_architecture_lab/native_2048_h2_crossover_hypothesis.py`

### Current closed result

The native model completed all 12,000 optimizer updates and 73,728,000 target
tokens. Its final checkpoint is:

~~~text
data/experiments/tinyllm_native_2048_pretrain/20260907_seed7/checkpoint_step12000.pt
SHA-256 b8cd8a75ae4daac7555dc87d7c8b8e07e41362bba9ef826dc8b8c9ed730edcd6
~~~

On the matched frozen length-256 population:

| Model | Token-mean loss |
| --- | ---: |
| original native-256 checkpoint | 4.8932135 |
| new native-2048 checkpoint | 5.3189554 |

The relative degradation was 8.70%, beyond the locked 5% competence margin.
The registered classification is
`native_long_context_model_competence_failed`.

The decision order therefore stopped the campaign before H² evaluation. No H²
cell ran, and this is a legitimate completed early-stop result, not an
incomplete campaign.

The immutable aggregate is:

~~~text
data/experiments/tinyllm_native_2048_h2_crossover/20260907_registered/campaign_results.json
SHA-256 1a11581da02f819e0a842092fe205ff94c56d15beff1b552a72001d302a448c4
~~~

It was registered and read back as completed. Do not run
`tinyllm_native_2048_h2_crossover.py` against this registered result expecting a
confirmatory H² answer.

The defensible next options both require a new explicitly scoped study:

1. preregister a larger-budget or context-curriculum native-2048 training
   protocol, then apply the same competence gate; or
2. run an explicitly exploratory, non-rescuing H² diagnostic on the failed-gate
   checkpoint under a new artifact root.

At this guide's creation, the A7R source, tests, preregistration, and report were
present as uncommitted work on branch `main` after commit `747c1a6`. A later
session must run `git status --short` rather than assuming that snapshot is still
current, and must not discard those files.

## 8. Focused verification for A5--A7 and A7R

For A5--A7:

~~~bash
pixi run pytest -q \
  tests/structure_net/test_tinyllm_ttno_cut_localization_parity.py \
  tests/structure_net/test_tinyllm_hss_shared_basis_nesting.py \
  tests/structure_net/test_tinyllm_causal_h2_attention.py
~~~

For native-2048:

~~~bash
pixi run pytest -q \
  tests/structure_net/test_tinyllm_native_2048_pretrain.py \
  tests/structure_net/test_tinyllm_native_2048_h2_crossover.py \
  tests/neural_architecture_lab/test_native_2048_h2_early_stop.py \
  tests/neural_architecture_lab/test_native_2048_h2_crossover_hypothesis.py
~~~

For the runner lifecycle:

~~~bash
pixi run pytest -q \
  tests/neural_architecture_lab/test_runner_lifecycle.py \
  tests/neural_architecture_lab/test_local_scheduler.py \
  tests/neural_architecture_lab/test_data_factory_integration.py
~~~

## 9. Failure and recovery rules

### GPU out of memory

Record whether the failure occurred during training, evaluation, aggregation, or
serialization. Check for unrelated resident processes. Reduce concurrency only
if that leaves the scientific computation unchanged, or amend before inspecting
new primary outcomes. Do not call an OOM a scientific failure.

If a final checkpoint was atomically written before an evaluation OOM, a fresh
process may reload that exact checkpoint and perform the unchanged locked
evaluation. Record the incident and verify that no optimizer update was
repeated.

### Interrupted worker

Check the terminal status and fingerprint of every existing `result.json`.
Resume only completed fingerprint matches. Inspect failure records before
retrying. Never retry a threshold miss merely to obtain a passing seed.

### Hash mismatch

Stop. A changed campaign file, checkpoint, source fingerprint, or registry hash
is a provenance failure until explained. Do not update the expected hash merely
to make finalization pass.

### Failed validity or positive control

Apply the preregistered invalidity classification and stop conditional claims.
Secondary measurements may diagnose the problem but cannot rescue the campaign.

### Changed analysis or threshold

Keep the old aggregate and report. Create a dated amendment and, when
machine-readable interpretation changes, a new schema revision or artifact
root.

### Registry warning

Treat only exact ID/classification read-back as success. Chroma or telemetry
warnings alone establish neither success nor failure.

## 10. Release checklist

- [ ] Exact hypothesis ID and current lifecycle state identified
- [ ] Preregistration read and unchanged after outcome inspection
- [ ] User worktree changes preserved
- [ ] Fixed controls, seed roles, splits, endpoints, and stop rules encoded
- [ ] Schema version and scientific fingerprint tested
- [ ] CPU, lifecycle, representative CUDA, and concurrency checks passed
- [ ] Producer frozen or fully content-hashed
- [ ] New dated artifact root used
- [ ] Required per-run evidence and strict aggregate JSON retained
- [ ] Completion counts distinguished from scientific gates
- [ ] Report classification matches the machine-readable decision
- [ ] Secondary findings do not rescue failed gates
- [ ] Registry hash check and exact read-back completed
- [ ] Focused tests and `git diff --check` recorded
- [ ] Source/tests/docs committed; large raw artifacts left in the data store

## 11. Short reading path

For a general new experiment, read in this order:

1. this guide;
2. the [NAL experiment standard](../05%20-%20Standards/NAL-STD-EXPERIMENT-v0.md);
3. the selected study's preregistration;
4. the [local GPU scheduler](../03%20-%20Architecture/nal-local-gpu-scheduler.md);
5. the selected entry point and focused tests;
6. the parent campaign aggregate and analysis, if the study depends on prior
   evidence.

For continuing the current long-context thread, read this guide followed by the
A7 result, native-2048 preregistration, native-2048 result, and current `git
status`. The registered native-2048 campaign is closed; continuation starts with
a new question and preregistration, not another invocation of the old campaign.
