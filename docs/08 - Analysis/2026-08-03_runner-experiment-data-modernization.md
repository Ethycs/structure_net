# Runner, Experiment, and Data Modernization Analysis

**Status:** VALIDATED  
**Date:** 2026-08-03  
**Applies to:** `src/neural_architecture_lab`, `src/structure_net/data_factory`, runner and storage integrations  
**Depends on:** `../03 - Architecture/structure-net-overview.md`, `../06 - Roadmaps/component-refactor.md`

## Executive verdict

The runner, experiment format, and data story now form one consistent local execution model. The modernization is no longer merely present in parallel implementations: the shared worker protocol, lifecycle timestamps, result shape, storage envelope, search API, and time-series round-trip are covered by active tests.

| Area | Canonical result | Readiness |
| --- | --- | --- |
| Runner | async local runner with logical GPU slots, spawned workers, retry/resume, and a shared `ExperimentWorker` protocol | ✅ Validated on CPU and local CUDA |
| Experiment format | direct-ID dataclasses with timezone-aware lifecycle, explicit result status, and duration compatibility | ✅ Canonical in memory |
| Data | schema-versioned searchable metadata plus JSON/HDF5 time-series storage | ✅ Round-trip validated |
| Canonical local GPU runner | fixed/memory-calibrated slots across logical devices | ✅ Real multi-process hardware shakedown |
| Ray | optional experimental branch, not a declared canonical dependency | ⚪ Deferred explicitly |

## 1. Canonical runner protocol

All local execution surfaces now converge on:

```text
ExperimentWorker := (Experiment, device_id) -> ExperimentResult

Experiment
  -> AsyncExperimentRunner / ParallelExperimentRunner / facade
  -> ExperimentWorker
  -> ExperimentResult
```

`AsyncExperimentRunner` accepts its worker through construction or `set_worker`, and its public run methods no longer require a backend-specific extra argument. The lab and facade use the same shape. A compatibility adapter still accepts the historical hypothesis callable that consumes a parameter dictionary and returns `(model, metrics)`; it normalizes that response into `ExperimentResult` at the boundary.

CPU work executes inline behind the async lifecycle and uses zero DataLoader subprocesses by default. The shakedown demonstrated that PyTorch evaluation could deadlock in a secondary `ThreadPoolExecutor` after Chroma initialized its own pools; inline CPU execution is therefore intentionally sequential. Accelerator work retains executor isolation, using a process for picklable workers and a thread compatibility fallback for local callables.

The advanced and parallel implementations preserve the shared contract. Ray-specific code is not included in the acceptance claim because Ray is not declared in the project environment.

## 2. Canonical experiment lifecycle

`neural_architecture_lab.core` owns the in-memory types:

| Type | Responsibility |
| --- | --- |
| `Hypothesis` | scientific question, callable, parameter space, controls, criteria, and provenance |
| `Experiment` | concrete parameter set, device/seed, lifecycle status, timestamps, and attached result |
| `ExperimentResult` | direct experiment/hypothesis IDs, metrics, architecture, parameter count, duration, history, artifacts, observations, error, and status |
| `HypothesisResult` | aggregate statistical verdict and detailed results |

Lifecycle timestamps are timezone-aware `datetime` values. `Experiment.__post_init__` accepts legacy Unix floats and naive datetimes, then exposes the current UTC-aware contract. Error-bearing results become failed results unless a caller records another terminal status. `ExperimentResult.duration` remains a compatibility alias for `training_time`.

The runtime dataclasses remain Python objects and may contain callables or flexible dictionaries. Durable storage therefore uses a separate versioned envelope rather than pretending every runtime field is directly portable.

## 3. Canonical data flow

```text
ExperimentResult
  -> NALChromaIntegration schema_version=1 envelope
     -> searchable experiment/config/metric summary in Chroma
     -> training-history reference when history is offloaded
        -> HDF5 for histories at/above the threshold
        -> compressed JSON below the threshold
  -> retrieval with IDs, metrics, metadata, and history preserved
```

The search package exports `ChromaSearchClient` as the canonical name and `ChromaDBClient` as a compatibility alias. Search helpers expose compatible add/get/search operations, and the hybrid storage layer uses installed-package imports and one versioned envelope.

`TimeSeriesStorage` uses one project-relative default root. HDF5 stores a canonical serialized history alongside numeric columns so mixed numeric and structured epoch data round-trip exactly; compressed JSON covers small histories. Both formats share key normalization, metadata, epoch selection, and cache behavior.

`NALChromaIntegration` consumes the current direct-ID `ExperimentResult` schema while tolerating the older embedded-experiment form. `MemoryEfficientNAL` can be constructed from an existing lab or configuration and offloads completed results through the same integration.

## 4. Validated evidence

Focused acceptance paths include:

- runner lifecycle and worker adaptation: `tests/neural_architecture_lab/test_runner_lifecycle.py`;
- result conversion, Chroma integration, hybrid history, and memory offload: `tests/neural_architecture_lab/test_data_factory_integration.py`;
- unified configuration compatibility: `tests/structure_net/test_config_migration.py`;
- tournament use of the current runner and experiment formats: `tests/structure_net/test_stress_test_memory.py`.

The repository-wide active suite completes with zero collection errors and zero failures. See `../07 - Status Reports/2026-08-03_component-migration-complete.md` for the exact final count.

### Tournament shakedown

The canonical CPU path was exercised outside pytest through `ultimate_stress_test_v2.py`: one generation, two competitors, six generated replicates, cached MNIST, a 0.1% dataset subset, zero training epochs, and one local runner slot. All evaluations, result logging, statistical aggregation, and population evolution completed with exit code 0.

Because this run intentionally used zero training epochs and only ten test samples per replicate, it validates execution rather than the tournament hypothesis. The persisted analysis recorded six successful experiments and an unconfirmed hypothesis (`p = 0.0953`).

The shakedown materially changed the implementation: all active source/tests/experiments now use installed package imports, CPU loader settings are always propagated, CPU PyTorch workers run inline, the CLI supports `--subset-fraction`, tournament replicas are averaged by competitor, and evolved competitors retain accuracy and parameter counts.

## 5. Boundaries and next decisions

1. The default `run_structure_net_experiment` training implementation still reaches legacy evolution shims. The protocol around it is modern; replacing every training primitive is separate retirement work.
2. A live external Chroma service is not required for the local acceptance path; tests use local persistence.
3. Slot calibration is admission control rather than memory isolation; each model family still needs a representative peak-memory pilot.
4. Ray must either become a declared optional dependency with equivalence tests or remain explicitly experimental.
5. A future schema version should be introduced only when the wire envelope changes incompatibly; flexible metric/config contents alone do not justify replacing the current in-memory types.
6. Chroma/PostHog telemetry callback errors and a Pydantic dataset-registry serialization warning remain non-fatal dependency noise; they should be resolved before treating logs as operationally clean.

## Verification

```bash
env PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache \
    UV_CACHE_DIR=/tmp/structure-net-uv-cache \
    pixi run pytest -q \
      tests/neural_architecture_lab/test_runner_lifecycle.py \
      tests/neural_architecture_lab/test_data_factory_integration.py \
      tests/structure_net/test_config_migration.py \
      tests/structure_net/test_stress_test_memory.py
```
