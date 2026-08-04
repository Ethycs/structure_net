# Developer Guide

**Status:** CURRENT  
**Date:** 2026-08-03  
**Applies to:** local setup, package inspection, and test execution  
**Depends on:** `pyproject.toml`, `pixi.lock`, `pytest.ini`

## Hard constraints

1. Python 3.11 is the declared runtime.
2. The wheel contains both `src/structure_net` and `src/neural_architecture_lab`.
3. CUDA and optional services must not be assumed available merely because their adapters exist.
4. **known rough edge:** repository-wide pytest discovery enters `archive/` and imports a script that calls `sys.exit` at collection time.

## Setup

```bash
pixi install
pixi run install-torch
pixi run test-cuda
```

The project metadata declares JAX in `[project.dependencies]`; most `structure_net` implementation paths use PyTorch, which is installed by a separate Pixi task rather than declared as a wheel dependency.

## Package surfaces

| Surface | Location | Role |
| --- | --- | --- |
| Public convenience API | `src/structure_net/__init__.py` | Sparse networks, evolution helper, LR schedulers |
| Component contracts | `src/structure_net/core/interfaces.py` | Contracts, contexts, reports, plans, component ABCs |
| Component implementation | `src/structure_net/components/` | Layers, models, metrics, analyzers, strategies, trainers, orchestrators |
| Legacy evolution API | `src/structure_net/evolution/` | Older implementations and compatibility paths |
| Research harness | `src/neural_architecture_lab/` | Hypotheses, runners, workers, orchestration, analysis |
| Infrastructure | `src/structure_net/{config,data_factory,logging,profiling,snapshots}/` | Cross-cutting services |

## Testing

Use the scoped suites until collection boundaries are corrected:

```bash
env PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache \
    UV_CACHE_DIR=/tmp/structure-net-uv-cache \
    pixi run pytest tests/structure_net tests/neural_architecture_lab -q
```

Collection-only inspection:

```bash
env PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache \
    UV_CACHE_DIR=/tmp/structure-net-uv-cache \
    pixi run pytest tests --collect-only -q
```

**validated:** on 2026-08-03, unscoped collection reached archived tests and terminated in `archive/root_cruft/test_stress_test_simple.py` after it tried to execute the missing `experiments/ultimate_stress_test_v2.py`. This is a collection defect, not evidence about the active suite's behavior.

**validated active-suite baseline (2026-08-03):** `pixi run pytest tests --collect-only -q` discovered 194 tests and stopped on five collection errors:

| Test surface | Collection failure |
| --- | --- |
| NAL data-factory integration | `ChromaDBClient` is not exported from `structure_net.data_factory.search` |
| Component models and snapshots | `StructuredLayer(nn.Module, ILayer)` has an inconsistent MRO because `ILayer` already inherits `nn.Module` |
| Metric components | `ComponentStatus` is not exported from `structure_net.core` |
| Stress-test memory | `TournamentExecutor` is not exported by the referenced experiment module |

The two MRO errors share one cause, yielding five failing test modules across four causes. No test-execution pass claim is justified until collection succeeds.

## Verification

Run the scoped collection command, then run the scoped suite. If collection or tests fail, record exact failures separately from documentation-only changes.
