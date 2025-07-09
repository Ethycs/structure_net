# Contributing to StructureNet

Thank you for your interest in contributing to the StructureNet project! This guide provides a roadmap for where to add new code and how to follow the existing architectural patterns.

## Core Project Structure

The `src/structure_net` directory is organized into several key packages. Understanding their purpose is the first step to contributing effectively.

```
src/structure_net/
├── core/          # Foundational components (layers, network factory)
├── evolution/     # Logic for network growth, adaptation, and evolution
├── models/        # Definitions of specific network architectures
├── metrics/       # Tools for analyzing network properties
├── logging/       # Standardized logging and experiment tracking
├── compactification/ # Network compression and sparse representations
└── ...
```

## Where to Add Your Code

Here’s a guide on where to place new contributions, based on the methodologies outlined in `docs/METHODOLOGY.md`.

### 1. Network Architecture Strategies (`src/structure_net/models/` and `src/structure_net/core/`)

-   **New Layer Types**: If you are creating a new fundamental layer (e.g., a novel type of sparse layer), it should be added to `src/structure_net/core/layers.py`.
-   **New Network Architectures**: If you are combining existing layers into a new, complete network architecture (like the `FiberBundle` network), create a new file in `src/structure_net/models/`. For example, a new `QuantumNetwork` would go in `src/structure_net/models/quantum_network.py`.
-   **Architecture Generation**: Logic for creating specific network architectures (e.g., "funnel" or "column" networks) should be added to the `NetworkFactory` in `src/structure_net/core/network_factory.py`.

### 2. Growth and Evolution Strategies (`src/structure_net/evolution/`)

This is the most complex part of the library and is home to the logic that dynamically changes the network.

-   **New Growth/Evolution Strategies**: If you are implementing a new high-level growth strategy (e.g., a new type of tournament, or a new way to decide when and how to grow), this logic belongs in the `src/structure_net/evolution/` directory. The `OptimalGrowthEvolver` in `network_evolver.py` is a good example of a high-level driver for the evolution process.
-   **New Composable Components**: For new, modular evolution components (the preferred method post-refactor), you should add them to `src/structure_net/evolution/components/`.
    -   A new **Analyzer** (like a new way to detect bottlenecks) should implement the `NetworkAnalyzer` interface.
    -   A new **Growth Strategy** (like a new way to add layers or connections) should implement the `GrowthStrategy` interface.
-   **Gauge Theory**: As we've established, new concepts related to gauge theory, such as `GaugeInvariantOptimizer`, belong in the `src/structure_net/evolution/gauge_theory/` subdirectory.

### 3. Learning Rate Adaptation Strategies (`src/structure_net/evolution/adaptive_learning_rates/`)

The project has a dedicated modular system for learning rate schedulers.

-   **New LR Schedulers**: If you are creating a new learning rate scheduling strategy (e.g., a new phase-based or component-specific scheduler), it should be added as a new class in the appropriate file within `src/structure_net/evolution/adaptive_learning_rates/`. For example, a new scheduler that adapts based on layer curvature would likely go in `layer_schedulers.py`.

### 4. Analysis and Metrics Strategies (`src/structure_net/evolution/metrics/`)

This package is for *measuring* and *analyzing* network properties, not for changing the network.

-   **New Metric Analyzers**: If you are implementing a new type of analysis (e.g., a new way to measure information flow or network health), it should be a new `Analyzer` class in its own file within `src/structure_net/evolution/metrics/`. For example, the `CatastropheAnalyzer` was recently added in `src/structure_net/evolution/metrics/catastrophe_analysis.py`.
-   **Extending Existing Analyzers**: If you are adding a new metric that fits within an existing category (e.g., a new graph metric), add it as a method to the corresponding analyzer (e.g., `GraphAnalyzer`).

### 5. Meta-Learning and Autocorrelation (`src/structure_net/evolution/autocorrelation/`)

This is a specialized part of the evolution package.

-   **New Meta-Learning Logic**: If you are improving the way the system learns from its own performance (e.g., a new way to correlate metrics with success, or a new recommendation algorithm), this code belongs in `src/structure_net/evolution/autocorrelation/`.

### 6. Training and MLOps (`src/integrations/` or `root-level scripts`)

-   **Experiment Scripts**: New high-level experiment scripts that tie everything together (e.g., a new stress test) should be placed in the root-level `experiments/` directory or, if they are for integrating multiple systems, in `src/integrations/`.
-   **Logging**: To add a new type of structured log, you should define a new Pydantic schema in `src/structure_net/logging/schemas.py`.
-   **Seed Hunting/Optimization**: Scripts related to training optimization, like the `GPUSaturatedSeedHunter`, are typically placed in the `utils/` directory if they are standalone tools, or within `src/structure_net/seed_search/` if they are part of the core library.

## Quick Reference Table

| If you are adding...                                     | The best place is...                                       | Example                                        |
| -------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------- |
| A new type of layer (e.g., `QuantumSparseLayer`)         | `src/structure_net/core/layers.py`                         | `StandardSparseLayer`                          |
| A new, complete network architecture (e.g., `Transformer`) | `src/structure_net/models/`                                | `FiberBundle`                                  |
| A new way to measure the network (e.g., "chaos theory")  | `src/structure_net/evolution/metrics/`                     | `CatastropheAnalyzer`                          |
| A new way to grow or change the network                  | `src/structure_net/evolution/components/`                  | `ExtremaGrowthStrategy`                        |
| A new learning rate scheduler                            | `src/structure_net/evolution/adaptive_learning_rates/`     | `LayerAgeAwareLR`                              |
| A new, high-level experiment script                      | `experiments/` or `src/integrations/`                      | `ultimate_stress_test.py`                      |
| A new structured log format                              | `src/structure_net/logging/schemas.py`                     | `GrowthExperiment` schema                      |

By following these guidelines, you can help keep the StructureNet library organized, maintainable, and easy for others to understand and contribute to.
