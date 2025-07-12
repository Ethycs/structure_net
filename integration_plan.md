# Integration Plan: `src/structure_net/evolution/` Refactoring

This document outlines the plan to refactor the `src/structure_net/evolution/` directory into a modular, component-based architecture as described in `docs/New Componentwise refactoring.md`.

## Phase 1: Foundational Components & De-Monolithing (The "Big Split")

The first and most critical step is to break apart the large, monolithic files into smaller, more focused components that align with the new architecture.

1.  **Create `components` Directory:** Create the new target directory: `src/structure_net/components/`. Inside, create subdirectories for each component type: `metrics`, `analyzers`, `strategies`, `evolvers`, `schedulers`, and `orchestrators`.

2.  **De-Monolith `adaptive_learning_rates`:** This is the most urgent task. The `adaptive_learning_rates.py` file is a massive, deprecated monolith.
    *   Create new files inside `src/structure_net/components/schedulers/` for each distinct scheduler type (e.g., `phase_schedulers.py`, `layer_schedulers.py`).
    *   Move the corresponding classes (`ExtremaPhaseScheduler`, `LayerAgeAwareLR`, etc.) from the old files into these new, focused files.
    *   Refactor each moved class to inherit from `BaseScheduler` (or a more specific base like `BasePhaseScheduler`) and implement the `IComponent` contract.

3.  **De-Monolith `metrics`:** The various metric analysis files (`mutual_information.py`, `activity_analysis.py`, etc.) will be split according to the guide.
    *   **Low-level measurements** will become individual `IMetric` components in `src/structure_net/components/metrics/`. For example, `mutual_information.py` will be split into `LayerMIMetric`, `EntropyMetric`, etc.
    *   **High-level analysis** logic will be moved into `IAnalyzer` components in `src/structure_net/components/analyzers/`. For example, the logic for combining MI and entropy to find bottlenecks will go into an `InformationFlowAnalyzer`.

4.  **De-Monolith `evolution` Components:**
    *   The `ExtremaGrowthStrategy` and other strategies from `components/strategies.py` will be moved to `src/structure_net/components/strategies/`.
    *   The `OptimalGrowthEvolver` from `network_evolver.py` will be refactored into one or more `IEvolver` components in `src/structure_net/components/evolvers/`.

## Phase 2: Implementing the `IComponent` Contract

For each newly created component file from Phase 1, perform the following "contract implementation" steps:

1.  **Inherit from Base:** Ensure each class inherits from the correct base component (e.g., `BaseMetric`, `BaseAnalyzer`, `BaseStrategy`).
2.  **Implement `contract` Property:** Add the `@property def contract(self) -> ComponentContract:` method to each class.
3.  **Define the Contract:** Inside the `contract` property, meticulously define:
    *   `component_name`, `version`, and `maturity`.
    *   `required_inputs`: What data does this component need from the `EvolutionContext` or `AnalysisReport`? (e.g., `{"metrics.activity", "model"}`).
    *   `provided_outputs`: What data does this component produce? (e.g., `{"analyzers.extrema_report"}`).
    *   `resources`: Estimate the `ResourceRequirements`.
4.  **Refactor `apply`/`analyze` Methods:** Refactor the core logic of each component to work with the new `EvolutionContext` and `AnalysisReport` data structures, ensuring they consume their `required_inputs` and produce their `provided_outputs`.

## Phase 3: Orchestration and Integration

Once the individual components are created and conform to their contracts, build the systems that use them.

1.  **Create `MetricsOrchestrator`:** Create a new `MetricsOrchestrator` in `src/structure_net/components/orchestrators/`. This class will replace the old `CompleteMetricsSystem`. Its job will be to run a list of `IMetric` components and assemble the results into a comprehensive `AnalysisReport`.
2.  **Create `EvolutionOrchestrator`:** This will be the main engine. It will take a list of `IAnalyzer`, `IStrategy`, and `IEvolver` components. Its `run_cycle` method will execute the full evolution loop as described in the guide.
3.  **Update `adaptive_learning_rates` Manager:** The `AdaptiveLearningRateManager` will be refactored into an `IOrchestrator` that specifically manages a collection of `IScheduler` components.

## Phase 4: Deprecation and Cleanup

1.  **Add Deprecation Warnings:** Go through all the old, now-monolithic files in `src/structure_net/evolution/` and add clear `DeprecationWarning`s at the top, pointing to the new component locations.
2.  **Update `__init__.py` Files:** Update the `__init__.py` files to ensure they still provide the old classes for backward compatibility, but also issue warnings.
3.  **Final Review:** Do a final pass to ensure all old files are properly marked and the new component system is the clearly recommended path forward.
