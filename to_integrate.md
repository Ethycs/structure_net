# Integration Plan: Refactoring `ultimate_stress_test_v2.py`

**Goal:** Refactor `experiments/ultimate_stress_test_v2.py` to align with the new component-based architecture defined in `docs/New Componentwise refactoring.md` and the NAL architecture.

**Problem:** The current script uses a monolithic `TournamentExecutor` class which handles population management, evolution, and hypothesis creation. This is not modular and does not follow the new component-based design.

**Refactoring Plan:**

1.  **Create `TournamentStrategy` Component:**
    *   **Location:** `src/structure_net/components/strategies/tournament_strategy.py`
    *   **Responsibility:** Inherit from `BaseStrategy`. Its `propose_plan` method will generate the `EvolutionPlan` for a single tournament generation, encapsulating the logic for creating the list of competitors to be evaluated.

2.  **Create `TournamentEvolver` Component:**
    *   **Location:** `src/structure_net/components/evolvers/tournament_evolver.py`
    *   **Responsibility:** Inherit from `BaseEvolver`. It will contain the `crossover` and `mutate` logic. Its `apply_plan` method will take the results of a generation and produce the next generation's population.

3.  **Create `TournamentOrchestrator` Component:**
    *   **Location:** `src/structure_net/components/orchestrators/tournament_orchestrator.py`
    *   **Responsibility:** This will be the main driver for the tournament. It will coordinate the `TournamentStrategy`, the `NeuralArchitectureLab`, and the `TournamentEvolver` to run the full tournament lifecycle.

4.  **Create `tournament_worker.py`:**
    *   **Location:** `src/neural_architecture_lab/workers/tournament_worker.py`
    *   **Responsibility:** This file will contain the `evaluate_competitor_task` function. This is the `test_function` that the NAL will execute for each individual experiment in the tournament.

5.  **Rewrite `ultimate_stress_test_v2.py`:**
    *   **Location:** `experiments/ultimate_stress_test_v2.py`
    *   **Responsibility:** The script will be simplified to be a high-level client. It will instantiate the `TournamentOrchestrator`, handle command-line arguments, and initiate the tournament.
