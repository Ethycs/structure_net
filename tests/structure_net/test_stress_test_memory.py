"""Tests for the component-based tournament stress-test migration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from experiments.neural_architecture_lab.ultimate_stress_test_v2 import (
    StressTestConfig,
    create_tournament_hypothesis,
    get_default_lab_config,
)
from neural_architecture_lab.core import ExperimentResult, HypothesisResult
from neural_architecture_lab.orchestrators import TournamentOrchestrator
from structure_net.components.evolvers import TournamentEvolver
from structure_net.components.strategies import TournamentStrategy
from structure_net.core import AnalysisReport, EvolutionContext, EvolutionPlan


def _population():
    return [
        {
            "id": "a",
            "architecture": [4, 3, 2],
            "sparsity": 0.1,
            "lr_strategy": "constant",
            "fitness": 0.0,
            "seed_path": None,
        },
        {
            "id": "b",
            "architecture": [4, 3, 2],
            "sparsity": 0.2,
            "lr_strategy": "cosine",
            "fitness": 0.0,
            "seed_path": None,
        },
    ]


def _result(competitor_id: str, fitness: float) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=f"exp-{competitor_id}",
        hypothesis_id="tournament_gen_0",
        metrics={
            "competitor_id": competitor_id,
            "fitness": fitness,
            "accuracy": fitness / 2,
            "parameters": 23,
        },
        primary_metric=fitness,
        model_architecture=[4, 3, 2],
        model_parameters=23,
        training_time=0.01,
    )


def test_default_lab_config_uses_current_schema():
    config = get_default_lab_config()
    assert config.experiment_timeout == 3600
    assert config.results_dir == "stress_test_results"
    assert config.log_level == "INFO"


def test_thin_client_builds_current_hypothesis_schema():
    config = StressTestConfig(tournament_size=2, generations=1, subset_fraction=0.01)
    hypothesis = create_tournament_hypothesis(config, 0, _population())

    assert hypothesis.parameter_space["params"][0]["competitor_id"] == "a"
    assert hypothesis.control_parameters["epochs"] == config.epochs_per_generation
    assert hypothesis.control_parameters["subset_fraction"] == 0.01
    assert hypothesis.test_function.__name__ == "evaluate_competitor_task"


def test_strategy_and_evolver_share_a_plan_contract():
    population = _population()
    strategy = TournamentStrategy(population)
    proposal = strategy.propose_plan(AnalysisReport(), EvolutionContext(generation=0))
    assert proposal["action_type"] == "evaluate_population"

    plan = EvolutionPlan(
        action_type=proposal["action_type"],
        results=[_result("a", 2.0), _result("a", 4.0), _result("b", 1.0)],
        population=population,
        generation=0,
    )
    evolver = TournamentEvolver(tournament_size=2, mutation_rate=0.0)
    assert evolver.can_execute_plan(plan)
    evolved = evolver.apply_plan(plan, None, None, None)

    assert len(evolved["new_population"]) == 2
    assert evolved["new_population"][0]["id"] == "a"
    assert evolved["new_population"][0]["fitness"] == 3.0
    assert evolved["new_population"][0]["accuracy"] == 1.5


@pytest.mark.asyncio
async def test_orchestrator_runs_one_generation_through_nal():
    config = StressTestConfig(tournament_size=2, generations=1, dataset_name="mnist")
    lab_config = get_default_lab_config()
    experiment_results = [_result("random_0", 2.0), _result("random_1", 1.0)]
    hypothesis_result = HypothesisResult(
        hypothesis_id="tournament_gen_0",
        num_experiments=2,
        successful_experiments=2,
        failed_experiments=0,
        confirmed=True,
        confidence=1.0,
        effect_size=1.0,
        best_parameters={},
        best_metrics={"fitness": 2.0},
        key_insights=[],
        unexpected_findings=[],
        suggested_hypotheses=[],
        experiment_results=experiment_results,
        statistical_summary={},
    )

    fake_lab = Mock()
    fake_lab.test_hypothesis = AsyncMock(return_value=hypothesis_result)

    with patch("neural_architecture_lab.NeuralArchitectureLab", return_value=fake_lab):
        orchestrator = TournamentOrchestrator(lab_config, config)
        generations = await orchestrator.run_cycle()

    assert len(generations) == 1
    assert len(generations[0]) == 2
    fake_lab.register_hypothesis.assert_called_once()
    fake_lab.test_hypothesis.assert_awaited_once_with("tournament_gen_0")
