"""
Tests for the Neural Architecture Lab (NAL) framework.
"""

import pytest
import asyncio
from typing import List, Dict, Any, Tuple

from src.neural_architecture_lab.core import (
    LabConfig,
    Hypothesis,
    HypothesisCategory,
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    ExperimentRunnerBase
)
from src.neural_architecture_lab.lab import NeuralArchitectureLab
from src.neural_architecture_lab.analyzers import StatisticalAnalyzer, InsightExtractor

# --- Mock Components ---

class MockExperimentRunner(ExperimentRunnerBase):
    """A mock runner that returns predefined results."""
    def __init__(self, config: LabConfig, results_to_return: List[ExperimentResult]):
        self.config = config
        self.results_map = {res.experiment_id: res for res in results_to_return}

    async def run_experiment(self, experiment: Experiment) -> ExperimentResult:
        await asyncio.sleep(0.01) # Simulate async work
        result = self.results_map.get(experiment.id)
        if result:
            experiment.status = ExperimentStatus.COMPLETED
            return result
        else:
            experiment.status = ExperimentStatus.FAILED
            return ExperimentResult(
                experiment_id=experiment.id,
                hypothesis_id=experiment.hypothesis_id,
                metrics={}, primary_metric=0.0, model_architecture=[],
                model_parameters=0, training_time=0.1, error="Mock result not found"
            )

    async def run_experiments(self, experiments: List[Experiment]) -> List[ExperimentResult]:
        tasks = [self.run_experiment(exp) for exp in experiments]
        return await asyncio.gather(*tasks)

def mock_test_function(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """A dummy test function for hypotheses."""
    model = "mock_model"
    metrics = {"accuracy": 0.5, "loss": 1.0}
    return model, metrics

# --- Test Fixtures ---

@pytest.fixture
def lab_config(tmp_path):
    """Provides a default LabConfig for tests."""
    return LabConfig(
        results_dir=str(tmp_path),
        min_experiments_per_hypothesis=2, # Keep it small for tests
        enable_adaptive_hypotheses=False
    )

@pytest.fixture
def simple_hypothesis():
    """Provides a simple, standard hypothesis for testing."""
    return Hypothesis(
        id="test_hypothesis_1",
        name="Test Hypothesis",
        description="A simple test case.",
        category=HypothesisCategory.ARCHITECTURE,
        question="Does this test work?",
        prediction="Yes, it should.",
        test_function=mock_test_function,
        parameter_space={'param1': [1, 2]},
        control_parameters={'control1': 10},
        success_metrics={'accuracy': 0.4}
    )

# --- NAL Tests ---

def test_lab_initialization(lab_config):
    """Tests if the NeuralArchitectureLab initializes correctly."""
    lab = NeuralArchitectureLab(lab_config)
    assert lab.config == lab_config
    assert isinstance(lab.runner, ExperimentRunnerBase)
    assert isinstance(lab.insight_extractor, InsightExtractor)
    assert isinstance(lab.statistical_analyzer, StatisticalAnalyzer)

def test_register_hypothesis(lab_config, simple_hypothesis):
    """Tests hypothesis registration."""
    lab = NeuralArchitectureLab(lab_config)
    lab.register_hypothesis(simple_hypothesis)
    assert "test_hypothesis_1" in lab.hypotheses
    assert lab.hypotheses["test_hypothesis_1"] == simple_hypothesis
    assert "test_hypothesis_1" in lab.pending_hypotheses

def test_generate_experiments(lab_config, simple_hypothesis):
    """Tests experiment generation from a hypothesis."""
    lab = NeuralArchitectureLab(lab_config)
    experiments = lab.generate_experiments(simple_hypothesis)
    
    # param1 has 2 values, min_experiments is 2.
    assert len(experiments) == 2 
    
    exp1 = experiments[0]
    assert exp1.hypothesis_id == "test_hypothesis_1"
    assert exp1.parameters['param1'] == 1
    assert exp1.parameters['control1'] == 10

    exp2 = experiments[1]
    assert exp2.parameters['param1'] == 2

@pytest.mark.asyncio
async def test_run_hypothesis_with_mock_runner(lab_config, simple_hypothesis):
    """Tests the full hypothesis testing workflow using a mock runner."""
    # Define the mock results the runner should return
    mock_results = [
        ExperimentResult(
            experiment_id="test_hypothesis_1_exp_000",
            hypothesis_id="test_hypothesis_1",
            metrics={'accuracy': 0.6}, primary_metric=0.6,
            model_architecture=[784, 10], model_parameters=7840, training_time=1.0
        ),
        ExperimentResult(
            experiment_id="test_hypothesis_1_exp_001",
            hypothesis_id="test_hypothesis_1",
            metrics={'accuracy': 0.7}, primary_metric=0.7,
            model_architecture=[784, 10], model_parameters=7840, training_time=1.0
        )
    ]
    
    # Setup lab with the mock runner
    lab = NeuralArchitectureLab(lab_config)
    lab.runner = MockExperimentRunner(lab_config, mock_results)
    lab.register_hypothesis(simple_hypothesis)
    
    # Run the hypothesis test
    result = await lab.test_hypothesis("test_hypothesis_1")
    
    assert result is not None
    assert result.hypothesis_id == "test_hypothesis_1"
    assert result.num_experiments == 2
    assert result.successful_experiments == 2
    
    # Check statistical analysis
    assert result.confirmed == True
    assert result.confidence > 0.95 # Should be statistically significant
    
    # Check insights
    assert len(result.key_insights) > 0
    
    # Check that the hypothesis is no longer pending
    assert "test_hypothesis_1" not in lab.pending_hypotheses
