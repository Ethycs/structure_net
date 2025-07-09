"""
Core data structures and interfaces for the Neural Architecture Lab.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import time
from datetime import datetime


class HypothesisCategory(Enum):
    """Categories of hypotheses we can test."""
    ARCHITECTURE = "architecture"
    TRAINING = "training"
    GROWTH = "growth"
    SPARSITY = "sparsity"
    OPTIMIZATION = "optimization"
    REGULARIZATION = "regularization"


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Hypothesis:
    """A testable hypothesis about neural network behavior."""
    
    # Identity
    id: str
    name: str
    description: str
    category: HypothesisCategory
    
    # Scientific method
    question: str  # What are we trying to answer?
    prediction: str  # What do we expect to happen?
    
    # Test specification
    test_function: Callable  # Function that runs the test
    parameter_space: Dict[str, Any]  # Parameters to explore
    control_parameters: Dict[str, Any]  # Parameters to keep fixed
    
    # Success criteria
    success_metrics: Dict[str, float]  # e.g., {"accuracy": 0.5, "efficiency": 0.001}
    statistical_significance: float = 0.05
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)  # Papers, previous experiments
    
    # Results
    tested: bool = False
    results: List['HypothesisResult'] = field(default_factory=list)


@dataclass
class Experiment:
    """A single experiment instance testing a hypothesis."""
    
    id: str
    hypothesis_id: str
    name: str
    
    # Specific parameters for this experiment
    parameters: Dict[str, Any]
    
    # Execution details
    device_id: Optional[int] = None
    seed: Optional[int] = None
    
    # Status tracking
    status: ExperimentStatus = ExperimentStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional['ExperimentResult'] = None


@dataclass
class ExperimentResult:
    """Result of running a single experiment."""
    
    experiment_id: str
    hypothesis_id: str
    
    # Core metrics
    metrics: Dict[str, float]  # All measured metrics
    primary_metric: float  # Main metric for this hypothesis
    
    # Model details
    model_architecture: List[int]
    model_parameters: int
    training_time: float
    
    # Additional data
    training_history: List[Dict[str, float]] = field(default_factory=list)
    model_checkpoint: Optional[str] = None  # Path to saved model
    
    # Insights
    observations: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


@dataclass
class HypothesisResult:
    """Aggregated results from testing a hypothesis."""
    
    hypothesis_id: str
    
    # Experiment summary
    num_experiments: int
    successful_experiments: int
    failed_experiments: int
    
    # Statistical analysis
    confirmed: bool  # Was the hypothesis confirmed?
    confidence: float  # Statistical confidence
    effect_size: float  # How big was the effect?
    
    # Best configuration
    best_parameters: Dict[str, Any]
    best_metrics: Dict[str, float]
    
    # Insights
    key_insights: List[str]
    unexpected_findings: List[str]
    
    # Follow-up suggestions
    suggested_hypotheses: List[str]
    
    # Detailed results
    experiment_results: List[ExperimentResult]
    statistical_summary: Dict[str, Any]
    
    # Metadata
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class LabConfig:
    """Configuration for the Neural Architecture Lab."""
    
    # Execution
    max_parallel_experiments: int = 8
    experiment_timeout: int = 3600  # seconds
    device_ids: List[int] = field(default_factory=lambda: [0, 1])
    
    # Scientific rigor
    min_experiments_per_hypothesis: int = 5
    require_statistical_significance: bool = True
    significance_level: float = 0.05
    
    # Resource management
    max_memory_per_experiment: float = 0.8  # Fraction of GPU memory
    checkpoint_frequency: int = 10  # epochs
    
    # Logging and output
    results_dir: str = "nal_results"
    save_all_models: bool = False
    save_best_models: bool = True
    verbose: bool = True
    
    # Adaptive exploration
    enable_adaptive_hypotheses: bool = True
    max_hypothesis_depth: int = 3  # How many follow-up hypotheses to generate


class ExperimentRunnerBase(ABC):
    """Base class for experiment runners."""
    
    @abstractmethod
    async def run_experiment(self, experiment: Experiment) -> ExperimentResult:
        """Run a single experiment."""
        pass
    
    @abstractmethod
    async def run_experiments(self, experiments: List[Experiment]) -> List[ExperimentResult]:
        """Run multiple experiments."""
        pass


class HypothesisTestFunction(ABC):
    """Base class for hypothesis test functions."""
    
    @abstractmethod
    def __call__(self, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """
        Run the hypothesis test with given configuration.
        
        Returns:
            Tuple of (model, metrics_dict)
        """
        pass