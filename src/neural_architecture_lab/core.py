"""
Core data structures and interfaces for the Neural Architecture Lab.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional, Protocol, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timezone


def _default_device_ids() -> List[int]:
    """Return logical CUDA ordinals, or the CPU sentinel when unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except (ImportError, RuntimeError):
        pass
    return [-1]


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for persisted experiment state."""
    return datetime.now(timezone.utc)


def _coerce_datetime(value: Optional[datetime]) -> Optional[datetime]:
    """Accept legacy Unix timestamps while exposing one datetime contract."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


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
    created_at: datetime = field(default_factory=utc_now)
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

    def __post_init__(self) -> None:
        self.started_at = _coerce_datetime(self.started_at)
        self.completed_at = _coerce_datetime(self.completed_at)


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
    timestamp: datetime = field(default_factory=utc_now)
    error: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.COMPLETED

    def __post_init__(self) -> None:
        self.timestamp = _coerce_datetime(self.timestamp) or utc_now()
        if isinstance(self.status, str):
            self.status = ExperimentStatus(self.status)
        # Error-bearing results are failures unless a caller explicitly records a
        # terminal status such as cancellation.
        if self.error is not None and self.status == ExperimentStatus.COMPLETED:
            self.status = ExperimentStatus.FAILED

    @property
    def duration(self) -> float:
        """Compatibility alias for consumers using the older result schema."""
        return self.training_time


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
    completed_at: datetime = field(default_factory=utc_now)


import argparse

@dataclass
class LabConfig:
    """Configuration for the Neural Architecture Lab."""
    
    # Execution
    max_parallel_experiments: int = 8
    experiment_timeout: int = 3600  # seconds
    # These are logical ordinals *after* CUDA_VISIBLE_DEVICES is applied.
    device_ids: List[int] = field(default_factory=_default_device_ids)
    gpu_slots_per_device: int = 1
    gpu_memory_per_experiment_gb: Optional[float] = None
    max_gpu_slots_per_device: int = 4
    max_experiment_retries: int = 0
    resume_completed_experiments: bool = False
    
    # Scientific rigor
    min_experiments_per_hypothesis: int = 5
    require_statistical_significance: bool = True
    significance_level: float = 0.05
    
    # Resource management
    max_memory_per_experiment: float = 0.8  # Fraction of GPU memory
    checkpoint_frequency: int = 10  # epochs
    
    # Logging and output
    project_name: str = "nal_experiments"
    results_dir: str = "nal_results"
    save_all_models: bool = False
    save_best_models: bool = True
    verbose: bool = True
    log_level: str = "INFO"
    module_log_levels: Optional[Dict[str, str]] = None
    log_file: Optional[str] = None
    enable_wandb: bool = True
    
    # Adaptive exploration
    enable_adaptive_hypotheses: bool = True
    max_hypothesis_depth: int = 3  # How many follow-up hypotheses to generate
    
    # Auto-balancing settings
    auto_balance: bool = True
    target_cpu_percent: float = 75.0
    max_cpu_percent: float = 90.0
    target_gpu_percent: float = 85.0
    max_gpu_percent: float = 95.0
    target_memory_percent: float = 80.0
    max_memory_percent: float = 90.0

class LabConfigFactory:
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        """Adds all standard NAL configuration arguments to a parser."""
        group = parser.add_argument_group('NAL Configuration')
        group.add_argument('--nal-project-name', type=str, help='Name of the project for logging and organization.')
        group.add_argument('--nal-results-dir', type=str, help='Directory to save all experiment results and logs.')
        group.add_argument('--nal-max-parallel', type=int, help='Maximum number of experiments to run in parallel.')
        group.add_argument('--nal-gpus', type=str, help='Comma-separated logical GPU IDs after CUDA visibility mapping, or "cpu"/"auto".')
        group.add_argument('--nal-gpu-slots', type=int, help='Concurrent experiment slots per logical GPU; 0 selects memory-based calibration.')
        group.add_argument('--nal-gpu-memory-gb', type=float, help='Estimated peak GPU memory per experiment, used when --nal-gpu-slots=0.')
        group.add_argument('--nal-max-retries', type=int, help='Retries after a failed experiment attempt.')
        group.add_argument('--nal-resume', action='store_true', help='Reuse fingerprint-matched completed per-experiment results from an earlier run.')
        group.add_argument('--nal-log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the global logging level for the lab.')
        group.add_argument('--nal-log-file', type=str, help='Path to a file to write all logs.')
        group.add_argument('--nal-enable-wandb', action='store_true', help='Enable logging to Weights & Biases.')
        group.add_argument('--nal-disable-wandb', action='store_true', help='Disable logging to Weights & Biases.')
        group.add_argument('--nal-verbose', action='store_true', help='Enable verbose output from the lab.')
        group.add_argument('--nal-disable-autobalance', action='store_true', help='Disable automatic resource balancing.')
        group.add_argument('--nal-target-cpu', type=float, help='Target CPU utilization percentage (default: 75).')
        group.add_argument('--nal-max-cpu', type=float, help='Maximum CPU utilization percentage (default: 90).')
        group.add_argument('--nal-target-gpu', type=float, help='Target GPU utilization percentage (default: 85).')
        group.add_argument('--nal-max-gpu', type=float, help='Maximum GPU utilization percentage (default: 95).')

    @staticmethod
    def from_args(args: argparse.Namespace, base_config: Optional[LabConfig] = None) -> LabConfig:
        """
        Creates a LabConfig by overriding a base config with provided command-line arguments.
        
        Args:
            args: The parsed arguments from argparse.
            base_config: A default LabConfig object. If None, a default one is created.
            
        Returns:
            A final, merged LabConfig object.
        """
        if base_config is None:
            base_config = LabConfig()

        # Create a dictionary of provided arguments (excluding defaults)
        provided_args = {k: v for k, v in vars(args).items() if v is not None}

        if 'nal_project_name' in provided_args:
            base_config.project_name = provided_args['nal_project_name']
        if 'nal_results_dir' in provided_args:
            base_config.results_dir = provided_args['nal_results_dir']
        if 'nal_max_parallel' in provided_args:
            base_config.max_parallel_experiments = provided_args['nal_max_parallel']
        if 'nal_gpus' in provided_args:
            requested_devices = provided_args['nal_gpus'].strip().lower()
            if requested_devices == 'cpu':
                base_config.device_ids = [-1]
            elif requested_devices == 'auto':
                base_config.device_ids = _default_device_ids()
            else:
                base_config.device_ids = [
                    int(g.strip()) for g in requested_devices.split(',')
                ]
        if 'nal_gpu_slots' in provided_args:
            base_config.gpu_slots_per_device = provided_args['nal_gpu_slots']
        if 'nal_gpu_memory_gb' in provided_args:
            base_config.gpu_memory_per_experiment_gb = provided_args['nal_gpu_memory_gb']
        if 'nal_max_retries' in provided_args:
            base_config.max_experiment_retries = provided_args['nal_max_retries']
        if provided_args.get('nal_resume'):
            base_config.resume_completed_experiments = True
        if 'nal_log_level' in provided_args:
            base_config.log_level = provided_args['nal_log_level']
        if 'nal_log_file' in provided_args:
            base_config.log_file = provided_args['nal_log_file']
        if 'nal_enable_wandb' in provided_args and provided_args['nal_enable_wandb']:
            base_config.enable_wandb = True
        if 'nal_disable_wandb' in provided_args and provided_args['nal_disable_wandb']:
            base_config.enable_wandb = False
        if 'nal_verbose' in provided_args:
            base_config.verbose = True
        if 'nal_disable_autobalance' in provided_args and provided_args['nal_disable_autobalance']:
            base_config.auto_balance = False
        if 'nal_target_cpu' in provided_args:
            base_config.target_cpu_percent = provided_args['nal_target_cpu']
        if 'nal_max_cpu' in provided_args:
            base_config.max_cpu_percent = provided_args['nal_max_cpu']
        if 'nal_target_gpu' in provided_args:
            base_config.target_gpu_percent = provided_args['nal_target_gpu']
        if 'nal_max_gpu' in provided_args:
            base_config.max_gpu_percent = provided_args['nal_max_gpu']
            
        return base_config



class ExperimentRunnerBase(ABC):
    """Backend-neutral asynchronous experiment runner contract."""

    def __init__(
        self,
        config: Optional[LabConfig] = None,
        worker: Optional['ExperimentWorker'] = None,
    ) -> None:
        self.config = config or LabConfig()
        self.worker = worker

    def set_worker(self, worker: 'ExperimentWorker') -> None:
        """Select the callable used by this runner's execution backend."""
        self.worker = worker
    
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


class ExperimentWorker(Protocol):
    """Canonical callable protocol used by experiment runner backends."""

    def __call__(self, experiment: Experiment, device_id: int) -> ExperimentResult:
        ...
