"""
Neural Architecture Lab (NAL)

A systematic framework for testing hypotheses about neural network architectures,
training strategies, and growth patterns. NAL enables scientific exploration of
the design space through reproducible experiments and automated insight extraction.
"""

# Configure environment before any torch imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from structure_net.config import setup_cuda_devices
    setup_cuda_devices()
except ImportError:
    # GPU visibility belongs to the parent launcher. Importing NAL must never
    # silently hide or renumber devices.
    pass

from .core import (
    Hypothesis,
    HypothesisResult,
    Experiment,
    ExperimentResult,
    LabConfig,
    HypothesisCategory,
    ExperimentStatus,
    ExperimentRunnerBase,
    ExperimentWorker,
    HypothesisTestFunction,
)

from .lab import NeuralArchitectureLab

# Import config adapter to enable unified config support
from . import config_adapter

from .hypothesis_library import (
    ArchitectureHypotheses,
    GrowthHypotheses,
    SparsityHypotheses,
    TrainingHypotheses
)

from .runners import (
    ExperimentRunner,
    AsyncExperimentRunner,
    ParallelExperimentRunner
)

from .advanced_runners import (
    AdvancedExperimentRunner,
    GPUMemoryManager
)

from .local_scheduler import (
    DeviceSlotPlan,
    ExperimentResultLedger,
    build_device_slot_plan,
)

from .analyzers import (
    InsightExtractor,
    StatisticalAnalyzer,
    PerformanceAnalyzer
)

__version__ = "0.1.0"

__all__ = [
    # Core types
    "Hypothesis",
    "HypothesisResult", 
    "Experiment",
    "ExperimentResult",
    "LabConfig",
    "HypothesisCategory",
    "ExperimentStatus",
    "ExperimentRunnerBase",
    "ExperimentWorker",
    "HypothesisTestFunction",
    
    # Main lab
    "NeuralArchitectureLab",
    
    # Hypothesis collections
    "ArchitectureHypotheses",
    "GrowthHypotheses",
    "SparsityHypotheses",
    "TrainingHypotheses",
    
    # Runners
    "ExperimentRunner",
    "AsyncExperimentRunner",
    "ParallelExperimentRunner",
    "AdvancedExperimentRunner",
    "GPUMemoryManager",
    "DeviceSlotPlan",
    "ExperimentResultLedger",
    "build_device_slot_plan",
    
    # Analyzers
    "InsightExtractor",
    "StatisticalAnalyzer",
    "PerformanceAnalyzer"
]
