"""
Core interfaces and contracts for the Structure Net component architecture.

This module defines the foundation for a self-aware, contract-driven research framework
where every component declares its capabilities, requirements, and compatibility rules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Set, Type, Union, Optional, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from datetime import datetime
import logging

# --- Maturity and Version System ---

class Maturity(Enum):
    EXPERIMENTAL = "experimental"  # No guarantees, evolving API
    STABLE = "stable"             # Reliable, backward-compatible
    DEPRECATED = "deprecated"     # Being phased out

@dataclass
class ComponentVersion:
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def is_compatible_with(self, other: 'ComponentVersion') -> bool:
        """Major version must match for compatibility"""
        return self.major == other.major

# --- Resource Requirements ---

class ResourceLevel(Enum):
    LOW = "low"        # < 1GB memory, CPU only
    MEDIUM = "medium"  # 1-4GB memory, GPU optional  
    HIGH = "high"      # > 4GB memory, GPU required
    EXTREME = "extreme" # > 16GB memory, multiple GPUs

@dataclass
class ResourceRequirements:
    memory_level: ResourceLevel = ResourceLevel.LOW
    requires_gpu: bool = False
    min_gpu_memory_mb: int = 0
    parallel_safe: bool = True
    estimated_runtime_seconds: Optional[float] = None

# --- Contract System ---

@dataclass
class ComponentContract:
    """A formal declaration of component capabilities and requirements"""
    
    # Identity
    component_name: str
    version: ComponentVersion
    maturity: Maturity
    
    # Data Dependencies  
    required_inputs: Set[str] = field(default_factory=set)
    provided_outputs: Set[str] = field(default_factory=set)
    optional_inputs: Set[str] = field(default_factory=set)
    
    # Resource Requirements
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Compatibility Rules
    requires_component_types: Set[Type] = field(default_factory=set)
    incompatible_with: Set[Type] = field(default_factory=set)
    compatible_maturity_levels: Set[Maturity] = field(default_factory=lambda: {Maturity.STABLE, Maturity.EXPERIMENTAL})
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    
    def validate_compatibility(self, other: 'ComponentContract') -> List[str]:
        """Returns list of compatibility issues, empty if compatible"""
        issues = []
        
        # Version compatibility
        if not self.version.is_compatible_with(other.version):
            issues.append(f"Version mismatch: {self.version} vs {other.version}")
        
        # Maturity compatibility
        if other.maturity not in self.compatible_maturity_levels:
            issues.append(f"Maturity incompatibility: {self.maturity} doesn't accept {other.maturity}")
        
        # Direct incompatibility
        if type(other) in self.incompatible_with:
            issues.append(f"Direct incompatibility with {other.component_name}")
        
        return issues

# --- Core Data Structures ---

class EvolutionContext(dict):
    """Real-time state of the training/evolution loop"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata = {
            'created_at': datetime.now(),
            'epoch': 0,
            'step': 0,
            'device': 'cpu'
        }
    
    @property 
    def epoch(self) -> int:
        return self._metadata['epoch']
    
    @epoch.setter
    def epoch(self, value: int):
        self._metadata['epoch'] = value
    
    @property
    def step(self) -> int:
        return self._metadata['step']
    
    @step.setter  
    def step(self, value: int):
        self._metadata['step'] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()

class AnalysisReport(dict):
    """Consolidated output from all Metrics and Analyzers"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp = datetime.now()
        self.sources = set()  # Track which components contributed
    
    def add_metric_data(self, component_name: str, data: Dict[str, Any]):
        """Add data from a metric component"""
        self[f"metrics.{component_name}"] = data
        self.sources.add(component_name)
    
    def add_analyzer_data(self, component_name: str, data: Dict[str, Any]):
        """Add data from an analyzer component"""
        self[f"analyzers.{component_name}"] = data
        self.sources.add(component_name)

class EvolutionPlan(dict):
    """Actionable plan produced by Strategy components"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority = 1.0  # Higher = more important
        self.estimated_impact = 0.0  # Predicted effect magnitude
        self.created_by = None  # Strategy component that created this
        self.timestamp = datetime.now()

# --- Base Component Interface ---

class IComponent(ABC):
    """The base interface for all self-aware components"""
    
    def __init__(self):
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._execution_count = 0
        self._last_execution_time = None
        self._total_execution_time = 0.0
    
    @property
    @abstractmethod
    def contract(self) -> ComponentContract:
        """This component's formal contract"""
        pass
    
    @property
    def name(self) -> str:
        """Component name, defaults to class name"""
        return self.__class__.__name__
    
    @property
    def maturity(self) -> Maturity:
        """Component maturity level"""
        return self.contract.maturity
    
    def log(self, level: int, message: str, **kwargs):
        """Structured logging with component context"""
        extra = {
            'component_name': self.name,
            'component_type': self.__class__.__name__,
            'execution_count': self._execution_count,
            **kwargs
        }
        self._logger.log(level, message, extra=extra)
    
    def _track_execution(self, func: Callable):
        """Decorator to track execution statistics"""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                self._execution_count += 1
                self._last_execution_time = datetime.now()
                execution_duration = (self._last_execution_time - start_time).total_seconds()
                self._total_execution_time += execution_duration
                
                self.log(logging.DEBUG, f"Execution completed in {execution_duration:.3f}s")
                return result
            except Exception as e:
                self.log(logging.ERROR, f"Execution failed: {str(e)}")
                raise
        return wrapper
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        return {
            'execution_count': self._execution_count,
            'last_execution': self._last_execution_time,
            'total_execution_time': self._total_execution_time,
            'average_execution_time': self._total_execution_time / max(1, self._execution_count)
        }

# --- Component-Specific Interfaces ---

class ILayer(IComponent, nn.Module):
    """Interface for atomic neural network building blocks"""
    
    @abstractmethod
    def get_analysis_properties(self) -> Dict[str, torch.Tensor]:
        """Properties available for metric analysis"""
        pass
    
    @abstractmethod
    def supports_modification(self) -> bool:
        """Can this layer be modified by evolvers?"""
        pass
    
    @abstractmethod
    def add_connections(self, num_connections: int, **kwargs) -> bool:
        """Add connections to this layer"""
        pass

class IModel(IComponent, nn.Module):
    """Interface for complete neural network models"""
    
    @abstractmethod
    def get_layers(self) -> List[ILayer]:
        """Get all layers in this model"""
        pass
    
    @abstractmethod
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get high-level architecture information"""
        pass
    
    @abstractmethod
    def supports_dynamic_growth(self) -> bool:
        """Can this model grow during training?"""
        pass

class ITrainer(IComponent):
    """Interface for training loop management"""
    
    @abstractmethod
    def train_step(self, model: IModel, batch: Any, context: EvolutionContext) -> Dict[str, float]:
        """Execute one training step"""
        pass
    
    @abstractmethod
    def supports_online_evolution(self) -> bool:
        """Can handle model changes during training?"""
        pass

class IMetric(IComponent):
    """Interface for low-level measurements"""
    
    @abstractmethod
    def analyze(self, target: Union[ILayer, IModel], context: EvolutionContext) -> Dict[str, Any]:
        """Analyze target and return measurements"""
        pass
    
    @abstractmethod
    def get_measurement_schema(self) -> Dict[str, type]:
        """Schema of measurements this metric produces"""
        pass

class IAnalyzer(IComponent):
    """Interface for high-level analysis"""
    
    @abstractmethod
    def analyze(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """Perform high-level analysis"""
        pass
    
    @abstractmethod
    def get_required_metrics(self) -> Set[str]:
        """Which metrics this analyzer needs"""
        pass

class IStrategy(IComponent):
    """Interface for high-level decision making"""
    
    @abstractmethod
    def propose_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Generate evolution plan based on analysis"""
        pass
    
    @abstractmethod
    def get_strategy_type(self) -> str:
        """Type of strategy (e.g., 'learning_rate', 'structural', 'hybrid')"""
        pass

class IEvolver(IComponent):
    """Interface for executing evolution plans"""
    
    @abstractmethod
    def can_execute_plan(self, plan: EvolutionPlan) -> bool:
        """Check if this evolver can execute the given plan"""
        pass
    
    @abstractmethod
    def apply_plan(self, plan: EvolutionPlan, model: IModel, trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Execute the evolution plan"""
        pass

class IScheduler(IComponent):
    """Interface for timing evolution cycles"""
    
    @abstractmethod
    def should_trigger(self, context: EvolutionContext) -> bool:
        """Should we trigger evolution now?"""
        pass
    
    @abstractmethod
    def get_next_trigger_estimate(self, context: EvolutionContext) -> Optional[int]:
        """Estimate when next trigger will occur (in steps)"""
        pass

class IOrchestrator(IComponent):
    """Interface for coordinating the entire system"""
    
    @abstractmethod
    def run_cycle(self, context: EvolutionContext) -> Dict[str, Any]:
        """Run one complete evolution cycle"""
        pass
    
    @abstractmethod
    def get_composition_health(self) -> Dict[str, Any]:
        """Get health status of component composition"""
        pass