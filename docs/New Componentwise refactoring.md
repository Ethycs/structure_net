# **The Structure Net Refactoring Guide: Complete System Transformation**

This is a comprehensive programmer's guide for transforming Structure Net into a self-aware, contract-driven research framework. Follow this guide step-by-step to migrate your existing codebase.

## **Table of Contents**
1. [Foundation: Core Interfaces and Contracts](#foundation)
2. [The Compatibility Manager System](#compatibility)
3. [Component Implementation Guide](#implementation)
4. [Migration Mapping: Old Files → New Components](#migration)
5. [Advanced Features: Self-Awareness and Validation](#advanced)
6. [Testing and Validation Framework](#testing)
7. [Usage Examples and Patterns](#examples)

---

## **1. Foundation: Core Interfaces and Contracts** {#foundation}

### **1.1 Create the Master Interface File**

```python
# src/structure_net/core/interfaces.py
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
```

---

## **2. The Compatibility Manager System** {#compatibility}

### **2.1 Core Compatibility Classes**

```python
# src/structure_net/core/compatibility.py
from typing import List, Dict, Set, Type, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from .interfaces import IComponent, ComponentContract, Maturity

class CompatibilityLevel(Enum):
    COMPATIBLE = "compatible"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class CompatibilityIssue:
    level: CompatibilityLevel
    component_a: str
    component_b: str
    description: str
    suggested_fix: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.level.value.upper()}: {self.description}"

class ComponentRegistry:
    """Central registry of all available components"""
    
    def __init__(self):
        self._components: Dict[str, Type[IComponent]] = {}
        self._contracts: Dict[str, ComponentContract] = {}
        self._logger = logging.getLogger(__name__)
    
    def register(self, component_class: Type[IComponent]):
        """Register a component class"""
        # Get contract from a temporary instance
        temp_instance = component_class()
        contract = temp_instance.contract
        
        self._components[contract.component_name] = component_class
        self._contracts[contract.component_name] = contract
        
        self._logger.info(f"Registered component: {contract.component_name} v{contract.version}")
    
    def get_available_components(self, component_type: Type[IComponent] = None) -> Dict[str, Type[IComponent]]:
        """Get all available components, optionally filtered by type"""
        if component_type is None:
            return self._components.copy()
        
        return {
            name: cls for name, cls in self._components.items()
            if issubclass(cls, component_type)
        }
    
    def get_contract(self, component_name: str) -> Optional[ComponentContract]:
        """Get contract for a component"""
        return self._contracts.get(component_name)
    
    def suggest_compatible_components(self, required_outputs: Set[str], 
                                   component_type: Type[IComponent] = None) -> List[str]:
        """Suggest components that can provide required outputs"""
        suggestions = []
        
        for name, contract in self._contracts.items():
            if component_type and not issubclass(self._components[name], component_type):
                continue
                
            if required_outputs.issubset(contract.provided_outputs):
                suggestions.append(name)
        
        return suggestions

class CompatibilityManager:
    """Validates component compositions and suggests fixes"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self._logger = logging.getLogger(__name__)
    
    def validate_composition(self, components: List[IComponent]) -> List[CompatibilityIssue]:
        """Validate a composition of components"""
        issues = []
        
        # Build contracts map
        contracts = {comp.name: comp.contract for comp in components}
        
        # 1. Check pairwise compatibility
        for i, comp_a in enumerate(components):
            for comp_b in components[i+1:]:
                issues.extend(self._check_pairwise_compatibility(comp_a, comp_b))
        
        # 2. Check data flow
        issues.extend(self._check_data_flow(components))
        
        # 3. Check resource conflicts
        issues.extend(self._check_resource_conflicts(components))
        
        # 4. Check maturity mixing
        issues.extend(self._check_maturity_mixing(components))
        
        return issues
    
    def _check_pairwise_compatibility(self, comp_a: IComponent, comp_b: IComponent) -> List[CompatibilityIssue]:
        """Check compatibility between two components"""
        issues = []
        
        contract_issues = comp_a.contract.validate_compatibility(comp_b.contract)
        
        for issue_desc in contract_issues:
            issues.append(CompatibilityIssue(
                level=CompatibilityLevel.ERROR,
                component_a=comp_a.name,
                component_b=comp_b.name,
                description=issue_desc,
                suggested_fix=f"Remove {comp_a.name} or {comp_b.name} from composition"
            ))
        
        return issues
    
    def _check_data_flow(self, components: List[IComponent]) -> List[CompatibilityIssue]:
        """Check that data dependencies are satisfied"""
        issues = []
        
        # Collect all provided outputs
        all_provided = set()
        for comp in components:
            all_provided.update(comp.contract.provided_outputs)
        
        # Check each component's requirements
        for comp in components:
            missing = comp.contract.required_inputs - all_provided
            if missing:
                # Try to suggest components that could provide missing data
                suggestions = []
                for missing_output in missing:
                    providers = self.registry.suggest_compatible_components({missing_output})
                    suggestions.extend(providers)
                
                suggested_fix = f"Add one of: {suggestions}" if suggestions else "No compatible components found"
                
                issues.append(CompatibilityIssue(
                    level=CompatibilityLevel.CRITICAL,
                    component_a=comp.name,
                    component_b="system",
                    description=f"Missing required inputs: {missing}",
                    suggested_fix=suggested_fix
                ))
        
        return issues
    
    def _check_resource_conflicts(self, components: List[IComponent]) -> List[CompatibilityIssue]:
        """Check for resource conflicts"""
        issues = []
        
        # Estimate total memory usage
        total_memory_score = 0
        gpu_required = False
        
        for comp in components:
            resources = comp.contract.resources
            
            # Simple scoring system
            memory_scores = {
                'low': 1, 'medium': 3, 'high': 8, 'extreme': 20
            }
            total_memory_score += memory_scores.get(resources.memory_level.value, 1)
            
            if resources.requires_gpu:
                gpu_required = True
        
        # Check for potential memory issues
        if total_memory_score > 15:  # Arbitrary threshold
            issues.append(CompatibilityIssue(
                level=CompatibilityLevel.WARNING,
                component_a="system",
                component_b="resources",
                description=f"High memory usage predicted (score: {total_memory_score})",
                suggested_fix="Consider running components separately or using smaller batch sizes"
            ))
        
        if gpu_required:
            issues.append(CompatibilityIssue(
                level=CompatibilityLevel.WARNING,
                component_a="system", 
                component_b="resources",
                description="GPU required for this composition",
                suggested_fix="Ensure GPU is available or use CPU-compatible alternatives"
            ))
        
        return issues
    
    def _check_maturity_mixing(self, components: List[IComponent]) -> List[CompatibilityIssue]:
        """Check for risky maturity combinations"""
        issues = []
        
        experimental_components = [c for c in components if c.maturity == Maturity.EXPERIMENTAL]
        stable_components = [c for c in components if c.maturity == Maturity.STABLE]
        
        if experimental_components and stable_components:
            issues.append(CompatibilityIssue(
                level=CompatibilityLevel.WARNING,
                component_a="experimental",
                component_b="stable",
                description=f"Mixing experimental ({len(experimental_components)}) and stable ({len(stable_components)}) components",
                suggested_fix="Use caution - experimental components may affect overall stability"
            ))
        
        return issues
    
    def suggest_composition_fixes(self, issues: List[CompatibilityIssue]) -> Dict[str, List[str]]:
        """Suggest ways to fix composition issues"""
        suggestions = {
            'remove_components': [],
            'add_components': [],
            'replace_components': [],
            'configuration_changes': []
        }
        
        for issue in issues:
            if issue.level == CompatibilityLevel.CRITICAL:
                if "Missing required inputs" in issue.description:
                    suggestions['add_components'].append(issue.suggested_fix)
                elif "incompatibility" in issue.description.lower():
                    suggestions['remove_components'].append(f"Remove {issue.component_a} or {issue.component_b}")
        
        return suggestions

class CompositionBuilder:
    """Helper for building valid component compositions"""
    
    def __init__(self, registry: ComponentRegistry, compatibility_manager: CompatibilityManager):
        self.registry = registry
        self.compatibility_manager = compatibility_manager
        self._components: List[IComponent] = []
    
    def add_component(self, component: IComponent) -> 'CompositionBuilder':
        """Add a component and validate incrementally"""
        # Check compatibility with existing components
        temp_composition = self._components + [component]
        issues = self.compatibility_manager.validate_composition(temp_composition)
        
        critical_issues = [i for i in issues if i.level == CompatibilityLevel.CRITICAL]
        if critical_issues:
            raise ValueError(f"Cannot add {component.name}: {critical_issues[0].description}")
        
        self._components.append(component)
        
        # Log warnings
        warnings = [i for i in issues if i.level == CompatibilityLevel.WARNING]
        for warning in warnings:
            logging.warning(f"Composition warning: {warning}")
        
        return self
    
    def auto_complete(self, target_functionality: Set[str]) -> 'CompositionBuilder':
        """Automatically add components to satisfy target functionality"""
        current_outputs = set()
        for comp in self._components:
            current_outputs.update(comp.contract.provided_outputs)
        
        missing = target_functionality - current_outputs
        
        while missing:
            # Find a component that provides some missing functionality
            found = False
            
            for component_name in self.registry.get_available_components():
                contract = self.registry.get_contract(component_name)
                if contract and missing.intersection(contract.provided_outputs):
                    # Try to add this component
                    try:
                        component_class = self.registry.get_available_components()[component_name]
                        component = component_class()
                        self.add_component(component)
                        
                        # Update missing set
                        current_outputs.update(contract.provided_outputs)
                        missing = target_functionality - current_outputs
                        found = True
                        break
                    except ValueError:
                        # This component didn't work, try next
                        continue
            
            if not found:
                raise ValueError(f"Cannot satisfy requirements: {missing}")
        
        return self
    
    def build(self) -> List[IComponent]:
        """Build and validate final composition"""
        issues = self.compatibility_manager.validate_composition(self._components)
        
        critical_issues = [i for i in issues if i.level == CompatibilityLevel.CRITICAL]
        if critical_issues:
            raise ValueError(f"Composition has critical issues: {[str(i) for i in critical_issues]}")
        
        return self._components.copy()
```

---

## **3. Component Implementation Guide** {#implementation}

### **3.1 Base Component Classes**

```python
# src/structure_net/core/base_components.py
from abc import abstractmethod
import time
from typing import Dict, Any, Set
import torch
import torch.nn as nn
from .interfaces import *

class BaseComponent(IComponent):
    """Provides common functionality for all components"""
    
    def __init__(self, name: Optional[str] = None, version: Optional[ComponentVersion] = None):
        super().__init__()
        self._name = name or self.__class__.__name__
        self._version = version or ComponentVersion()
        self._performance_tracker = {
            'call_count': 0,
            'total_time': 0.0,
            'last_call_time': 0.0
        }
    
    def _measure_performance(self, func):
        """Decorator to measure performance"""
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                self._performance_tracker['call_count'] += 1
                self._performance_tracker['total_time'] += duration
                self._performance_tracker['last_call_time'] = duration
                
                return result
            except Exception as e:
                self.log(logging.ERROR, f"Component {self.name} failed: {str(e)}")
                raise
        return wrapper
    
    @property
    def name(self) -> str:
        return self._name
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance statistics"""
        count = self._performance_tracker['call_count']
        return {
            'call_count': count,
            'total_time': self._performance_tracker['total_time'],
            'average_time': self._performance_tracker['total_time'] / max(1, count),
            'last_call_time': self._performance_tracker['last_call_time']
        }

class BaseLayer(BaseComponent, ILayer):
    """Base implementation for neural network layers"""
    
    def __init__(self, *args, **kwargs):
        BaseComponent.__init__(self)
        nn.Module.__init__(self)
        
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            provided_outputs={"layer.weights", "layer.activations"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW)
        )
    
    def get_analysis_properties(self) -> Dict[str, torch.Tensor]:
        """Default implementation - override in subclasses"""
        properties = {}
        
        # Add weight information if available
        for name, param in self.named_parameters():
            if 'weight' in name:
                properties[f"weights.{name}"] = param.data
                
        return properties
    
    def supports_modification(self) -> bool:
        """Default: layers support modification"""
        return True
    
    def add_connections(self, num_connections: int, **kwargs) -> bool:
        """Default implementation - override in subclasses"""
        self.log(logging.WARNING, f"add_connections not implemented for {self.__class__.__name__}")
        return False

class BaseModel(BaseComponent, IModel):
    """Base implementation for neural network models"""
    
    def __init__(self, *args, **kwargs):
        BaseComponent.__init__(self)
        nn.Module.__init__(self)
        self._layers: List[ILayer] = []
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            provided_outputs={"model.architecture", "model.parameters", "model.forward"},
            resources=ResourceRequirements(memory_level=ResourceLevel.MEDIUM)
        )
    
    def get_layers(self) -> List[ILayer]:
        """Get all layers that implement ILayer interface"""
        layers = []
        for module in self.modules():
            if isinstance(module, ILayer) and module != self:
                layers.append(module)
        return layers
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get high-level architecture information"""
        layers = self.get_layers()
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'num_layers': len(layers),
            'total_parameters': total_params,
            'layer_types': [type(layer).__name__ for layer in layers],
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def supports_dynamic_growth(self) -> bool:
        """Default: models support growth if all layers do"""
        return all(layer.supports_modification() for layer in self.get_layers())

class BaseMetric(BaseComponent, IMetric):
    """Base implementation for metrics"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._schema = {}
    
    @property 
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "layer"},
            provided_outputs={f"metrics.{self.name.lower()}"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW)
        )
    
    @abstractmethod
    def _compute_metric(self, target: Union[ILayer, IModel], context: EvolutionContext) -> Dict[str, Any]:
        """Override this to implement metric logic"""
        pass
    
    def analyze(self, target: Union[ILayer, IModel], context: EvolutionContext) -> Dict[str, Any]:
        """Public interface with performance tracking"""
        return self._measure_performance(self._compute_metric)(target, context)
    
    def get_measurement_schema(self) -> Dict[str, type]:
        return self._schema

class BaseAnalyzer(BaseComponent, IAnalyzer):
    """Base implementation for analyzers"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._required_metrics: Set[str] = set()
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs=self._required_metrics,
            provided_outputs={f"analyzers.{self.name.lower()}"},
            resources=ResourceRequirements(memory_level=ResourceLevel.MEDIUM)
        )
    
    def get_required_metrics(self) -> Set[str]:
        return self._required_metrics
    
    @abstractmethod
    def _perform_analysis(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """Override this to implement analysis logic"""
        pass
    
    def analyze(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """Public interface with validation and performance tracking"""
        # Validate required metrics are present
        missing_metrics = self._required_metrics - set(report.keys())
        if missing_metrics:
            raise ValueError(f"Missing required metrics: {missing_metrics}")
        
        return self._measure_performance(self._perform_analysis)(model, report, context)

class BaseStrategy(BaseComponent, IStrategy):
    """Base implementation for strategies"""
    
    def __init__(self, name: str = None, strategy_type: str = "generic"):
        super().__init__(name)
        self._strategy_type = strategy_type
        self._required_analysis: Set[str] = set()
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs=self._required_analysis,
            provided_outputs={f"plans.{self._strategy_type}"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW)
        )
    
    def get_strategy_type(self) -> str:
        return self._strategy_type
    
    @abstractmethod
    def _create_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Override this to implement strategy logic"""
        pass
    
    def propose_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Public interface with validation"""
        # Validate required analysis data is present
        missing_data = self._required_analysis - set(report.keys())
        if missing_data:
            self.log(logging.WARNING, f"Missing recommended analysis data: {missing_data}")
        
        plan = self._measure_performance(self._create_plan)(report, context)
        plan.created_by = self.name
        return plan

class BaseEvolver(BaseComponent, IEvolver):
    """Base implementation for evolvers"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._supported_plan_types: Set[str] = set()
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={f"plans.{pt}" for pt in self._supported_plan_types},
            provided_outputs={"evolution.applied_changes"},
            resources=ResourceRequirements(memory_level=ResourceLevel.MEDIUM)
        )
    
    def can_execute_plan(self, plan: EvolutionPlan) -> bool:
        """Check if this evolver can handle the plan"""
        # Check if plan contains any keys this evolver can handle
        plan_keys = set(plan.keys())
        evolver_capabilities = self._supported_plan_types
        
        return bool(plan_keys.intersection(evolver_capabilities))
    
    @abstractmethod
    def _execute_plan(self, plan: EvolutionPlan, model: IModel, trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Override this to implement evolution logic"""
        pass
    
    def apply_plan(self, plan: EvolutionPlan, model: IModel, trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Public interface with validation"""
        if not self.can_execute_plan(plan):
            raise ValueError(f"Evolver {self.name} cannot execute this plan")
        
        return self._measure_performance(self._execute_plan)(plan, model, trainer, optimizer)
```

---

## **4. Migration Mapping: Old Files → New Components** {#migration}

### **4.1 Migration Strategy**

```python
# migration/migration_plan.py
"""
Migration mapping from old Structure Net files to new component architecture
"""

MIGRATION_MAP = {
    # LAYERS: core/layers.py -> components/layers/
    "core/layers.py": {
        "CompactLayer": "components/layers/compact_layer.py",
        "ExtremaAwareSparseLayer": "components/layers/extrema_aware_layer.py",
        "TemporaryPatchLayer": "components/layers/temporary_patch_layer.py"
    },
    
    # MODELS: models/ -> components/models/
    "models/minimal_network.py": "components/models/minimal_network.py",
    "models/fiber_bundle_network.py": "components/models/fiber_bundle_network.py", 
    "models/modern_multi_scale_network.py": "components/models/multi_scale_network.py",
    
    # METRICS: evolution/metrics/ -> components/metrics/
    "evolution/metrics/mutual_information.py": [
        "components/metrics/layer_mi_metric.py",
        "components/metrics/entropy_metric.py", 
        "components/metrics/advanced_mi_metric.py",
        "components/metrics/information_flow_metric.py",
        "components/metrics/redundancy_metric.py",
        "components/analyzers/information_flow_analyzer.py"
    ],
    "evolution/metrics/homological_analysis.py": [
        "components/metrics/chain_complex_metric.py",
        "components/metrics/rank_metric.py",
        "components/metrics/betti_number_metric.py",
        "components/metrics/homology_metric.py",
        "components/metrics/information_efficiency_metric.py",
        "components/analyzers/homological_analyzer.py"
    ],
    "evolution/metrics/sensitivity_analysis.py": [
        "components/metrics/gradient_sensitivity_metric.py",
        "components/metrics/bottleneck_metric.py",
        "components/analyzers/sensitivity_analyzer.py"
    ],
    "evolution/metrics/topological_analysis.py": [
        "components/metrics/extrema_metric.py",
        "components/metrics/persistence_metric.py",
        "components/metrics/connectivity_metric.py",
        "components/metrics/topological_signature_metric.py",
        "components/analyzers/topological_analyzer.py"
    ],
    "evolution/metrics/activity_analysis.py": "components/metrics/activity_metric.py",  # Pending
    "evolution/metrics/graph_analysis.py": "components/metrics/graph_metric.py",  # Pending
    "evolution/metrics/catastrophe_analysis.py": "components/metrics/catastrophe_metric.py",  # Pending
    "evolution/metrics/compactification_metrics.py": "components/metrics/compactification_metric.py",  # Pending
    
    # ANALYZERS: evolution/components/analyzers.py + evolution/extrema_analyzer.py -> components/analyzers/
    "evolution/components/analyzers.py": "components/analyzers/network_analyzer.py",
    "evolution/extrema_analyzer.py": "components/analyzers/extrema_analyzer.py",
    
    # STRATEGIES: evolution/adaptive_learning_rates/ -> components/strategies/
    "evolution/adaptive_learning_rates/layer_schedulers.py": "components/strategies/layerwise_rate_strategy.py",
    "evolution/adaptive_learning_rates/connection_schedulers.py": "components/strategies/connection_rate_strategy.py",
    "evolution/adaptive_learning_rates/phase_schedulers.py": "components/strategies/phase_strategy.py",
    "evolution/adaptive_learning_rates/unified_manager.py": "components/orchestrators/unified_orchestrator.py",
    
    # EVOLVERS: evolution/components/strategies.py + evolution/network_evolver.py -> components/evolvers/
    "evolution/components/strategies.py": "components/evolvers/structural_evolver.py",
    "evolution/network_evolver.py": "components/evolvers/network_evolver.py",
    "evolution/residual_blocks.py": "components/evolvers/residual_evolver.py",
    
    # SCHEDULERS: Create new from timing logic in existing files
    "evolution/adaptive_learning_rates/": "components/schedulers/",
    
    # TRAINERS: Extract from experiment files -> components/trainers/
    "experiments/ultimate_stress_test_v2.py": "components/trainers/standard_trainer.py"
}
```

### **4.2 Example Migration: Compact Layer**

```python
# OLD: core/layers.py (relevant parts)
class CompactLayer(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.98):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.sparsity = sparsity
        # ... existing logic

# NEW: components/layers/compact_layer.py
from src.structure_net.core.base_components import BaseLayer
from src.structure_net.core.interfaces import ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel

class CompactLayer(BaseLayer):
    """Refactored CompactLayer with self-aware contract"""
    
    def __init__(self, in_features: int, out_features: int, sparsity: float = 0.98, name: str = None):
        super().__init__(name or f"CompactLayer_{in_features}x{out_features}")
        
        # Original CompactLayer logic
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Initialize sparsity pattern
        self._initialize_sparsity()
    
    @property
    def contract(self) -> ComponentContract:
        """Self-aware contract declaration"""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(2, 0, 0),  # Upgraded for new architecture
            maturity=Maturity.STABLE,
            
            # This layer provides weight matrices for analysis
            provided_outputs={
                "layer.weights.full_matrix",
                "layer.weights.sparse_pattern", 
                "layer.properties.sparsity_ratio",
                "layer.properties.in_features",
                "layer.properties.out_features"
            },
            
            # Resource requirements
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def get_analysis_properties(self) -> Dict[str, torch.Tensor]:
        """Provide properties for metric analysis"""
        return {
            "full_weight_matrix": self.weight.data,
            "sparse_pattern": (self.weight.data != 0).float(),
            "bias": self.bias.data,
            "sparsity_mask": self._get_sparsity_mask()
        }
    
    def supports_modification(self) -> bool:
        """This layer supports dynamic modification"""
        return True
    
    def add_connections(self, num_connections: int, **kwargs) -> bool:
        """Add random connections to the layer"""
        try:
            # Find zero positions
            zero_positions = (self.weight.data == 0).nonzero()
            
            if len(zero_positions) < num_connections:
                self.log(logging.WARNING, f"Only {len(zero_positions)} zero positions available")
                num_connections = len(zero_positions)
            
            # Randomly select positions to activate
            indices = torch.randperm(len(zero_positions))[:num_connections]
            selected_positions = zero_positions[indices]
            
            # Initialize new connections
            for pos in selected_positions:
                i, j = pos[0], pos[1]
                self.weight.data[i, j] = torch.randn(1) * 0.01
            
            self.log(logging.INFO, f"Added {num_connections} connections")
            return True
            
        except Exception as e:
            self.log(logging.ERROR, f"Failed to add connections: {str(e)}")
            return False
    
    def _initialize_sparsity(self):
        """Initialize the sparse connection pattern"""
        # Keep original initialization logic
        pass
    
    def _get_sparsity_mask(self) -> torch.Tensor:
        """Get current sparsity mask"""
        return (self.weight.data != 0).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - keep original logic"""
        return torch.nn.functional.linear(x, self.weight, self.bias)
```

### **4.3 Example Migration: Extrema Analyzer**

```python
# OLD: evolution/extrema_analyzer.py (simplified)
class StandardExtremaAnalyzer:
    def __init__(self, max_batches=10, dead_threshold=0.01):
        self.max_batches = max_batches
        self.dead_threshold = dead_threshold
    
    def analyze(self, context):
        # ... existing logic
        pass

# NEW: components/analyzers/extrema_analyzer.py
from src.structure_net.core.base_components import BaseAnalyzer
from src.structure_net.core.interfaces import *

class ExtremaAnalyzer(BaseAnalyzer):
    """Refactored extrema analyzer with self-aware contract"""
    
    def __init__(self, max_batches: int = 10, dead_threshold: float = 0.01, 
                 saturated_multiplier: float = 2.5, name: str = None):
        super().__init__(name or "ExtremaAnalyzer")
        
        # Configuration
        self.max_batches = max_batches
        self.dead_threshold = dead_threshold
        self.saturated_multiplier = saturated_multiplier
        
        # Declare what metrics we need
        self._required_metrics = {
            "metrics.activity",
            "metrics.sparsity"
        }
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(2, 0, 0),
            maturity=Maturity.STABLE,
            
            # We need activity and sparsity metrics
            required_inputs=self._required_metrics,
            
            # We provide extrema analysis
            provided_outputs={
                "analyzers.extrema_report",
                "analyzers.dead_neurons",
                "analyzers.saturated_neurons",
                "analyzers.extrema_ratio"
            },
            
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False  # Can work on CPU
            )
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """Perform extrema analysis using available metrics"""
        
        # Extract required data from the analysis report
        activity_data = report.get("metrics.activity", {})
        sparsity_data = report.get("metrics.sparsity", {})
        
        if not activity_data:
            raise ValueError("Activity metric data not available")
        
        # Perform extrema analysis
        dead_neurons = self._find_dead_neurons(activity_data)
        saturated_neurons = self._find_saturated_neurons(activity_data)
        extrema_ratio = len(dead_neurons + saturated_neurons) / max(1, self._count_total_neurons(model))
        
        # Create analysis report
        analysis_result = {
            "dead_neurons": dead_neurons,
            "saturated_neurons": saturated_neurons, 
            "extrema_ratio": extrema_ratio,
            "total_neurons_analyzed": self._count_total_neurons(model),
            "analysis_timestamp": context.get_metadata()["created_at"],
            "configuration": {
                "max_batches": self.max_batches,
                "dead_threshold": self.dead_threshold,
                "saturated_multiplier": self.saturated_multiplier
            }
        }
        
        self.log(logging.INFO, f"Found {len(dead_neurons)} dead and {len(saturated_neurons)} saturated neurons")
        
        return analysis_result
    
    def _find_dead_neurons(self, activity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find neurons with very low activity"""
        dead_neurons = []
        
        for layer_name, activities in activity_data.items():
            if isinstance(activities, torch.Tensor):
                # Find neurons below threshold
                mean_activity = activities.mean(dim=0)  # Average over batch
                dead_indices = (mean_activity < self.dead_threshold).nonzero().flatten()
                
                for idx in dead_indices:
                    dead_neurons.append({
                        "layer": layer_name,
                        "neuron_index": idx.item(),
                        "activity_level": mean_activity[idx].item()
                    })
        
        return dead_neurons
    
    def _find_saturated_neurons(self, activity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find neurons with very high activity"""
        saturated_neurons = []
        
        for layer_name, activities in activity_data.items():
            if isinstance(activities, torch.Tensor):
                mean_activity = activities.mean(dim=0)
                std_activity = activities.std(dim=0)
                
                # Saturated = mean + std * multiplier
                saturation_threshold = mean_activity + (std_activity * self.saturated_multiplier)
                saturated_indices = (mean_activity > saturation_threshold).nonzero().flatten()
                
                for idx in saturated_indices:
                    saturated_neurons.append({
                        "layer": layer_name,
                        "neuron_index": idx.item(),
                        "activity_level": mean_activity[idx].item(),
                        "saturation_threshold": saturation_threshold[idx].item()
                    })
        
        return saturated_neurons
    
    def _count_total_neurons(self, model: IModel) -> int:
        """Count total neurons in the model"""
        total = 0
        for layer in model.get_layers():
            if hasattr(layer, 'out_features'):
                total += layer.out_features
        return total
```

### **4.4 Example Migration: Monolithic Analyzer to Multiple Components**

```python
# OLD: evolution/metrics/topological_analysis.py (monolithic)
class TopologicalAnalyzer:
    """Single class doing everything - extrema detection, persistence, connectivity, etc."""
    
    def __init__(self, threshold_config=None, patch_size: int = 8):
        self.threshold_config = threshold_config
        self.patch_size = patch_size
    
    def compute_metrics(self, weight_matrix: torch.Tensor) -> Dict[str, Any]:
        # All logic mixed together in one place
        extrema_info = self._detect_extrema(weight_matrix)
        persistence_diagram = self._compute_persistence_diagram(weight_matrix)
        connectivity_analysis = self._analyze_connectivity(weight_matrix)
        # ... etc
        
        return {
            'extrema_info': extrema_info,
            'persistence_diagram': persistence_diagram,
            'connectivity_analysis': connectivity_analysis,
            # ... all metrics together
        }

# NEW: Split into focused metric components and a high-level analyzer

# 1. components/metrics/extrema_metric.py
class ExtremaMetric(BaseMetric):
    """Focused on extrema detection only"""
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="ExtremaMetric",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"weight_matrix"},
            provided_outputs={
                "metrics.extrema_points",
                "metrics.num_extrema",
                "metrics.extrema_density"
            }
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        # Only extrema detection logic
        weight_matrix = context.get('weight_matrix')
        extrema_points = self._detect_extrema(weight_matrix)
        return {
            "extrema_points": extrema_points,
            "num_extrema": len(extrema_points),
            "extrema_density": len(extrema_points) / weight_matrix.numel()
        }

# 2. components/metrics/persistence_metric.py  
class PersistenceMetric(BaseMetric):
    """Focused on persistence diagrams only"""
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="PersistenceMetric",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"weight_matrix"},
            provided_outputs={
                "metrics.persistence_features",
                "metrics.persistence_entropy",
                "metrics.total_persistence"
            }
        )

# 3. components/analyzers/topological_analyzer.py
class TopologicalAnalyzer(BaseAnalyzer):
    """High-level analyzer that combines multiple topological metrics"""
    
    def __init__(self, patch_size: int = 8, name: str = None):
        super().__init__(name or "TopologicalAnalyzer")
        
        # Initialize the metrics we'll use
        self._extrema_metric = ExtremaMetric(patch_size=patch_size)
        self._persistence_metric = PersistenceMetric()
        self._connectivity_metric = ConnectivityMetric()
        
        self._required_metrics = {
            "ExtremaMetric",
            "PersistenceMetric",
            "ConnectivityMetric"
        }
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="TopologicalAnalyzer",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model"},
            provided_outputs={
                "analysis.topological_summary",
                "analysis.extrema_analysis",
                "analysis.persistence_analysis",
                "analysis.patch_placement_suggestions"
            }
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, 
                         context: EvolutionContext) -> Dict[str, Any]:
        # Combine insights from multiple metrics
        extrema_analysis = self._aggregate_extrema_analysis(report)
        persistence_analysis = self._aggregate_persistence_analysis(report)
        
        # Generate high-level insights
        recommendations = self._generate_topological_recommendations(
            extrema_analysis, persistence_analysis
        )
        
        return {
            "topological_summary": summary,
            "extrema_analysis": extrema_analysis,
            "persistence_analysis": persistence_analysis,
            "topological_recommendations": recommendations
        }

# Migration Pattern Benefits:
# 1. Each metric is focused and reusable
# 2. Clear contracts for each component
# 3. Analyzer combines metrics for insights
# 4. Better testability and maintainability
# 5. Components can be used independently
```

### **4.5 Example Migration: Learning Rate Strategy**

```python
# OLD: evolution/adaptive_learning_rates/layer_schedulers.py
class LayerwiseAdaptiveRates:
    def __init__(self):
        pass
    
    def get_layer_specific_rates(self, context):
        # ... existing logic
        pass

# NEW: components/strategies/layerwise_rate_strategy.py
from src.structure_net.core.base_components import BaseStrategy
from src.structure_net.core.interfaces import *

class LayerwiseRateStrategy(BaseStrategy):
    """Strategy for adapting learning rates per layer"""
    
    def __init__(self, base_lr: float = 0.01, adaptation_factor: float = 0.1, name: str = None):
        super().__init__(name or "LayerwiseRateStrategy", strategy_type="learning_rate")
        
        self.base_lr = base_lr
        self.adaptation_factor = adaptation_factor
        
        # Declare what analysis data we need
        self._required_analysis = {
            "analyzers.extrema_report",
            "metrics.activity"
        }
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(2, 0, 0),
            maturity=Maturity.STABLE,
            
            required_inputs=self._required_analysis,
            provided_outputs={
                "plans.learning_rate",
                "plans.optimizer_updates"
            },
            
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False
            )
        )
    
    def _create_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Create learning rate adaptation plan"""
        
        # Get extrema analysis
        extrema_data = report.get("analyzers.extrema_report", {})
        activity_data = report.get("metrics.activity", {})
        
        # Create layer-specific learning rate plan
        lr_adjustments = {}
        
        if extrema_data:
            # Adjust learning rates based on extrema analysis
            dead_neurons = extrema_data.get("dead_neurons", [])
            saturated_neurons = extrema_data.get("saturated_neurons", [])
            
            # Group by layer
            layer_dead_counts = {}
            layer_saturated_counts = {}
            
            for neuron in dead_neurons:
                layer = neuron["layer"]
                layer_dead_counts[layer] = layer_dead_counts.get(layer, 0) + 1
            
            for neuron in saturated_neurons:
                layer = neuron["layer"] 
                layer_saturated_counts[layer] = layer_saturated_counts.get(layer, 0) + 1
            
            # Calculate adjustments
            for layer_name in set(list(layer_dead_counts.keys()) + list(layer_saturated_counts.keys())):
                dead_count = layer_dead_counts.get(layer_name, 0)
                saturated_count = layer_saturated_counts.get(layer_name, 0)
                
                # Strategy: increase LR for layers with many dead neurons, decrease for saturated
                if dead_count > saturated_count:
                    # More dead neurons - increase learning rate
                    adjustment = 1.0 + (self.adaptation_factor * dead_count / 10)
                elif saturated_count > dead_count:
                    # More saturated neurons - decrease learning rate  
                    adjustment = 1.0 - (self.adaptation_factor * saturated_count / 10)
                else:
                    adjustment = 1.0
                
                lr_adjustments[layer_name] = {
                    "multiplier": adjustment,
                    "new_lr": self.base_lr * adjustment,
                    "reason": f"dead={dead_count}, saturated={saturated_count}"
                }
        
        # Create evolution plan
        plan = EvolutionPlan({
            "learning_rate_adjustments": lr_adjustments,
            "strategy_type": "layerwise_adaptation",
            "base_lr": self.base_lr,
            "epoch": context.epoch
        })
        
        plan.priority = 0.8  # Medium priority
        plan.estimated_impact = len(lr_adjustments) * 0.1  # Rough estimate
        
        self.log(logging.INFO, f"Created LR plan for {len(lr_adjustments)} layers")
        
        return plan
```

---

## **5. Advanced Features: Self-Awareness and Validation** {#advanced}

### **5.1 Component Health Monitoring**

```python
# src/structure_net/core/health_monitor.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import threading
import logging

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    component_name: str
    status: HealthStatus
    message: str
    timestamp: float
    metrics: Dict[str, float]
    
class ComponentHealthMonitor:
    """Monitors component health and performance"""
    
    def __init__(self):
        self._health_data: Dict[str, List[HealthCheck]] = {}
        self._alert_thresholds = {
            'execution_time': 10.0,  # seconds
            'error_rate': 0.1,       # 10% error rate
            'memory_usage': 1000     # MB
        }
        self._lock = threading.Lock()
    
    def record_health_check(self, component: IComponent, status: HealthStatus, 
                          message: str, additional_metrics: Dict[str, float] = None):
        """Record a health check for a component"""
        
        # Get performance metrics from component
        perf_metrics = component.get_performance_metrics()
        
        # Combine with additional metrics
        all_metrics = {**perf_metrics}
        if additional_metrics:
            all_metrics.update(additional_metrics)
        
        health_check = HealthCheck(
            component_name=component.name,
            status=status,
            message=message,
            timestamp=time.time(),
            metrics=all_metrics
        )
        
        with self._lock:
            if component.name not in self._health_data:
                self._health_data[component.name] = []
            
            self._health_data[component.name].append(health_check)
            
            # Keep only recent history (last 100 checks)
            if len(self._health_data[component.name]) > 100:
                self._health_data[component.name] = self._health_data[component.name][-100:]
    
    def get_component_health(self, component_name: str) -> Optional[HealthCheck]:
        """Get most recent health check for a component"""
        with self._lock:
            checks = self._health_data.get(component_name, [])
            return checks[-1] if checks else None
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        with self._lock:
            total_components = len(self._health_data)
            if total_components == 0:
                return {"status": "unknown", "components": 0}
            
            status_counts = {status.value: 0 for status in HealthStatus}
            
            for component_name, checks in self._health_data.items():
                if checks:
                    latest_status = checks[-1].status
                    status_counts[latest_status.value] += 1
            
            # Determine overall status
            if status_counts["critical"] > 0:
                overall_status = "critical"
            elif status_counts["warning"] > total_components * 0.3:  # >30% warnings
                overall_status = "warning"
            elif status_counts["healthy"] > total_components * 0.8:  # >80% healthy
                overall_status = "healthy"
            else:
                overall_status = "warning"
            
            return {
                "status": overall_status,
                "total_components": total_components,
                "status_breakdown": status_counts,
                "timestamp": time.time()
            }
    
    def auto_diagnose_component(self, component: IComponent) -> HealthCheck:
        """Automatically diagnose component health"""
        
        perf_metrics = component.get_performance_metrics()
        issues = []
        status = HealthStatus.HEALTHY
        
        # Check execution time
        if perf_metrics.get('average_time', 0) > self._alert_thresholds['execution_time']:
            issues.append(f"Slow execution: {perf_metrics['average_time']:.2f}s average")
            status = HealthStatus.WARNING
        
        # Check if component is being called
        if perf_metrics.get('call_count', 0) == 0:
            issues.append("Component never executed")
            status = HealthStatus.WARNING
        
        # Check for contract violations (if available)
        try:
            contract = component.contract
            if contract.maturity == Maturity.EXPERIMENTAL:
                issues.append("Using experimental component")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
        except Exception as e:
            issues.append(f"Contract access failed: {str(e)}")
            status = HealthStatus.CRITICAL
        
        message = "Healthy" if not issues else "; ".join(issues)
        
        health_check = HealthCheck(
            component_name=component.name,
            status=status,
            message=message,
            timestamp=time.time(),
            metrics=perf_metrics
        )
        
        # Record the diagnosis
        self.record_health_check(component, status, message)
        
        return health_check

class ComponentProfiler:
    """Advanced profiling for component performance"""
    
    def __init__(self):
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def start_profiling(self, component: IComponent):
        """Start profiling a component"""
        with self._lock:
            self._profiles[component.name] = {
                "start_time": time.time(),
                "memory_start": self._get_memory_usage(),
                "call_count": 0,
                "total_time": 0.0,
                "max_time": 0.0,
                "min_time": float('inf')
            }
    
    def record_execution(self, component: IComponent, execution_time: float):
        """Record an execution for profiling"""
        with self._lock:
            if component.name in self._profiles:
                profile = self._profiles[component.name]
                profile["call_count"] += 1
                profile["total_time"] += execution_time
                profile["max_time"] = max(profile["max_time"], execution_time)
                profile["min_time"] = min(profile["min_time"], execution_time)
    
    def get_profile_report(self, component_name: str) -> Dict[str, Any]:
        """Get profiling report for a component"""
        with self._lock:
            profile = self._profiles.get(component_name)
            if not profile:
                return {}
            
            call_count = profile["call_count"]
            average_time = profile["total_time"] / max(1, call_count)
            
            return {
                "component_name": component_name,
                "total_executions": call_count,
                "total_time": profile["total_time"],
                "average_time": average_time,
                "max_time": profile["max_time"],
                "min_time": profile["min_time"] if call_count > 0 else 0.0,
                "executions_per_second": call_count / max(1, time.time() - profile["start_time"]),
                "memory_usage": self._get_memory_usage() - profile["memory_start"]
            }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available
```

### **5.2 Intelligent Composition Assistant**

```python
# src/structure_net/core/composition_assistant.py
from typing import List, Dict, Set, Any, Optional, Tuple
import logging
from .interfaces import IComponent, ComponentContract
from .compatibility import CompatibilityManager, ComponentRegistry, CompatibilityLevel

class CompositionAssistant:
    """AI-like assistant for building component compositions"""
    
    def __init__(self, registry: ComponentRegistry, compatibility_manager: CompatibilityManager):
        self.registry = registry
        self.compatibility_manager = compatibility_manager
        self._logger = logging.getLogger(__name__)
        
        # Knowledge base of successful compositions
        self._successful_patterns: List[Dict[str, Any]] = []
        
    def suggest_composition(self, research_goal: str, constraints: Dict[str, Any] = None) -> List[IComponent]:
        """Suggest a complete composition for a research goal"""
        
        constraints = constraints or {}
        
        # Parse research goal to understand requirements
        requirements = self._parse_research_goal(research_goal)
        
        # Find base components
        suggested_components = []
        
        # 1. Always need a model
        model_suggestions = self._suggest_models(requirements)
        if model_suggestions:
            suggested_components.append(model_suggestions[0])
        
        # 2. Suggest metrics based on research goal
        metric_suggestions = self._suggest_metrics(requirements)
        suggested_components.extend(metric_suggestions[:3])  # Limit to 3 metrics
        
        # 3. Suggest analyzers
        analyzer_suggestions = self._suggest_analyzers(requirements)
        suggested_components.extend(analyzer_suggestions[:2])  # Limit to 2 analyzers
        
        # 4. Suggest strategies
        strategy_suggestions = self._suggest_strategies(requirements)
        suggested_components.extend(strategy_suggestions[:2])  # Limit to 2 strategies
        
        # 5. Suggest evolvers
        evolver_suggestions = self._suggest_evolvers(requirements)
        suggested_components.extend(evolver_suggestions[:2])  # Limit to 2 evolvers
        
        # 6. Add trainer if needed
        trainer_suggestions = self._suggest_trainers(requirements)
        if trainer_suggestions:
            suggested_components.append(trainer_suggestions[0])
        
        # Validate and fix composition
        final_composition = self._validate_and_fix_composition(suggested_components, constraints)
        
        return final_composition
    
    def _parse_research_goal(self, goal: str) -> Dict[str, Any]:
        """Parse research goal to extract requirements"""
        goal_lower = goal.lower()
        
        requirements = {
            "keywords": goal_lower.split(),
            "focus_areas": [],
            "complexity_level": "medium",
            "experimental_tolerance": "medium"
        }
        
        # Detect focus areas
        if any(word in goal_lower for word in ["sparse", "sparsity", "pruning"]):
            requirements["focus_areas"].append("sparsity")
            
        if any(word in goal_lower for word in ["growth", "evolution", "dynamic"]):
            requirements["focus_areas"].append("growth")
            
        if any(word in goal_lower for word in ["topology", "topological", "homology"]):
            requirements["focus_areas"].append("topology")
            
        if any(word in goal_lower for word in ["learning rate", "lr", "optimization"]):
            requirements["focus_areas"].append("optimization")
            
        if any(word in goal_lower for word in ["geometric", "fiber", "curvature"]):
            requirements["focus_areas"].append("geometry")
        
        # Detect complexity level
        if any(word in goal_lower for word in ["simple", "basic", "quick"]):
            requirements["complexity_level"] = "low"
        elif any(word in goal_lower for word in ["complex", "advanced", "comprehensive"]):
            requirements["complexity_level"] = "high"
        
        # Detect experimental tolerance
        if any(word in goal_lower for word in ["experimental", "cutting edge", "novel"]):
            requirements["experimental_tolerance"] = "high"
        elif any(word in goal_lower for word in ["stable", "reliable", "production"]):
            requirements["experimental_tolerance"] = "low"
        
        return requirements
    
    def _suggest_models(self, requirements: Dict[str, Any]) -> List[IComponent]:
        """Suggest appropriate models"""
        suggestions = []
        
        # Get available model components
        available_models = self.registry.get_available_components()
        model_classes = [cls for cls in available_models.values() 
                        if hasattr(cls, '__bases__') and any('Model' in base.__name__ for base in cls.__bases__)]
        
        for model_class in model_classes:
            try:
                temp_instance = model_class()
                
                # Score based on requirements
                score = 0
                
                # Check focus areas
                if "geometry" in requirements["focus_areas"]:
                    if "fiber" in temp_instance.name.lower() or "geometric" in temp_instance.name.lower():
                        score += 3
                
                if "sparsity" in requirements["focus_areas"]:
                    if "sparse" in temp_instance.name.lower() or "compact" in temp_instance.name.lower():
                        score += 2
                
                # Check complexity matching
                if requirements["complexity_level"] == "low" and temp_instance.contract.maturity == Maturity.STABLE:
                    score += 1
                elif requirements["complexity_level"] == "high" and "advanced" in temp_instance.name.lower():
                    score += 1
                
                # Check experimental tolerance
                if requirements["experimental_tolerance"] == "high" or temp_instance.contract.maturity == Maturity.STABLE:
                    score += 1
                
                suggestions.append((score, temp_instance))
                
            except Exception as e:
                self._logger.warning(f"Could not instantiate model {model_class}: {e}")
        
        # Sort by score and return instances
        suggestions.sort(key=lambda x: x[0], reverse=True)
        return [instance for score, instance in suggestions]
    
    def _suggest_metrics(self, requirements: Dict[str, Any]) -> List[IComponent]:
        """Suggest appropriate metrics"""
        suggestions = []
        
        # Basic metrics everyone needs
        basic_metrics = ["sparsity", "activity"]
        
        # Focus-specific metrics
        if "topology" in requirements["focus_areas"]:
            basic_metrics.extend(["topological", "homological"])
        
        if "geometry" in requirements["focus_areas"]:
            basic_metrics.extend(["curvature", "catastrophe"])
        
        if "growth" in requirements["focus_areas"]:
            basic_metrics.extend(["sensitivity", "mutual_information"])
        
        # Try to instantiate suggested metrics
        available_components = self.registry.get_available_components()
        
        for component_name, component_class in available_components.items():
            if any(metric_type in component_name.lower() for metric_type in basic_metrics):
                try:
                    instance = component_class()
                    if hasattr(instance, 'analyze') and hasattr(instance.contract, 'provided_outputs'):
                        suggestions.append(instance)
                except Exception as e:
                    self._logger.warning(f"Could not instantiate metric {component_name}: {e}")
        
        return suggestions
    
    def _suggest_analyzers(self, requirements: Dict[str, Any]) -> List[IComponent]:
        """Suggest appropriate analyzers"""
        suggestions = []
        
        # Common analyzers
        analyzer_types = ["extrema"]
        
        if "topology" in requirements["focus_areas"]:
            analyzer_types.append("topological")
        
        if "growth" in requirements["focus_areas"]:
            analyzer_types.append("network")
        
        # Try to instantiate suggested analyzers
        available_components = self.registry.get_available_components()
        
        for component_name, component_class in available_components.items():
            if any(analyzer_type in component_name.lower() for analyzer_type in analyzer_types):
                try:
                    instance = component_class()
                    if hasattr(instance, 'analyze') and 'analyzer' in instance.contract.provided_outputs.__str__():
                        suggestions.append(instance)
                except Exception as e:
                    self._logger.warning(f"Could not instantiate analyzer {component_name}: {e}")
        
        return suggestions
    
    def _suggest_strategies(self, requirements: Dict[str, Any]) -> List[IComponent]:
        """Suggest appropriate strategies"""
        suggestions = []
        
        strategy_types = []
        
        if "optimization" in requirements["focus_areas"]:
            strategy_types.append("rate")
        
        if "growth" in requirements["focus_areas"]:
            strategy_types.extend(["growth", "structural"])
        
        if not strategy_types:  # Default strategies
            strategy_types = ["rate", "phase"]
        
        # Try to instantiate suggested strategies
        available_components = self.registry.get_available_components()
        
        for component_name, component_class in available_components.items():
            if any(strategy_type in component_name.lower() for strategy_type in strategy_types):
                try:
                    instance = component_class()
                    if hasattr(instance, 'propose_plan'):
                        suggestions.append(instance)
                except Exception as e:
                    self._logger.warning(f"Could not instantiate strategy {component_name}: {e}")
        
        return suggestions
    
    def _suggest_evolvers(self, requirements: Dict[str, Any]) -> List[IComponent]:
        """Suggest appropriate evolvers"""
        suggestions = []
        
        evolver_types = []
        
        if "optimization" in requirements["focus_areas"]:
            evolver_types.append("rate")
        
        if "growth" in requirements["focus_areas"]:
            evolver_types.extend(["structural", "growth"])
        
        if not evolver_types:  # Default evolvers
            evolver_types = ["rate"]
        
        # Try to instantiate suggested evolvers
        available_components = self.registry.get_available_components()
        
        for component_name, component_class in available_components.items():
            if any(evolver_type in component_name.lower() for evolver_type in evolver_types):
                try:
                    instance = component_class()
                    if hasattr(instance, 'apply_plan'):
                        suggestions.append(instance)
                except Exception as e:
                    self._logger.warning(f"Could not instantiate evolver {component_name}: {e}")
        
        return suggestions
    
    def _suggest_trainers(self, requirements: Dict[str, Any]) -> List[IComponent]:
        """Suggest appropriate trainers"""
        suggestions = []
        
        # For now, suggest standard trainer
        available_components = self.registry.get_available_components()
        
        for component_name, component_class in available_components.items():
            if "trainer" in component_name.lower():
                try:
                    instance = component_class()
                    if hasattr(instance, 'train_step'):
                        suggestions.append(instance)
                except Exception as e:
                    self._logger.warning(f"Could not instantiate trainer {component_name}: {e}")
        
        return suggestions
    
    def _validate_and_fix_composition(self, components: List[IComponent], constraints: Dict[str, Any]) -> List[IComponent]:
        """Validate composition and try to fix issues"""
        
        # First validation
        issues = self.compatibility_manager.validate_composition(components)
        
        # Filter out critical issues
        critical_issues = [i for i in issues if i.level == CompatibilityLevel.CRITICAL]
        
        if not critical_issues:
            self._logger.info(f"Composition validated successfully with {len(issues)} warnings")
            return components
        
        # Try to fix critical issues
        self._logger.warning(f"Found {len(critical_issues)} critical issues, attempting to fix...")
        
        # For now, just remove problematic components
        # More sophisticated fixing could be implemented
        working_components = []
        
        for component in components:
            test_composition = working_components + [component]
            test_issues = self.compatibility_manager.validate_composition(test_composition)
            test_critical = [i for i in test_issues if i.level == CompatibilityLevel.CRITICAL]
            
            if not test_critical:
                working_components.append(component)
            else:
                self._logger.warning(f"Removing {component.name} due to critical issues")
        
        return working_components
    
    def explain_composition(self, components: List[IComponent]) -> str:
        """Generate human-readable explanation of a composition"""
        
        if not components:
            return "Empty composition"
        
        explanation = f"Composition with {len(components)} components:\n\n"
        
        # Group by component type
        by_type = {}
        for comp in components:
            comp_type = type(comp).__name__
            if comp_type not in by_type:
                by_type[comp_type] = []
            by_type[comp_type].append(comp)
        
        # Explain each type
        for comp_type, comps in by_type.items():
            explanation += f"**{comp_type}s**: {len(comps)} component(s)\n"
            for comp in comps:
                explanation += f"  - {comp.name} (maturity: {comp.contract.maturity.value})\n"
            explanation += "\n"
        
        # Explain data flow
        explanation += "**Data Flow**:\n"
        for comp in components:
            if comp.contract.required_inputs:
                explanation += f"  {comp.name} needs: {', '.join(comp.contract.required_inputs)}\n"
            if comp.contract.provided_outputs:
                explanation += f"  {comp.name} provides: {', '.join(comp.contract.provided_outputs)}\n"
        
        return explanation
```

---

## **6. Testing and Validation Framework** {#testing}

### **6.1 Component Testing Infrastructure**

```python
# tests/test_framework/component_tester.py
import unittest
import torch
from typing import List, Dict, Any
from src.structure_net.core.interfaces import IComponent, EvolutionContext, AnalysisReport

class ComponentTestCase(unittest.TestCase):
    """Base test case for component testing"""
    
    def setUp(self):
        """Set up common test fixtures"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context = EvolutionContext({
            'device': str(self.device),
            'batch_size': 32,
            'test_mode': True
        })
        self.context.epoch = 1
        self.context.step = 10
    
    def assert_component_contract_valid(self, component: IComponent):
        """Assert that a component's contract is valid"""
        contract = component.contract
        
        # Basic contract validation
        self.assertIsNotNone(contract.component_name)
        self.assertIsInstance(contract.required_inputs, set)
        self.assertIsInstance(contract.provided_outputs, set)
        self.assertIsNotNone(contract.version)
        self.assertIsNotNone(contract.maturity)
        
        # Component name should match
        self.assertEqual(contract.component_name, component.name)
    
    def assert_component_performance_acceptable(self, component: IComponent, 
                                              max_execution_time: float = 5.0):
        """Assert that component performance is acceptable"""
        perf_metrics = component.get_performance_metrics()
        
        if perf_metrics['call_count'] > 0:
            avg_time = perf_metrics['average_time']
            self.assertLess(avg_time, max_execution_time, 
                          f"Component {component.name} too slow: {avg_time:.3f}s average")

class MetricTestCase(ComponentTestCase):
    """Test case for metric components"""
    
    def create_test_layer(self) -> torch.nn.Module:
        """Create a test layer for metric testing"""
        return torch.nn.Linear(10, 5)
    
    def create_test_model(self) -> torch.nn.Module:
        """Create a test model for metric testing"""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
    
    def assert_metric_output_valid(self, metric_output: Dict[str, Any]):
        """Assert that metric output is valid"""
        self.assertIsInstance(metric_output, dict)
        
        # All values should be numeric or tensors
        for key, value in metric_output.items():
            self.assertTrue(
                isinstance(value, (int, float, torch.Tensor)),
                f"Metric output '{key}' is not numeric: {type(value)}"
            )

class AnalyzerTestCase(ComponentTestCase):
    """Test case for analyzer components"""
    
    def create_test_report(self) -> AnalysisReport:
        """Create a test analysis report"""
        report = AnalysisReport()
        
        # Add some mock metric data
        report.add_metric_data("sparsity", {
            "layer_0": {"sparsity_ratio": 0.8},
            "layer_1": {"sparsity_ratio": 0.6}
        })
        
        report.add_metric_data("activity", {
            "layer_0": torch.randn(32, 20),  # Batch of activations
            "layer_1": torch.randn(32, 5)
        })
        
        return report
    
    def assert_analyzer_output_valid(self, analyzer_output: Dict[str, Any]):
        """Assert that analyzer output is valid"""
        self.assertIsInstance(analyzer_output, dict)
        
        # Should contain some analysis results
        self.assertGreater(len(analyzer_output), 0)

class StrategyTestCase(ComponentTestCase):
    """Test case for strategy components"""
    
    def assert_plan_valid(self, plan: Dict[str, Any]):
        """Assert that an evolution plan is valid"""
        self.assertIsInstance(plan, dict)
        
        # Plan should have some actions
        self.assertGreater(len(plan), 0)
        
        # Check for plan metadata
        if hasattr(plan, 'priority'):
            self.assertIsInstance(plan.priority, (int, float))
            self.assertGreaterEqual(plan.priority, 0)

class EvolverTestCase(ComponentTestCase):
    """Test case for evolver components"""
    
    def assert_evolution_successful(self, evolution_result: Dict[str, Any]):
        """Assert that evolution was successful"""
        self.assertIsInstance(evolution_result, dict)
        
        # Should indicate what was changed
        self.assertIn('changes_applied', evolution_result)
        
# Example specific test implementations

class TestSparsityMetric(MetricTestCase):
    """Test the SparsityMetric component"""
    
    def setUp(self):
        super().setUp()
        # Import your actual SparsityMetric here
        # from src.structure_net.components.metrics.sparsity_metric import SparsityMetric
        # self.metric = SparsityMetric()
        pass
    
    def test_contract_valid(self):
        """Test that the contract is valid"""
        # self.assert_component_contract_valid(self.metric)
        pass
    
    def test_analyze_layer(self):
        """Test analyzing a layer"""
        # layer = self.create_test_layer()
        # result = self.metric.analyze(layer, self.context)
        # self.assert_metric_output_valid(result)
        # self.assertIn('sparsity_ratio', result)
        pass
    
    def test_performance(self):
        """Test performance is acceptable"""
        # layer = self.create_test_layer()
        # for _ in range(10):
        #     self.metric.analyze(layer, self.context)
        # self.assert_component_performance_acceptable(self.metric, max_execution_time=1.0)
        pass

class TestExtremaAnalyzer(AnalyzerTestCase):
    """Test the ExtremaAnalyzer component"""
    
    def setUp(self):
        super().setUp()
        # from src.structure_net.components.analyzers.extrema_analyzer import ExtremaAnalyzer
        # self.analyzer = ExtremaAnalyzer()
        pass
    
    def test_contract_valid(self):
        """Test that the contract is valid"""
        # self.assert_component_contract_valid(self.analyzer)
        pass
    
    def test_analyze_with_valid_report(self):
        """Test analyzing with valid report"""
        # model = self.create_test_model()
        # report = self.create_test_report()
        # result = self.analyzer.analyze(model, report, self.context)
        # self.assert_analyzer_output_valid(result)
        pass
    
    def test_missing_required_metrics(self):
        """Test behavior when required metrics are missing"""
        # model = self.create_test_model()
        # empty_report = AnalysisReport()
        # with self.assertRaises(ValueError):
        #     self.analyzer.analyze(model, empty_report, self.context)
        pass
```

### **6.2 Integration Testing**

```python
# tests/test_integration/test_composition.py
import unittest
from src.structure_net.core.compatibility import ComponentRegistry, CompatibilityManager
from src.structure_net.core.composition_assistant import CompositionAssistant

class TestComponentComposition(unittest.TestCase):
    """Test component composition and compatibility"""
    
    def setUp(self):
        self.registry = ComponentRegistry()
        self.compatibility_manager = CompatibilityManager(self.registry)
        self.assistant = CompositionAssistant(self.registry, self.compatibility_manager)
        
        # Register test components
        self._register_test_components()
    
    def _register_test_components(self):
        """Register test components"""
        # This would register your actual components
        # self.registry.register(SparsityMetric)
        # self.registry.register(ExtremaAnalyzer)
        # etc.
        pass
    
    def test_basic_composition_validation(self):
        """Test basic composition validation"""
        # Create a simple composition
        # components = [
        #     TestModel(),
        #     SparsityMetric(),
        #     ExtremaAnalyzer(),
        #     LayerwiseRateStrategy(),
        #     LearningRateEvolver()
        # ]
        
        # issues = self.compatibility_manager.validate_composition(components)
        # critical_issues = [i for i in issues if i.level == CompatibilityLevel.CRITICAL]
        # self.assertEqual(len(critical_issues), 0, f"Critical issues found: {critical_issues}")
        pass
    
    def test_composition_assistant_suggestions(self):
        """Test composition assistant suggestions"""
        # composition = self.assistant.suggest_composition(
        #     "Study sparsity patterns in neural networks",
        #     constraints={"experimental_tolerance": "low"}
        # )
        
        # self.assertGreater(len(composition), 0)
        # 
        # # Validate suggested composition
        # issues = self.compatibility_manager.validate_composition(composition)
        # critical_issues = [i for i in issues if i.level == CompatibilityLevel.CRITICAL]
        # self.assertEqual(len(critical_issues), 0)
        pass
    
    def test_missing_dependencies_detection(self):
        """Test detection of missing dependencies"""
        # Create composition with missing dependencies
        # components = [
        #     ExtremaAnalyzer(),  # Needs activity metrics but no metric provides it
        # ]
        
        # issues = self.compatibility_manager.validate_composition(components)
        # missing_input_issues = [i for i in issues if "Missing required inputs" in i.description]
        # self.assertGreater(len(missing_input_issues), 0)
        pass

# tests/test_integration/test_orchestrator.py
class TestOrchestrator(unittest.TestCase):
    """Test the full orchestration system"""
    
    def test_full_evolution_cycle(self):
        """Test a complete evolution cycle"""
        # This would test the entire flow:
        # Model -> Metrics -> Analyzers -> Strategies -> Evolvers
        pass
    
    def test_error_recovery(self):
        """Test system behavior when components fail"""
        pass
    
    def test_performance_monitoring(self):
        """Test performance monitoring during execution"""
        pass
```

---

## **7. Usage Examples and Patterns** {#examples}

### **7.1 Basic Research Experiment**

```python
# examples/basic_research_experiment.py
"""
Example: Basic sparsity research experiment using the new architecture
"""

from src.structure_net.core.compatibility import ComponentRegistry, CompatibilityManager
from src.structure_net.core.composition_assistant import CompositionAssistant
from src.structure_net.components.models.minimal_network import MinimalNetwork
from src.structure_net.components.metrics.sparsity_metric import SparsityMetric
from src.structure_net.components.analyzers.extrema_analyzer import ExtremaAnalyzer
from src.structure_net.components.strategies.layerwise_rate_strategy import LayerwiseRateStrategy
from src.structure_net.components.evolvers.learning_rate_evolver import LearningRateEvolver
from src.structure_net.components.orchestrators.research_orchestrator import ResearchOrchestrator

def run_basic_sparsity_experiment():
    """Run a basic sparsity research experiment"""
    
    print("🔬 Setting up Neural Architecture Lab experiment...")
    
    # 1. Set up the component registry and compatibility system
    registry = ComponentRegistry()
    compatibility_manager = CompatibilityManager(registry)
    assistant = CompositionAssistant(registry, compatibility_manager)
    
    # Register components (this would be done automatically in real system)
    registry.register(MinimalNetwork)
    registry.register(SparsityMetric)
    registry.register(ExtremaAnalyzer)
    registry.register(LayerwiseRateStrategy)
    registry.register(LearningRateEvolver)
    
    # 2. Create components
    print("📦 Creating components...")
    
    # Model
    model = MinimalNetwork(
        architecture=[784, 256, 128, 10],
        sparsity=0.02,
        name="SparseMLPExperiment"
    )
    
    # Metrics
    sparsity_metric = SparsityMetric(name="SparsityAnalysis")
    
    # Analyzers  
    extrema_analyzer = ExtremaAnalyzer(
        max_batches=10,
        dead_threshold=0.01,
        name="ExtremaAnalysis"
    )
    
    # Strategies
    lr_strategy = LayerwiseRateStrategy(
        base_lr=0.01,
        adaptation_factor=0.1,
        name="AdaptiveLRStrategy"
    )
    
    # Evolvers
    lr_evolver = LearningRateEvolver(name="LREvolver")
    
    # 3. Validate composition
    print("✅ Validating component composition...")
    
    components = [model, sparsity_metric, extrema_analyzer, lr_strategy, lr_evolver]
    issues = compatibility_manager.validate_composition(components)
    
    # Print any issues
    for issue in issues:
        print(f"  {issue.level.value.upper()}: {issue.description}")
    
    # Check for critical issues
    critical_issues = [i for i in issues if i.level.value == "critical"]
    if critical_issues:
        print("❌ Critical issues found, cannot proceed")
        return
    
    # 4. Create orchestrator and run experiment
    print("🎯 Creating orchestrator...")
    
    orchestrator = ResearchOrchestrator(
        model=model,
        components=components[1:],  # All except model
        name="SparsityResearchOrchestrator"
    )
    
    # 5. Run experiment
    print("🚀 Running experiment...")
    
    # Create dummy data for demonstration
    import torch
    train_data = torch.randn(1000, 784)
    train_labels = torch.randint(0, 10, (1000,))
    
    # Run for several epochs
    for epoch in range(10):
        print(f"\n📊 Epoch {epoch + 1}/10")
        
        # Update context
        context = orchestrator.create_context(epoch=epoch, train_data=train_data)
        
        # Run evolution cycle
        results = orchestrator.run_cycle(context)
        
        # Print results
        if results.get('evolution_applied'):
            print(f"  🔄 Evolution applied: {results['changes_summary']}")
        
        # Print component health
        health = orchestrator.get_composition_health()
        print(f"  💚 System health: {health['status']}")
        
        # Print performance metrics
        for component in components:
            perf = component.get_performance_metrics()
            if perf['call_count'] > 0:
                print(f"    {component.name}: {perf['average_time']:.3f}s avg")
    
    print("\n✅ Experiment completed successfully!")
    
    # 6. Generate experiment summary
    print("\n📋 Experiment Summary:")
    print(assistant.explain_composition(components))

if __name__ == "__main__":
    run_basic_sparsity_experiment()
```

### **7.2 Advanced Research with Composition Assistant**

```python
# examples/advanced_research_with_assistant.py
"""
Example: Using the composition assistant for complex research
"""

def run_advanced_experiment_with_assistant():
    """Use the composition assistant for an advanced experiment"""
    
    # Set up infrastructure
    registry = ComponentRegistry()
    compatibility_manager = CompatibilityManager(registry)
    assistant = CompositionAssistant(registry, compatibility_manager)
    
    # Register all available components automatically
    _auto_register_components(registry)
    
    print("🎯 Starting AI-assisted experiment design...")
    
    # 1. Define research goal
    research_goal = """
    Study the relationship between topological properties and growth patterns 
    in sparse neural networks. Focus on how homological features change 
    during network evolution and their impact on learning efficiency.
    """
    
    print(f"🔬 Research Goal: {research_goal}")
    
    # 2. Let assistant suggest composition
    print("\n🤖 Getting AI suggestions...")
    
    suggested_composition = assistant.suggest_composition(
        research_goal=research_goal,
        constraints={
            "max_components": 8,
            "experimental_tolerance": "high",  # Allow experimental components
            "complexity_level": "high",
            "resource_limit": "medium"
        }
    )
    
    print(f"💡 Assistant suggested {len(suggested_composition)} components:")
    for comp in suggested_composition:
        print(f"  - {comp.name} ({comp.contract.maturity.value})")
    
    # 3. Validate and explain composition
    print("\n📋 Composition Analysis:")
    explanation = assistant.explain_composition(suggested_composition)
    print(explanation)
    
    # 4. Check for issues
    issues = compatibility_manager.validate_composition(suggested_composition)
    if issues:
        print("\n⚠️  Composition Issues:")
        for issue in issues:
            print(f"  {issue.level.value}: {issue.description}")
            if issue.suggested_fix:
                print(f"    💡 Suggestion: {issue.suggested_fix}")
    
    # 5. Set up monitoring
    print("\n📊 Setting up advanced monitoring...")
    
    from src.structure_net.core.health_monitor import ComponentHealthMonitor, ComponentProfiler
    
    health_monitor = ComponentHealthMonitor()
    profiler = ComponentProfiler()
    
    # Start profiling all components
    for component in suggested_composition:
        profiler.start_profiling(component)
    
    # 6. Run experiment with monitoring
    print("\n🚀 Running monitored experiment...")
    
    orchestrator = ResearchOrchestrator(
        model=None,  # Will be set from composition
        components=suggested_composition,
        health_monitor=health_monitor,
        profiler=profiler
    )
    
    # Run with health monitoring
    for epoch in range(5):
        print(f"\n🔄 Epoch {epoch + 1}/5")
        
        # Create context
        context = orchestrator.create_context(epoch=epoch)
        
        # Run cycle with monitoring
        try:
            results = orchestrator.run_monitored_cycle(context)
            
            # Auto-diagnose component health
            for component in suggested_composition:
                health_check = health_monitor.auto_diagnose_component(component)
                if health_check.status.value != "healthy":
                    print(f"  ⚠️  {component.name}: {health_check.message}")
            
            # Print system health summary
            system_health = health_monitor.get_system_health_summary()
            print(f"  🏥 System Health: {system_health['status']} ({system_health['total_components']} components)")
            
        except Exception as e:
            print(f"  ❌ Error in epoch {epoch + 1}: {str(e)}")
            
            # Try to auto-fix issues
            print("  🔧 Attempting automatic recovery...")
            # Implementation would try to remove problematic components and continue
    
    # 7. Generate comprehensive report
    print("\n📊 Final Performance Report:")
    
    for component in suggested_composition:
        profile_report = profiler.get_profile_report(component.name)
        if profile_report:
            print(f"\n{component.name}:")
            print(f"  Executions: {profile_report['total_executions']}")
            print(f"  Average time: {profile_report['average_time']:.3f}s")
            print(f"  Max time: {profile_report['max_time']:.3f}s")
            if profile_report['memory_usage'] > 0:
                print(f"  Memory usage: {profile_report['memory_usage']:.1f}MB")
    
    print("\n✅ Advanced experiment completed!")

def _auto_register_components(registry: ComponentRegistry):
    """Auto-register all available components"""
    # This would scan the components directory and register everything
    # Implementation would use importlib to discover and register components
    pass

if __name__ == "__main__":
    run_advanced_experiment_with_assistant()
```

### **7.3 Custom Component Development**

```python
# examples/custom_component_development.py
"""
Example: How to create custom components in the new architecture
"""

from src.structure_net.core.base_components import BaseMetric, BaseAnalyzer, BaseStrategy, BaseEvolver
from src.structure_net.core.interfaces import *
import torch
import numpy as np

class CustomInformationFlowMetric(BaseMetric):
    """Custom metric for measuring information flow"""
    
    def __init__(self, name: str = "InformationFlowMetric"):
        super().__init__(name)
        self._schema = {
            'entropy': float,
            'mutual_information': float,
            'information_bottleneck': float
        }
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,  # New metric, mark as experimental
            
            required_inputs={"layer.activations", "layer.weights"},
            provided_outputs={"metrics.information_flow"},
            
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                estimated_runtime_seconds=2.0
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], context: EvolutionContext) -> Dict[str, Any]:
        """Compute information flow metrics"""
        
        if isinstance(target, ILayer):
            # Get layer properties
            properties = target.get_analysis_properties()
            
            if "weights" in properties:
                weights = properties["weights"]
                
                # Compute information-theoretic measures
                entropy = self._compute_entropy(weights)
                mutual_info = self._compute_mutual_information(weights)
                bottleneck = self._compute_information_bottleneck(weights)
                
                return {
                    'entropy': entropy,
                    'mutual_information': mutual_info,
                    'information_bottleneck': bottleneck,
                    'layer_name': target.name if hasattr(target, 'name') else 'unknown'
                }
        
        return {}
    
    def _compute_entropy(self, weights: torch.Tensor) -> float:
        """Compute entropy of weight distribution"""
        # Flatten weights and compute histogram
        flat_weights = weights.flatten()
        hist, _ = np.histogram(flat_weights.cpu().numpy(), bins=50, density=True)
        
        # Remove zeros and compute entropy
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return float(entropy)
    
    def _compute_mutual_information(self, weights: torch.Tensor) -> float:
        """Compute mutual information between input and output neurons"""
        # Simplified MI calculation
        # In practice, this would be more sophisticated
        correlation_matrix = torch.corrcoef(weights)
        mi_estimate = torch.abs(correlation_matrix).mean().item()
        
        return float(mi_estimate)
    
    def _compute_information_bottleneck(self, weights: torch.Tensor) -> float:
        """Compute information bottleneck measure"""
        # Simplified bottleneck calculation
        singular_values = torch.linalg.svdvals(weights)
        effective_rank = (singular_values.sum() ** 2) / (singular_values ** 2).sum()
        
        # Normalize by layer size
        max_rank = min(weights.shape)
        bottleneck = effective_rank / max_rank
        
        return float(bottleneck.item())

class CustomAdaptiveGrowthStrategy(BaseStrategy):
    """Custom strategy for adaptive network growth"""
    
    def __init__(self, growth_threshold: float = 0.1, max_growth_rate: float = 0.2):
        super().__init__(name="AdaptiveGrowthStrategy", strategy_type="structural")
        
        self.growth_threshold = growth_threshold
        self.max_growth_rate = max_growth_rate
        
        # Declare what analysis data we need
        self._required_analysis = {
            "metrics.information_flow",
            "analyzers.extrema_report"
        }
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            
            required_inputs=self._required_analysis,
            provided_outputs={
                "plans.structural_growth",
                "plans.adaptive_parameters"
            },
            
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False
            )
        )
    
    def _create_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Create adaptive growth plan"""
        
        # Get analysis data
        info_flow_data = report.get("metrics.information_flow", {})
        extrema_data = report.get("analyzers.extrema_report", {})
        
        # Analyze information bottlenecks
        bottlenecks = []
        for layer_name, flow_metrics in info_flow_data.items():
            bottleneck_score = flow_metrics.get('information_bottleneck', 0)
            if bottleneck_score < self.growth_threshold:
                bottlenecks.append({
                    'layer': layer_name,
                    'bottleneck_score': bottleneck_score,
                    'suggested_growth': min(self.max_growth_rate, self.growth_threshold - bottleneck_score)
                })
        
        # Consider extrema analysis
        growth_priorities = {}
        if extrema_data:
            dead_neurons = extrema_data.get("dead_neurons", [])
            for neuron_info in dead_neurons:
                layer = neuron_info["layer"]
                if layer not in growth_priorities:
                    growth_priorities[layer] = 0
                growth_priorities[layer] += 0.1  # Increase priority for layers with dead neurons
        
        # Create growth plan
        growth_actions = []
        for bottleneck in bottlenecks:
            layer = bottleneck['layer']
            priority = growth_priorities.get(layer, 1.0)
            
            growth_actions.append({
                'action': 'add_connections',
                'layer': layer,
                'amount': bottleneck['suggested_growth'],
                'priority': priority,
                'reason': f"Information bottleneck detected (score: {bottleneck['bottleneck_score']:.3f})"
            })
        
        # Create evolution plan
        plan = EvolutionPlan({
            'structural_growth': growth_actions,
            'strategy_reasoning': {
                'bottlenecks_detected': len(bottlenecks),
                'total_growth_actions': len(growth_actions),
                'growth_threshold': self.growth_threshold
            }
        })
        
        plan.priority = 0.9  # High priority for structural changes
        plan.estimated_impact = len(growth_actions) * 0.15
        
        return plan

def demonstrate_custom_components():
    """Demonstrate custom component usage"""
    
    print("🔧 Demonstrating custom component development...")
    
    # 1. Create custom components
    info_flow_metric = CustomInformationFlowMetric()
    adaptive_strategy = CustomAdaptiveGrowthStrategy(
        growth_threshold=0.1,
        max_growth_rate=0.2
    )
    
    # 2. Validate contracts
    print("\n📝 Custom component contracts:")
    print(f"Information Flow Metric: {info_flow_metric.contract.component_name}")
    print(f"  Maturity: {info_flow_metric.contract.maturity.value}")
    print(f"  Provides: {info_flow_metric.contract.provided_outputs}")
    
    print(f"\nAdaptive Growth Strategy: {adaptive_strategy.contract.component_name}")
    print(f"  Maturity: {adaptive_strategy.contract.maturity.value}")
    print(f"  Requires: {adaptive_strategy.contract.required_inputs}")
    print(f"  Provides: {adaptive_strategy.contract.provided_outputs}")
    
    # 3. Test compatibility
    from src.structure_net.core.compatibility import ComponentRegistry, CompatibilityManager
    
    registry = ComponentRegistry()
    compatibility_manager = CompatibilityManager(registry)
    
    # Register custom components
    registry.register(CustomInformationFlowMetric)
    registry.register(CustomAdaptiveGrowthStrategy)
    
    # Test composition with custom components
    components = [info_flow_metric, adaptive_strategy]
    issues = compatibility_manager.validate_composition(components)
    
    print(f"\n✅ Compatibility validation:")
    if not issues:
        print("  No issues found - components are compatible!")
    else:
        for issue in issues:
            print(f"  {issue.level.value}: {issue.description}")
    
    # 4. Demonstrate usage
    print("\n🧪 Testing custom components...")
    
    # Create test data
    test_layer = torch.nn.Linear(20, 10)
    context = EvolutionContext({'epoch': 1, 'device': 'cpu'})
    
    # Test metric
    try:
        metric_result = info_flow_metric.analyze(test_layer, context)
        print(f"  Information flow analysis: {list(metric_result.keys())}")
    except Exception as e:
        print(f"  Metric test failed: {e}")
    
    # Test strategy (would need proper analysis report)
    try:
        # Create mock analysis report
        report = AnalysisReport()
        report['metrics.information_flow'] = {
            'layer_0': {
                'entropy': 2.5,
                'mutual_information': 0.3,
                'information_bottleneck': 0.05  # Below threshold
            }
        }
        
        plan = adaptive_strategy.propose_plan(report, context)
        print(f"  Growth plan created with {len(plan.get('structural_growth', []))} actions")
    except Exception as e:
        print(f"  Strategy test failed: {e}")
    
    print("\n✅ Custom component demonstration completed!")

if __name__ == "__main__":
    demonstrate_custom_components()
```

---

## **Summary: The Complete Refactoring Roadmap**

This guide provides a complete transformation of Structure Net into a world-class, self-aware research framework. Here's your implementation roadmap:

### **Phase 1: Foundation (Week 1-2)**
1. Implement the core interfaces (`src/structure_net/core/interfaces.py`)
2. Create base component classes (`src/structure_net/core/base_components.py`)
3. Build the compatibility system (`src/structure_net/core/compatibility.py`)

### **Phase 2: Component Migration (Week 3-4)**
1. Refactor existing layers to implement the new interfaces
2. Migrate metrics from `evolution/metrics/` to the new component structure
3. Transform analyzers and create the strategy abstraction layer

#### **Migration Status Tracking**
Track migration progress in `src/structure_net/evolution/metrics/MIGRATION_STATUS.md`:

**✅ Completed Migrations:**
- **MutualInformationAnalyzer** → 5 metrics + 1 analyzer
- **HomologicalAnalyzer** → 5 metrics + 1 analyzer  
- **SensitivityAnalyzer** → 2 metrics + 1 analyzer
- **TopologicalAnalyzer** → 4 metrics + 1 analyzer

**⏳ Pending Migrations:**
- ActivityAnalyzer
- GraphAnalyzer
- CatastropheAnalyzer
- CompactificationAnalyzer

**Migration Pattern for Monolithic Analyzers:**
1. Identify distinct metric computations (low-level measurements)
2. Create focused IMetric components for each measurement
3. Create IAnalyzer component that combines metrics for insights
4. Add deprecation warnings to old classes with migration examples
5. Update component exports in __init__.py files

### **Phase 3: Advanced Features (Week 5-6)**
1. Implement health monitoring and profiling
2. Build the composition assistant
3. Create comprehensive testing framework

### **Phase 4: Integration and Validation (Week 7-8)**
1. Test all components in the new architecture
2. Validate the orchestration system
3. Create examples and documentation

The result will be a research framework that is:
- **Self-aware**: Components understand their capabilities and requirements
- **Intelligent**: Automatic composition validation and suggestions
- **Robust**: Health monitoring and error recovery
- **Extensible**: Easy to add new research components
- **Scientific**: Built-in hypothesis testing and statistical analysis

This architecture transforms Structure Net from a collection of scripts into a sophisticated research platform that scales with your ambitions.