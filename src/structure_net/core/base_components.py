from abc import abstractmethod
import time
from typing import Dict, Any, Set, Optional, Union, List
import torch
import torch.nn as nn
import logging
from .interfaces import (
    IComponent,
    ILayer,
    IModel,
    IMetric,
    IAnalyzer,
    IStrategy,
    IEvolver,
    IOrchestrator,
    ComponentContract,
    ComponentVersion,
    Maturity,
    ResourceRequirements,
    ResourceLevel,
    EvolutionContext,
    AnalysisReport,
    EvolutionPlan,
    ITrainer,
)


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

class BaseTrainer(BaseComponent, ITrainer):
    """Base implementation for trainers"""

    def __init__(self, name: str = None):
        super().__init__(name)

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "dataset"},
            provided_outputs={"training_metrics"},
            resources=ResourceRequirements(memory_level=ResourceLevel.MEDIUM, requires_gpu=True)
        )

    def supports_online_evolution(self) -> bool:
        return True

    @abstractmethod
    def _train_step(self, model: IModel, batch: Any, context: EvolutionContext) -> Dict[str, float]:
        """Override this to implement the training logic for a single step."""
        pass

    def train_step(self, model: IModel, batch: Any, context: EvolutionContext) -> Dict[str, float]:
        """Public training step with performance tracking."""
        return self._measure_performance(self._train_step)(model, batch, context)

class BaseOrchestrator(BaseComponent, IOrchestrator):
    """Base implementation for orchestrators"""

    def __init__(self, name: str = None):
        super().__init__(name)

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"context"},
            provided_outputs={"system_report"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW)
        )

    @abstractmethod
    def run_cycle(self, context: EvolutionContext) -> Dict[str, Any]:
        pass

    def get_composition_health(self) -> Dict[str, Any]:
        return {"status": "ok"}

class BaseStrategyOrchestrator(BaseOrchestrator):
    """Base class for orchestrators that manage and select from multiple strategies."""

    def __init__(self, strategies: List[IComponent], name: str = None):
        super().__init__(name)
        self.strategies = strategies

    @abstractmethod
    def select_best_plan(self, plans: List[EvolutionPlan]) -> EvolutionPlan:
        """Selects the best plan from a list of proposed plans."""
        pass

    def run_cycle(self, context: EvolutionContext, report: AnalysisReport) -> EvolutionPlan:
        """
        Runs all strategies, collects their proposed plans, and selects the best one.
        """
        self.log(logging.INFO, f"Running {len(self.strategies)} strategies to find the best plan.")
        proposed_plans = []
        for strategy in self.strategies:
            if all(req in report for req in strategy.contract.required_inputs):
                try:
                    plan = strategy.propose_plan(report, context)
                    if plan:
                        proposed_plans.append(plan)
                except Exception as e:
                    self.log(logging.ERROR, f"Strategy {strategy.name} failed: {e}")
        
        if not proposed_plans:
            self.log(logging.INFO, "No plans were proposed by any strategy.")
            return EvolutionPlan()

        best_plan = self.select_best_plan(proposed_plans)
        self.log(logging.INFO, f"Selected plan from {best_plan.created_by} with priority {best_plan.priority:.2f}.")
        return best_plan
