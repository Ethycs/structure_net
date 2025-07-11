"""
Base component implementations providing common functionality.

These abstract base classes implement the boilerplate for each component type,
allowing concrete implementations to focus on their specific logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Set, Union, Optional
import logging
import time
from datetime import datetime
import torch
import torch.nn as nn

from .interfaces import (
    IComponent, ILayer, IModel, ITrainer, IMetric, IAnalyzer, 
    IStrategy, IEvolver, IScheduler, IOrchestrator,
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel,
    EvolutionContext, AnalysisReport, EvolutionPlan
)


class BaseComponent(IComponent):
    """Base implementation for all components with common functionality"""
    
    def __init__(self, name: str = None):
        super().__init__()
        self._name = name or self.__class__.__name__
        self._execution_history = []
        self._error_count = 0
    
    @property
    def name(self) -> str:
        return self._name
    
    def _measure_performance(self, func):
        """Decorator to measure performance of key methods"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage() if torch.cuda.is_available() else 0
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                memory_delta = (self._get_memory_usage() if torch.cuda.is_available() else 0) - start_memory
                
                self._execution_history.append({
                    'timestamp': datetime.now(),
                    'method': func.__name__,
                    'execution_time': execution_time,
                    'memory_delta': memory_delta,
                    'success': True
                })
                
                return result
            except Exception as e:
                self._error_count += 1
                self._execution_history.append({
                    'timestamp': datetime.now(),
                    'method': func.__name__,
                    'error': str(e),
                    'success': False
                })
                raise
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        if not self._execution_history:
            return {
                'call_count': 0,
                'error_count': 0,
                'average_time': 0.0,
                'total_time': 0.0
            }
        
        successful_executions = [e for e in self._execution_history if e['success']]
        
        return {
            'call_count': len(self._execution_history),
            'error_count': self._error_count,
            'success_rate': len(successful_executions) / len(self._execution_history),
            'average_time': sum(e.get('execution_time', 0) for e in successful_executions) / max(1, len(successful_executions)),
            'total_time': sum(e.get('execution_time', 0) for e in successful_executions),
            'average_memory_delta': sum(e.get('memory_delta', 0) for e in successful_executions) / max(1, len(successful_executions))
        }


class BaseLayer(BaseComponent, ILayer, nn.Module):
    """Base implementation for neural network layers"""
    
    def __init__(self, name: str = None):
        BaseComponent.__init__(self, name)
        nn.Module.__init__(self)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward pass - must be overridden"""
        raise NotImplementedError("Layer must implement forward pass")
    
    @property
    def contract(self) -> ComponentContract:
        """Default contract for layers"""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            provided_outputs={"layer.output", "layer.properties"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW)
        )


class BaseModel(BaseComponent, IModel, nn.Module):
    """Base implementation for neural network models"""
    
    def __init__(self, name: str = None):
        BaseComponent.__init__(self, name)
        nn.Module.__init__(self)
        self._layers: List[ILayer] = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward pass - must be overridden"""
        raise NotImplementedError("Model must implement forward pass")
    
    @property
    def contract(self) -> ComponentContract:
        """Default contract for models"""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            provided_outputs={"model.output", "model.architecture"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=torch.cuda.is_available()
            )
        )
    
    def get_layers(self) -> List[ILayer]:
        """Get all layers in this model"""
        return self._layers
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Default architecture summary"""
        return {
            'name': self.name,
            'num_layers': len(self._layers),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def supports_dynamic_growth(self) -> bool:
        """By default, models don't support dynamic growth"""
        return False


class BaseTrainer(BaseComponent, ITrainer):
    """Base implementation for training components"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._steps_completed = 0
    
    @property
    def contract(self) -> ComponentContract:
        """Default contract for trainers"""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "data", "optimizer"},
            provided_outputs={"training.loss", "training.metrics"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=torch.cuda.is_available()
            )
        )
    
    def supports_online_evolution(self) -> bool:
        """By default, trainers don't support online evolution"""
        return False


class BaseMetric(BaseComponent, IMetric):
    """Base implementation for metric components"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._measurement_schema: Dict[str, type] = {}
    
    @property
    def contract(self) -> ComponentContract:
        """Default contract for metrics"""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"target"},
            provided_outputs={f"metrics.{self.name}"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW)
        )
    
    def get_measurement_schema(self) -> Dict[str, type]:
        """Return the measurement schema"""
        return self._measurement_schema
    
    @abstractmethod
    def _compute_metric(self, target: Union[ILayer, IModel], context: EvolutionContext) -> Dict[str, Any]:
        """Override this to implement metric computation"""
        pass
    
    def analyze(self, target: Union[ILayer, IModel], context: EvolutionContext) -> Dict[str, Any]:
        """Public interface with validation"""
        return self._measure_performance(self._compute_metric)(target, context)


class BaseAnalyzer(BaseComponent, IAnalyzer):
    """Base implementation for analyzer components"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._required_metrics: Set[str] = set()
    
    @property
    def contract(self) -> ComponentContract:
        """Default contract for analyzers"""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model", "analysis_report"}.union(self._required_metrics),
            provided_outputs={f"analysis.{self.name}"},
            resources=ResourceRequirements(memory_level=ResourceLevel.MEDIUM)
        )
    
    def get_required_metrics(self) -> Set[str]:
        """Return required metrics"""
        return self._required_metrics
    
    @abstractmethod
    def _perform_analysis(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """Override this to implement analysis logic"""
        pass
    
    def analyze(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """Public interface with validation"""
        # Check if required metrics are present
        missing_metrics = self._required_metrics - set(report.sources)
        if missing_metrics:
            self.log(logging.WARNING, f"Missing required metrics: {missing_metrics}")
        
        return self._measure_performance(self._perform_analysis)(model, report, context)


class BaseStrategy(BaseComponent, IStrategy):
    """Base implementation for strategy components"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._required_analysis: Set[str] = set()
    
    @property
    def contract(self) -> ComponentContract:
        """Default contract for strategies"""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"analysis_report"}.union(self._required_analysis),
            provided_outputs={f"strategy.{self.get_strategy_type()}"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW)
        )
    
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


class BaseScheduler(BaseComponent, IScheduler):
    """Base implementation for scheduler components"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._trigger_history: List[int] = []
    
    @property
    def contract(self) -> ComponentContract:
        """Default contract for schedulers"""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"context"},
            provided_outputs={"scheduler.trigger_decision"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW)
        )
    
    def should_trigger(self, context: EvolutionContext) -> bool:
        """Check if evolution should trigger now"""
        trigger = self._check_trigger_condition(context)
        
        if trigger:
            self._trigger_history.append(context.step)
            self.log(logging.INFO, f"Evolution triggered at step {context.step}")
        
        return trigger
    
    @abstractmethod
    def _check_trigger_condition(self, context: EvolutionContext) -> bool:
        """Override this to implement trigger logic"""
        pass
    
    def get_next_trigger_estimate(self, context: EvolutionContext) -> Optional[int]:
        """Estimate when next trigger will occur"""
        # Default implementation based on history
        if len(self._trigger_history) < 2:
            return None
        
        # Calculate average interval
        intervals = [self._trigger_history[i] - self._trigger_history[i-1] 
                    for i in range(1, len(self._trigger_history))]
        avg_interval = sum(intervals) / len(intervals)
        
        last_trigger = self._trigger_history[-1]
        return int(last_trigger + avg_interval)


class BaseOrchestrator(BaseComponent, IOrchestrator):
    """Base implementation for orchestrator components"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._components: Dict[str, IComponent] = {}
        self._execution_count = 0
    
    @property
    def contract(self) -> ComponentContract:
        """Default contract for orchestrators"""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"context"},
            provided_outputs={"orchestration.cycle_results"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=torch.cuda.is_available()
            )
        )
    
    def register_component(self, component: IComponent):
        """Register a component with the orchestrator"""
        self._components[component.name] = component
        self.log(logging.INFO, f"Registered component: {component.name}")
    
    def get_composition_health(self) -> Dict[str, Any]:
        """Get health status of component composition"""
        health_status = {
            'status': 'healthy',
            'components': {},
            'issues': []
        }
        
        for name, component in self._components.items():
            try:
                perf_metrics = component.get_performance_metrics()
                health_status['components'][name] = {
                    'status': 'healthy',
                    'call_count': perf_metrics.get('call_count', 0),
                    'error_count': perf_metrics.get('error_count', 0),
                    'maturity': component.maturity.value
                }
                
                # Check for issues
                if perf_metrics.get('error_count', 0) > 0:
                    health_status['issues'].append(f"{name} has errors")
                    health_status['components'][name]['status'] = 'degraded'
                
            except Exception as e:
                health_status['components'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['issues'].append(f"{name} health check failed")
        
        # Overall status
        if any(c['status'] == 'error' for c in health_status['components'].values()):
            health_status['status'] = 'critical'
        elif health_status['issues']:
            health_status['status'] = 'degraded'
        
        return health_status