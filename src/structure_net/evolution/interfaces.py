#!/usr/bin/env python3
"""
Core Interfaces for Composable Network Evolution

This module defines the fundamental interfaces that enable composable,
modular network evolution components. All evolution strategies, analyzers,
and learning rate schedulers should implement these interfaces.

Key principles:
- Single responsibility per component
- Composable through interfaces
- Testable in isolation
- Configurable and extensible
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from torch.utils.data import DataLoader


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class ActionType(Enum):
    """Types of growth actions that can be performed."""
    ADD_LAYER = "add_layer"
    ADD_PATCHES = "add_patches"
    ADD_RESIDUAL_BLOCK = "add_residual_block"
    ADD_SKIP_CONNECTION = "add_skip_connection"
    INCREASE_DENSITY = "increase_density"
    PRUNE_CONNECTIONS = "prune_connections"
    HYBRID_GROWTH = "hybrid_growth"
    NO_ACTION = "no_action"


@dataclass
class GrowthAction:
    """Represents a specific growth action to be applied to a network."""
    action_type: ActionType
    position: Optional[int] = None
    size: Optional[int] = None
    layer_count: Optional[int] = None
    reason: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NetworkContext:
    """Context information for network operations."""
    network: nn.Module
    data_loader: DataLoader
    device: torch.device
    epoch: int = 0
    iteration: int = 0
    performance_history: List[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnalysisResult:
    """Result of network analysis."""
    analyzer_name: str
    metrics: Dict[str, Any]
    recommendations: List[str] = None
    confidence: float = 1.0
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


# ============================================================================
# CORE INTERFACES
# ============================================================================

class NetworkComponent(ABC):
    """
    Base interface for all network components.
    
    This is the fundamental interface that all network evolution components
    should implement, providing basic lifecycle and capability methods.
    """
    
    @abstractmethod
    def can_apply(self, context: NetworkContext) -> bool:
        """
        Check if this component can be applied to the given network context.
        
        Args:
            context: Network context containing network, data, and metadata
            
        Returns:
            True if component can be applied, False otherwise
        """
        pass
    
    @abstractmethod
    def apply(self, context: NetworkContext) -> bool:
        """
        Apply this component to the network context.
        
        Args:
            context: Network context to modify
            
        Returns:
            True if application was successful, False otherwise
        """
        pass
    
    def get_name(self) -> str:
        """Get human-readable name for this component."""
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters for this component."""
        return {}
    
    def validate_context(self, context: NetworkContext) -> bool:
        """Validate that the context is suitable for this component."""
        return (context.network is not None and 
                context.data_loader is not None and
                context.device is not None)


class NetworkAnalyzer(NetworkComponent):
    """
    Interface for network analysis components.
    
    Analyzers examine networks and provide insights about their state,
    performance, and potential improvements.
    """
    
    @abstractmethod
    def analyze(self, context: NetworkContext) -> AnalysisResult:
        """
        Analyze the network and return insights.
        
        Args:
            context: Network context to analyze
            
        Returns:
            Analysis result with metrics and recommendations
        """
        pass
    
    def get_analysis_type(self) -> str:
        """Get the type of analysis this analyzer performs."""
        return "generic"
    
    def get_required_batches(self) -> int:
        """Get number of data batches required for analysis."""
        return 1


class GrowthStrategy(NetworkComponent):
    """
    Interface for network growth strategies.
    
    Growth strategies determine how to modify network architecture
    based on analysis results and performance requirements.
    """
    
    @abstractmethod
    def analyze_growth_potential(self, context: NetworkContext) -> AnalysisResult:
        """
        Analyze the network to determine growth potential.
        
        Args:
            context: Network context to analyze
            
        Returns:
            Analysis result indicating growth opportunities
        """
        pass
    
    @abstractmethod
    def calculate_growth_action(self, 
                              analysis: AnalysisResult, 
                              context: NetworkContext) -> Optional[GrowthAction]:
        """
        Calculate the specific growth action to take.
        
        Args:
            analysis: Analysis result from analyze_growth_potential
            context: Current network context
            
        Returns:
            Growth action to perform, or None if no action needed
        """
        pass
    
    def apply(self, context: NetworkContext) -> bool:
        """
        Apply growth strategy to the network.
        
        This default implementation follows the standard pattern:
        1. Analyze growth potential
        2. Calculate growth action
        3. Execute the action
        """
        if not self.can_apply(context):
            return False
        
        # Analyze growth potential
        analysis = self.analyze_growth_potential(context)
        
        # Calculate action
        action = self.calculate_growth_action(analysis, context)
        
        if action is None or action.action_type == ActionType.NO_ACTION:
            return False
        
        # Execute action
        return self.execute_growth_action(action, context)
    
    @abstractmethod
    def execute_growth_action(self, action: GrowthAction, context: NetworkContext) -> bool:
        """
        Execute a specific growth action.
        
        Args:
            action: Growth action to execute
            context: Network context to modify
            
        Returns:
            True if action was executed successfully
        """
        pass
    
    def get_strategy_type(self) -> str:
        """Get the type of growth strategy."""
        return "generic"


class LearningRateStrategy(NetworkComponent):
    """
    Interface for learning rate strategies.
    
    Learning rate strategies determine how to adjust learning rates
    based on network state, training progress, and other factors.
    """
    
    @abstractmethod
    def get_learning_rate(self, context: NetworkContext) -> Union[float, Dict[str, float]]:
        """
        Get learning rate(s) for the current context.
        
        Args:
            context: Current network context
            
        Returns:
            Learning rate (single value) or dict of learning rates by component
        """
        pass
    
    def apply(self, context: NetworkContext) -> bool:
        """
        Apply learning rate strategy by updating optimizer.
        
        This default implementation updates the optimizer in context metadata.
        """
        if not self.can_apply(context):
            return False
        
        lr = self.get_learning_rate(context)
        
        # Update optimizer if available in context
        optimizer = context.metadata.get('optimizer')
        if optimizer is not None:
            if isinstance(lr, dict):
                # Update parameter groups by name
                for param_group in optimizer.param_groups:
                    group_name = param_group.get('name', 'default')
                    if group_name in lr:
                        param_group['lr'] = lr[group_name]
            else:
                # Update all parameter groups
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            return True
        
        return False
    
    def get_strategy_type(self) -> str:
        """Get the type of learning rate strategy."""
        return "generic"


class NetworkTrainer(NetworkComponent):
    """
    Interface for network training strategies.
    
    Trainers handle the actual training process, including
    optimization, loss calculation, and performance monitoring.
    """
    
    @abstractmethod
    def train_epoch(self, context: NetworkContext) -> Dict[str, float]:
        """
        Train the network for one epoch.
        
        Args:
            context: Network context with training data
            
        Returns:
            Training metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def evaluate(self, context: NetworkContext) -> Dict[str, float]:
        """
        Evaluate the network on validation/test data.
        
        Args:
            context: Network context with evaluation data
            
        Returns:
            Evaluation metrics
        """
        pass
    
    def apply(self, context: NetworkContext) -> bool:
        """
        Apply training strategy (train for one epoch).
        """
        if not self.can_apply(context):
            return False
        
        metrics = self.train_epoch(context)
        
        # Store metrics in context
        if 'training_metrics' not in context.metadata:
            context.metadata['training_metrics'] = []
        context.metadata['training_metrics'].append(metrics)
        
        return True


class NetworkEvolutionSystem(ABC):
    """
    Interface for complete network evolution systems.
    
    Evolution systems coordinate multiple components (analyzers, strategies,
    trainers) to evolve networks over time.
    """
    
    @abstractmethod
    def evolve_network(self, 
                      context: NetworkContext, 
                      num_iterations: int = 1) -> NetworkContext:
        """
        Evolve the network for the specified number of iterations.
        
        Args:
            context: Initial network context
            num_iterations: Number of evolution iterations
            
        Returns:
            Updated network context after evolution
        """
        pass
    
    @abstractmethod
    def add_component(self, component: NetworkComponent):
        """Add a component to the evolution system."""
        pass
    
    @abstractmethod
    def remove_component(self, component: NetworkComponent):
        """Remove a component from the evolution system."""
        pass
    
    def get_components(self) -> List[NetworkComponent]:
        """Get all components in the evolution system."""
        return []


# ============================================================================
# SPECIALIZED INTERFACES
# ============================================================================

class ExtremaAnalyzer(NetworkAnalyzer):
    """Specialized interface for extrema analysis."""
    
    @abstractmethod
    def detect_extrema(self, context: NetworkContext) -> Dict[str, Any]:
        """Detect extrema patterns in network activations."""
        pass
    
    def get_analysis_type(self) -> str:
        return "extrema"


class InformationFlowAnalyzer(NetworkAnalyzer):
    """Specialized interface for information flow analysis."""
    
    @abstractmethod
    def analyze_information_flow(self, context: NetworkContext) -> Dict[str, Any]:
        """Analyze information flow through network layers."""
        pass
    
    def get_analysis_type(self) -> str:
        return "information_flow"


class ArchitectureModifier(ABC):
    """
    Interface for components that modify network architecture.
    
    This is a specialized interface for components that need to
    modify the actual structure of the network.
    """
    
    @abstractmethod
    def modify_architecture(self, 
                          network: nn.Module, 
                          action: GrowthAction) -> nn.Module:
        """
        Modify network architecture according to the growth action.
        
        Args:
            network: Network to modify
            action: Growth action specifying the modification
            
        Returns:
            Modified network (may be the same object or a new one)
        """
        pass
    
    @abstractmethod
    def can_modify(self, network: nn.Module, action: GrowthAction) -> bool:
        """Check if this modifier can handle the given action."""
        pass


class ComponentFactory(ABC):
    """
    Interface for creating network components.
    
    Factories enable dynamic component creation and configuration.
    """
    
    @abstractmethod
    def create_component(self, 
                        component_type: str, 
                        config: Dict[str, Any]) -> NetworkComponent:
        """
        Create a component of the specified type with given configuration.
        
        Args:
            component_type: Type of component to create
            config: Configuration parameters
            
        Returns:
            Created component instance
        """
        pass
    
    @abstractmethod
    def get_available_types(self) -> List[str]:
        """Get list of available component types."""
        pass


# ============================================================================
# UTILITY INTERFACES
# ============================================================================

class Configurable(ABC):
    """Interface for configurable components."""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]):
        """Configure the component with given parameters."""
        pass
    
    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        pass


class Serializable(ABC):
    """Interface for serializable components."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Deserialize component from dictionary."""
        pass


class Monitorable(ABC):
    """Interface for components that can be monitored."""
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for monitoring."""
        pass
    
    @abstractmethod
    def reset_metrics(self):
        """Reset monitoring metrics."""
        pass


# ============================================================================
# COMPOSITE INTERFACES
# ============================================================================

class FullyConfigurableComponent(NetworkComponent, Configurable, Serializable, Monitorable):
    """
    Composite interface for fully-featured components.
    
    Components implementing this interface support configuration,
    serialization, and monitoring in addition to basic functionality.
    """
    pass


# Export all interfaces
__all__ = [
    # Core data structures
    'ActionType', 'GrowthAction', 'NetworkContext', 'AnalysisResult',
    
    # Core interfaces
    'NetworkComponent', 'NetworkAnalyzer', 'GrowthStrategy', 
    'LearningRateStrategy', 'NetworkTrainer', 'NetworkEvolutionSystem',
    
    # Specialized interfaces
    'ExtremaAnalyzer', 'InformationFlowAnalyzer', 'ArchitectureModifier',
    'ComponentFactory',
    
    # Utility interfaces
    'Configurable', 'Serializable', 'Monitorable',
    
    # Composite interfaces
    'FullyConfigurableComponent'
]
