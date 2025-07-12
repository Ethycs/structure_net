#!/usr/bin/env python3
"""
Scheduler Strategy Selector Component

A strategy component that selects and configures appropriate learning rate
schedulers based on the training context and strategy level.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from enum import Enum

from ...core.base_components import BaseStrategy
from ...core.interfaces import (
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)
from ..schedulers import (
    MultiScaleLearningScheduler,
    LayerAgeAwareScheduler,
    ExtremaPhaseScheduler
)
from ..orchestrators import AdaptiveLearningRateOrchestrator


class SchedulingStrategy(Enum):
    """Available scheduling strategies."""
    BASIC = "basic"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"
    ULTIMATE = "ultimate"
    CUSTOM = "custom"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"
    STRUCTURE_NET = "structure_net"


class SchedulerStrategySelector(BaseStrategy):
    """
    Strategy component for selecting and configuring learning rate schedulers.
    
    This component replaces the factory pattern with a proper strategy pattern,
    providing intelligent scheduler selection based on context.
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="SchedulerStrategySelector",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"network", "strategy"},
            provided_outputs={"scheduler_config", "orchestrator", "optimizer"},
            optional_inputs={"base_lr", "custom_schedulers", "scheduler_configs"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def __init__(self):
        """Initialize the Scheduler Strategy Selector."""
        super().__init__("SchedulerStrategySelector")
        
        # Define strategy presets
        self.strategy_presets = self._create_strategy_presets()
        
        # Define scheduler presets
        self.scheduler_presets = self._create_scheduler_presets()
    
    def _create_strategy_presets(self) -> Dict[SchedulingStrategy, Dict[str, Any]]:
        """Create predefined strategy configurations."""
        return {
            SchedulingStrategy.BASIC: {
                'schedulers': ['warmup', 'exponential_backoff', 'layerwise_rates'],
                'enable_extrema_phase': False,
                'enable_layer_age': True,
                'enable_multi_scale': False,
                'description': 'Essential adaptive strategies for stable training'
            },
            
            SchedulingStrategy.ADVANCED: {
                'schedulers': ['extrema_phase', 'layer_age_aware', 'soft_clamping'],
                'enable_extrema_phase': True,
                'enable_layer_age': True,
                'enable_multi_scale': False,
                'description': 'Advanced strategies with extrema-based adaptation'
            },
            
            SchedulingStrategy.COMPREHENSIVE: {
                'schedulers': ['extrema_phase', 'layer_age_aware', 'multi_scale', 
                              'cascading_decay', 'progressive_freezing'],
                'enable_extrema_phase': True,
                'enable_layer_age': True,
                'enable_multi_scale': True,
                'description': 'Comprehensive strategies for complex training scenarios'
            },
            
            SchedulingStrategy.ULTIMATE: {
                'schedulers': ['extrema_phase', 'layer_age_aware', 'multi_scale',
                              'unified_system', 'all_advanced_features'],
                'enable_extrema_phase': True,
                'enable_layer_age': True,
                'enable_multi_scale': True,
                'enable_unified_system': True,
                'description': 'All available strategies combined intelligently'
            },
            
            SchedulingStrategy.TRANSFER_LEARNING: {
                'schedulers': ['progressive_freezing', 'pretrained_new_layer', 
                              'layer_age_aware'],
                'enable_extrema_phase': False,
                'enable_layer_age': True,
                'enable_multi_scale': False,
                'freeze_pretrained_initially': True,
                'description': 'Optimized for transfer learning scenarios'
            },
            
            SchedulingStrategy.CONTINUAL_LEARNING: {
                'schedulers': ['sedimentary_learning', 'soft_clamping', 
                              'connection_strength'],
                'enable_extrema_phase': True,
                'enable_layer_age': True,
                'enable_multi_scale': True,
                'protect_old_knowledge': True,
                'description': 'Designed for continual learning without forgetting'
            },
            
            SchedulingStrategy.STRUCTURE_NET: {
                'schedulers': ['extrema_phase', 'multi_scale', 'layer_age_aware',
                              'sparsity_aware', 'growth_phase'],
                'enable_extrema_phase': True,
                'enable_layer_age': True,
                'enable_multi_scale': True,
                'structure_aware': True,
                'description': 'Specialized for Structure Net architecture'
            }
        }
    
    def _create_scheduler_presets(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined scheduler configurations."""
        return {
            'conservative': {
                'base_lr': 0.0001,
                'min_lr': 1e-7,
                'max_lr': 0.001,
                'warmup_epochs': 10,
                'decay_rate': 0.95
            },
            
            'aggressive': {
                'base_lr': 0.01,
                'min_lr': 1e-5,
                'max_lr': 0.1,
                'warmup_epochs': 3,
                'decay_rate': 0.9
            },
            
            'adaptive': {
                'base_lr': 0.001,
                'min_lr': 1e-6,
                'max_lr': 0.01,
                'warmup_epochs': 5,
                'decay_rate': 0.95,
                'extrema_boost': 2.0
            },
            
            'fine_tuning': {
                'base_lr': 0.00001,
                'min_lr': 1e-8,
                'max_lr': 0.0001,
                'warmup_epochs': 0,
                'decay_rate': 0.99
            }
        }
    
    def select_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate scheduling strategy based on context.
        
        Args:
            context: Dictionary containing:
                - network: The neural network
                - strategy: Strategy name or enum
                - base_lr: Base learning rate
                - task_type: Type of task (classification, regression, etc.)
                - dataset_size: Size of the dataset
                - is_pretrained: Whether using pretrained weights
                - custom_schedulers: List of specific schedulers to use
                - scheduler_configs: Custom scheduler configurations
        
        Returns:
            Strategy configuration dictionary
        """
        network = context['network']
        strategy = context.get('strategy', SchedulingStrategy.BASIC)
        base_lr = context.get('base_lr', 0.001)
        custom_schedulers = context.get('custom_schedulers', None)
        scheduler_configs = context.get('scheduler_configs', {})
        
        # Convert string strategy to enum
        if isinstance(strategy, str):
            strategy = SchedulingStrategy(strategy.lower())
        
        # Get base strategy configuration
        if strategy == SchedulingStrategy.CUSTOM and custom_schedulers:
            config = {
                'schedulers': custom_schedulers,
                'base_lr': base_lr,
                'custom': True
            }
        else:
            config = self.strategy_presets.get(strategy, self.strategy_presets[SchedulingStrategy.BASIC]).copy()
            config['base_lr'] = base_lr
        
        # Apply automatic adjustments based on context
        config = self._adjust_for_context(config, context)
        
        # Merge custom configurations
        if scheduler_configs:
            config['scheduler_configs'] = scheduler_configs
        
        return config
    
    def _adjust_for_context(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust strategy configuration based on context."""
        # Adjust for dataset size
        dataset_size = context.get('dataset_size', 'medium')
        if dataset_size == 'small':
            config['base_lr'] *= 0.5  # More conservative for small datasets
            config['enable_extrema_phase'] = True  # More important to detect overfitting
        elif dataset_size == 'large':
            config['base_lr'] *= 1.5  # Can be more aggressive
            config['warmup_epochs'] = config.get('warmup_epochs', 5) * 2
        
        # Adjust for pretrained models
        if context.get('is_pretrained', False):
            config['base_lr'] *= 0.1  # Much lower learning rate
            config['freeze_pretrained_initially'] = True
            if 'progressive_freezing' not in config.get('schedulers', []):
                config['schedulers'].append('progressive_freezing')
        
        # Adjust for task type
        task_type = context.get('task_type', 'classification')
        if task_type == 'regression':
            config['enable_extrema_phase'] = True  # Important for detecting saturation
            config['extrema_boost'] = 1.5  # Less aggressive boosting
        elif task_type == 'generation':
            config['enable_multi_scale'] = True  # Important for generative models
            config['temporal_decay'] = 0.02  # Slower decay
        
        return config
    
    def create_orchestrator(self, 
                          network: nn.Module,
                          strategy_config: Dict[str, Any]) -> AdaptiveLearningRateOrchestrator:
        """
        Create an orchestrator based on strategy configuration.
        
        Args:
            network: The neural network
            strategy_config: Strategy configuration from select_strategy
        
        Returns:
            Configured AdaptiveLearningRateOrchestrator
        """
        base_lr = strategy_config.get('base_lr', 0.001)
        scheduler_configs = strategy_config.get('scheduler_configs', {})
        
        # Extract enable flags
        enable_extrema = strategy_config.get('enable_extrema_phase', False)
        enable_layer_age = strategy_config.get('enable_layer_age', False)
        enable_multi_scale = strategy_config.get('enable_multi_scale', False)
        
        # Create orchestrator with merged configurations
        orchestrator_config = {
            'base_lr': base_lr,
            'enable_extrema_phase': enable_extrema,
            'enable_layer_age': enable_layer_age,
            'enable_multi_scale': enable_multi_scale,
            **scheduler_configs.get('orchestrator', {})
        }
        
        orchestrator = AdaptiveLearningRateOrchestrator(**orchestrator_config)
        
        # Configure individual schedulers if custom configs provided
        if 'extrema_phase' in scheduler_configs and orchestrator.extrema_phase:
            orchestrator.extrema_phase.__dict__.update(scheduler_configs['extrema_phase'])
        if 'layer_age' in scheduler_configs and orchestrator.layer_age:
            orchestrator.layer_age.__dict__.update(scheduler_configs['layer_age'])
        if 'multi_scale' in scheduler_configs and orchestrator.multi_scale:
            orchestrator.multi_scale.__dict__.update(scheduler_configs['multi_scale'])
        
        return orchestrator
    
    def create_adaptive_optimizer(self,
                                network: nn.Module,
                                orchestrator: AdaptiveLearningRateOrchestrator,
                                optimizer_class: type = optim.Adam,
                                optimizer_kwargs: Optional[Dict[str, Any]] = None) -> optim.Optimizer:
        """
        Create an optimizer managed by the orchestrator.
        
        Args:
            network: The neural network
            orchestrator: The learning rate orchestrator
            optimizer_class: Optimizer class to use
            optimizer_kwargs: Additional optimizer arguments
        
        Returns:
            Configured optimizer
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        # Create base optimizer
        optimizer = optimizer_class(
            network.parameters(),
            lr=orchestrator.base_lr,
            **optimizer_kwargs
        )
        
        return optimizer
    
    def get_strategy_info(self, strategy: Union[str, SchedulingStrategy]) -> Dict[str, Any]:
        """Get information about a specific strategy."""
        if isinstance(strategy, str):
            strategy = SchedulingStrategy(strategy.lower())
        
        preset = self.strategy_presets.get(strategy, {})
        return {
            'name': strategy.value,
            'description': preset.get('description', 'Custom strategy'),
            'schedulers': preset.get('schedulers', []),
            'features': {
                'extrema_detection': preset.get('enable_extrema_phase', False),
                'layer_adaptation': preset.get('enable_layer_age', False),
                'multi_scale': preset.get('enable_multi_scale', False),
                'unified_system': preset.get('enable_unified_system', False)
            }
        }
    
    def list_available_strategies(self) -> List[Dict[str, Any]]:
        """List all available strategies with their descriptions."""
        return [
            self.get_strategy_info(strategy)
            for strategy in SchedulingStrategy
        ]
    
    def recommend_strategy(self, context: Dict[str, Any]) -> SchedulingStrategy:
        """Recommend a strategy based on the context."""
        # Simple recommendation logic
        if context.get('is_pretrained', False):
            return SchedulingStrategy.TRANSFER_LEARNING
        
        if context.get('continual_learning', False):
            return SchedulingStrategy.CONTINUAL_LEARNING
        
        if 'structure_net' in str(type(context.get('network', ''))).lower():
            return SchedulingStrategy.STRUCTURE_NET
        
        dataset_size = context.get('dataset_size', 'medium')
        if dataset_size == 'small':
            return SchedulingStrategy.BASIC
        elif dataset_size == 'large':
            return SchedulingStrategy.COMPREHENSIVE
        
        return SchedulingStrategy.ADVANCED