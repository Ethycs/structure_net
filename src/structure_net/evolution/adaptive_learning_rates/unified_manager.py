"""
Unified adaptive learning rate manager.

This module provides a comprehensive system that combines multiple learning rate
strategies into a cohesive, easy-to-use interface.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import math

from .base import (
    BaseLearningRateScheduler, 
    LearningRateStrategy, 
    AdaptiveOptimizerWrapper,
    LearningRateHistory,
    ParameterGroup
)
from .phase_schedulers import (
    ExtremaPhaseScheduler,
    GrowthPhaseScheduler,
    ExponentialBackoffScheduler,
    WarmupScheduler
)
from .layer_schedulers import (
    LayerAgeAwareLR,
    CascadingDecayScheduler,
    LayerwiseAdaptiveRates,
    ProgressiveFreezingScheduler
)
from .connection_schedulers import (
    MultiScaleLearning,
    SoftClampingScheduler,
    SparsityAwareScheduler,
    AgeBasedScheduler
)

try:
    from ..extrema_analyzer import detect_network_extrema
except ImportError:
    # Fallback if extrema_analyzer is not available
    def detect_network_extrema(*args, **kwargs):
        return {}


class UnifiedAdaptiveLearning(BaseLearningRateScheduler):
    """
    The Ultimate Combination: Unified Adaptive System
    
    Combines all techniques into one coherent system:
    1. Phase detection from extrema
    2. Layer-wise adaptation
    3. Scale-based rates (connection birth time)
    4. Age-based soft decay
    5. Extrema proximity bonus
    6. All the additional sophisticated strategies
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 min_lr: float = 1e-6,
                 max_lr: float = 0.1,
                 extrema_boost: float = 2.0,
                 enable_extrema_phase: bool = True,
                 enable_layer_age: bool = True,
                 enable_multi_scale: bool = True,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.extrema_boost = extrema_boost
        
        # Initialize component systems
        self.extrema_phase = ExtremaPhaseScheduler(base_lr) if enable_extrema_phase else None
        self.layer_age = LayerAgeAwareLR(base_lr) if enable_layer_age else None
        self.multi_scale = MultiScaleLearning(base_lr) if enable_multi_scale else None
        
        # Track extrema proximity
        self.extrema_connections = set()
        self.total_layers = 1
    
    def get_learning_rate(self, 
                         connection_id: str = "", 
                         layer_idx: int = 0,
                         network=None,
                         data_loader=None,
                         device='cuda',
                         **kwargs) -> float:
        """Get unified learning rate combining all factors"""
        
        # Start with base learning rate
        final_lr = self.base_lr
        
        # 1. Base phase detection from extrema
        if self.extrema_phase and network is not None and data_loader is not None:
            phase_lr = self.extrema_phase.get_learning_rate(network, data_loader, device)
            phase_multiplier = phase_lr / self.base_lr
            final_lr *= phase_multiplier
        
        # 2. Layer-wise adaptation
        if self.layer_age:
            layer_multiplier = self.layer_age.get_layer_rate(layer_idx)
            final_lr *= layer_multiplier
        
        # 3. Scale-based rate (when was connection born)
        if self.multi_scale and connection_id:
            scale_lr = self.multi_scale.get_connection_rate(connection_id)
            scale_multiplier = scale_lr / self.base_lr
            final_lr *= scale_multiplier
        
        # 4. Age-based soft decay
        if self.layer_age and connection_id:
            age = self.layer_age.connection_ages.get(connection_id, 0)
            age_multiplier = np.exp(-age / self.layer_age.decay_constant)
            final_lr *= age_multiplier
        
        # 5. Extrema proximity bonus
        extrema_multiplier = self.get_extrema_proximity_multiplier(connection_id)
        final_lr *= extrema_multiplier
        
        return np.clip(final_lr, self.min_lr, self.max_lr)
    
    def get_extrema_proximity_multiplier(self, connection_id: str) -> float:
        """Connections near extrema get different treatment"""
        if connection_id in self.extrema_connections:
            return self.extrema_boost  # Boost learning for critical routes
        else:
            return 1.0
    
    def update_extrema_connections(self, extrema_connection_ids: set):
        """Update which connections are near extrema"""
        self.extrema_connections = extrema_connection_ids
    
    def update_epoch(self, epoch: int):
        """Update current epoch for all subsystems"""
        super().update_epoch(epoch)
        if self.extrema_phase:
            self.extrema_phase.update_epoch(epoch)
        if self.layer_age:
            self.layer_age.update_epoch(epoch)
        if self.multi_scale:
            self.multi_scale.update_epoch(epoch)
    
    def update_connection_age(self, connection_id: str):
        """Update age for a connection"""
        if self.layer_age:
            self.layer_age.update_connection_age(connection_id)
    
    def register_new_connection(self, connection_id: str, layer_idx: int):
        """Register a new connection"""
        if self.multi_scale:
            self.multi_scale.register_connection(connection_id, birth_epoch=self.current_epoch)
        # Age starts at 0 automatically
    
    def set_network_structure(self, total_layers: int):
        """Set network structure information"""
        self.total_layers = total_layers
        if self.layer_age:
            self.layer_age.set_total_layers(total_layers)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the unified system state"""
        summary = {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'total_layers': self.total_layers,
            'extrema_connections': len(self.extrema_connections)
        }
        
        if self.extrema_phase:
            summary['extrema_phase'] = {
                'recent_extrema_rates': self.extrema_phase.extrema_history[-5:],
                'trend': self.extrema_phase.get_phase_trend(),
                'current_phase': self.extrema_phase.current_phase
            }
        
        if self.layer_age:
            summary['layer_age'] = {
                'total_layers': self.layer_age.total_layers,
                'tracked_connections': len(self.layer_age.connection_ages)
            }
        
        if self.multi_scale:
            summary['multi_scale'] = {
                'tracked_connections': len(self.multi_scale.connection_registry),
                'scale_thresholds': self.multi_scale.scale_snapshots
            }
        
        return summary


class AdaptiveLearningRateManager:
    """
    Unified manager for all adaptive learning rate strategies.
    
    Combines all strategies into a cohesive system that can be easily
    integrated into structure_net training loops.
    """
    
    def __init__(self, 
                 network: nn.Module,
                 base_lr: float = 0.001,
                 strategy: Union[str, LearningRateStrategy] = LearningRateStrategy.BASIC,
                 # Basic strategy flags
                 enable_exponential_backoff: bool = True,
                 enable_layerwise_rates: bool = True,
                 enable_soft_clamping: bool = True,
                 enable_scale_dependent: bool = True,
                 enable_phase_based: bool = True,
                 # Advanced strategy flags
                 enable_extrema_phase: bool = False,
                 enable_layer_age_aware: bool = False,
                 enable_multi_scale: bool = False,
                 enable_unified_system: bool = False,
                 # Custom scheduler configurations
                 scheduler_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                 **kwargs):
        
        self.network = network
        self.base_lr = base_lr
        self.strategy = LearningRateStrategy(strategy) if isinstance(strategy, str) else strategy
        
        # Initialize schedulers based on strategy and flags
        self.schedulers = []
        self.scheduler_configs = scheduler_configs or {}
        
        # Apply strategy presets
        if self.strategy == LearningRateStrategy.BASIC:
            enable_extrema_phase = False
            enable_layer_age_aware = False
            enable_multi_scale = False
            enable_unified_system = False
        elif self.strategy == LearningRateStrategy.ADVANCED:
            enable_extrema_phase = True
            enable_layer_age_aware = True
            enable_multi_scale = False
            enable_unified_system = False
        elif self.strategy == LearningRateStrategy.COMPREHENSIVE:
            enable_extrema_phase = True
            enable_layer_age_aware = True
            enable_multi_scale = True
            enable_unified_system = False
        elif self.strategy == LearningRateStrategy.ULTIMATE:
            enable_extrema_phase = True
            enable_layer_age_aware = True
            enable_multi_scale = True
            enable_unified_system = True
        
        # Initialize basic strategies
        if enable_exponential_backoff:
            config = self.scheduler_configs.get('exponential_backoff', {})
            self.schedulers.append(ExponentialBackoffScheduler(base_lr, **config))
        
        if enable_layerwise_rates:
            config = self.scheduler_configs.get('layerwise_rates', {})
            self.schedulers.append(LayerwiseAdaptiveRates(base_lr, **config))
        
        if enable_soft_clamping:
            config = self.scheduler_configs.get('soft_clamping', {})
            self.schedulers.append(SoftClampingScheduler(base_lr, **config))
        
        if enable_scale_dependent:
            config = self.scheduler_configs.get('scale_dependent', {})
            self.schedulers.append(MultiScaleLearning(base_lr, **config))
        
        if enable_phase_based:
            config = self.scheduler_configs.get('phase_based', {})
            self.schedulers.append(GrowthPhaseScheduler(base_lr, **config))
        
        # Initialize advanced combination systems
        if enable_extrema_phase:
            config = self.scheduler_configs.get('extrema_phase', {})
            self.schedulers.append(ExtremaPhaseScheduler(base_lr, **config))
        
        if enable_layer_age_aware:
            config = self.scheduler_configs.get('layer_age_aware', {})
            self.schedulers.append(LayerAgeAwareLR(base_lr, **config))
        
        if enable_multi_scale and not any(isinstance(s, MultiScaleLearning) for s in self.schedulers):
            config = self.scheduler_configs.get('multi_scale', {})
            self.schedulers.append(MultiScaleLearning(base_lr, **config))
        
        if enable_unified_system:
            config = self.scheduler_configs.get('unified_system', {})
            self.unified_system = UnifiedAdaptiveLearning(base_lr, **config)
            self.schedulers.append(self.unified_system)
        else:
            self.unified_system = None
        
        # Track network structure
        self.sparse_layers = self._get_sparse_layers(network)
        self.n_layers = len(self.sparse_layers)
        
        # Set up layer structure for schedulers that need it
        for scheduler in self.schedulers:
            if hasattr(scheduler, 'set_total_layers'):
                scheduler.set_total_layers(self.n_layers)
            if hasattr(scheduler, 'set_network_structure'):
                scheduler.set_network_structure(self.n_layers)
        
        # History tracking
        self.history = LearningRateHistory()
        
        print(f"🎯 Initialized AdaptiveLearningRateManager")
        print(f"   Strategy: {self.strategy.value}")
        print(f"   Base LR: {base_lr}")
        print(f"   Active schedulers: {len(self.schedulers)}")
        for i, scheduler in enumerate(self.schedulers):
            print(f"     {i+1}. {scheduler.__class__.__name__}")
    
    def _get_sparse_layers(self, network):
        """Extract sparse layers from network"""
        try:
            from ...core.layers import StandardSparseLayer
            return [layer for layer in network.modules() if isinstance(layer, StandardSparseLayer)]
        except ImportError:
            # Fallback: return all linear layers
            return [layer for layer in network.modules() if isinstance(layer, nn.Linear)]
    
    def create_adaptive_optimizer(self, optimizer_class=optim.Adam, **optimizer_kwargs) -> AdaptiveOptimizerWrapper:
        """Create optimizer with adaptive parameter groups"""
        param_groups = self._create_parameter_groups()
        
        # Create base optimizer
        base_optimizer = optimizer_class(param_groups, **optimizer_kwargs)
        
        # Wrap with adaptive functionality
        adaptive_optimizer = AdaptiveOptimizerWrapper(base_optimizer, self.schedulers)
        
        return adaptive_optimizer
    
    def _create_parameter_groups(self) -> List[Dict]:
        """Create parameter groups for each layer with adaptive rates"""
        param_groups = []
        
        # Create parameter groups for each layer
        for layer_idx, layer in enumerate(self.sparse_layers):
            # Get base rate for this layer from first applicable scheduler
            layer_rate = self.base_lr
            for scheduler in self.schedulers:
                if hasattr(scheduler, 'get_layer_rate'):
                    layer_rate = scheduler.get_learning_rate(layer_idx=layer_idx)
                    break
            
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': layer_rate,
                'layer_idx': layer_idx,
                'layer_type': 'sparse',
                'name': f'sparse_layer_{layer_idx}'
            })
        
        # Add any remaining parameters (non-sparse layers)
        remaining_params = []
        sparse_param_ids = set()
        for layer in self.sparse_layers:
            for param in layer.parameters():
                sparse_param_ids.add(id(param))
        
        for module in self.network.modules():
            for param in module.parameters():
                if id(param) not in sparse_param_ids:
                    remaining_params.append(param)
        
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': self.base_lr,
                'layer_type': 'other',
                'name': 'other_params'
            })
        
        return param_groups
    
    def update_learning_rates(self, optimizer: Union[optim.Optimizer, AdaptiveOptimizerWrapper], 
                            epoch: int, **context):
        """Update learning rates for current epoch"""
        # Update all schedulers
        for scheduler in self.schedulers:
            scheduler.update_epoch(epoch)
        
        # If using adaptive wrapper, let it handle the updates
        if isinstance(optimizer, AdaptiveOptimizerWrapper):
            optimizer.update_learning_rates(epoch, **context)
            return
        
        # Otherwise, manually update parameter groups
        lr_data = {'epoch': epoch}
        
        for group in optimizer.param_groups:
            if 'layer_idx' in group:
                layer_idx = group['layer_idx']
                group_name = group.get('name', f'layer_{layer_idx}')
                
                # Get learning rate from unified system or first applicable scheduler
                if self.unified_system:
                    new_lr = self.unified_system.get_learning_rate(
                        layer_idx=layer_idx, **context
                    )
                else:
                    new_lr = self.base_lr
                    for scheduler in self.schedulers:
                        if hasattr(scheduler, 'get_layer_rate'):
                            new_lr = scheduler.get_learning_rate(layer_idx=layer_idx, **context)
                            break
                
                group['lr'] = new_lr
                lr_data[group_name] = new_lr
            else:
                # For other parameter groups, use base learning rate
                group_name = group.get('name', 'other')
                group['lr'] = self.base_lr
                lr_data[group_name] = self.base_lr
        
        # Record in history
        self.history.record(epoch, lr_data)
    
    def apply_gradient_modifications(self, optimizer: Union[optim.Optimizer, AdaptiveOptimizerWrapper]):
        """Apply gradient modifications (soft clamping, scale-dependent adjustments)"""
        if isinstance(optimizer, AdaptiveOptimizerWrapper):
            # Let the wrapper handle this
            return
        
        # Apply modifications from connection-level schedulers
        for scheduler in self.schedulers:
            if hasattr(scheduler, 'apply_to_gradient'):
                for group_idx, group in enumerate(optimizer.param_groups):
                    if 'layer_idx' in group:
                        layer_idx = group['layer_idx']
                        if layer_idx < len(self.sparse_layers):
                            layer = self.sparse_layers[layer_idx]
                            
                            # Apply modifications to layer parameters
                            for param_name, param in layer.named_parameters():
                                if param.grad is not None:
                                    connection_id = f"layer_{layer_idx}_{param_name}"
                                    param.grad = scheduler.apply_to_gradient(param.grad, connection_id)
    
    def get_current_rates_summary(self) -> Dict[str, Any]:
        """Get summary of current learning rates and strategies"""
        summary = {
            'epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'strategy': self.strategy.value,
            'n_layers': self.n_layers,
            'schedulers': {}
        }
        
        for scheduler in self.schedulers:
            scheduler_name = scheduler.__class__.__name__
            if hasattr(scheduler, 'get_config'):
                summary['schedulers'][scheduler_name] = scheduler.get_config()
            else:
                summary['schedulers'][scheduler_name] = {'active': True}
        
        return summary
    
    def print_rates_summary(self):
        """Print current learning rates summary"""
        summary = self.get_current_rates_summary()
        
        print(f"\n📊 Learning Rates Summary (Epoch {summary['epoch']})")
        print("=" * 60)
        print(f"Strategy: {summary['strategy']}")
        print(f"Base LR: {summary['base_lr']:.6f}")
        print(f"Network Layers: {summary['n_layers']}")
        print(f"Active Schedulers: {len(summary['schedulers'])}")
        
        for scheduler_name, config in summary['schedulers'].items():
            print(f"\n🎯 {scheduler_name}:")
            if isinstance(config, dict):
                for key, value in list(config.items())[:5]:  # Show first 5 items
                    if isinstance(value, (list, tuple)) and len(value) > 3:
                        print(f"   {key}: [{value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f}, ...]")
                    elif isinstance(value, float):
                        print(f"   {key}: {value:.6f}")
                    else:
                        print(f"   {key}: {value}")
    
    def save_state(self) -> Dict[str, Any]:
        """Save manager state for checkpointing"""
        state = {
            'base_lr': self.base_lr,
            'strategy': self.strategy.value,
            'current_epoch': self.current_epoch,
            'scheduler_configs': self.scheduler_configs,
            'schedulers': []
        }
        
        for scheduler in self.schedulers:
            scheduler_state = {
                'class_name': scheduler.__class__.__name__,
                'config': scheduler.get_config() if hasattr(scheduler, 'get_config') else {}
            }
            state['schedulers'].append(scheduler_state)
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load manager state from checkpoint"""
        self.base_lr = state.get('base_lr', self.base_lr)
        self.current_epoch = state.get('current_epoch', 0)
        
        # Update scheduler epochs
        for scheduler in self.schedulers:
            scheduler.update_epoch(self.current_epoch)
    
    @property
    def current_epoch(self) -> int:
        """Get current epoch from schedulers"""
        if self.schedulers:
            return self.schedulers[0].current_epoch
        return 0
