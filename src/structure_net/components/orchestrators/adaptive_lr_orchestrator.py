#!/usr/bin/env python3
"""
Adaptive Learning Rate Orchestrator Component

Unified adaptive learning rate management system that combines multiple scheduler
strategies into a cohesive orchestration framework. Coordinates phase detection,
layer-wise adaptation, connection-level rates, and extrema-based adjustments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from ...core.base_components import BaseOrchestrator
from ...core.interfaces import (
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)
from ..schedulers import (
    MultiScaleLearningScheduler,
    LayerAgeAwareScheduler,
    ExtremaPhaseScheduler
)


class AdaptiveLearningRateOrchestrator(BaseOrchestrator):
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
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="AdaptiveLearningRateOrchestrator",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"optimizer", "network"},
            provided_outputs={"adapted_optimizer", "lr_info", "phase_info"},
            optional_inputs={"data_loader", "extrema_patterns"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=True,
                parallel_safe=False
            ),
            requires_component_types={MultiScaleLearningScheduler, LayerAgeAwareScheduler, ExtremaPhaseScheduler}
        )
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 min_lr: float = 1e-6,
                 max_lr: float = 0.1,
                 extrema_boost: float = 2.0,
                 enable_extrema_phase: bool = True,
                 enable_layer_age: bool = True,
                 enable_multi_scale: bool = True,
                 **kwargs):
        """
        Initialize Adaptive Learning Rate Orchestrator.
        
        Args:
            base_lr: Base learning rate
            min_lr: Minimum allowed learning rate
            max_lr: Maximum allowed learning rate
            extrema_boost: Boost factor for connections near extrema
            enable_extrema_phase: Enable extrema-based phase detection
            enable_layer_age: Enable layer-age aware scheduling
            enable_multi_scale: Enable multi-scale learning rates
        """
        super().__init__("AdaptiveLearningRateOrchestrator")
        
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.extrema_boost = extrema_boost
        self.current_epoch = 0
        
        # Initialize component systems
        self.components = {
            'schedulers': []
        }
        
        if enable_extrema_phase:
            self.extrema_phase = ExtremaPhaseScheduler(base_lr)
            self.components['schedulers'].append(self.extrema_phase)
        else:
            self.extrema_phase = None
            
        if enable_layer_age:
            self.layer_age = LayerAgeAwareScheduler(base_lr)
            self.components['schedulers'].append(self.layer_age)
        else:
            self.layer_age = None
            
        if enable_multi_scale:
            self.multi_scale = MultiScaleLearningScheduler(base_lr)
            self.components['schedulers'].append(self.multi_scale)
        else:
            self.multi_scale = None
        
        # Track extrema proximity
        self.extrema_connections: Set[str] = set()
        self.total_layers = 1
        self._config = kwargs
        
        # Learning rate history
        self.lr_history = []
    
    def orchestrate(self, 
                   components: Dict[str, Any],
                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate adaptive learning rate scheduling.
        
        Args:
            components: Dictionary containing optimizer, network, etc.
            context: Context with epoch, step, data_loader, etc.
            
        Returns:
            Dictionary with adapted optimizer and scheduling info
        """
        optimizer = components.get('optimizer')
        network = components.get('network')
        data_loader = context.get('data_loader')
        epoch = context.get('epoch', 0)
        global_step = context.get('global_step', 0)
        device = context.get('device', 'cuda')
        
        self.current_epoch = epoch
        
        # Compute learning rates for all parameter groups
        param_lr_map = self._compute_unified_learning_rates(
            network=network,
            data_loader=data_loader,
            device=device,
            global_step=global_step,
            epoch=epoch
        )
        
        # Apply learning rates to optimizer
        adapted_optimizer = self._apply_learning_rates(optimizer, param_lr_map, network)
        
        # Collect scheduling info
        scheduling_info = {
            'current_lr_map': param_lr_map,
            'phase_info': self._get_phase_info(),
            'layer_distribution': self._get_layer_distribution(),
            'scale_distribution': self._get_scale_distribution(),
            'extrema_connections': len(self.extrema_connections)
        }
        
        # Update history
        self.lr_history.append({
            'epoch': epoch,
            'global_step': global_step,
            'avg_lr': np.mean(list(param_lr_map.values())),
            'min_lr': min(param_lr_map.values()),
            'max_lr': max(param_lr_map.values())
        })
        
        return {
            'adapted_optimizer': adapted_optimizer,
            'lr_info': param_lr_map,
            'phase_info': scheduling_info
        }
    
    def _compute_unified_learning_rates(self,
                                      network: nn.Module,
                                      data_loader: Any,
                                      device: str,
                                      global_step: int,
                                      epoch: int) -> Dict[str, float]:
        """Compute unified learning rates for all parameters."""
        param_lr_map = {}
        
        # Model state for schedulers
        model_state = {
            'network': network,
            'data_loader': data_loader,
            'device': device,
            'layer_info': self._extract_layer_info(network)
        }
        
        # Get base rates from each scheduler
        rates = {}
        
        # 1. Phase-based rate
        if self.extrema_phase:
            phase_rates = self.extrema_phase.compute_learning_rate(
                global_step, epoch, model_state=model_state
            )
            rates['phase'] = phase_rates.get('default', self.base_lr)
        
        # 2. Layer-age based rates
        if self.layer_age:
            layer_rates = self.layer_age.compute_learning_rate(
                global_step, epoch, model_state=model_state
            )
            rates['layers'] = layer_rates
        
        # 3. Multi-scale connection rates
        if self.multi_scale:
            scale_rates = self.multi_scale.compute_learning_rate(
                global_step, epoch, model_state=model_state
            )
            rates['connections'] = scale_rates
        
        # Combine rates for each parameter
        for name, param in network.named_parameters():
            if param.requires_grad:
                # Start with base rate
                final_lr = self.base_lr
                
                # Apply phase multiplier
                if 'phase' in rates:
                    final_lr *= rates['phase'] / self.base_lr
                
                # Apply layer-specific rate
                layer_idx = self._get_layer_index(name)
                if 'layers' in rates and f'layer_{layer_idx}' in rates['layers']:
                    layer_factor = rates['layers'][f'layer_{layer_idx}'] / self.base_lr
                    final_lr *= layer_factor
                
                # Apply connection-specific rate if tracked
                conn_id = self._get_connection_id(name)
                if 'connections' in rates and conn_id in rates['connections']:
                    conn_factor = rates['connections'][conn_id] / self.base_lr
                    final_lr *= conn_factor
                
                # Apply extrema boost if near extrema
                if conn_id in self.extrema_connections:
                    final_lr *= self.extrema_boost
                
                # Clamp to valid range
                final_lr = max(self.min_lr, min(self.max_lr, final_lr))
                
                param_lr_map[name] = final_lr
        
        return param_lr_map
    
    def _apply_learning_rates(self, 
                            optimizer: optim.Optimizer,
                            param_lr_map: Dict[str, float],
                            network: nn.Module) -> optim.Optimizer:
        """Apply computed learning rates to optimizer parameter groups."""
        # Create parameter groups if needed
        if len(optimizer.param_groups) == 1:
            # Need to split into per-parameter groups
            optimizer.param_groups = []
            for name, param in network.named_parameters():
                if param.requires_grad:
                    optimizer.add_param_group({
                        'params': [param],
                        'lr': param_lr_map.get(name, self.base_lr),
                        'name': name
                    })
        else:
            # Update existing groups
            for group in optimizer.param_groups:
                if 'name' in group:
                    group['lr'] = param_lr_map.get(group['name'], self.base_lr)
        
        return optimizer
    
    def register_extrema_connections(self, extrema_patterns: Dict[str, Any]):
        """Register connections that are near extrema neurons."""
        self.extrema_connections.clear()
        
        for layer_name, pattern in extrema_patterns.items():
            if isinstance(pattern, dict) and 'extrema_indices' in pattern:
                for idx in pattern['extrema_indices']:
                    # Create connection IDs for weights connected to extrema
                    self.extrema_connections.add(f"{layer_name}.weight[{idx}]")
    
    def _extract_layer_info(self, network: nn.Module) -> List[Dict[str, Any]]:
        """Extract layer information from network."""
        layer_info = []
        layer_idx = 0
        
        for name, module in network.named_modules():
            if hasattr(module, 'weight'):
                info = {
                    'name': name,
                    'index': layer_idx,
                    'type': type(module).__name__,
                    'shape': tuple(module.weight.shape)
                }
                layer_info.append(info)
                layer_idx += 1
        
        self.total_layers = layer_idx
        if self.layer_age:
            self.layer_age.set_total_layers(self.total_layers)
        
        return layer_info
    
    def _get_layer_index(self, param_name: str) -> int:
        """Extract layer index from parameter name."""
        # Simple heuristic: count dots before 'weight' or 'bias'
        parts = param_name.split('.')
        layer_idx = 0
        for i, part in enumerate(parts):
            if part in ['weight', 'bias']:
                break
            if part.isdigit():
                layer_idx = int(part)
        return layer_idx
    
    def _get_connection_id(self, param_name: str) -> str:
        """Generate connection ID from parameter name."""
        return param_name
    
    def _get_phase_info(self) -> Dict[str, Any]:
        """Get current phase information."""
        if self.extrema_phase:
            return self.extrema_phase.get_phase_info()
        return {'phase': 'unknown'}
    
    def _get_layer_distribution(self) -> List[float]:
        """Get layer-wise learning rate distribution."""
        if self.layer_age:
            return self.layer_age.get_layer_distribution()
        return []
    
    def _get_scale_distribution(self) -> Dict[str, int]:
        """Get scale distribution of connections."""
        if self.multi_scale:
            return self.multi_scale.get_scale_distribution()
        return {}
    
    def get_orchestrator_state(self) -> Dict[str, Any]:
        """Get orchestrator state for checkpointing."""
        state = {
            'current_epoch': self.current_epoch,
            'extrema_connections': list(self.extrema_connections),
            'total_layers': self.total_layers,
            'lr_history': self.lr_history,
            'config': self._config
        }
        
        # Add component states
        if self.extrema_phase:
            state['extrema_phase_state'] = self.extrema_phase.get_scheduler_state()
        if self.layer_age:
            state['layer_age_state'] = self.layer_age.get_scheduler_state()
        if self.multi_scale:
            state['multi_scale_state'] = self.multi_scale.get_scheduler_state()
        
        return state
    
    def load_orchestrator_state(self, state: Dict[str, Any]):
        """Load orchestrator state from checkpoint."""
        self.current_epoch = state.get('current_epoch', 0)
        self.extrema_connections = set(state.get('extrema_connections', []))
        self.total_layers = state.get('total_layers', 1)
        self.lr_history = state.get('lr_history', [])
        self._config = state.get('config', {})
        
        # Load component states
        if self.extrema_phase and 'extrema_phase_state' in state:
            self.extrema_phase.load_scheduler_state(state['extrema_phase_state'])
        if self.layer_age and 'layer_age_state' in state:
            self.layer_age.load_scheduler_state(state['layer_age_state'])
        if self.multi_scale and 'multi_scale_state' in state:
            self.multi_scale.load_scheduler_state(state['multi_scale_state'])