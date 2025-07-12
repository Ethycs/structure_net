#!/usr/bin/env python3
"""
Layer Age-Aware Learning Rate Scheduler Component

Combines layer-specific learning rates with age-based decay for sophisticated
connection-level adaptation. Early layers learn slower (general features) while
late layers learn faster (task-specific features).
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List

from ...core.base_components import BaseScheduler
from ...core.interfaces import (
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)


class LayerAgeAwareScheduler(BaseScheduler):
    """
    Layer-wise Rates + Connection Age Soft Clamping
    
    Combines layer-specific learning rates with age-based decay
    for sophisticated connection-level adaptation.
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="LayerAgeAwareScheduler",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"layer_idx"},
            provided_outputs={"learning_rate", "layer_info"},
            optional_inputs={"connection_id", "total_layers"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 decay_constant: float = 50.0,
                 early_layer_rate: float = 0.1,
                 late_layer_boost: float = 0.9,
                 **kwargs):
        """
        Initialize Layer Age-Aware Scheduler.
        
        Args:
            base_lr: Base learning rate
            total_layers: Total number of layers in the network
            decay_constant: Decay constant for age-based soft clamping
            early_layer_rate: Learning rate multiplier for early layers
            late_layer_boost: Additional boost for late layers
        """
        super().__init__("LayerAgeAwareScheduler")
        
        self.base_lr = base_lr
        self.total_layers = total_layers
        self.decay_constant = decay_constant
        self.early_layer_rate = early_layer_rate
        self.late_layer_boost = late_layer_boost
        self.connection_ages = {}
        self.current_epoch = 0
        self._config = kwargs
    
    def set_total_layers(self, total_layers: int):
        """Update the total number of layers."""
        self.total_layers = total_layers
    
    def get_layer_rate(self, layer_idx: int) -> float:
        """
        Get layer-specific rate multiplier.
        Early layers: slower (more general features)
        Late layers: faster (task-specific)
        """
        if self.total_layers <= 1:
            return 1.0
        
        depth_ratio = layer_idx / (self.total_layers - 1)
        return self.early_layer_rate + self.late_layer_boost * depth_ratio
    
    def get_connection_lr(self, connection_id: str, layer_idx: int) -> float:
        """Get learning rate for specific connection combining layer and age factors."""
        # Layer-specific base rate
        layer_rate = self.get_layer_rate(layer_idx)
        
        # Age-based decay (soft clamping)
        age = self.connection_ages.get(connection_id, 0)
        age_factor = np.exp(-age / self.decay_constant)
        
        # Combine both factors
        return self.base_lr * layer_rate * age_factor
    
    def update_connection_age(self, connection_id: str):
        """Increment age of a connection."""
        self.connection_ages[connection_id] = self.connection_ages.get(connection_id, 0) + 1
    
    def update_all_connection_ages(self):
        """Age all tracked connections by one epoch."""
        for connection_id in self.connection_ages:
            self.connection_ages[connection_id] += 1
    
    def compute_learning_rate(self, 
                            global_step: int,
                            epoch: int,
                            optimizer_state: Optional[Dict[str, Any]] = None,
                            model_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Compute learning rates for all layers.
        
        Returns:
            Dictionary mapping layer indices to learning rates
        """
        self.current_epoch = epoch
        
        # Get layer information from model state if provided
        if model_state and 'layer_info' in model_state:
            layer_info = model_state['layer_info']
            rates = {}
            
            for layer_idx, info in enumerate(layer_info):
                # Base layer rate
                base_rate = self.base_lr * self.get_layer_rate(layer_idx)
                rates[f"layer_{layer_idx}"] = base_rate
                
                # If connection IDs are provided, compute connection-specific rates
                if 'connections' in info:
                    for conn_id in info['connections']:
                        rates[conn_id] = self.get_connection_lr(conn_id, layer_idx)
            
            return rates
        else:
            # Return rates for all layers
            rates = {}
            for layer_idx in range(self.total_layers):
                rates[f"layer_{layer_idx}"] = self.base_lr * self.get_layer_rate(layer_idx)
            return rates
    
    def get_scheduler_state(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'connection_ages': self.connection_ages,
            'total_layers': self.total_layers,
            'decay_constant': self.decay_constant,
            'early_layer_rate': self.early_layer_rate,
            'late_layer_boost': self.late_layer_boost,
            'base_lr': self.base_lr,
            'config': self._config
        }
    
    def load_scheduler_state(self, state: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state.get('current_epoch', 0)
        self.connection_ages = state.get('connection_ages', {})
        self.total_layers = state.get('total_layers', self.total_layers)
        self.decay_constant = state.get('decay_constant', self.decay_constant)
        self.early_layer_rate = state.get('early_layer_rate', self.early_layer_rate)
        self.late_layer_boost = state.get('late_layer_boost', self.late_layer_boost)
        self.base_lr = state.get('base_lr', self.base_lr)
        self._config = state.get('config', {})
    
    def get_layer_distribution(self) -> List[float]:
        """Get distribution of learning rates across layers."""
        return [self.get_layer_rate(i) for i in range(self.total_layers)]
    
    def update_epoch(self, epoch: int):
        """Update current epoch and age all connections."""
        self.current_epoch = epoch
        self.update_all_connection_ages()