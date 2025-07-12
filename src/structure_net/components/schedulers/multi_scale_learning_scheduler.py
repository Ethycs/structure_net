#!/usr/bin/env python3
"""
Multi-Scale Learning Scheduler Component

Scale-dependent and temporal snapshot integration scheduler that adapts learning rates
based on when connections were created and their scale in network evolution.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional

from ...core.base_components import BaseScheduler
from ...core.interfaces import (
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)


class MultiScaleLearningScheduler(BaseScheduler):
    """
    Scale-Dependent + Temporal Snapshot Integration
    
    Different learning rates based on when connections were created
    and their scale in the network evolution.
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="MultiScaleLearningScheduler",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"connection_id"},
            provided_outputs={"learning_rate", "scale_info"},
            optional_inputs={"connection_metadata"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 coarse_threshold: int = 20,
                 medium_threshold: int = 50,
                 fine_threshold: int = 100,
                 coarse_multiplier: float = 0.1,
                 medium_multiplier: float = 0.5,
                 fine_multiplier: float = 1.0,
                 temporal_decay: float = 0.01,
                 **kwargs):
        """
        Initialize Multi-Scale Learning Scheduler.
        
        Args:
            base_lr: Base learning rate
            coarse_threshold: Epoch threshold for coarse scale
            medium_threshold: Epoch threshold for medium scale
            fine_threshold: Epoch threshold for fine scale
            coarse_multiplier: Learning rate multiplier for coarse scale
            medium_multiplier: Learning rate multiplier for medium scale
            fine_multiplier: Learning rate multiplier for fine scale
            temporal_decay: Exponential decay factor within scale
        """
        super().__init__("MultiScaleLearningScheduler")
        
        self.base_lr = base_lr
        self.current_epoch = 0
        self.connection_registry = {}
        
        self.scale_snapshots = {
            'coarse': {'epoch_threshold': coarse_threshold, 'lr_multiplier': coarse_multiplier},
            'medium': {'epoch_threshold': medium_threshold, 'lr_multiplier': medium_multiplier},
            'fine': {'epoch_threshold': fine_threshold, 'lr_multiplier': fine_multiplier}
        }
        self.temporal_decay = temporal_decay
        self._config = kwargs
    
    def register_connection(self, connection_id: str, **metadata):
        """Register a new connection with metadata."""
        self.connection_registry[connection_id] = {
            'birth_epoch': self.current_epoch,
            **metadata
        }
    
    def get_connection_metadata(self, connection_id: str) -> Dict[str, Any]:
        """Get metadata for a connection."""
        return self.connection_registry.get(connection_id, {})
    
    def get_connection_scale(self, connection_id: str) -> str:
        """Determine which scale this connection belongs to."""
        metadata = self.get_connection_metadata(connection_id)
        birth_epoch = metadata.get('birth_epoch', self.current_epoch)
        
        if birth_epoch < self.scale_snapshots['coarse']['epoch_threshold']:
            return 'coarse'
        elif birth_epoch < self.scale_snapshots['medium']['epoch_threshold']:
            return 'medium'
        else:
            return 'fine'
    
    def get_connection_rate(self, connection_id: str, **kwargs) -> float:
        """Get learning rate for connection based on its birth epoch and scale."""
        metadata = self.get_connection_metadata(connection_id)
        birth_epoch = metadata.get('birth_epoch', self.current_epoch)
        
        # Determine scale
        scale = self.get_connection_scale(connection_id)
        
        # Scale-specific learning rate
        scale_lr = self.base_lr * self.scale_snapshots[scale]['lr_multiplier']
        
        # Add exponential decay within scale
        epochs_in_scale = self.current_epoch - birth_epoch
        decay_factor = np.exp(-self.temporal_decay * epochs_in_scale)
        
        return scale_lr * decay_factor
    
    def compute_learning_rate(self, 
                            global_step: int,
                            epoch: int,
                            optimizer_state: Optional[Dict[str, Any]] = None,
                            model_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Compute learning rate for all connections.
        
        Returns:
            Dictionary mapping connection IDs to learning rates
        """
        self.current_epoch = epoch
        
        # If no connections registered, return base rate
        if not self.connection_registry:
            return {"default": self.base_lr}
        
        # Compute rates for all registered connections
        rates = {}
        for connection_id in self.connection_registry:
            rates[connection_id] = self.get_connection_rate(connection_id)
        
        return rates
    
    def get_scheduler_state(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'connection_registry': self.connection_registry,
            'scale_snapshots': self.scale_snapshots,
            'temporal_decay': self.temporal_decay,
            'base_lr': self.base_lr,
            'config': self._config
        }
    
    def load_scheduler_state(self, state: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state.get('current_epoch', 0)
        self.connection_registry = state.get('connection_registry', {})
        self.scale_snapshots = state.get('scale_snapshots', self.scale_snapshots)
        self.temporal_decay = state.get('temporal_decay', self.temporal_decay)
        self.base_lr = state.get('base_lr', self.base_lr)
        self._config = state.get('config', {})
    
    def get_scale_distribution(self) -> Dict[str, int]:
        """Get distribution of connections across scales."""
        distribution = {scale: 0 for scale in ['coarse', 'medium', 'fine']}
        
        for connection_id in self.connection_registry:
            scale = self.get_connection_scale(connection_id)
            distribution[scale] += 1
        
        return distribution
    
    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch