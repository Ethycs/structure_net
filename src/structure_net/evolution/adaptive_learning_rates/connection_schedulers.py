"""
Connection-level adaptive learning rate schedulers.

This module contains schedulers that adapt learning rates at the individual
connection level, considering factors like connection age, sparsity, and scale.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math

from .base import BaseConnectionScheduler, SchedulerMixin


class MultiScaleLearning(BaseConnectionScheduler, SchedulerMixin):
    """
    Scale-Dependent + Temporal Snapshot Integration
    
    Different learning rates based on when connections were created
    and their scale in the network evolution.
    """
    
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
        super().__init__(base_lr, **kwargs)
        self.scale_snapshots = {
            'coarse': {'epoch_threshold': coarse_threshold, 'lr_multiplier': coarse_multiplier},
            'medium': {'epoch_threshold': medium_threshold, 'lr_multiplier': medium_multiplier},
            'fine': {'epoch_threshold': fine_threshold, 'lr_multiplier': fine_multiplier}
        }
        self.temporal_decay = temporal_decay
    
    def get_connection_scale(self, connection_id: str) -> str:
        """Determine which scale this connection belongs to"""
        metadata = self.get_connection_metadata(connection_id)
        birth_epoch = metadata.get('birth_epoch', self.current_epoch)
        
        if birth_epoch < self.scale_snapshots['coarse']['epoch_threshold']:
            return 'coarse'
        elif birth_epoch < self.scale_snapshots['medium']['epoch_threshold']:
            return 'medium'
        else:
            return 'fine'
    
    def get_connection_rate(self, connection_id: str, **kwargs) -> float:
        """Get learning rate for connection based on its birth epoch and scale"""
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
    
    def get_learning_rate(self, connection_id: str = "", **kwargs) -> float:
        """Get learning rate for a specific connection"""
        if connection_id:
            return self.get_connection_rate(connection_id, **kwargs)
        return self.base_lr
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'scale_snapshots': self.scale_snapshots,
            'temporal_decay': self.temporal_decay,
            'scale_distribution': {
                scale: sum(1 for conn_id in self.connection_registry.keys() 
                          if self.get_connection_scale(conn_id) == scale)
                for scale in ['coarse', 'medium', 'fine']
            }
        })
        return config


class SoftClampingScheduler(BaseConnectionScheduler, SchedulerMixin):
    """
    Soft Clamping (Gradual Freezing) - Gradual freezing instead of hard clamp
    
    Reduces learning rate with connection age, doesn't freeze completely.
    Allows old connections to still adapt but more slowly.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 max_age: int = 100,
                 min_clamp_factor: float = 0.1,
                 decay_rate: float = 1.0,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.max_age = max_age
        self.min_clamp_factor = min_clamp_factor
        self.decay_rate = decay_rate
        self.connection_ages = defaultdict(int)
    
    def get_connection_rate(self, connection_id: str, age: Optional[int] = None, **kwargs) -> float:
        """Get learning rate with soft clamping applied"""
        if age is None:
            age = self.connection_ages[connection_id]
        
        # Reduce learning rate with age, don't freeze completely
        clamp_factor = 1.0 - (age / self.max_age) * self.decay_rate
        clamp_factor = max(clamp_factor, self.min_clamp_factor)
        
        return self.base_lr * clamp_factor
    
    def get_learning_rate(self, connection_id: str = "", **kwargs) -> float:
        """Get learning rate for a specific connection"""
        if connection_id:
            return self.get_connection_rate(connection_id, **kwargs)
        return self.base_lr
    
    def update_connection_age(self, connection_id: str):
        """Increment age of a connection"""
        self.connection_ages[connection_id] += 1
    
    def get_connection_factor(self, connection_id: str) -> float:
        """Get current clamping factor for a connection"""
        age = self.connection_ages[connection_id]
        clamp_factor = 1.0 - (age / self.max_age) * self.decay_rate
        return max(clamp_factor, self.min_clamp_factor)
    
    def apply_to_gradient(self, gradient: torch.Tensor, connection_id: str) -> torch.Tensor:
        """Apply soft clamping to gradient"""
        clamp_factor = self.get_connection_factor(connection_id)
        return gradient * clamp_factor
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'max_age': self.max_age,
            'min_clamp_factor': self.min_clamp_factor,
            'decay_rate': self.decay_rate,
            'active_connections': len(self.connection_ages),
            'age_distribution': {
                'young': sum(1 for age in self.connection_ages.values() if age < self.max_age // 3),
                'middle': sum(1 for age in self.connection_ages.values() 
                            if self.max_age // 3 <= age < 2 * self.max_age // 3),
                'old': sum(1 for age in self.connection_ages.values() if age >= 2 * self.max_age // 3)
            }
        })
        return config


class SparsityAwareScheduler(BaseConnectionScheduler, SchedulerMixin):
    """
    Sparsity-Aware Learning Rates
    
    Adjust LR based on connection density.
    Purpose: Sparse connections may need higher learning rates.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 sparse_multiplier: float = 2.0,
                 dense_multiplier: float = 0.5,
                 sparse_threshold: float = 0.02,
                 dense_threshold: float = 0.1,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.sparse_multiplier = sparse_multiplier
        self.dense_multiplier = dense_multiplier
        self.sparse_threshold = sparse_threshold
        self.dense_threshold = dense_threshold
        self.layer_densities = {}
    
    def register_layer_density(self, layer_id: str, density: float):
        """Register the density of a layer"""
        self.layer_densities[layer_id] = density
    
    def get_sparsity_adjusted_lr(self, layer_id: str, density: Optional[float] = None) -> float:
        """Adjust learning rate based on layer sparsity"""
        if density is None:
            density = self.layer_densities.get(layer_id, 0.05)  # Default moderate density
        
        if density < self.sparse_threshold:  # Very sparse
            return self.base_lr * self.sparse_multiplier  # Need higher LR
        elif density > self.dense_threshold:  # Dense patches
            return self.base_lr * self.dense_multiplier  # Lower LR for stability
        else:
            return self.base_lr
    
    def get_connection_rate(self, connection_id: str, layer_id: Optional[str] = None, 
                          density: Optional[float] = None, **kwargs) -> float:
        """Get learning rate for connection based on sparsity"""
        if layer_id is not None:
            return self.get_sparsity_adjusted_lr(layer_id, density)
        
        # Try to extract layer_id from connection_id
        if '_layer_' in connection_id:
            layer_id = connection_id.split('_layer_')[1].split('_')[0]
            return self.get_sparsity_adjusted_lr(layer_id, density)
        
        return self.base_lr
    
    def get_learning_rate(self, connection_id: str = "", layer_id: Optional[str] = None, 
                         density: Optional[float] = None, **kwargs) -> float:
        """Get learning rate for a specific connection or layer"""
        if connection_id or layer_id:
            return self.get_connection_rate(connection_id, layer_id, density, **kwargs)
        return self.base_lr
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'sparse_multiplier': self.sparse_multiplier,
            'dense_multiplier': self.dense_multiplier,
            'sparse_threshold': self.sparse_threshold,
            'dense_threshold': self.dense_threshold,
            'tracked_layers': len(self.layer_densities),
            'density_distribution': {
                'sparse': sum(1 for d in self.layer_densities.values() if d < self.sparse_threshold),
                'moderate': sum(1 for d in self.layer_densities.values() 
                              if self.sparse_threshold <= d <= self.dense_threshold),
                'dense': sum(1 for d in self.layer_densities.values() if d > self.dense_threshold)
            }
        })
        return config


class AgeBasedScheduler(BaseConnectionScheduler, SchedulerMixin):
    """
    Age-Based Learning Rates for Connections
    
    Older connections learn slower, newer connections learn faster.
    Purpose: Stabilize old knowledge while allowing new adaptation.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 age_decay_base: float = 0.1,
                 max_age_effect: int = 50,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.age_decay_base = age_decay_base
        self.max_age_effect = max_age_effect
    
    def get_connection_rate(self, connection_id: str, **kwargs) -> float:
        """Get learning rate based on connection age"""
        metadata = self.get_connection_metadata(connection_id)
        birth_epoch = metadata.get('birth_epoch', self.current_epoch)
        age = self.current_epoch - birth_epoch
        
        # Apply age-based decay with saturation
        effective_age = min(age, self.max_age_effect)
        decay = self.age_decay_base ** (effective_age / self.max_age_effect)
        
        return self.base_lr * decay
    
    def get_learning_rate(self, connection_id: str = "", **kwargs) -> float:
        """Get learning rate for a specific connection"""
        if connection_id:
            return self.get_connection_rate(connection_id, **kwargs)
        return self.base_lr
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'age_decay_base': self.age_decay_base,
            'max_age_effect': self.max_age_effect,
            'connection_ages': {
                conn_id: self.current_epoch - metadata.get('birth_epoch', self.current_epoch)
                for conn_id, metadata in list(self.connection_registry.items())[:5]
            }
        })
        return config


class ScaleDependentRates(BaseConnectionScheduler, SchedulerMixin):
    """
    Scale-Dependent Learning Rates - Different rates for different scales
    
    Coarse scale: Slow learning for major pathways
    Medium scale: Moderate for features  
    Fine scale: Fast learning for details
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 coarse_scale_lr: float = 0.001,
                 medium_scale_lr: float = 0.01,
                 fine_scale_lr: float = 0.1,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.learning_rates = {
            'coarse_scale': coarse_scale_lr,   # Slow learning for major pathways
            'medium_scale': medium_scale_lr,   # Moderate for features
            'fine_scale': fine_scale_lr        # Fast learning for details
        }
        self.connection_scales = {}
    
    def register_connection_scale(self, connection_id: str, scale: str):
        """Register the scale of a connection"""
        assert scale in self.learning_rates, f"Invalid scale: {scale}"
        self.connection_scales[connection_id] = scale
    
    def get_scale_rate(self, scale: str) -> float:
        """Get learning rate for specific scale"""
        return self.learning_rates.get(scale, self.learning_rates['medium_scale'])
    
    def determine_connection_scale(self, 
                                 layer_idx: int, 
                                 n_layers: int,
                                 connection_strength: float) -> str:
        """Determine scale of a connection based on layer position and strength"""
        # Early layers with strong connections = coarse scale
        if layer_idx < n_layers // 3 and connection_strength > 0.5:
            return 'coarse_scale'
        # Late layers or weak connections = fine scale
        elif layer_idx > 2 * n_layers // 3 or connection_strength < 0.1:
            return 'fine_scale'
        # Everything else = medium scale
        else:
            return 'medium_scale'
    
    def get_connection_rate(self, connection_id: str, 
                          layer_idx: Optional[int] = None, 
                          n_layers: Optional[int] = None,
                          connection_strength: Optional[float] = None, 
                          **kwargs) -> float:
        """Get learning rate for a specific connection"""
        # Check if scale is already registered
        if connection_id in self.connection_scales:
            scale = self.connection_scales[connection_id]
            return self.get_scale_rate(scale)
        
        # Determine scale dynamically if parameters provided
        if all(x is not None for x in [layer_idx, n_layers, connection_strength]):
            scale = self.determine_connection_scale(layer_idx, n_layers, connection_strength)
            self.connection_scales[connection_id] = scale  # Cache for future use
            return self.get_scale_rate(scale)
        
        # Default to medium scale
        return self.learning_rates['medium_scale']
    
    def get_learning_rate(self, connection_id: str = "", **kwargs) -> float:
        """Get learning rate for a specific connection"""
        if connection_id:
            return self.get_connection_rate(connection_id, **kwargs)
        return self.base_lr
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'learning_rates': self.learning_rates,
            'registered_scales': len(self.connection_scales),
            'scale_distribution': {
                scale: sum(1 for s in self.connection_scales.values() if s == scale)
                for scale in self.learning_rates.keys()
            }
        })
        return config


class ConnectionStrengthScheduler(BaseConnectionScheduler, SchedulerMixin):
    """
    Connection Strength-Based Learning Rates
    
    Adjust learning rates based on connection weight magnitudes.
    Strong connections get lower rates, weak connections get higher rates.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 strong_threshold: float = 0.5,
                 weak_threshold: float = 0.1,
                 strong_multiplier: float = 0.5,
                 weak_multiplier: float = 2.0,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.strong_multiplier = strong_multiplier
        self.weak_multiplier = weak_multiplier
        self.connection_strengths = {}
    
    def update_connection_strength(self, connection_id: str, strength: float):
        """Update the strength of a connection"""
        self.connection_strengths[connection_id] = strength
    
    def get_connection_rate(self, connection_id: str, 
                          connection_strength: Optional[float] = None, **kwargs) -> float:
        """Get learning rate based on connection strength"""
        if connection_strength is None:
            connection_strength = self.connection_strengths.get(connection_id, 0.25)  # Default moderate
        
        if connection_strength > self.strong_threshold:
            # Strong connections: lower learning rate for stability
            return self.base_lr * self.strong_multiplier
        elif connection_strength < self.weak_threshold:
            # Weak connections: higher learning rate to strengthen
            return self.base_lr * self.weak_multiplier
        else:
            # Moderate connections: base learning rate
            return self.base_lr
    
    def get_learning_rate(self, connection_id: str = "", **kwargs) -> float:
        """Get learning rate for a specific connection"""
        if connection_id:
            return self.get_connection_rate(connection_id, **kwargs)
        return self.base_lr
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'strong_threshold': self.strong_threshold,
            'weak_threshold': self.weak_threshold,
            'strong_multiplier': self.strong_multiplier,
            'weak_multiplier': self.weak_multiplier,
            'tracked_strengths': len(self.connection_strengths),
            'strength_distribution': {
                'strong': sum(1 for s in self.connection_strengths.values() if s > self.strong_threshold),
                'moderate': sum(1 for s in self.connection_strengths.values() 
                              if self.weak_threshold <= s <= self.strong_threshold),
                'weak': sum(1 for s in self.connection_strengths.values() if s < self.weak_threshold)
            }
        })
        return config


class GradientBasedScheduler(BaseConnectionScheduler, SchedulerMixin):
    """
    Gradient-Based Learning Rate Adaptation
    
    Adjust learning rates based on gradient magnitudes and patterns.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 high_grad_threshold: float = 1.0,
                 low_grad_threshold: float = 0.01,
                 high_grad_multiplier: float = 0.5,
                 low_grad_multiplier: float = 2.0,
                 momentum: float = 0.9,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.high_grad_threshold = high_grad_threshold
        self.low_grad_threshold = low_grad_threshold
        self.high_grad_multiplier = high_grad_multiplier
        self.low_grad_multiplier = low_grad_multiplier
        self.momentum = momentum
        self.gradient_history = {}
    
    def update_gradient_history(self, connection_id: str, grad_magnitude: float):
        """Update gradient history for a connection"""
        if connection_id not in self.gradient_history:
            self.gradient_history[connection_id] = grad_magnitude
        else:
            # Exponential moving average
            self.gradient_history[connection_id] = (
                self.momentum * self.gradient_history[connection_id] + 
                (1 - self.momentum) * grad_magnitude
            )
    
    def get_connection_rate(self, connection_id: str, 
                          grad_magnitude: Optional[float] = None, **kwargs) -> float:
        """Get learning rate based on gradient magnitude"""
        if grad_magnitude is not None:
            self.update_gradient_history(connection_id, grad_magnitude)
        
        avg_grad = self.gradient_history.get(connection_id, 0.1)  # Default moderate
        
        if avg_grad > self.high_grad_threshold:
            # High gradients: reduce learning rate to prevent instability
            return self.base_lr * self.high_grad_multiplier
        elif avg_grad < self.low_grad_threshold:
            # Low gradients: increase learning rate to accelerate learning
            return self.base_lr * self.low_grad_multiplier
        else:
            # Moderate gradients: base learning rate
            return self.base_lr
    
    def get_learning_rate(self, connection_id: str = "", **kwargs) -> float:
        """Get learning rate for a specific connection"""
        if connection_id:
            return self.get_connection_rate(connection_id, **kwargs)
        return self.base_lr
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'high_grad_threshold': self.high_grad_threshold,
            'low_grad_threshold': self.low_grad_threshold,
            'high_grad_multiplier': self.high_grad_multiplier,
            'low_grad_multiplier': self.low_grad_multiplier,
            'momentum': self.momentum,
            'tracked_gradients': len(self.gradient_history),
            'gradient_distribution': {
                'high': sum(1 for g in self.gradient_history.values() if g > self.high_grad_threshold),
                'moderate': sum(1 for g in self.gradient_history.values() 
                              if self.low_grad_threshold <= g <= self.high_grad_threshold),
                'low': sum(1 for g in self.gradient_history.values() if g < self.low_grad_threshold)
            }
        })
        return config


class ExtremaProximityScheduler(BaseConnectionScheduler, SchedulerMixin):
    """
    Extrema Proximity-Based Learning Rates
    
    Connections near extrema get different learning rates.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 extrema_boost: float = 2.0,
                 proximity_threshold: float = 0.1,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.extrema_boost = extrema_boost
        self.proximity_threshold = proximity_threshold
        self.extrema_connections = set()
        self.connection_proximities = {}
    
    def update_extrema_connections(self, extrema_connection_ids: set):
        """Update which connections are near extrema"""
        self.extrema_connections = extrema_connection_ids
    
    def register_connection_proximity(self, connection_id: str, proximity: float):
        """Register proximity to extrema for a connection"""
        self.connection_proximities[connection_id] = proximity
    
    def get_connection_rate(self, connection_id: str, **kwargs) -> float:
        """Get learning rate based on extrema proximity"""
        # Check direct extrema connection
        if connection_id in self.extrema_connections:
            return self.base_lr * self.extrema_boost
        
        # Check proximity-based adjustment
        proximity = self.connection_proximities.get(connection_id, 1.0)  # Default far
        if proximity < self.proximity_threshold:
            # Close to extrema: boost learning
            boost_factor = 1.0 + (self.extrema_boost - 1.0) * (1.0 - proximity / self.proximity_threshold)
            return self.base_lr * boost_factor
        
        return self.base_lr
    
    def get_learning_rate(self, connection_id: str = "", **kwargs) -> float:
        """Get learning rate for a specific connection"""
        if connection_id:
            return self.get_connection_rate(connection_id, **kwargs)
        return self.base_lr
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'extrema_boost': self.extrema_boost,
            'proximity_threshold': self.proximity_threshold,
            'extrema_connections': len(self.extrema_connections),
            'tracked_proximities': len(self.connection_proximities),
            'proximity_distribution': {
                'close': sum(1 for p in self.connection_proximities.values() if p < self.proximity_threshold),
                'moderate': sum(1 for p in self.connection_proximities.values() 
                              if self.proximity_threshold <= p < 0.5),
                'far': sum(1 for p in self.connection_proximities.values() if p >= 0.5)
            }
        })
        return config
