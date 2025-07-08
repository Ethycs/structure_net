#!/usr/bin/env python3
"""
Adaptive Learning Rate Strategies for Structure Net

This module implements sophisticated differential learning rate strategies
that adapt based on network growth phase, layer position, connection age,
and scale-dependent factors.

Key strategies:
1. Exponential Backoff for Loss
2. Layer-wise Adaptive Growth Rates  
3. Soft Clamping (Gradual Freezing)
4. Scale-Dependent Learning Rates
5. Growth Phase-Based Adjustment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math

from ..core.layers import StandardSparseLayer
from .extrema_analyzer import detect_network_extrema


class ExtremaPhaseScheduler:
    """
    Extrema-Driven Phase Detection + Phase-Based Learning Rates
    
    Uses extrema patterns to automatically detect growth phases and
    adjust learning rates accordingly.
    """
    
    def __init__(self, 
                 explosive_threshold: float = 0.1,
                 steady_threshold: float = 0.01,
                 explosive_multiplier: float = 1.0,
                 steady_multiplier: float = 0.1,
                 refinement_multiplier: float = 0.01):
        self.explosive_threshold = explosive_threshold
        self.steady_threshold = steady_threshold
        self.explosive_multiplier = explosive_multiplier
        self.steady_multiplier = steady_multiplier
        self.refinement_multiplier = refinement_multiplier
        
        # Track extrema history
        self.extrema_history = []
        self.total_neurons = 0
    
    def detect_growth_phase(self, network, data_loader, device='cuda'):
        """Use extrema patterns to detect phases automatically"""
        # Analyze current extrema
        extrema_patterns = detect_network_extrema(network, data_loader, device, max_batches=3)
        
        # Count total extrema and neurons
        total_extrema = 0
        total_neurons = 0
        
        for layer_data in extrema_patterns.values():
            if isinstance(layer_data, dict) and 'extrema_count' in layer_data:
                total_extrema += layer_data['extrema_count']
                total_neurons += layer_data.get('layer_size', 0)
        
        self.total_neurons = total_neurons
        extrema_rate = total_extrema / max(total_neurons, 1)
        
        # Store in history
        self.extrema_history.append(extrema_rate)
        if len(self.extrema_history) > 10:  # Keep last 10 measurements
            self.extrema_history.pop(0)
        
        # Detect phase based on extrema rate
        if extrema_rate > self.explosive_threshold:  # Many extrema
            return "explosive_growth"
        elif extrema_rate > self.steady_threshold:  # Moderate extrema
            return "steady_growth"
        else:  # Few extrema
            return "refinement"
    
    def get_learning_rate(self, base_lr, network, data_loader, device='cuda'):
        """Get learning rate based on detected growth phase"""
        phase = self.detect_growth_phase(network, data_loader, device)
        
        multipliers = {
            "explosive_growth": self.explosive_multiplier,   # Full LR for structure
            "steady_growth": self.steady_multiplier,         # Reduced for stability
            "refinement": self.refinement_multiplier         # Tiny for fine-tuning
        }
        
        return base_lr * multipliers[phase], phase
    
    def get_phase_trend(self):
        """Get trend in extrema rate over recent history"""
        if len(self.extrema_history) < 3:
            return "stable"
        
        recent = self.extrema_history[-3:]
        if recent[-1] > recent[0] * 1.2:
            return "increasing"
        elif recent[-1] < recent[0] * 0.8:
            return "decreasing"
        else:
            return "stable"


class LayerAgeAwareLR:
    """
    Layer-wise Rates + Connection Age Soft Clamping
    
    Combines layer-specific learning rates with age-based decay
    for sophisticated connection-level adaptation.
    """
    
    def __init__(self, 
                 decay_constant: float = 50.0,
                 early_layer_rate: float = 0.1,
                 late_layer_boost: float = 0.9):
        self.decay_constant = decay_constant
        self.early_layer_rate = early_layer_rate
        self.late_layer_boost = late_layer_boost
        self.connection_ages = {}
        self.total_layers = 0
    
    def get_connection_lr(self, connection_id: str, layer_idx: int, base_lr: float) -> float:
        """Get learning rate for specific connection combining layer and age factors"""
        # Layer-specific base rate
        layer_rate = self.get_layer_rate(layer_idx)
        
        # Age-based decay (soft clamping)
        age = self.connection_ages.get(connection_id, 0)
        age_factor = np.exp(-age / self.decay_constant)
        
        # Combine both factors
        return base_lr * layer_rate * age_factor
    
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
    
    def update_connection_age(self, connection_id: str):
        """Increment age of a connection"""
        self.connection_ages[connection_id] = self.connection_ages.get(connection_id, 0) + 1
    
    def set_total_layers(self, total_layers: int):
        """Set total number of layers for depth ratio calculation"""
        self.total_layers = total_layers


class MultiScaleLearning:
    """
    Scale-Dependent + Temporal Snapshot Integration
    
    Different learning rates based on when connections were created
    and their scale in the network evolution.
    """
    
    def __init__(self):
        self.scale_snapshots = {
            'coarse': {'epoch_threshold': 20, 'lr_multiplier': 0.1},
            'medium': {'epoch_threshold': 50, 'lr_multiplier': 0.5},
            'fine': {'epoch_threshold': 100, 'lr_multiplier': 1.0}
        }
        self.connection_birth_epochs = {}
        self.current_epoch = 0
    
    def register_connection(self, connection_id: str, birth_epoch: int):
        """Register when a connection was created"""
        self.connection_birth_epochs[connection_id] = birth_epoch
    
    def get_lr_for_connection(self, connection_id: str, base_lr: float) -> float:
        """Get learning rate for connection based on its birth epoch and scale"""
        birth_epoch = self.connection_birth_epochs.get(connection_id, self.current_epoch)
        
        # Determine which scale this connection belongs to
        if birth_epoch < self.scale_snapshots['coarse']['epoch_threshold']:
            scale = 'coarse'
        elif birth_epoch < self.scale_snapshots['medium']['epoch_threshold']:
            scale = 'medium'
        else:
            scale = 'fine'
        
        # Scale-specific learning rate
        scale_lr = base_lr * self.scale_snapshots[scale]['lr_multiplier']
        
        # Add exponential decay within scale
        epochs_in_scale = self.current_epoch - birth_epoch
        decay_factor = np.exp(-0.01 * epochs_in_scale)
        
        return scale_lr * decay_factor
    
    def update_epoch(self, epoch: int):
        """Update current epoch"""
        self.current_epoch = epoch
    
    def get_scale_for_connection(self, connection_id: str) -> str:
        """Get the scale category for a connection"""
        birth_epoch = self.connection_birth_epochs.get(connection_id, self.current_epoch)
        
        if birth_epoch < self.scale_snapshots['coarse']['epoch_threshold']:
            return 'coarse'
        elif birth_epoch < self.scale_snapshots['medium']['epoch_threshold']:
            return 'medium'
        else:
            return 'fine'


class UnifiedAdaptiveLearning:
    """
    The Ultimate Combination: Unified Adaptive System
    
    Combines all techniques into one coherent system:
    1. Phase detection from extrema
    2. Layer-wise adaptation
    3. Scale-based rates (connection birth time)
    4. Age-based soft decay
    5. Extrema proximity bonus
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 min_lr: float = 1e-6,
                 max_lr: float = 0.1,
                 extrema_boost: float = 2.0):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.extrema_boost = extrema_boost
        
        # Initialize component systems
        self.extrema_phase = ExtremaPhaseScheduler()
        self.layer_age = LayerAgeAwareLR()
        self.multi_scale = MultiScaleLearning()
        
        # Track extrema proximity
        self.extrema_connections = set()
        self.current_epoch = 0
    
    def get_learning_rate(self, 
                         connection_id: str, 
                         layer_idx: int,
                         network,
                         data_loader,
                         device='cuda') -> float:
        """Get unified learning rate combining all factors"""
        
        # 1. Base phase detection from extrema
        phase_lr, phase = self.extrema_phase.get_learning_rate(
            self.base_lr, network, data_loader, device
        )
        phase_multiplier = phase_lr / self.base_lr
        
        # 2. Layer-wise adaptation
        layer_multiplier = self.layer_age.get_layer_rate(layer_idx)
        
        # 3. Scale-based rate (when was connection born)
        scale_lr = self.multi_scale.get_lr_for_connection(connection_id, self.base_lr)
        scale_multiplier = scale_lr / self.base_lr
        
        # 4. Age-based soft decay
        age = self.layer_age.connection_ages.get(connection_id, 0)
        age_multiplier = np.exp(-age / self.layer_age.decay_constant)
        
        # 5. Extrema proximity bonus
        extrema_multiplier = self.get_extrema_proximity_multiplier(connection_id)
        
        # Combine all factors
        final_lr = (self.base_lr * 
                   phase_multiplier * 
                   layer_multiplier * 
                   scale_multiplier * 
                   age_multiplier * 
                   extrema_multiplier)
        
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
        self.current_epoch = epoch
        self.multi_scale.update_epoch(epoch)
    
    def update_connection_age(self, connection_id: str):
        """Update age for a connection"""
        self.layer_age.update_connection_age(connection_id)
    
    def register_new_connection(self, connection_id: str, layer_idx: int):
        """Register a new connection"""
        self.multi_scale.register_connection(connection_id, self.current_epoch)
        # Age starts at 0 automatically
    
    def set_network_structure(self, total_layers: int):
        """Set network structure information"""
        self.layer_age.set_total_layers(total_layers)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the unified system state"""
        return {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'extrema_phase': {
                'recent_extrema_rates': self.extrema_phase.extrema_history[-5:],
                'trend': self.extrema_phase.get_phase_trend()
            },
            'layer_age': {
                'total_layers': self.layer_age.total_layers,
                'tracked_connections': len(self.layer_age.connection_ages)
            },
            'multi_scale': {
                'tracked_connections': len(self.multi_scale.connection_birth_epochs),
                'scale_thresholds': self.multi_scale.scale_snapshots
            },
            'extrema_connections': len(self.extrema_connections)
        }


class ExponentialBackoffScheduler:
    """
    Exponential Backoff for Loss - Aggressive early → Gentle late
    
    Creates natural curriculum from finding major highways to refining local roads.
    """
    
    def __init__(self, initial_lr: float = 1.0, decay_rate: float = 0.95):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.epoch = 0
    
    def get_loss_weight(self, epoch: Optional[int] = None) -> float:
        """Aggressive early → Gentle late"""
        if epoch is not None:
            self.epoch = epoch
        
        # Exponential decay: lr = initial_lr * (decay_rate)^epoch
        weight = self.initial_lr * (self.decay_rate ** self.epoch)
        return weight
    
    def step(self):
        """Increment epoch counter"""
        self.epoch += 1
    
    def get_current_weight(self) -> float:
        """Get current loss weight"""
        return self.get_loss_weight()


class LayerwiseAdaptiveRates:
    """
    Layer-wise Adaptive Growth Rates - Different layers grow at different rates
    
    Early layers: Faster growth (feature extraction)
    Middle layers: Medium growth (feature combination)
    Late layers: Slower growth (sparse bridges to output)
    """
    
    def __init__(self, 
                 early_rate: float = 0.02,
                 middle_rate: float = 0.01, 
                 late_rate: float = 0.005):
        self.early_rate = early_rate
        self.middle_rate = middle_rate
        self.late_rate = late_rate
    
    def adaptive_growth_rate(self, layer_idx: int, n_layers: int) -> float:
        """Different layers grow at different rates"""
        if layer_idx < n_layers // 3:  # Early layers
            return self.early_rate  # Grow faster (feature extraction)
        elif layer_idx > 2 * n_layers // 3:  # Late layers
            return self.late_rate  # Grow slower (sparse bridges)
        else:  # Middle layers
            return self.middle_rate  # Medium growth
    
    def get_layer_rates(self, n_layers: int) -> List[float]:
        """Get learning rates for all layers"""
        return [self.adaptive_growth_rate(i, n_layers) for i in range(n_layers)]


class SoftClampingScheduler:
    """
    Soft Clamping (Gradual Freezing) - Gradual freezing instead of hard clamp
    
    Reduces learning rate with connection age, doesn't freeze completely.
    Allows old connections to still adapt but more slowly.
    """
    
    def __init__(self, max_age: int = 100, min_clamp_factor: float = 0.1):
        self.max_age = max_age
        self.min_clamp_factor = min_clamp_factor
        self.connection_ages = defaultdict(int)
    
    def soft_clamping(self, connection_id: str, age: Optional[int] = None) -> float:
        """Gradual freezing instead of hard clamp"""
        if age is None:
            age = self.connection_ages[connection_id]
        
        # Reduce learning rate with age, don't freeze completely
        clamp_factor = 1.0 - (age / self.max_age)
        clamp_factor = max(clamp_factor, self.min_clamp_factor)
        
        return clamp_factor
    
    def update_connection_age(self, connection_id: str):
        """Increment age of a connection"""
        self.connection_ages[connection_id] += 1
    
    def get_connection_factor(self, connection_id: str) -> float:
        """Get current clamping factor for a connection"""
        return self.soft_clamping(connection_id)
    
    def apply_to_gradient(self, gradient: torch.Tensor, connection_id: str) -> torch.Tensor:
        """Apply soft clamping to gradient"""
        clamp_factor = self.get_connection_factor(connection_id)
        return gradient * clamp_factor


class ScaleDependentRates:
    """
    Scale-Dependent Learning Rates - Different rates for different scales
    
    Coarse scale: Slow learning for major pathways
    Medium scale: Moderate for features  
    Fine scale: Fast learning for details
    """
    
    def __init__(self, 
                 coarse_scale_lr: float = 0.001,
                 medium_scale_lr: float = 0.01,
                 fine_scale_lr: float = 0.1):
        self.learning_rates = {
            'coarse_scale': coarse_scale_lr,   # Slow learning for major pathways
            'medium_scale': medium_scale_lr,   # Moderate for features
            'fine_scale': fine_scale_lr        # Fast learning for details
        }
    
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
    
    def get_connection_rate(self, 
                          layer_idx: int, 
                          n_layers: int,
                          connection_strength: float) -> float:
        """Get learning rate for a specific connection"""
        scale = self.determine_connection_scale(layer_idx, n_layers, connection_strength)
        return self.get_scale_rate(scale)


class GrowthPhaseScheduler:
    """
    Growth Phase-Based Adjustment - Learning rates based on training phase
    
    Early: Aggressive learning for structure discovery
    Middle: Moderate for feature development  
    Late: Gentle for fine-tuning
    """
    
    def __init__(self, 
                 early_lr: float = 0.1,
                 middle_lr: float = 0.01,
                 late_lr: float = 0.001,
                 early_phase_end: int = 20,
                 middle_phase_end: int = 50):
        self.early_lr = early_lr
        self.middle_lr = middle_lr
        self.late_lr = late_lr
        self.early_phase_end = early_phase_end
        self.middle_phase_end = middle_phase_end
    
    def phase_based_lr(self, epoch: int) -> float:
        """Get learning rate based on training phase"""
        if epoch < self.early_phase_end:
            # Early: Aggressive learning for structure discovery
            return self.early_lr
        elif epoch < self.middle_phase_end:
            # Middle: Moderate for feature development
            return self.middle_lr
        else:
            # Late: Gentle for fine-tuning
            return self.late_lr
    
    def get_current_phase(self, epoch: int) -> str:
        """Get current training phase name"""
        if epoch < self.early_phase_end:
            return 'early'
        elif epoch < self.middle_phase_end:
            return 'middle'
        else:
            return 'late'


class AdaptiveLearningRateManager:
    """
    Unified manager for all adaptive learning rate strategies.
    
    Combines all strategies into a cohesive system that can be easily
    integrated into structure_net training loops.
    
    Now includes advanced combination systems:
    - Extrema-driven phase detection
    - Layer-age aware learning
    - Multi-scale temporal integration
    - Unified adaptive system
    """
    
    def __init__(self, 
                 network: nn.Module,
                 base_lr: float = 0.001,
                 enable_exponential_backoff: bool = True,
                 enable_layerwise_rates: bool = True,
                 enable_soft_clamping: bool = True,
                 enable_scale_dependent: bool = True,
                 enable_phase_based: bool = True,
                 enable_extrema_phase: bool = False,
                 enable_layer_age_aware: bool = False,
                 enable_multi_scale: bool = False,
                 enable_unified_system: bool = False):
        
        self.network = network
        self.base_lr = base_lr
        
        # Initialize basic strategies
        self.exponential_backoff = ExponentialBackoffScheduler() if enable_exponential_backoff else None
        self.layerwise_rates = LayerwiseAdaptiveRates() if enable_layerwise_rates else None
        self.soft_clamping = SoftClampingScheduler() if enable_soft_clamping else None
        self.scale_dependent = ScaleDependentRates() if enable_scale_dependent else None
        self.phase_scheduler = GrowthPhaseScheduler() if enable_phase_based else None
        
        # Initialize advanced combination systems
        self.extrema_phase = ExtremaPhaseScheduler() if enable_extrema_phase else None
        self.layer_age_aware = LayerAgeAwareLR() if enable_layer_age_aware else None
        self.multi_scale = MultiScaleLearning() if enable_multi_scale else None
        self.unified_system = UnifiedAdaptiveLearning(base_lr) if enable_unified_system else None
        
        # Track network structure
        self.sparse_layers = [layer for layer in network if isinstance(layer, StandardSparseLayer)]
        self.n_layers = len(self.sparse_layers)
        
        # Set up layer structure for advanced systems
        if self.layer_age_aware:
            self.layer_age_aware.set_total_layers(self.n_layers)
        if self.unified_system:
            self.unified_system.set_network_structure(self.n_layers)
        
        # Current epoch
        self.current_epoch = 0
        
        print(f"🎯 Initialized AdaptiveLearningRateManager")
        print(f"   Base LR: {base_lr}")
        print(f"   Basic strategies enabled:")
        print(f"     - Exponential Backoff: {enable_exponential_backoff}")
        print(f"     - Layerwise Rates: {enable_layerwise_rates}")
        print(f"     - Soft Clamping: {enable_soft_clamping}")
        print(f"     - Scale Dependent: {enable_scale_dependent}")
        print(f"     - Phase Based: {enable_phase_based}")
        print(f"   Advanced combination systems:")
        print(f"     - Extrema Phase Detection: {enable_extrema_phase}")
        print(f"     - Layer-Age Aware: {enable_layer_age_aware}")
        print(f"     - Multi-Scale Learning: {enable_multi_scale}")
        print(f"     - Unified System: {enable_unified_system}")
    
    def create_adaptive_optimizer(self) -> optim.Optimizer:
        """Create optimizer with adaptive parameter groups"""
        param_groups = []
        
        # Create parameter groups for each layer with adaptive rates
        for layer_idx, layer in enumerate(self.sparse_layers):
            # Get base rate for this layer
            if self.layerwise_rates:
                layer_rate = self.layerwise_rates.adaptive_growth_rate(layer_idx, self.n_layers)
            else:
                layer_rate = self.base_lr
            
            # Adjust for current phase
            if self.phase_scheduler:
                phase_multiplier = self.phase_scheduler.phase_based_lr(self.current_epoch) / self.base_lr
                layer_rate *= phase_multiplier
            
            # Adjust for exponential backoff
            if self.exponential_backoff:
                backoff_multiplier = self.exponential_backoff.get_current_weight()
                layer_rate *= backoff_multiplier
            
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': layer_rate,
                'layer_idx': layer_idx,
                'layer_type': 'sparse'
            })
        
        # Add any remaining parameters (non-sparse layers)
        remaining_params = []
        for module in self.network.modules():
            if not isinstance(module, StandardSparseLayer) and len(list(module.parameters())) > 0:
                remaining_params.extend(list(module.parameters()))
        
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': self.base_lr,
                'layer_type': 'other'
            })
        
        return optim.Adam(param_groups)
    
    def update_learning_rates(self, optimizer: optim.Optimizer, epoch: int):
        """Update learning rates for current epoch"""
        self.current_epoch = epoch
        
        # Update exponential backoff
        if self.exponential_backoff:
            self.exponential_backoff.epoch = epoch
        
        # Update optimizer parameter groups
        for group in optimizer.param_groups:
            if 'layer_idx' in group:
                layer_idx = group['layer_idx']
                
                # Get base rate for this layer
                if self.layerwise_rates:
                    layer_rate = self.layerwise_rates.adaptive_growth_rate(layer_idx, self.n_layers)
                else:
                    layer_rate = self.base_lr
                
                # Adjust for current phase
                if self.phase_scheduler:
                    phase_multiplier = self.phase_scheduler.phase_based_lr(epoch) / self.base_lr
                    layer_rate *= phase_multiplier
                
                # Adjust for exponential backoff
                if self.exponential_backoff:
                    backoff_multiplier = self.exponential_backoff.get_current_weight()
                    layer_rate *= backoff_multiplier
                
                group['lr'] = layer_rate
    
    def apply_gradient_modifications(self, optimizer: optim.Optimizer):
        """Apply gradient modifications (soft clamping, scale-dependent adjustments)"""
        if not (self.soft_clamping or self.scale_dependent):
            return
        
        for group_idx, group in enumerate(optimizer.param_groups):
            if 'layer_idx' in group:
                layer_idx = group['layer_idx']
                layer = self.sparse_layers[layer_idx]
                
                # Apply soft clamping to gradients
                if self.soft_clamping and hasattr(layer, 'mask'):
                    if layer.linear.weight.grad is not None:
                        # Create connection IDs and apply soft clamping
                        for i in range(layer.linear.weight.shape[0]):
                            for j in range(layer.linear.weight.shape[1]):
                                if layer.mask[i, j] > 0:  # Active connection
                                    connection_id = f"layer_{layer_idx}_conn_{i}_{j}"
                                    clamp_factor = self.soft_clamping.get_connection_factor(connection_id)
                                    layer.linear.weight.grad[i, j] *= clamp_factor
                                    
                                    # Update connection age
                                    self.soft_clamping.update_connection_age(connection_id)
                
                # Apply scale-dependent adjustments
                if self.scale_dependent and hasattr(layer, 'mask'):
                    if layer.linear.weight.grad is not None:
                        for i in range(layer.linear.weight.shape[0]):
                            for j in range(layer.linear.weight.shape[1]):
                                if layer.mask[i, j] > 0:  # Active connection
                                    connection_strength = abs(layer.linear.weight.data[i, j].item())
                                    scale_rate = self.scale_dependent.get_connection_rate(
                                        layer_idx, self.n_layers, connection_strength
                                    )
                                    # Adjust gradient based on scale
                                    scale_multiplier = scale_rate / self.base_lr
                                    layer.linear.weight.grad[i, j] *= scale_multiplier
    
    def get_current_rates_summary(self) -> Dict[str, Any]:
        """Get summary of current learning rates and strategies"""
        summary = {
            'epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'strategies': {}
        }
        
        if self.exponential_backoff:
            summary['strategies']['exponential_backoff'] = {
                'current_weight': self.exponential_backoff.get_current_weight(),
                'decay_rate': self.exponential_backoff.decay_rate
            }
        
        if self.layerwise_rates:
            summary['strategies']['layerwise_rates'] = {
                'layer_rates': self.layerwise_rates.get_layer_rates(self.n_layers)
            }
        
        if self.phase_scheduler:
            summary['strategies']['phase_based'] = {
                'current_phase': self.phase_scheduler.get_current_phase(self.current_epoch),
                'current_lr': self.phase_scheduler.phase_based_lr(self.current_epoch)
            }
        
        if self.scale_dependent:
            summary['strategies']['scale_dependent'] = {
                'rates': self.scale_dependent.learning_rates
            }
        
        if self.soft_clamping:
            summary['strategies']['soft_clamping'] = {
                'max_age': self.soft_clamping.max_age,
                'active_connections': len(self.soft_clamping.connection_ages)
            }
        
        return summary
    
    def print_rates_summary(self):
        """Print current learning rates summary"""
        summary = self.get_current_rates_summary()
        
        print(f"\n📊 Learning Rates Summary (Epoch {summary['epoch']})")
        print("=" * 50)
        
        for strategy_name, strategy_data in summary['strategies'].items():
            print(f"🎯 {strategy_name.replace('_', ' ').title()}:")
            for key, value in strategy_data.items():
                if isinstance(value, list):
                    print(f"   {key}: {[f'{v:.4f}' for v in value[:5]]}{'...' if len(value) > 5 else ''}")
                elif isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")


def create_adaptive_training_loop(network: nn.Module, 
                                train_loader, 
                                val_loader,
                                epochs: int = 50,
                                **adaptive_kwargs) -> Tuple[nn.Module, List[Dict]]:
    """
    Create a complete training loop with adaptive learning rate strategies.
    
    Returns:
        Trained network and training history
    """
    
    # Initialize adaptive learning rate manager
    lr_manager = AdaptiveLearningRateManager(network, **adaptive_kwargs)
    
    # Create adaptive optimizer
    optimizer = lr_manager.create_adaptive_optimizer()
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = []
    
    device = next(network.parameters()).device
    
    print(f"\n🚀 Starting Adaptive Training Loop")
    print(f"   Epochs: {epochs}")
    print(f"   Device: {device}")
    
    for epoch in range(epochs):
        # Update learning rates for current epoch
        lr_manager.update_learning_rates(optimizer, epoch)
        
        # Training phase
        network.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Apply gradient modifications
            lr_manager.apply_gradient_modifications(optimizer)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += len(target)
        
        # Validation phase
        network.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = network(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += len(target)
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Record history
        epoch_data = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'learning_rates': lr_manager.get_current_rates_summary()
        }
        history.append(epoch_data)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2%}, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2%}")
            
            if epoch % 20 == 0:
                lr_manager.print_rates_summary()
    
    print(f"\n✅ Adaptive Training Complete")
    print(f"   Final Val Accuracy: {history[-1]['val_acc']:.2%}")
    
    return network, history
