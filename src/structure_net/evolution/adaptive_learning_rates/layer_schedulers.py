"""
Layer-wise adaptive learning rate schedulers.

This module contains schedulers that adapt learning rates based on layer properties
such as depth, age, and architectural position within the network.

DEPRECATED: This module is deprecated. Please use the new component-based
schedulers in structure_net.components.schedulers instead:
- LayerAgeAwareLR -> LayerAgeAwareScheduler
- Other schedulers will be migrated in future updates
"""

import warnings

warnings.warn(
    "The adaptive_learning_rates.layer_schedulers module is deprecated. "
    "Please use structure_net.components.schedulers instead.",
    DeprecationWarning,
    stacklevel=2
)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math

from .base import BaseLayerScheduler, SchedulerMixin


class LayerAgeAwareLR(BaseLayerScheduler, SchedulerMixin):
    """
    Layer-wise Rates + Connection Age Soft Clamping
    
    Combines layer-specific learning rates with age-based decay
    for sophisticated connection-level adaptation.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 decay_constant: float = 50.0,
                 early_layer_rate: float = 0.1,
                 late_layer_boost: float = 0.9,
                 **kwargs):
        super().__init__(base_lr, total_layers, **kwargs)
        self.decay_constant = decay_constant
        self.early_layer_rate = early_layer_rate
        self.late_layer_boost = late_layer_boost
        self.connection_ages = {}
    
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
        """Get learning rate for specific connection combining layer and age factors"""
        # Layer-specific base rate
        layer_rate = self.get_layer_rate(layer_idx)
        
        # Age-based decay (soft clamping)
        age = self.connection_ages.get(connection_id, 0)
        age_factor = np.exp(-age / self.decay_constant)
        
        # Combine both factors
        return self.base_lr * layer_rate * age_factor
    
    def get_learning_rate(self, layer_idx: int = 0, connection_id: Optional[str] = None, **kwargs) -> float:
        """Get learning rate for layer or connection"""
        if connection_id is not None:
            return self.get_connection_lr(connection_id, layer_idx)
        else:
            return self.base_lr * self.get_layer_rate(layer_idx)
    
    def update_connection_age(self, connection_id: str):
        """Increment age of a connection"""
        self.connection_ages[connection_id] = self.connection_ages.get(connection_id, 0) + 1
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'decay_constant': self.decay_constant,
            'early_layer_rate': self.early_layer_rate,
            'late_layer_boost': self.late_layer_boost,
            'tracked_connections': len(self.connection_ages)
        })
        return config


class CascadingDecayScheduler(BaseLayerScheduler, SchedulerMixin):
    """
    Cascading/Exponential Decay Learning Rates
    
    Each layer gets exponentially smaller LR based on depth.
    Purpose: Preserve learned features in early layers while allowing later layers to adapt.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 decay_base: float = 0.1,
                 **kwargs):
        super().__init__(base_lr, total_layers, **kwargs)
        self.decay_base = decay_base
    
    def get_layer_rate(self, layer_idx: int) -> float:
        """Get exponentially decaying LR based on layer depth"""
        if self.total_layers <= 1:
            return 1.0
        
        decay = self.decay_base ** (layer_idx / self.total_layers)
        return decay
    
    def get_learning_rate(self, layer_idx: int = 0, **kwargs) -> float:
        """Get learning rate for specific layer"""
        return self.base_lr * self.get_layer_rate(layer_idx)
    
    def create_param_groups(self, layers: List[nn.Module]) -> List[Dict]:
        """Create parameter groups with cascading decay"""
        param_groups = []
        for i, layer in enumerate(layers):
            decay = self.get_layer_rate(i)
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': self.base_lr * decay,
                'name': f'scaffold_layer_{i}',
                'layer_idx': i
            })
        return param_groups
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'decay_base': self.decay_base,
            'layer_rates': [self.get_layer_rate(i) for i in range(min(self.total_layers, 10))]
        })
        return config


class LayerwiseAdaptiveRates(BaseLayerScheduler, SchedulerMixin):
    """
    Layer-wise Adaptive Growth Rates - Different layers grow at different rates
    
    Early layers: Faster growth (feature extraction)
    Middle layers: Medium growth (feature combination)
    Late layers: Slower growth (sparse bridges to output)
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 early_rate: float = 0.02,
                 middle_rate: float = 0.01, 
                 late_rate: float = 0.005,
                 **kwargs):
        super().__init__(base_lr, total_layers, **kwargs)
        self.early_rate = early_rate
        self.middle_rate = middle_rate
        self.late_rate = late_rate
    
    def get_layer_rate(self, layer_idx: int) -> float:
        """Different layers grow at different rates"""
        if self.total_layers <= 1:
            return self.middle_rate / self.base_lr
        
        if layer_idx < self.total_layers // 3:  # Early layers
            return self.early_rate / self.base_lr  # Grow faster (feature extraction)
        elif layer_idx > 2 * self.total_layers // 3:  # Late layers
            return self.late_rate / self.base_lr  # Grow slower (sparse bridges)
        else:  # Middle layers
            return self.middle_rate / self.base_lr  # Medium growth
    
    def get_learning_rate(self, layer_idx: int = 0, **kwargs) -> float:
        """Get learning rate for specific layer"""
        return self.base_lr * self.get_layer_rate(layer_idx)
    
    def get_layer_rates(self) -> List[float]:
        """Get learning rates for all layers"""
        return [self.get_learning_rate(i) for i in range(self.total_layers)]
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'early_rate': self.early_rate,
            'middle_rate': self.middle_rate,
            'late_rate': self.late_rate,
            'layer_rates': self.get_layer_rates()[:10]  # Show first 10 layers
        })
        return config


class ProgressiveFreezingScheduler(BaseLayerScheduler, SchedulerMixin):
    """
    Progressive Freezing Schedule
    
    Gradually freeze layers as training progresses.
    Purpose: Focus learning on later layers over time.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 warmup_lr: float = 0.001,
                 refinement_early_lr: float = 0.00001,
                 refinement_late_lr: float = 0.0001,
                 final_lr: float = 0.0001,
                 warmup_epochs: int = 10,
                 refinement_epochs: int = 30,
                 **kwargs):
        super().__init__(base_lr, total_layers, **kwargs)
        self.warmup_lr = warmup_lr
        self.refinement_early_lr = refinement_early_lr
        self.refinement_late_lr = refinement_late_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.refinement_epochs = refinement_epochs
    
    def get_current_phase(self) -> str:
        """Get current training phase"""
        if self.current_epoch < self.warmup_epochs:
            return 'warmup'
        elif self.current_epoch < self.refinement_epochs:
            return 'refinement'
        else:
            return 'final'
    
    def get_layer_rate(self, layer_idx: int) -> float:
        """Get learning rate based on training phase and layer position"""
        phase = self.get_current_phase()
        
        if phase == 'warmup':
            return self.warmup_lr / self.base_lr  # All layers active
        elif phase == 'refinement':
            if layer_idx < 2:
                return self.refinement_early_lr / self.base_lr  # Nearly frozen early layers
            else:
                return self.refinement_late_lr / self.base_lr   # Slow for later layers
        elif phase == 'final':
            if layer_idx < self.total_layers - 1:
                return 0  # Freeze all but last
            return self.final_lr / self.base_lr  # Only tune final layer
        
        return 1.0
    
    def get_learning_rate(self, layer_idx: int = 0, **kwargs) -> float:
        """Get learning rate for specific layer"""
        return self.base_lr * self.get_layer_rate(layer_idx)
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'current_phase': self.get_current_phase(),
            'warmup_epochs': self.warmup_epochs,
            'refinement_epochs': self.refinement_epochs,
            'phase_lrs': {
                'warmup': self.warmup_lr,
                'refinement_early': self.refinement_early_lr,
                'refinement_late': self.refinement_late_lr,
                'final': self.final_lr
            }
        })
        return config


class AgeBasedScheduler(BaseLayerScheduler, SchedulerMixin):
    """
    Age-Based Learning Rates
    
    Older layers learn slower, newer layers learn faster.
    Purpose: Stabilize old knowledge while allowing new adaptation.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 age_decay_base: float = 0.1,
                 **kwargs):
        super().__init__(base_lr, total_layers, **kwargs)
        self.age_decay_base = age_decay_base
        self.layer_birth_epochs = {}
    
    def register_layer(self, layer_id: str, birth_epoch: Optional[int] = None):
        """Register when a layer was created"""
        if birth_epoch is None:
            birth_epoch = self.current_epoch
        self.layer_birth_epochs[layer_id] = birth_epoch
    
    def get_layer_rate(self, layer_idx: int) -> float:
        """Get learning rate based on layer age (using layer_idx as layer_id)"""
        layer_id = str(layer_idx)
        return self.get_age_based_lr(layer_id) / self.base_lr
    
    def get_age_based_lr(self, layer_id: str) -> float:
        """Get learning rate based on layer age"""
        birth_epoch = self.layer_birth_epochs.get(layer_id, self.current_epoch)
        age = self.current_epoch - birth_epoch
        decay = self.age_decay_base ** age
        return self.base_lr * decay
    
    def get_learning_rate(self, layer_idx: int = 0, layer_id: Optional[str] = None, **kwargs) -> float:
        """Get learning rate for specific layer"""
        if layer_id is not None:
            return self.get_age_based_lr(layer_id)
        else:
            return self.base_lr * self.get_layer_rate(layer_idx)
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'age_decay_base': self.age_decay_base,
            'registered_layers': len(self.layer_birth_epochs),
            'layer_ages': {
                layer_id: self.current_epoch - birth_epoch 
                for layer_id, birth_epoch in list(self.layer_birth_epochs.items())[:5]
            }
        })
        return config


class ComponentSpecificScheduler(BaseLayerScheduler, SchedulerMixin):
    """
    Component-Specific Learning Rates
    
    Different components (scaffold, patches, necks, new layers) get different rates.
    Purpose: Different components need different learning speeds.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 scaffold_lr: float = 0.0001,
                 patch_lr: float = 0.0005,
                 neck_lr: float = 0.001,
                 new_layer_lr: float = 0.001,
                 **kwargs):
        super().__init__(base_lr, total_layers, **kwargs)
        self.component_rates = {
            'scaffold': scaffold_lr,
            'patches': patch_lr,
            'necks': neck_lr,
            'new_layers': new_layer_lr
        }
        self.layer_components = {}  # Maps layer_idx to component type
    
    def register_layer_component(self, layer_idx: int, component_type: str):
        """Register the component type for a layer"""
        self.layer_components[layer_idx] = component_type
    
    def get_layer_rate(self, layer_idx: int) -> float:
        """Get learning rate based on layer's component type"""
        component_type = self.layer_components.get(layer_idx, 'new_layers')
        component_lr = self.component_rates.get(component_type, self.base_lr)
        return component_lr / self.base_lr
    
    def get_learning_rate(self, layer_idx: int = 0, component_type: Optional[str] = None, **kwargs) -> float:
        """Get learning rate for specific layer or component"""
        if component_type is not None:
            return self.component_rates.get(component_type, self.base_lr)
        else:
            return self.base_lr * self.get_layer_rate(layer_idx)
    
    def create_component_groups(self, components: Dict[str, List[nn.Module]]) -> List[Dict]:
        """Create parameter groups for different components"""
        param_groups = []
        
        for component_type, layers in components.items():
            if component_type in self.component_rates:
                lr = self.component_rates[component_type]
                for i, layer in enumerate(layers):
                    param_groups.append({
                        'params': list(layer.parameters()),
                        'lr': lr,
                        'name': f'{component_type}_{i}',
                        'component_type': component_type
                    })
        
        return param_groups
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'component_rates': self.component_rates,
            'registered_components': len(self.layer_components),
            'component_distribution': {
                comp_type: sum(1 for ct in self.layer_components.values() if ct == comp_type)
                for comp_type in self.component_rates.keys()
            }
        })
        return config


class PretrainedNewLayerScheduler(BaseLayerScheduler, SchedulerMixin):
    """
    Pretrained + New Layer Strategy
    
    Pretrained layers get very low LR, new layers get high LR.
    Purpose: Preserve pretrained features while training new components.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 pretrained_lr: float = 1e-5,
                 new_lr: float = 1e-3,
                 adapter_lr: float = 5e-4,
                 **kwargs):
        super().__init__(base_lr, total_layers, **kwargs)
        self.pretrained_lr = pretrained_lr
        self.new_lr = new_lr
        self.adapter_lr = adapter_lr
        self.layer_types = {}  # Maps layer_idx to type ('pretrained', 'new', 'adapter')
    
    def register_layer_type(self, layer_idx: int, layer_type: str):
        """Register the type for a layer"""
        assert layer_type in ['pretrained', 'new', 'adapter'], f"Invalid layer type: {layer_type}"
        self.layer_types[layer_idx] = layer_type
    
    def get_layer_rate(self, layer_idx: int) -> float:
        """Get learning rate based on layer type"""
        layer_type = self.layer_types.get(layer_idx, 'new')
        
        if layer_type == 'pretrained':
            return self.pretrained_lr / self.base_lr
        elif layer_type == 'adapter':
            return self.adapter_lr / self.base_lr
        else:  # new
            return self.new_lr / self.base_lr
    
    def get_learning_rate(self, layer_idx: int = 0, layer_type: Optional[str] = None, **kwargs) -> float:
        """Get learning rate for specific layer or type"""
        if layer_type is not None:
            if layer_type == 'pretrained':
                return self.pretrained_lr
            elif layer_type == 'adapter':
                return self.adapter_lr
            else:
                return self.new_lr
        else:
            return self.base_lr * self.get_layer_rate(layer_idx)
    
    def create_pretrained_groups(self, 
                               pretrained_layers: List[nn.Module],
                               new_layers: List[nn.Module],
                               adapter_layers: Optional[List[nn.Module]] = None) -> List[Dict]:
        """Create parameter groups for pretrained vs new components"""
        param_groups = []
        
        # Pretrained layers - nearly frozen
        for i, layer in enumerate(pretrained_layers):
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': self.pretrained_lr,
                'name': f'pretrained_{i}',
                'layer_type': 'pretrained'
            })
        
        # New layers - full learning rate
        for i, layer in enumerate(new_layers):
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': self.new_lr,
                'name': f'new_{i}',
                'layer_type': 'new'
            })
        
        # Adapter layers - medium speed
        if adapter_layers:
            for i, layer in enumerate(adapter_layers):
                param_groups.append({
                    'params': list(layer.parameters()),
                    'lr': self.adapter_lr,
                    'name': f'adapter_{i}',
                    'layer_type': 'adapter'
                })
        
        return param_groups
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'pretrained_lr': self.pretrained_lr,
            'new_lr': self.new_lr,
            'adapter_lr': self.adapter_lr,
            'layer_type_distribution': {
                layer_type: sum(1 for lt in self.layer_types.values() if lt == layer_type)
                for layer_type in ['pretrained', 'new', 'adapter']
            }
        })
        return config


class LARSScheduler(BaseLayerScheduler, SchedulerMixin):
    """
    Layer-wise Adaptive Rate Scaling (LARS)
    
    Adapt learning rate based on gradient/weight ratio per layer.
    Purpose: Automatic scaling based on layer dynamics.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 eps: float = 1e-8,
                 trust_coefficient: float = 0.001,
                 **kwargs):
        super().__init__(base_lr, total_layers, **kwargs)
        self.eps = eps
        self.trust_coefficient = trust_coefficient
        self.layer_lars_rates = {}
    
    def get_layer_rate(self, layer_idx: int) -> float:
        """Get LARS rate for layer (requires layer object for computation)"""
        # This is a placeholder - actual LARS computation requires the layer object
        return self.layer_lars_rates.get(layer_idx, 1.0)
    
    def get_lars_lr(self, layer: nn.Module, layer_idx: Optional[int] = None) -> float:
        """Get LARS-adjusted learning rate for a layer"""
        if not hasattr(layer, 'weight') or layer.weight.grad is None:
            return self.base_lr
        
        weight_norm = layer.weight.norm().item()
        grad_norm = layer.weight.grad.norm().item()
        
        if grad_norm > 0:
            layer_lr = self.trust_coefficient * (weight_norm / (grad_norm + self.eps))
            layer_lr = min(layer_lr, self.base_lr)  # Cap at base_lr
            
            # Cache the rate if layer_idx provided
            if layer_idx is not None:
                self.layer_lars_rates[layer_idx] = layer_lr / self.base_lr
            
            return layer_lr
        return self.base_lr
    
    def get_learning_rate(self, layer: Optional[nn.Module] = None, layer_idx: int = 0, **kwargs) -> float:
        """Get learning rate for specific layer"""
        if layer is not None:
            return self.get_lars_lr(layer, layer_idx)
        else:
            return self.base_lr * self.get_layer_rate(layer_idx)
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'eps': self.eps,
            'trust_coefficient': self.trust_coefficient,
            'cached_lars_rates': dict(list(self.layer_lars_rates.items())[:5])
        })
        return config


class SedimentaryLearningScheduler(BaseLayerScheduler, SchedulerMixin):
    """
    The "Sedimentary" Learning Strategy
    
    Natural stratification of learning speeds by component age.
    Purpose: Geological-like learning where older layers barely change.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 total_layers: int = 1,
                 geological_lr: float = 0.00001,
                 sediment_lr: float = 0.0001,
                 active_lr: float = 0.001,
                 patch_lr: float = 0.0005,
                 geological_age: int = 100,
                 sediment_age: int = 20,
                 **kwargs):
        super().__init__(base_lr, total_layers, **kwargs)
        self.learning_rates = {
            'geological_layers': geological_lr,   # Oldest, barely change
            'sediment_layers': sediment_lr,       # Middle age, slow drift
            'active_layers': active_lr,           # Newest, rapid change
            'patches': patch_lr                   # Targeted fixes
        }
        self.geological_age = geological_age
        self.sediment_age = sediment_age
        self.layer_ages = {}
        self.layer_patches = set()
    
    def register_layer_age(self, layer_idx: int, age: int):
        """Register the age of a layer"""
        self.layer_ages[layer_idx] = age
    
    def register_patch_layer(self, layer_idx: int):
        """Register a layer as a patch"""
        self.layer_patches.add(layer_idx)
    
    def classify_layer_age(self, layer_age: int) -> str:
        """Classify layer into geological age category"""
        if layer_age > self.geological_age:
            return 'geological_layers'
        elif layer_age > self.sediment_age:
            return 'sediment_layers'
        else:
            return 'active_layers'
    
    def get_layer_rate(self, layer_idx: int) -> float:
        """Get learning rate based on sedimentary classification"""
        # Check if it's a patch first
        if layer_idx in self.layer_patches:
            return self.learning_rates['patches'] / self.base_lr
        
        # Get layer age
        layer_age = self.layer_ages.get(layer_idx, 0)
        age_category = self.classify_layer_age(layer_age)
        
        return self.learning_rates[age_category] / self.base_lr
    
    def get_sedimentary_lr(self, layer_age: int, is_patch: bool = False) -> float:
        """Get learning rate based on sedimentary classification"""
        if is_patch:
            return self.learning_rates['patches']
        
        age_category = self.classify_layer_age(layer_age)
        return self.learning_rates[age_category]
    
    def get_learning_rate(self, layer_idx: int = 0, layer_age: Optional[int] = None, 
                         is_patch: bool = False, **kwargs) -> float:
        """Get learning rate for specific layer"""
        if layer_age is not None:
            return self.get_sedimentary_lr(layer_age, is_patch)
        else:
            return self.base_lr * self.get_layer_rate(layer_idx)
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'learning_rates': self.learning_rates,
            'geological_age': self.geological_age,
            'sediment_age': self.sediment_age,
            'registered_ages': len(self.layer_ages),
            'patch_layers': len(self.layer_patches),
            'age_distribution': {
                category: sum(1 for age in self.layer_ages.values() 
                            if self.classify_layer_age(age) == category)
                for category in ['geological_layers', 'sediment_layers', 'active_layers']
            }
        })
        return config
