"""
Base classes and interfaces for adaptive learning rate strategies.

This module defines the core abstractions that all learning rate schedulers
should implement, providing a consistent interface and common functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum


class LearningRateStrategy(Enum):
    """Enumeration of available learning rate strategies."""
    BASIC = "basic"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"
    ULTIMATE = "ultimate"


class BaseLearningRateScheduler(ABC):
    """
    Abstract base class for all learning rate schedulers.
    
    Provides common interface and functionality that all schedulers should implement.
    """
    
    def __init__(self, base_lr: float = 0.001, **kwargs):
        """
        Initialize the scheduler.
        
        Args:
            base_lr: Base learning rate
            **kwargs: Additional scheduler-specific parameters
        """
        self.base_lr = base_lr
        self.current_epoch = 0
        self._config = kwargs
    
    @abstractmethod
    def get_learning_rate(self, *args, **kwargs) -> Union[float, Dict[str, float]]:
        """
        Get the learning rate(s) for the current state.
        
        Returns:
            Learning rate(s) - can be a single float or dict of rates
        """
        pass
    
    def update_epoch(self, epoch: int):
        """Update the current epoch."""
        self.current_epoch = epoch
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration."""
        return {
            'base_lr': self.base_lr,
            'current_epoch': self.current_epoch,
            **self._config
        }
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_epoch = 0


class BasePhaseScheduler(BaseLearningRateScheduler):
    """Base class for phase-based learning rate schedulers."""
    
    @abstractmethod
    def detect_phase(self, *args, **kwargs) -> str:
        """Detect the current training phase."""
        pass
    
    @abstractmethod
    def get_phase_multiplier(self, phase: str) -> float:
        """Get learning rate multiplier for a specific phase."""
        pass


class BaseLayerScheduler(BaseLearningRateScheduler):
    """Base class for layer-wise learning rate schedulers."""
    
    def __init__(self, base_lr: float = 0.001, total_layers: int = 1, **kwargs):
        super().__init__(base_lr, **kwargs)
        self.total_layers = total_layers
    
    @abstractmethod
    def get_layer_rate(self, layer_idx: int) -> float:
        """Get learning rate for a specific layer."""
        pass
    
    def set_total_layers(self, total_layers: int):
        """Update the total number of layers."""
        self.total_layers = total_layers


class BaseConnectionScheduler(BaseLearningRateScheduler):
    """Base class for connection-level learning rate schedulers."""
    
    def __init__(self, base_lr: float = 0.001, **kwargs):
        super().__init__(base_lr, **kwargs)
        self.connection_registry = {}
    
    @abstractmethod
    def get_connection_rate(self, connection_id: str, **kwargs) -> float:
        """Get learning rate for a specific connection."""
        pass
    
    def register_connection(self, connection_id: str, **metadata):
        """Register a new connection with metadata."""
        self.connection_registry[connection_id] = {
            'birth_epoch': self.current_epoch,
            **metadata
        }
    
    def get_connection_metadata(self, connection_id: str) -> Dict[str, Any]:
        """Get metadata for a connection."""
        return self.connection_registry.get(connection_id, {})


class SchedulerMixin:
    """Mixin class providing common scheduler utilities."""
    
    @staticmethod
    def exponential_decay(initial_value: float, decay_rate: float, step: int) -> float:
        """Apply exponential decay."""
        return initial_value * (decay_rate ** step)
    
    @staticmethod
    def linear_interpolation(start: float, end: float, progress: float) -> float:
        """Linear interpolation between start and end values."""
        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        return start + (end - start) * progress
    
    @staticmethod
    def cosine_annealing(initial_lr: float, min_lr: float, current_step: int, total_steps: int) -> float:
        """Cosine annealing schedule."""
        import math
        progress = current_step / total_steps
        return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
    
    @staticmethod
    def warmup_schedule(current_step: int, warmup_steps: int) -> float:
        """Linear warmup schedule."""
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, current_step / warmup_steps)
    
    @staticmethod
    def clamp_lr(lr: float, min_lr: float = 1e-8, max_lr: float = 1.0) -> float:
        """Clamp learning rate to reasonable bounds."""
        return max(min_lr, min(max_lr, lr))


class ParameterGroup:
    """
    Represents a parameter group with associated learning rate and metadata.
    """
    
    def __init__(self, 
                 params: List[torch.nn.Parameter],
                 lr: float,
                 name: str = "",
                 metadata: Optional[Dict[str, Any]] = None):
        self.params = params
        self.lr = lr
        self.name = name
        self.metadata = metadata or {}
    
    def to_optimizer_group(self) -> Dict[str, Any]:
        """Convert to optimizer parameter group format."""
        group = {
            'params': self.params,
            'lr': self.lr
        }
        if self.name:
            group['name'] = self.name
        group.update(self.metadata)
        return group
    
    def update_lr(self, new_lr: float):
        """Update learning rate."""
        self.lr = new_lr
    
    def __repr__(self) -> str:
        return f"ParameterGroup(name='{self.name}', lr={self.lr:.6f}, params={len(self.params)})"


class LearningRateHistory:
    """
    Tracks learning rate history for analysis and debugging.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history = []
    
    def record(self, epoch: int, lr_data: Dict[str, Any]):
        """Record learning rate data for an epoch."""
        entry = {
            'epoch': epoch,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
            **lr_data
        }
        
        self.history.append(entry)
        
        # Maintain max history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_recent_history(self, n_epochs: int = 10) -> List[Dict[str, Any]]:
        """Get recent learning rate history."""
        return self.history[-n_epochs:]
    
    def get_lr_trend(self, parameter_group: str = 'default', n_epochs: int = 5) -> str:
        """Analyze learning rate trend for a parameter group."""
        if len(self.history) < 2:
            return "insufficient_data"
        
        recent = self.get_recent_history(n_epochs)
        if len(recent) < 2:
            return "insufficient_data"
        
        # Extract learning rates for the specified group
        lrs = []
        for entry in recent:
            if parameter_group in entry:
                lrs.append(entry[parameter_group])
            elif 'lr' in entry:
                lrs.append(entry['lr'])
        
        if len(lrs) < 2:
            return "no_data"
        
        # Analyze trend
        first_lr = lrs[0]
        last_lr = lrs[-1]
        
        if last_lr > first_lr * 1.1:
            return "increasing"
        elif last_lr < first_lr * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def clear(self):
        """Clear history."""
        self.history.clear()


class AdaptiveOptimizerWrapper:
    """
    Wrapper around PyTorch optimizers to provide adaptive learning rate functionality.
    """
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 schedulers: List[BaseLearningRateScheduler]):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.history = LearningRateHistory()
    
    def step(self, closure=None):
        """Perform optimization step."""
        return self.optimizer.step(closure)
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def update_learning_rates(self, epoch: int, **context):
        """Update learning rates using all schedulers."""
        lr_data = {'epoch': epoch}
        
        for scheduler in self.schedulers:
            scheduler.update_epoch(epoch)
            
            # Apply scheduler to parameter groups
            for i, param_group in enumerate(self.optimizer.param_groups):
                group_name = param_group.get('name', f'group_{i}')
                
                # Get learning rate from scheduler
                if hasattr(scheduler, 'get_layer_rate') and 'layer_idx' in param_group:
                    new_lr = scheduler.get_layer_rate(param_group['layer_idx'])
                elif hasattr(scheduler, 'get_connection_rate'):
                    # For connection-level schedulers, use base rate
                    new_lr = scheduler.get_learning_rate(**context)
                else:
                    new_lr = scheduler.get_learning_rate(**context)
                
                # Update parameter group
                if isinstance(new_lr, dict):
                    if group_name in new_lr:
                        param_group['lr'] = new_lr[group_name]
                        lr_data[group_name] = new_lr[group_name]
                else:
                    param_group['lr'] = new_lr
                    lr_data[group_name] = new_lr
        
        # Record in history
        self.history.record(epoch, lr_data)
    
    def get_current_lrs(self) -> Dict[str, float]:
        """Get current learning rates for all parameter groups."""
        lrs = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            lrs[group_name] = param_group['lr']
        return lrs
    
    def state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)
