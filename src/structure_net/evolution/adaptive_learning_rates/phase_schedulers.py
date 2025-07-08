"""
Phase-based learning rate schedulers.

This module contains schedulers that adapt learning rates based on training phases,
such as growth phases, extrema patterns, and exponential backoff strategies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math

from .base import BasePhaseScheduler, SchedulerMixin
from ..extrema_analyzer import detect_network_extrema


class ExtremaPhaseScheduler(BasePhaseScheduler, SchedulerMixin):
    """
    Extrema-Driven Phase Detection + Phase-Based Learning Rates
    
    Uses extrema patterns to automatically detect growth phases and
    adjust learning rates accordingly.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 explosive_threshold: float = 0.1,
                 steady_threshold: float = 0.01,
                 explosive_multiplier: float = 1.0,
                 steady_multiplier: float = 0.1,
                 refinement_multiplier: float = 0.01,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.explosive_threshold = explosive_threshold
        self.steady_threshold = steady_threshold
        self.explosive_multiplier = explosive_multiplier
        self.steady_multiplier = steady_multiplier
        self.refinement_multiplier = refinement_multiplier
        
        # Track extrema history
        self.extrema_history = []
        self.total_neurons = 0
        self.current_phase = "steady_growth"
    
    def detect_phase(self, network, data_loader, device='cuda') -> str:
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
            phase = "explosive_growth"
        elif extrema_rate > self.steady_threshold:  # Moderate extrema
            phase = "steady_growth"
        else:  # Few extrema
            phase = "refinement"
        
        self.current_phase = phase
        return phase
    
    def get_phase_multiplier(self, phase: str) -> float:
        """Get learning rate multiplier for a specific phase"""
        multipliers = {
            "explosive_growth": self.explosive_multiplier,   # Full LR for structure
            "steady_growth": self.steady_multiplier,         # Reduced for stability
            "refinement": self.refinement_multiplier         # Tiny for fine-tuning
        }
        return multipliers.get(phase, self.steady_multiplier)
    
    def get_learning_rate(self, network=None, data_loader=None, device='cuda', **kwargs) -> float:
        """Get learning rate based on detected growth phase"""
        if network is not None and data_loader is not None:
            phase = self.detect_phase(network, data_loader, device)
        else:
            phase = self.current_phase
        
        multiplier = self.get_phase_multiplier(phase)
        return self.base_lr * multiplier
    
    def get_phase_trend(self) -> str:
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
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'explosive_threshold': self.explosive_threshold,
            'steady_threshold': self.steady_threshold,
            'current_phase': self.current_phase,
            'extrema_history_length': len(self.extrema_history),
            'phase_trend': self.get_phase_trend()
        })
        return config


class GrowthPhaseScheduler(BasePhaseScheduler, SchedulerMixin):
    """
    Growth Phase-Based Adjustment - Learning rates based on training phase
    
    Early: Aggressive learning for structure discovery
    Middle: Moderate for feature development  
    Late: Gentle for fine-tuning
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 early_lr: float = 0.1,
                 middle_lr: float = 0.01,
                 late_lr: float = 0.001,
                 early_phase_end: int = 20,
                 middle_phase_end: int = 50,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.early_lr = early_lr
        self.middle_lr = middle_lr
        self.late_lr = late_lr
        self.early_phase_end = early_phase_end
        self.middle_phase_end = middle_phase_end
    
    def detect_phase(self, epoch: Optional[int] = None, **kwargs) -> str:
        """Detect current training phase based on epoch"""
        if epoch is None:
            epoch = self.current_epoch
            
        if epoch < self.early_phase_end:
            return 'early'
        elif epoch < self.middle_phase_end:
            return 'middle'
        else:
            return 'late'
    
    def get_phase_multiplier(self, phase: str) -> float:
        """Get learning rate multiplier for a specific phase"""
        multipliers = {
            'early': self.early_lr / self.base_lr,
            'middle': self.middle_lr / self.base_lr,
            'late': self.late_lr / self.base_lr
        }
        return multipliers.get(phase, 1.0)
    
    def get_learning_rate(self, epoch: Optional[int] = None, **kwargs) -> float:
        """Get learning rate based on training phase"""
        if epoch is None:
            epoch = self.current_epoch
            
        phase = self.detect_phase(epoch)
        
        if phase == 'early':
            # Early: Aggressive learning for structure discovery
            return self.early_lr
        elif phase == 'middle':
            # Middle: Moderate for feature development
            return self.middle_lr
        else:
            # Late: Gentle for fine-tuning
            return self.late_lr
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'early_phase_end': self.early_phase_end,
            'middle_phase_end': self.middle_phase_end,
            'current_phase': self.detect_phase(),
            'phase_lrs': {
                'early': self.early_lr,
                'middle': self.middle_lr,
                'late': self.late_lr
            }
        })
        return config


class ExponentialBackoffScheduler(BasePhaseScheduler, SchedulerMixin):
    """
    Exponential Backoff for Loss - Aggressive early → Gentle late
    
    Creates natural curriculum from finding major highways to refining local roads.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 initial_lr: float = 1.0,
                 decay_rate: float = 0.95,
                 min_lr: float = 1e-6,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.min_lr = min_lr
    
    def detect_phase(self, epoch: Optional[int] = None, **kwargs) -> str:
        """Detect phase based on current learning rate level"""
        current_lr = self.get_learning_rate(epoch)
        
        if current_lr > self.base_lr * 0.5:
            return "aggressive"
        elif current_lr > self.base_lr * 0.1:
            return "moderate"
        else:
            return "gentle"
    
    def get_phase_multiplier(self, phase: str) -> float:
        """Get multiplier for phase (not used directly in exponential backoff)"""
        return 1.0
    
    def get_learning_rate(self, epoch: Optional[int] = None, **kwargs) -> float:
        """Get exponentially decaying learning rate"""
        if epoch is None:
            epoch = self.current_epoch
        
        # Exponential decay: lr = initial_lr * (decay_rate)^epoch
        lr = self.initial_lr * (self.decay_rate ** epoch)
        return self.clamp_lr(lr, self.min_lr, self.initial_lr)
    
    def get_loss_weight(self, epoch: Optional[int] = None) -> float:
        """Get loss weight for current epoch (alias for get_learning_rate)"""
        return self.get_learning_rate(epoch)
    
    def step(self):
        """Increment epoch counter"""
        self.current_epoch += 1
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'initial_lr': self.initial_lr,
            'decay_rate': self.decay_rate,
            'min_lr': self.min_lr,
            'current_phase': self.detect_phase(),
            'current_lr': self.get_learning_rate()
        })
        return config


class WarmupScheduler(BasePhaseScheduler, SchedulerMixin):
    """
    Warm-Up for New Components
    
    Gradually increase learning rate for new components.
    Purpose: Prevent new components from disrupting existing features.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 warmup_epochs: int = 5,
                 warmup_start_lr: float = 1e-6,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
    
    def detect_phase(self, epoch: Optional[int] = None, **kwargs) -> str:
        """Detect if we're in warmup phase"""
        if epoch is None:
            epoch = self.current_epoch
        return "warmup" if epoch < self.warmup_epochs else "normal"
    
    def get_phase_multiplier(self, phase: str) -> float:
        """Get multiplier for phase"""
        if phase == "warmup":
            return self.warmup_schedule(self.current_epoch, self.warmup_epochs)
        return 1.0
    
    def get_learning_rate(self, epoch: Optional[int] = None, **kwargs) -> float:
        """Get learning rate with warm-up applied"""
        if epoch is None:
            epoch = self.current_epoch
            
        if epoch < self.warmup_epochs:
            # Linear warm-up from warmup_start_lr to base_lr
            progress = epoch / self.warmup_epochs
            return self.linear_interpolation(self.warmup_start_lr, self.base_lr, progress)
        else:
            return self.base_lr
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'current_phase': self.detect_phase(),
            'warmup_progress': min(1.0, self.current_epoch / self.warmup_epochs)
        })
        return config


class CosineAnnealingScheduler(BasePhaseScheduler, SchedulerMixin):
    """
    Cosine Annealing Learning Rate Schedule
    
    Smoothly decreases learning rate following a cosine curve.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 min_lr: float = 1e-6,
                 total_epochs: int = 100,
                 restart_epochs: Optional[int] = None,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.restart_epochs = restart_epochs
    
    def detect_phase(self, epoch: Optional[int] = None, **kwargs) -> str:
        """Detect phase based on cosine position"""
        if epoch is None:
            epoch = self.current_epoch
            
        if self.restart_epochs:
            cycle_epoch = epoch % self.restart_epochs
            progress = cycle_epoch / self.restart_epochs
        else:
            progress = epoch / self.total_epochs
        
        if progress < 0.25:
            return "high"
        elif progress < 0.75:
            return "declining"
        else:
            return "low"
    
    def get_phase_multiplier(self, phase: str) -> float:
        """Get multiplier for phase (computed dynamically)"""
        return self.get_learning_rate() / self.base_lr
    
    def get_learning_rate(self, epoch: Optional[int] = None, **kwargs) -> float:
        """Get cosine annealed learning rate"""
        if epoch is None:
            epoch = self.current_epoch
        
        if self.restart_epochs:
            # Cosine annealing with restarts
            cycle_epoch = epoch % self.restart_epochs
            total_steps = self.restart_epochs
        else:
            cycle_epoch = epoch
            total_steps = self.total_epochs
        
        return self.cosine_annealing(self.base_lr, self.min_lr, cycle_epoch, total_steps)
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'min_lr': self.min_lr,
            'total_epochs': self.total_epochs,
            'restart_epochs': self.restart_epochs,
            'current_phase': self.detect_phase(),
            'current_lr': self.get_learning_rate()
        })
        return config


class AdaptivePhaseScheduler(BasePhaseScheduler, SchedulerMixin):
    """
    Adaptive Phase Scheduler that combines multiple phase detection strategies.
    
    Uses multiple signals to determine the current training phase and adjust
    learning rates accordingly.
    """
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 loss_patience: int = 5,
                 loss_threshold: float = 0.01,
                 phase_multipliers: Optional[Dict[str, float]] = None,
                 **kwargs):
        super().__init__(base_lr, **kwargs)
        self.loss_patience = loss_patience
        self.loss_threshold = loss_threshold
        self.loss_history = []
        self.current_phase = "exploration"
        
        # Default phase multipliers
        self.phase_multipliers = phase_multipliers or {
            "exploration": 1.0,
            "exploitation": 0.5,
            "refinement": 0.1,
            "plateau": 0.05
        }
    
    def detect_phase(self, loss: Optional[float] = None, **kwargs) -> str:
        """Detect phase based on loss trends and other signals"""
        if loss is not None:
            self.loss_history.append(loss)
            if len(self.loss_history) > 20:  # Keep last 20 losses
                self.loss_history.pop(0)
        
        if len(self.loss_history) < self.loss_patience:
            return "exploration"
        
        # Analyze loss trend
        recent_losses = self.loss_history[-self.loss_patience:]
        loss_trend = self._analyze_loss_trend(recent_losses)
        
        # Determine phase based on trend
        if loss_trend == "decreasing_fast":
            phase = "exploration"
        elif loss_trend == "decreasing_slow":
            phase = "exploitation"
        elif loss_trend == "stable":
            phase = "refinement"
        else:  # increasing or plateau
            phase = "plateau"
        
        self.current_phase = phase
        return phase
    
    def _analyze_loss_trend(self, losses: List[float]) -> str:
        """Analyze trend in loss values"""
        if len(losses) < 2:
            return "stable"
        
        # Calculate relative change
        first_half = np.mean(losses[:len(losses)//2])
        second_half = np.mean(losses[len(losses)//2:])
        
        relative_change = (first_half - second_half) / first_half
        
        if relative_change > self.loss_threshold:
            return "decreasing_fast"
        elif relative_change > self.loss_threshold / 2:
            return "decreasing_slow"
        elif abs(relative_change) < self.loss_threshold / 4:
            return "stable"
        else:
            return "increasing"
    
    def get_phase_multiplier(self, phase: str) -> float:
        """Get learning rate multiplier for a specific phase"""
        return self.phase_multipliers.get(phase, 1.0)
    
    def get_learning_rate(self, loss: Optional[float] = None, **kwargs) -> float:
        """Get adaptive learning rate based on current phase"""
        phase = self.detect_phase(loss, **kwargs)
        multiplier = self.get_phase_multiplier(phase)
        return self.base_lr * multiplier
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        config = super().get_config()
        config.update({
            'loss_patience': self.loss_patience,
            'loss_threshold': self.loss_threshold,
            'current_phase': self.current_phase,
            'phase_multipliers': self.phase_multipliers,
            'loss_history_length': len(self.loss_history)
        })
        return config
