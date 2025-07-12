#!/usr/bin/env python3
"""
Extrema-Driven Phase Scheduler Component

Uses extrema patterns to automatically detect growth phases and adjust learning rates
accordingly. Adapts training dynamics based on network health indicators.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ...core.base_components import BaseScheduler
from ...core.interfaces import (
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)


class ExtremaPhaseScheduler(BaseScheduler):
    """
    Extrema-Driven Phase Detection + Phase-Based Learning Rates
    
    Uses extrema patterns to automatically detect growth phases and
    adjust learning rates accordingly.
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="ExtremaPhaseScheduler",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"network", "data_loader"},
            provided_outputs={"learning_rate", "phase", "extrema_info"},
            optional_inputs={"device"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=True,
                parallel_safe=False,
                estimated_runtime_seconds=5.0
            )
        )
    
    def __init__(self, 
                 base_lr: float = 0.001,
                 explosive_threshold: float = 0.1,
                 steady_threshold: float = 0.01,
                 explosive_multiplier: float = 1.0,
                 steady_multiplier: float = 0.1,
                 refinement_multiplier: float = 0.01,
                 **kwargs):
        """
        Initialize Extrema Phase Scheduler.
        
        Args:
            base_lr: Base learning rate
            explosive_threshold: Extrema rate threshold for explosive growth phase
            steady_threshold: Extrema rate threshold for steady growth phase
            explosive_multiplier: LR multiplier for explosive growth phase
            steady_multiplier: LR multiplier for steady growth phase
            refinement_multiplier: LR multiplier for refinement phase
        """
        super().__init__("ExtremaPhaseScheduler")
        
        self.base_lr = base_lr
        self.explosive_threshold = explosive_threshold
        self.steady_threshold = steady_threshold
        self.explosive_multiplier = explosive_multiplier
        self.steady_multiplier = steady_multiplier
        self.refinement_multiplier = refinement_multiplier
        
        # Track extrema history
        self.extrema_history = []
        self.total_neurons = 0
        self.current_phase = "steady_growth"
        self.current_epoch = 0
        self._config = kwargs
    
    def detect_phase(self, network: nn.Module, data_loader: Any, device: str = 'cuda') -> str:
        """Use extrema patterns to detect phases automatically."""
        # Import here to avoid circular dependency
        from ...evolution.extrema_analyzer import detect_network_extrema
        
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
        """Get learning rate multiplier for a specific phase."""
        multipliers = {
            "explosive_growth": self.explosive_multiplier,   # Full LR for structure
            "steady_growth": self.steady_multiplier,         # Reduced for stability
            "refinement": self.refinement_multiplier         # Tiny for fine-tuning
        }
        return multipliers.get(phase, self.steady_multiplier)
    
    def compute_learning_rate(self, 
                            global_step: int,
                            epoch: int,
                            optimizer_state: Optional[Dict[str, Any]] = None,
                            model_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Compute learning rate based on detected growth phase.
        
        Returns:
            Dictionary with single learning rate for all parameters
        """
        self.current_epoch = epoch
        
        # If network and data_loader are provided in model_state, detect phase
        if model_state and 'network' in model_state and 'data_loader' in model_state:
            network = model_state['network']
            data_loader = model_state['data_loader']
            device = model_state.get('device', 'cuda')
            
            phase = self.detect_phase(network, data_loader, device)
        else:
            phase = self.current_phase
        
        multiplier = self.get_phase_multiplier(phase)
        lr = self.base_lr * multiplier
        
        return {"default": lr}
    
    def get_phase_trend(self) -> str:
        """Get trend in extrema rate over recent history."""
        if len(self.extrema_history) < 3:
            return "insufficient_data"
        
        # Compare recent average to older average
        recent_avg = np.mean(self.extrema_history[-3:])
        older_avg = np.mean(self.extrema_history[:-3])
        
        if recent_avg > older_avg * 1.2:
            return "increasing"
        elif recent_avg < older_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def get_scheduler_state(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'current_phase': self.current_phase,
            'extrema_history': self.extrema_history,
            'total_neurons': self.total_neurons,
            'explosive_threshold': self.explosive_threshold,
            'steady_threshold': self.steady_threshold,
            'explosive_multiplier': self.explosive_multiplier,
            'steady_multiplier': self.steady_multiplier,
            'refinement_multiplier': self.refinement_multiplier,
            'base_lr': self.base_lr,
            'config': self._config
        }
    
    def load_scheduler_state(self, state: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state.get('current_epoch', 0)
        self.current_phase = state.get('current_phase', 'steady_growth')
        self.extrema_history = state.get('extrema_history', [])
        self.total_neurons = state.get('total_neurons', 0)
        self.explosive_threshold = state.get('explosive_threshold', self.explosive_threshold)
        self.steady_threshold = state.get('steady_threshold', self.steady_threshold)
        self.explosive_multiplier = state.get('explosive_multiplier', self.explosive_multiplier)
        self.steady_multiplier = state.get('steady_multiplier', self.steady_multiplier)
        self.refinement_multiplier = state.get('refinement_multiplier', self.refinement_multiplier)
        self.base_lr = state.get('base_lr', self.base_lr)
        self._config = state.get('config', {})
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get detailed phase information."""
        return {
            'current_phase': self.current_phase,
            'phase_trend': self.get_phase_trend(),
            'extrema_history': self.extrema_history,
            'current_extrema_rate': self.extrema_history[-1] if self.extrema_history else 0.0,
            'total_neurons': self.total_neurons,
            'phase_multiplier': self.get_phase_multiplier(self.current_phase)
        }
    
    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch