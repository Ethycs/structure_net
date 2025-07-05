"""
Growth Scheduler for Multi-Scale Snapshots Experiment

This module implements the growth detection and scheduling logic based on:
- Gradient variance spike detection
- Credit system (10 credits per spike, 100 threshold)
- Growth phases (coarse, medium, fine)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging


class GrowthScheduler:
    """
    Manages when and how network growth occurs based on gradient signals.
    
    Implements the growth economy rules:
    - Gradient signals: 10 credits per spike
    - CIT signals: 80 credits per boundary (optional)
    - Growth threshold: 100 credits
    - Spend all credits after growth
    """
    
    def __init__(
        self,
        gradient_window: int = 10,
        variance_threshold: float = 0.5,
        gradient_credits: int = 10,
        cit_credits: int = 80,
        growth_threshold: int = 25,  # Further reduced to trigger growth more often
        stabilization_epochs: int = 5,  # Reduced cooldown period
        bootstrap_epochs: int = 10,    # Shorter bootstrap period
        bootstrap_credits_per_epoch: int = 8  # More bootstrap credits
    ):
        """
        Initialize growth scheduler.
        
        Args:
            gradient_window: Window size for gradient variance calculation
            variance_threshold: Threshold for detecting gradient variance spikes (50% change)
            gradient_credits: Credits awarded per gradient spike
            cit_credits: Credits awarded per CIT boundary detection
            growth_threshold: Credits needed to trigger growth
            stabilization_epochs: Epochs to wait after growth before allowing new growth
        """
        self.gradient_window = gradient_window
        self.variance_threshold = variance_threshold
        self.gradient_credits = gradient_credits
        self.cit_credits = cit_credits
        self.growth_threshold = growth_threshold
        self.stabilization_epochs = stabilization_epochs
        self.bootstrap_epochs = bootstrap_epochs
        self.bootstrap_credits_per_epoch = bootstrap_credits_per_epoch
        
        # State tracking
        self.gradient_history = deque(maxlen=gradient_window * 2)  # Keep extra history
        self.credits = 0
        self.growth_events = []
        self.last_growth_epoch = -1
        self.current_epoch = 0
        self.last_bootstrap_epoch = -1  # Track when bootstrap was last applied
        
        # Statistics
        self.variance_history = []
        self.credit_history = []
        self.spike_count = 0
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch
    
    def add_gradient_norm(self, gradient_norm: float) -> bool:
        """
        Add gradient norm and check for growth trigger.
        
        Args:
            gradient_norm: Current gradient norm
            
        Returns:
            True if growth should occur, False otherwise
        """
        self.gradient_history.append(gradient_norm)
        
        # Bootstrap mechanism: Add credits once per epoch early on if no growth has occurred
        bootstrap_credits_added = False
        if (self.current_epoch < self.bootstrap_epochs and 
            len(self.growth_events) == 0 and 
            self.current_epoch > 0 and
            self.last_bootstrap_epoch != self.current_epoch):  # Only once per epoch
            self.credits += self.bootstrap_credits_per_epoch
            self.last_bootstrap_epoch = self.current_epoch
            bootstrap_credits_added = True
            self.logger.info(f"Bootstrap credits added at epoch {self.current_epoch}. Credits: {self.credits}")
        
        # Need sufficient history for variance calculation
        if len(self.gradient_history) < self.gradient_window * 2:
            # Record credit history even if we can't detect spikes yet
            self.credit_history.append(self.credits)
            # Check for bootstrap growth trigger
            if self.credits >= self.growth_threshold:
                self._trigger_growth()
                return True
            return False
        
        # Check if we're in stabilization period
        if self.current_epoch - self.last_growth_epoch < self.stabilization_epochs:
            self.credit_history.append(self.credits)
            return False
        
        # Calculate gradient variance spike
        spike_detected = self._detect_variance_spike()
        
        if spike_detected:
            self.credits += self.gradient_credits
            self.spike_count += 1
            self.logger.info(f"Gradient spike detected at epoch {self.current_epoch}. Credits: {self.credits}")
        
        # Record credit history
        self.credit_history.append(self.credits)
        
        # Check for growth trigger
        if self.credits >= self.growth_threshold:
            self._trigger_growth()
            return True
        
        return False
    
    def _detect_variance_spike(self) -> bool:
        """
        Detect gradient variance spike using sliding window.
        
        Returns:
            True if spike detected, False otherwise
        """
        if len(self.gradient_history) < self.gradient_window * 2:
            return False
        
        # Get recent and past windows
        recent_window = list(self.gradient_history)[-self.gradient_window:]
        past_window = list(self.gradient_history)[-2*self.gradient_window:-self.gradient_window]
        
        # Calculate variances
        recent_variance = np.var(recent_window)
        past_variance = np.var(past_window)
        
        # Store variance history
        self.variance_history.append({
            'epoch': self.current_epoch,
            'recent_variance': recent_variance,
            'past_variance': past_variance
        })
        
        # Avoid division by zero
        if past_variance == 0:
            return recent_variance > 0
        
        # Check for significant change
        variance_change = abs(recent_variance - past_variance) / past_variance
        
        return variance_change > self.variance_threshold
    
    def add_cit_boundary(self):
        """
        Add CIT boundary detection signal.
        
        This is called when CIT boundary detection occurs (optional feature).
        """
        self.credits += self.cit_credits
        self.logger.info(f"CIT boundary detected at epoch {self.current_epoch}. Credits: {self.credits}")
    
    def _trigger_growth(self):
        """Trigger growth event and reset credits."""
        growth_event = {
            'epoch': self.current_epoch,
            'credits_spent': self.credits,
            'spike_count': self.spike_count,
            'growth_type': self._determine_growth_phase()
        }
        
        self.growth_events.append(growth_event)
        self.last_growth_epoch = self.current_epoch
        
        # Spend all credits
        self.credits = 0
        
        self.logger.info(f"Growth triggered at epoch {self.current_epoch}: {growth_event}")
    
    def _determine_growth_phase(self) -> str:
        """
        Determine current growth phase based on epoch.
        
        Returns:
            Growth phase: 'coarse', 'medium', or 'fine'
        """
        if self.current_epoch < 50:
            return 'coarse'
        elif self.current_epoch < 100:
            return 'medium'
        else:
            return 'fine'
    
    def should_grow(self) -> bool:
        """
        Check if growth should occur based on current state.
        
        Returns:
            True if growth conditions are met
        """
        # Check stabilization period
        if self.current_epoch - self.last_growth_epoch < self.stabilization_epochs:
            return False
        
        # Check credit threshold
        return self.credits >= self.growth_threshold
    
    def get_current_phase(self) -> str:
        """Get current growth phase."""
        return self._determine_growth_phase()
    
    def get_stats(self) -> Dict:
        """Get growth scheduler statistics."""
        return {
            'current_epoch': self.current_epoch,
            'current_credits': self.credits,
            'total_spikes': self.spike_count,
            'total_growth_events': len(self.growth_events),
            'last_growth_epoch': self.last_growth_epoch,
            'current_phase': self.get_current_phase(),
            'epochs_since_growth': self.current_epoch - self.last_growth_epoch,
            'in_stabilization': self.current_epoch - self.last_growth_epoch < self.stabilization_epochs
        }
    
    def get_growth_history(self) -> List[Dict]:
        """Get complete growth event history."""
        return self.growth_events.copy()
    
    def get_variance_history(self) -> List[Dict]:
        """Get gradient variance history."""
        return self.variance_history.copy()
    
    def reset(self):
        """Reset scheduler state."""
        self.gradient_history.clear()
        self.credits = 0
        self.growth_events.clear()
        self.last_growth_epoch = -1
        self.current_epoch = 0
        self.variance_history.clear()
        self.credit_history.clear()
        self.spike_count = 0


class StructuralLimits:
    """
    Manages structural limits for different growth phases.
    
    Implements the structural limit rules:
    - Coarse structures (early): max 10
    - Medium structures (middle): max 50
    - Fine structures (late): max 200
    """
    
    def __init__(self):
        self.limits = {
            'coarse': 10,    # Early phase (epochs 0-50)
            'medium': 50,    # Middle phase (epochs 50-100)
            'fine': 200      # Late phase (epochs 100+)
        }
        
        self.current_counts = {
            'coarse': 0,
            'medium': 0,
            'fine': 0
        }
        
        self.growth_log = []
    
    def get_phase(self, epoch: int) -> str:
        """Determine growth phase based on epoch."""
        if epoch < 50:
            return 'coarse'
        elif epoch < 100:
            return 'medium'
        else:
            return 'fine'
    
    def can_grow(self, epoch: int, num_structures: int = 1) -> bool:
        """
        Check if growth is allowed based on structural limits.
        
        Args:
            epoch: Current epoch
            num_structures: Number of structures to add
            
        Returns:
            True if growth is allowed
        """
        phase = self.get_phase(epoch)
        return self.current_counts[phase] + num_structures <= self.limits[phase]
    
    def record_growth(self, epoch: int, num_structures: int):
        """
        Record growth event and update counts.
        
        Args:
            epoch: Current epoch
            num_structures: Number of structures added
        """
        phase = self.get_phase(epoch)
        self.current_counts[phase] += num_structures
        
        self.growth_log.append({
            'epoch': epoch,
            'phase': phase,
            'structures_added': num_structures,
            'total_in_phase': self.current_counts[phase],
            'limit': self.limits[phase]
        })
    
    def get_remaining_capacity(self, epoch: int) -> int:
        """Get remaining growth capacity for current phase."""
        phase = self.get_phase(epoch)
        return max(0, self.limits[phase] - self.current_counts[phase])
    
    def get_stats(self) -> Dict:
        """Get structural limits statistics."""
        return {
            'limits': self.limits.copy(),
            'current_counts': self.current_counts.copy(),
            'growth_log': self.growth_log.copy()
        }
    
    def reset(self):
        """Reset structural limits."""
        self.current_counts = {phase: 0 for phase in self.limits}
        self.growth_log.clear()


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test growth scheduler
    scheduler = GrowthScheduler()
    limits = StructuralLimits()
    
    # Simulate training with gradient norms
    np.random.seed(42)
    epochs = 200
    gradient_norms = []
    growth_epochs = []
    
    for epoch in range(epochs):
        scheduler.update_epoch(epoch)
        
        # Simulate gradient norm with occasional spikes
        base_norm = 1.0 + 0.1 * np.sin(epoch / 10)  # Baseline variation
        if epoch % 25 == 0 and epoch > 0:  # Periodic spikes
            gradient_norm = base_norm * (2 + np.random.random())
        else:
            gradient_norm = base_norm * (0.8 + 0.4 * np.random.random())
        
        gradient_norms.append(gradient_norm)
        
        # Check for growth
        should_grow = scheduler.add_gradient_norm(gradient_norm)
        
        if should_grow and limits.can_grow(epoch, 1):
            growth_epochs.append(epoch)
            limits.record_growth(epoch, 1)
            print(f"Growth at epoch {epoch}, phase: {scheduler.get_current_phase()}")
    
    # Print final statistics
    print("\nFinal Statistics:")
    print("Scheduler:", scheduler.get_stats())
    print("Limits:", limits.get_stats())
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(gradient_norms)
    plt.axhline(y=np.mean(gradient_norms), color='r', linestyle='--', alpha=0.5)
    plt.title('Gradient Norms')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    
    plt.subplot(2, 2, 2)
    plt.plot(scheduler.credit_history)
    plt.axhline(y=scheduler.growth_threshold, color='r', linestyle='--', alpha=0.5)
    for epoch in growth_epochs:
        plt.axvline(x=epoch, color='g', alpha=0.3)
    plt.title('Credit Accumulation')
    plt.xlabel('Epoch')
    plt.ylabel('Credits')
    
    plt.subplot(2, 2, 3)
    variance_epochs = [v['epoch'] for v in scheduler.get_variance_history()]
    recent_variances = [v['recent_variance'] for v in scheduler.get_variance_history()]
    past_variances = [v['past_variance'] for v in scheduler.get_variance_history()]
    
    plt.plot(variance_epochs, recent_variances, label='Recent')
    plt.plot(variance_epochs, past_variances, label='Past')
    plt.title('Gradient Variance')
    plt.xlabel('Epoch')
    plt.ylabel('Variance')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    phases = ['coarse', 'medium', 'fine']
    counts = [limits.current_counts[phase] for phase in phases]
    limits_vals = [limits.limits[phase] for phase in phases]
    
    x = np.arange(len(phases))
    plt.bar(x - 0.2, counts, 0.4, label='Current')
    plt.bar(x + 0.2, limits_vals, 0.4, label='Limit', alpha=0.5)
    plt.xticks(x, phases)
    plt.title('Structural Limits')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('growth_scheduler_test.png', dpi=150, bbox_inches='tight')
    plt.show()
