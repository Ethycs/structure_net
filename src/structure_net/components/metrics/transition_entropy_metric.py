"""
Transition entropy metric component.

This component computes the entropy of activation pattern transitions,
measuring the complexity and unpredictability of network dynamics.
"""

from typing import Dict, Any, Union, Optional, List
import torch
import torch.nn as nn
import logging
import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class TransitionEntropyMetric(BaseMetric):
    """
    Computes entropy of activation pattern transitions.
    
    This metric quantifies the complexity and unpredictability of how
    activation patterns change over time, which can indicate stability
    or chaos in network dynamics.
    """
    
    def __init__(self, n_symbols: int = 64, name: str = None):
        """
        Initialize transition entropy metric.
        
        Args:
            n_symbols: Number of symbols for discretizing activation patterns
            name: Optional custom name
        """
        super().__init__(name or "TransitionEntropyMetric")
        self.n_symbols = n_symbols
        self._measurement_schema = {
            "transition_entropy": float,
            "entropy_rate": float,
            "predictability_score": float,
            "transition_diversity": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"activation_trajectories"},
            provided_outputs={
                "metrics.transition_entropy",
                "metrics.entropy_rate",
                "metrics.predictability_score",
                "metrics.transition_diversity"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute transition entropy metrics.
        
        Args:
            target: Not used directly
            context: Must contain 'activation_trajectories'
            
        Returns:
            Dictionary containing entropy measurements
        """
        if not SKLEARN_AVAILABLE:
            self.log(logging.WARNING, "scikit-learn not available, returning empty metrics")
            return self._empty_metrics()
        
        # Get activation trajectories
        trajectories = context.get('activation_trajectories')
        if trajectories is None or not trajectories:
            raise ValueError("TransitionEntropyMetric requires 'activation_trajectories' in context")
        
        # Flatten all activation patterns
        patterns = []
        for traj in trajectories:
            for activation in traj:
                patterns.append(activation.flatten().cpu().numpy())
        
        if len(patterns) < self.n_symbols:
            self.log(logging.WARNING, 
                    f"Too few patterns ({len(patterns)}) for {self.n_symbols} symbols")
            return self._empty_metrics()
        
        patterns = np.array(patterns)
        
        # Cluster patterns into symbols
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_symbols,
                batch_size=min(256, len(patterns)),
                n_init=3,
                random_state=42
            )
            kmeans.fit(patterns)
        except Exception as e:
            self.log(logging.WARNING, f"Clustering failed: {e}")
            return self._empty_metrics()
        
        # Build transition matrix
        transitions = np.zeros((self.n_symbols, self.n_symbols))
        
        for traj in trajectories:
            if len(traj) < 2:
                continue
            
            # Convert trajectory to symbols
            traj_patterns = np.array([act.flatten().cpu().numpy() for act in traj])
            symbols = kmeans.predict(traj_patterns)
            
            # Count transitions
            for i in range(len(symbols) - 1):
                transitions[symbols[i], symbols[i + 1]] += 1
        
        # Normalize to get transition probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums
        
        # Compute entropy
        transition_entropy = self._compute_entropy(transition_probs)
        
        # Compute entropy rate (average entropy per symbol)
        entropy_rate = transition_entropy / self.n_symbols if self.n_symbols > 0 else 0
        
        # Predictability score (inverse of normalized entropy)
        max_entropy = np.log2(self.n_symbols) if self.n_symbols > 1 else 1
        predictability_score = 1 - (entropy_rate / max_entropy) if max_entropy > 0 else 0
        
        # Transition diversity (how many different transitions are used)
        used_transitions = np.sum(transitions > 0)
        max_transitions = self.n_symbols * self.n_symbols
        transition_diversity = used_transitions / max_transitions
        
        self.log(logging.DEBUG, 
                f"Transition entropy: {transition_entropy:.3f}, "
                f"predictability: {predictability_score:.3f}")
        
        return {
            "transition_entropy": transition_entropy,
            "entropy_rate": entropy_rate,
            "predictability_score": predictability_score,
            "transition_diversity": transition_diversity
        }
    
    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of transition probability matrix."""
        # Flatten and filter out zeros
        flat_probs = probs.flatten()
        flat_probs = flat_probs[flat_probs > 0]
        
        if len(flat_probs) == 0:
            return 0.0
        
        # Shannon entropy
        entropy = -np.sum(flat_probs * np.log2(flat_probs))
        
        return entropy
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when computation cannot proceed."""
        return {
            "transition_entropy": 0.0,
            "entropy_rate": 0.0,
            "predictability_score": 1.0,
            "transition_diversity": 0.0
        }