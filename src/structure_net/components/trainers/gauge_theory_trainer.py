#!/usr/bin/env python3
"""
Gauge Theory Trainer Component

Migrated from evolution.gauge_theory to use the ITrainer interface.
Implements gauge-invariant optimization and training strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from ...core.interfaces import (
    ITrainer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)


class GaugeTheoryTrainer(ITrainer):
    """
    Trainer component implementing gauge-invariant optimization techniques.
    
    This trainer exploits gauge symmetries in neural networks to improve
    training stability and achieve gauge-invariant optimization.
    
    Features:
    - Gauge-invariant optimization in quotient space
    - Canonical gauge enforcement
    - Gauge-aware compression
    - Neuron importance-based ordering
    """
    
    def __init__(self,
                 base_optimizer_class: type = optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = None,
                 canonical_gauge_frequency: int = 10,
                 importance_metric: str = 'l2_norm',
                 gauge_enforcement: bool = True,
                 name: str = None):
        super().__init__()
        self.base_optimizer_class = base_optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 0.001}
        self.canonical_gauge_frequency = canonical_gauge_frequency
        self.importance_metric = importance_metric
        self.gauge_enforcement = gauge_enforcement
        self._name = name or "GaugeTheoryTrainer"
        
        # Training state
        self.base_optimizer = None
        self.step_count = 0
        self.last_gauge_enforcement = 0
        
        # Component contract
        self._contract = ComponentContract(
            component_name=self._name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={'model', 'data_batch', 'loss_function'},
            provided_outputs={'training_metrics', 'gauge_transformations'},
            optional_inputs={'optimizer_state', 'gauge_parameters'},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=True,
                parallel_safe=False,  # Gauge transformations need careful coordination
                estimated_runtime_seconds=1.0
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Component contract declaration."""
        return self._contract
    
    @property
    def name(self) -> str:
        """Component name."""
        return self._name
    
    def train_step(self, model: IModel, batch: Any, context: EvolutionContext) -> Dict[str, float]:
        """
        Execute one training step with gauge-invariant optimization.
        
        Args:
            model: Model to train
            batch: Training batch data
            context: Evolution context containing loss function and other info
            
        Returns:
            Training metrics including loss and gauge statistics
        """
        self._track_execution(self._perform_training_step)
        return self._perform_training_step(model, batch, context)
    
    def _perform_training_step(self, model: IModel, batch: Any, context: EvolutionContext) -> Dict[str, float]:
        """Internal training step implementation."""
        try:
            # Initialize optimizer if needed
            if self.base_optimizer is None:
                self.base_optimizer = self.base_optimizer_class(
                    model.parameters(), **self.optimizer_kwargs
                )
            
            # Get loss function and device
            loss_fn = context.get('loss_function')
            device = context.get('device', 'cpu')
            
            if loss_fn is None:
                raise ValueError("loss_function not found in context")
            
            # Prepare batch data
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                raise ValueError("Batch must contain inputs and targets")
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            self.base_optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gauge-invariant optimization step
            self._gauge_invariant_step(model)
            
            # Compute metrics
            metrics = {
                'loss': loss.item(),
                'step_count': self.step_count
            }
            
            # Add gauge-specific metrics
            if self.gauge_enforcement:
                gauge_metrics = self._compute_gauge_metrics(model)
                metrics.update(gauge_metrics)
            
            self.step_count += 1
            
            self.log(logging.DEBUG, f"Training step completed, loss: {loss.item():.6f}")
            
            return metrics
            
        except Exception as e:
            self.log(logging.ERROR, f"Training step failed: {str(e)}")
            raise
    
    def _gauge_invariant_step(self, model: IModel):
        """
        Perform gauge-invariant optimization step.
        
        Args:
            model: Model to optimize
        """
        # Regular gradient step
        self.base_optimizer.step()
        
        # Enforce canonical gauge periodically
        if (self.gauge_enforcement and 
            self.step_count - self.last_gauge_enforcement >= self.canonical_gauge_frequency):
            self._enforce_canonical_gauge(model)
            self.last_gauge_enforcement = self.step_count
    
    def _enforce_canonical_gauge(self, model: IModel):
        """
        Fix gauge to canonical form - ordered by neuron importance.
        
        Args:
            model: Model to apply gauge transformation to
        """
        try:
            layers = model.get_layers()
            
            # Apply gauge transformation to each hidden layer
            for layer_idx in range(1, len(layers) - 1):
                current_layer = layers[layer_idx]
                prev_layer = layers[layer_idx - 1]
                
                # Skip if layers don't support gauge transformation
                if not self._can_apply_gauge_transform(current_layer, prev_layer):
                    continue
                
                # Compute neuron importance
                importance = self._compute_neuron_importance(
                    current_layer, prev_layer, layer_idx
                )
                
                # Get permutation to sort by importance
                perm = torch.argsort(importance, descending=True)
                
                # Apply gauge transformation
                self._apply_gauge_transform(current_layer, prev_layer, perm)
            
            self.log(logging.DEBUG, f"Applied canonical gauge enforcement")
            
        except Exception as e:
            self.log(logging.WARNING, f"Gauge enforcement failed: {str(e)}")
    
    def _can_apply_gauge_transform(self, current_layer, prev_layer) -> bool:
        """
        Check if gauge transformation can be applied to these layers.
        
        Args:
            current_layer: Current layer
            prev_layer: Previous layer
            
        Returns:
            True if gauge transformation is applicable
        """
        return (
            hasattr(current_layer, 'weight') and 
            hasattr(prev_layer, 'weight') and
            isinstance(current_layer.weight, nn.Parameter) and
            isinstance(prev_layer.weight, nn.Parameter)
        )
    
    def _compute_neuron_importance(self, current_layer, prev_layer, layer_idx: int) -> torch.Tensor:
        """
        Compute importance scores for neurons in the current layer.
        
        Args:
            current_layer: Current layer
            prev_layer: Previous layer
            layer_idx: Layer index
            
        Returns:
            Importance scores tensor
        """
        if self.importance_metric == 'l2_norm':
            return self._compute_l2_importance(current_layer, prev_layer)
        elif self.importance_metric == 'gradient_magnitude':
            return self._compute_gradient_importance(current_layer)
        elif self.importance_metric == 'hybrid':
            return self._compute_hybrid_importance(current_layer, prev_layer)
        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")
    
    def _compute_l2_importance(self, current_layer, prev_layer) -> torch.Tensor:
        """Compute importance based on L2 norm of weights."""
        W_in = prev_layer.weight
        W_out = current_layer.weight
        
        # L2 norm of incoming + outgoing weights
        importance = torch.norm(W_in, p=2, dim=0) + torch.norm(W_out, p=2, dim=1)
        return importance
    
    def _compute_gradient_importance(self, current_layer) -> torch.Tensor:
        """Compute importance based on gradient magnitude."""
        if current_layer.weight.grad is not None:
            importance = torch.norm(current_layer.weight.grad, p=2, dim=0)
        else:
            # Fallback to weight norms if no gradients available
            importance = torch.norm(current_layer.weight, p=2, dim=1)
        return importance
    
    def _compute_hybrid_importance(self, current_layer, prev_layer) -> torch.Tensor:
        """Compute importance using hybrid method."""
        l2_importance = self._compute_l2_importance(current_layer, prev_layer)
        
        if current_layer.weight.grad is not None:
            grad_importance = torch.norm(current_layer.weight.grad, p=2, dim=0)
            # Weighted combination
            importance = 0.7 * l2_importance + 0.3 * grad_importance
        else:
            importance = l2_importance
        
        return importance
    
    def _apply_gauge_transform(self, current_layer, prev_layer, perm: torch.Tensor):
        """
        Apply gauge transformation using the given permutation.
        
        Args:
            current_layer: Current layer
            prev_layer: Previous layer
            perm: Permutation indices
        """
        with torch.no_grad():
            # Apply permutation to previous layer's output weights
            W_in = prev_layer.weight
            prev_layer.weight.data = W_in[:, perm]
            
            # Apply inverse permutation to current layer's input weights
            W_out = current_layer.weight
            perm_inv = torch.argsort(perm)
            current_layer.weight.data = W_out[perm_inv, :]
            
            # Apply to bias if present
            if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                bias = current_layer.bias
                current_layer.bias.data = bias[perm_inv]
    
    def _compute_gauge_metrics(self, model: IModel) -> Dict[str, float]:
        """
        Compute gauge-specific metrics for monitoring.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary of gauge metrics
        """
        metrics = {}
        
        try:
            layers = model.get_layers()
            total_gauge_invariance = 0.0
            valid_layers = 0
            
            for layer_idx in range(1, len(layers) - 1):
                current_layer = layers[layer_idx]
                prev_layer = layers[layer_idx - 1]
                
                if self._can_apply_gauge_transform(current_layer, prev_layer):
                    # Measure how well-ordered neurons are by importance
                    importance = self._compute_neuron_importance(
                        current_layer, prev_layer, layer_idx
                    )
                    
                    # Compute ordering quality (how close to sorted)
                    sorted_importance = torch.sort(importance, descending=True)[0]
                    ordering_quality = torch.cosine_similarity(
                        importance.unsqueeze(0), 
                        sorted_importance.unsqueeze(0)
                    ).item()
                    
                    total_gauge_invariance += ordering_quality
                    valid_layers += 1
                    
                    metrics[f'layer_{layer_idx}_gauge_quality'] = ordering_quality
            
            if valid_layers > 0:
                metrics['avg_gauge_quality'] = total_gauge_invariance / valid_layers
            
            metrics['gauge_enforcements'] = self.last_gauge_enforcement
            
        except Exception as e:
            self.log(logging.WARNING, f"Failed to compute gauge metrics: {str(e)}")
        
        return metrics
    
    def supports_online_evolution(self) -> bool:
        """Check if this trainer can handle model changes during training."""
        return True  # Gauge theory can adapt to architectural changes
    
    def can_apply(self, context: EvolutionContext) -> bool:
        """Check if this trainer can be applied to the given context."""
        return (
            self.validate_context(context) and
            'model' in context and
            'loss_function' in context
        )
    
    def apply(self, context: EvolutionContext) -> bool:
        """Apply this trainer (training happens via train_step)."""
        return self.can_apply(context)
    
    # Additional gauge theory methods
    
    def compress_network_gauge_aware(self, model: IModel, compression_ratio: float = 0.5) -> IModel:
        """
        Compress network by removing least important neurons using gauge theory.
        
        Args:
            model: Model to compress
            compression_ratio: Fraction of neurons to keep
            
        Returns:
            Compressed model
        """
        compressed_model = copy.deepcopy(model)
        layers = compressed_model.get_layers()
        
        for layer_idx in range(1, len(layers) - 1):
            current_layer = layers[layer_idx]
            prev_layer = layers[layer_idx - 1]
            
            if not self._can_apply_gauge_transform(current_layer, prev_layer):
                continue
            
            # Find optimal gauge for compression
            importance = self._compute_neuron_importance(
                current_layer, prev_layer, layer_idx
            )
            
            # Sort by importance and apply gauge transformation
            perm = torch.argsort(importance, descending=True)
            self._apply_gauge_transform(current_layer, prev_layer, perm)
            
            # Remove least important neurons
            n_neurons = importance.shape[0]
            n_keep = int(n_neurons * compression_ratio)
            
            # Truncate weights
            with torch.no_grad():
                W_in = prev_layer.weight
                W_out = current_layer.weight
                
                prev_layer.weight.data = W_in[:, :n_keep]
                current_layer.weight.data = W_out[:n_keep, :]
                
                if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                    current_layer.bias.data = current_layer.bias.data[:n_keep]
        
        self.log(logging.INFO, f"Compressed model with ratio {compression_ratio}")
        return compressed_model
    
    def get_gauge_transformation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of gauge transformations applied.
        
        Returns:
            List of transformation records
        """
        # This would be implemented with actual tracking in a full version
        return []
    
    def reset_gauge_state(self):
        """Reset gauge enforcement state."""
        self.step_count = 0
        self.last_gauge_enforcement = 0
        self.log(logging.INFO, "Reset gauge state")