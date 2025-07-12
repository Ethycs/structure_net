#!/usr/bin/env python3
"""
Optimal Growth Evolver Component

Migrated from evolution.network_evolver to use the IEvolver interface.
Advanced evolver that uses information theory to precisely identify
bottlenecks and calculate minimal, optimal interventions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging
from torch.utils.data import DataLoader

from ...core.interfaces import (
    IEvolver, IModel, ITrainer, EvolutionPlan, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)
from ...core.network_factory import create_standard_network
from ...core.layers import StandardSparseLayer


class PlateauDetector:
    """Detects performance plateaus to trigger strategy changes."""
    
    def __init__(self, patience: int = 5, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.single_layer_history = []

    def add_result(self, improvement: float):
        """Add a performance improvement result."""
        self.single_layer_history.append(improvement)

    def should_switch_to_multilayer(self) -> tuple[bool, str]:
        """Check if it's time to switch to multi-layer insertions."""
        if len(self.single_layer_history) < self.patience:
            return False, "Not enough history for single-layer additions."
            
        recent_improvements = self.single_layer_history[-self.patience:]
        avg_improvement = np.mean(recent_improvements)
        
        if avg_improvement < self.min_improvement:
            return True, f"Single layers plateaued (avg. improvement {avg_improvement:.2%})"
            
        return False, f"Single layers still effective (avg. improvement {avg_improvement:.2%})"


class AdaptiveLayerInsertionStrategy:
    """Switch from single to multi-layer insertion based on plateau detection."""
    
    def __init__(self):
        self.mode = 'single'
        self.plateau_detector = PlateauDetector()
        
    def determine_growth_action(self, evolver: 'OptimalGrowthEvolver', analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Decide on the insertion strategy based on performance history."""
        if self.mode == 'single':
            action = self._single_layer_strategy(evolver, analysis)
            
            # Get improvement from evolver
            improvement = evolver.get_recent_improvement()
            self.plateau_detector.add_result(improvement)
            
            # Check if we should switch modes
            should_switch, reason = self.plateau_detector.should_switch_to_multilayer()
            
            if should_switch:
                evolver.log(logging.INFO, f"Switching to multi-layer mode: {reason}")
                self.mode = 'multi'
                return action
            else:
                return action
                
        else:  # Already in multi mode
            return self._multi_layer_strategy(evolver, analysis)

    def _single_layer_strategy(self, evolver: 'OptimalGrowthEvolver', analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert a single layer at the worst bottleneck."""
        current_arch = evolver.get_current_architecture()
        position = len(current_arch) // 2
        width = current_arch[position] if position < len(current_arch) else 128
        
        return {
            'type': 'insert_layer',
            'position': position,
            'width': width,
            'reason': 'single_layer_bottleneck_relief'
        }

    def _multi_layer_strategy(self, evolver: 'OptimalGrowthEvolver', analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """When single layers plateau, try inserting multiple layers."""
        current_arch = evolver.get_current_architecture()
        position = len(current_arch) // 2
        width1 = current_arch[position] if position < len(current_arch) else 128
        width2 = width1 // 2
        
        return {
            'type': 'insert_multiple_layers',
            'position': position,
            'widths': [width1, width2],
            'reason': 'multi_layer_plateau_breaking'
        }


class OptimalGrowthEvolver(IEvolver):
    """
    Advanced evolver component that uses information theory to precisely identify
    bottlenecks and calculate minimal, optimal interventions.
    
    Features:
    - Information theory-based bottleneck detection
    - Adaptive single vs multi-layer growth strategies
    - Plateau detection and strategy switching
    - Precise layer insertion calculations
    - Neuron sorting for improved efficiency
    """
    
    def __init__(self,
                 seed_architecture: List[int],
                 seed_sparsity: float = 0.02,
                 enable_sorting: bool = True,
                 sort_frequency: int = 3,
                 plateau_patience: int = 5,
                 min_improvement: float = 0.01,
                 name: str = None):
        super().__init__()
        self.seed_architecture = seed_architecture
        self.seed_sparsity = seed_sparsity
        self.enable_sorting = enable_sorting
        self.sort_frequency = sort_frequency
        self.evolution_step_count = 0
        self._name = name or "OptimalGrowthEvolver"
        
        # Strategy management
        self.adaptive_strategy = AdaptiveLayerInsertionStrategy()
        self.adaptive_strategy.plateau_detector.patience = plateau_patience
        self.adaptive_strategy.plateau_detector.min_improvement = min_improvement
        
        # Performance tracking
        self.performance_history = []
        self.current_architecture = seed_architecture.copy()
        
        # Component contract
        self._contract = ComponentContract(
            component_name=self._name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={'model', 'data_loader', 'performance_metrics'},
            provided_outputs={'evolved_model', 'growth_actions', 'bottleneck_analysis'},
            optional_inputs={'device', 'evolution_plan_override'},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=True,
                parallel_safe=False,  # Network modification needs careful coordination
                estimated_runtime_seconds=10.0
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
    
    def can_execute_plan(self, plan: EvolutionPlan) -> bool:
        """
        Check if this evolver can execute the given plan.
        
        Args:
            plan: Evolution plan to check
            
        Returns:
            True if this evolver can handle the plan
        """
        return plan.get('type') in [
            'optimal_growth',
            'insert_layer',
            'insert_multiple_layers',
            'information_theory_growth',
            'adaptive_layer_insertion'
        ]
    
    def apply_plan(self, plan: EvolutionPlan, model: IModel, trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """
        Execute the evolution plan.
        
        Args:
            plan: Evolution plan to execute
            model: Model to evolve
            trainer: Trainer for the model
            optimizer: Optimizer for training
            
        Returns:
            Results of the evolution
        """
        self._track_execution(self._execute_plan)
        return self._execute_plan(plan, model, trainer, optimizer)
    
    def _execute_plan(self, plan: EvolutionPlan, model: IModel, trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Internal plan execution implementation."""
        try:
            plan_type = plan.get('type', 'optimal_growth')
            
            if plan_type == 'insert_layer':
                return self._execute_single_layer_insertion(plan, model, trainer, optimizer)
            elif plan_type == 'insert_multiple_layers':
                return self._execute_multi_layer_insertion(plan, model, trainer, optimizer)
            elif plan_type in ['optimal_growth', 'information_theory_growth', 'adaptive_layer_insertion']:
                return self._execute_optimal_growth(plan, model, trainer, optimizer)
            else:
                raise ValueError(f"Unsupported plan type: {plan_type}")
                
        except Exception as e:
            self.log(logging.ERROR, f"Plan execution failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _execute_optimal_growth(self, plan: EvolutionPlan, model: IModel, trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Execute optimal growth strategy."""
        # Analyze current model for bottlenecks
        analysis = self._analyze_information_bottlenecks(model, plan.get('context', {}))
        
        # Determine growth action using adaptive strategy
        growth_action = self.adaptive_strategy.determine_growth_action(self, analysis)
        
        if growth_action is None:
            return {
                'success': True,
                'action_taken': 'no_growth_needed',
                'analysis': analysis
            }
        
        # Execute the determined action
        if growth_action['type'] == 'insert_layer':
            result = self._insert_single_layer(model, growth_action, trainer, optimizer)
        elif growth_action['type'] == 'insert_multiple_layers':
            result = self._insert_multiple_layers(model, growth_action, trainer, optimizer)
        else:
            raise ValueError(f"Unknown growth action type: {growth_action['type']}")
        
        # Update evolution state
        self.evolution_step_count += 1
        
        # Apply neuron sorting if enabled
        if self.enable_sorting and self.evolution_step_count % self.sort_frequency == 0:
            self._apply_neuron_sorting(model)
        
        return {
            'success': True,
            'action_taken': growth_action,
            'evolution_step': self.evolution_step_count,
            'analysis': analysis,
            'model_updated': True,
            **result
        }
    
    def _execute_single_layer_insertion(self, plan: EvolutionPlan, model: IModel, trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Execute single layer insertion."""
        growth_action = {
            'type': 'insert_layer',
            'position': plan.get('position', len(self.current_architecture) // 2),
            'width': plan.get('width', 128),
            'reason': 'plan_specified'
        }
        
        result = self._insert_single_layer(model, growth_action, trainer, optimizer)
        self.evolution_step_count += 1
        
        return {'success': True, 'action_taken': growth_action, **result}
    
    def _execute_multi_layer_insertion(self, plan: EvolutionPlan, model: IModel, trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Execute multi-layer insertion."""
        growth_action = {
            'type': 'insert_multiple_layers',
            'position': plan.get('position', len(self.current_architecture) // 2),
            'widths': plan.get('widths', [128, 64]),
            'reason': 'plan_specified'
        }
        
        result = self._insert_multiple_layers(model, growth_action, trainer, optimizer)
        self.evolution_step_count += 1
        
        return {'success': True, 'action_taken': growth_action, **result}
    
    def _analyze_information_bottlenecks(self, model: IModel, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze model for information bottlenecks using information theory.
        
        Args:
            model: Model to analyze
            context: Context with data and other info
            
        Returns:
            Analysis results
        """
        # This would use the information flow analyzer
        # For now, provide a simplified analysis
        layers = model.get_layers() if hasattr(model, 'get_layers') else []
        
        analysis = {
            'num_layers': len(layers),
            'architecture': self.current_architecture,
            'bottleneck_layers': [],
            'information_flow': [],
            'growth_potential': 0.7  # Simplified score
        }
        
        # Identify potential bottleneck layers (simplified)
        for i, layer in enumerate(layers):
            if hasattr(layer, 'linear'):
                layer_size = layer.linear.out_features
                if i > 0 and i < len(layers) - 1:  # Not input or output layer
                    # Simple bottleneck detection based on size
                    prev_size = layers[i-1].linear.out_features if hasattr(layers[i-1], 'linear') else layer_size
                    next_size = layers[i+1].linear.out_features if i+1 < len(layers) and hasattr(layers[i+1], 'linear') else layer_size
                    
                    if layer_size < min(prev_size, next_size) * 0.5:
                        analysis['bottleneck_layers'].append({
                            'layer_index': i,
                            'layer_size': layer_size,
                            'bottleneck_severity': 0.8,
                            'recommended_expansion': int(layer_size * 1.5)
                        })
        
        return analysis
    
    def _insert_single_layer(self, model: IModel, action: Dict[str, Any], trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Insert a single layer into the model."""
        position = action['position']
        width = action['width']
        
        self.log(logging.INFO, f"Inserting single layer at position {position} with width {width}")
        
        # Update architecture record
        self.current_architecture.insert(position, width)
        
        # Create new network with inserted layer
        new_network = create_standard_network(
            architecture=self.current_architecture,
            sparsity=self.seed_sparsity,
            device=next(model.parameters()).device
        )
        
        # Copy weights from old model (simplified transfer)
        self._transfer_weights(model, new_network, position)
        
        # Replace model layers (this is a simplified approach)
        if hasattr(model, 'layers'):
            model.layers = new_network.layers
        
        return {
            'layers_added': 1,
            'new_architecture': self.current_architecture.copy(),
            'insertion_position': position,
            'layer_width': width
        }
    
    def _insert_multiple_layers(self, model: IModel, action: Dict[str, Any], trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Insert multiple layers into the model."""
        position = action['position']
        widths = action['widths']
        
        self.log(logging.INFO, f"Inserting {len(widths)} layers at position {position} with widths {widths}")
        
        # Update architecture record
        for i, width in enumerate(widths):
            self.current_architecture.insert(position + i, width)
        
        # Create new network with inserted layers
        new_network = create_standard_network(
            architecture=self.current_architecture,
            sparsity=self.seed_sparsity,
            device=next(model.parameters()).device
        )
        
        # Transfer weights (simplified)
        self._transfer_weights(model, new_network, position)
        
        # Replace model layers
        if hasattr(model, 'layers'):
            model.layers = new_network.layers
        
        return {
            'layers_added': len(widths),
            'new_architecture': self.current_architecture.copy(),
            'insertion_position': position,
            'layer_widths': widths
        }
    
    def _transfer_weights(self, old_model: IModel, new_model: nn.Module, insertion_position: int):
        """
        Transfer weights from old model to new model (simplified implementation).
        
        Args:
            old_model: Original model
            new_model: New model with inserted layers
            insertion_position: Where layers were inserted
        """
        # This is a simplified weight transfer
        # In practice, this would be more sophisticated
        old_layers = old_model.get_layers() if hasattr(old_model, 'get_layers') else []
        new_layers = [layer for layer in new_model.modules() if isinstance(layer, StandardSparseLayer)]
        
        # Copy weights for layers before insertion point
        for i in range(min(insertion_position, len(old_layers))):
            if i < len(new_layers):
                self._copy_layer_weights(old_layers[i], new_layers[i])
        
        # Skip the inserted layers and copy the rest
        old_idx = insertion_position
        new_idx = insertion_position + 1  # Assuming one layer inserted for simplicity
        
        while old_idx < len(old_layers) and new_idx < len(new_layers):
            self._copy_layer_weights(old_layers[old_idx], new_layers[new_idx])
            old_idx += 1
            new_idx += 1
    
    def _copy_layer_weights(self, old_layer, new_layer):
        """Copy weights between layers if dimensions match."""
        if (hasattr(old_layer, 'linear') and hasattr(new_layer, 'linear') and
            old_layer.linear.weight.shape == new_layer.linear.weight.shape):
            
            with torch.no_grad():
                new_layer.linear.weight.data.copy_(old_layer.linear.weight.data)
                if old_layer.linear.bias is not None and new_layer.linear.bias is not None:
                    new_layer.linear.bias.data.copy_(old_layer.linear.bias.data)
                
                if hasattr(old_layer, 'mask') and hasattr(new_layer, 'mask'):
                    new_layer.mask.data.copy_(old_layer.mask.data)
    
    def _apply_neuron_sorting(self, model: IModel):
        """Apply importance-based neuron sorting to the model."""
        self.log(logging.DEBUG, "Applying neuron sorting")
        
        layers = model.get_layers() if hasattr(model, 'get_layers') else []
        
        # Sort every hidden layer, but not the final output layer
        for i in range(len(layers) - 1):
            self._sort_layer_neurons(layers, i)
    
    def _sort_layer_neurons(self, layers: List, layer_idx: int):
        """Sort neurons in a specific layer by importance."""
        if layer_idx >= len(layers) - 1:  # Don't sort output layer
            return
        
        current_layer = layers[layer_idx]
        
        if not hasattr(current_layer, 'linear'):
            return
        
        # Calculate importance of each output neuron based on weight norm
        importance = torch.linalg.norm(current_layer.linear.weight, ord=2, dim=1)
        
        # Get the indices that would sort the neurons by importance (descending)
        perm_indices = torch.argsort(importance, descending=True)
        
        with torch.no_grad():
            # Apply permutation to current layer (output neurons)
            current_layer.linear.weight.data = current_layer.linear.weight.data[perm_indices, :]
            current_layer.linear.bias.data = current_layer.linear.bias.data[perm_indices]
            
            if hasattr(current_layer, 'mask'):
                current_layer.mask.data = current_layer.mask.data[perm_indices, :]
            
            # Apply permutation to next layer (input connections)
            if layer_idx + 1 < len(layers):
                next_layer = layers[layer_idx + 1]
                if hasattr(next_layer, 'linear'):
                    next_layer.linear.weight.data = next_layer.linear.weight.data[:, perm_indices]
                    
                    if hasattr(next_layer, 'mask'):
                        next_layer.mask.data = next_layer.mask.data[:, perm_indices]
    
    def get_recent_improvement(self) -> float:
        """Get recent performance improvement for plateau detection."""
        if len(self.performance_history) < 2:
            return 0.01  # Default small improvement
        
        return self.performance_history[-1] - self.performance_history[-2]
    
    def add_performance_result(self, performance: float):
        """Add a performance result to history."""
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
    
    def get_current_architecture(self) -> List[int]:
        """Get current network architecture."""
        return self.current_architecture.copy()
    
    def can_apply(self, context: EvolutionContext) -> bool:
        """Check if this evolver can be applied to the given context."""
        return (
            self.validate_context(context) and
            'model' in context and
            'data_loader' in context
        )
    
    def apply(self, context: EvolutionContext) -> bool:
        """Apply this evolver (evolution happens via apply_plan)."""
        return self.can_apply(context)
    
    def reset_state(self):
        """Reset evolver state for new evolution cycle."""
        self.evolution_step_count = 0
        self.performance_history.clear()
        self.current_architecture = self.seed_architecture.copy()
        self.adaptive_strategy = AdaptiveLayerInsertionStrategy()
        self.log(logging.INFO, "Reset evolver state")