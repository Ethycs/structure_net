#!/usr/bin/env python3
"""
Concrete Growth Strategy Implementations

This module provides concrete implementations of growth strategies
that wrap existing growth functionality into the composable interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List, Any, Optional
import time

from ..interfaces import (
    GrowthStrategy, ArchitectureModifier, NetworkContext, 
    AnalysisResult, GrowthAction, ActionType, FullyConfigurableComponent
)
from ...core.network_factory import create_standard_network
from ...core.network_analysis import get_network_stats
from ...core.layers import StandardSparseLayer
from ..residual_blocks import SparseResidualBlock


class ExtremaGrowthStrategy(GrowthStrategy, FullyConfigurableComponent):
    """
    Growth strategy based on extrema analysis.
    
    This strategy analyzes dead and saturated neurons and adds patches
    or layers to address extrema patterns.
    """
    
    def __init__(self, 
                 extrema_threshold: float = 0.3,
                 dead_neuron_threshold: int = 5,
                 saturated_neuron_threshold: int = 5,
                 patch_size: int = 3):
        self.extrema_threshold = extrema_threshold
        self.dead_neuron_threshold = dead_neuron_threshold
        self.saturated_neuron_threshold = saturated_neuron_threshold
        self.patch_size = patch_size
        self._metrics = {}
    
    def can_apply(self, context: NetworkContext) -> bool:
        """Check if extrema growth strategy can be applied."""
        return (self.validate_context(context) and
                'extrema_analysis' in context.metadata)
    
    def analyze_growth_potential(self, context: NetworkContext) -> AnalysisResult:
        """Analyze growth potential based on extrema patterns."""
        start_time = time.time()
        
        # Get extrema analysis from context
        extrema_result = context.metadata.get('extrema_analysis')
        if extrema_result is None:
            # No extrema analysis available
            return AnalysisResult(
                analyzer_name=self.get_name(),
                metrics={'error': 'No extrema analysis available'},
                confidence=0.0
            )
        
        extrema_metrics = extrema_result.metrics
        
        # Analyze growth potential
        extrema_ratio = extrema_metrics.get('extrema_ratio', 0.0)
        dead_neurons = extrema_metrics.get('dead_neurons', {})
        saturated_neurons = extrema_metrics.get('saturated_neurons', {})
        
        total_dead = sum(len(neurons) for neurons in dead_neurons.values())
        total_saturated = sum(len(neurons) for neurons in saturated_neurons.values())
        
        growth_potential = {
            'extrema_ratio': extrema_ratio,
            'total_dead_neurons': total_dead,
            'total_saturated_neurons': total_saturated,
            'needs_patches': total_dead >= self.dead_neuron_threshold or 
                           total_saturated >= self.saturated_neuron_threshold,
            'needs_layer': extrema_ratio > self.extrema_threshold,
            'analysis_time': time.time() - start_time
        }
        
        recommendations = []
        if growth_potential['needs_patches']:
            recommendations.append("add_extrema_aware_patches")
        if growth_potential['needs_layer']:
            recommendations.append("add_layer_for_extrema")
        
        confidence = min(1.0, extrema_ratio * 2) if extrema_ratio > 0.1 else 0.3
        
        return AnalysisResult(
            analyzer_name=self.get_name(),
            metrics=growth_potential,
            recommendations=recommendations,
            confidence=confidence,
            timestamp=start_time
        )
    
    def calculate_growth_action(self, 
                              analysis: AnalysisResult, 
                              context: NetworkContext) -> Optional[GrowthAction]:
        """Calculate specific growth action based on extrema analysis."""
        metrics = analysis.metrics
        
        if metrics.get('needs_layer', False):
            # Add layer for high extrema ratio
            return GrowthAction(
                action_type=ActionType.ADD_LAYER,
                position=self._find_best_layer_position(context),
                size=self._calculate_layer_size(context),
                reason=f"High extrema ratio: {metrics.get('extrema_ratio', 0):.2f}",
                confidence=analysis.confidence
            )
        elif metrics.get('needs_patches', False):
            # Add patches for dead/saturated neurons
            return GrowthAction(
                action_type=ActionType.ADD_PATCHES,
                size=self.patch_size,
                reason=f"Dead neurons: {metrics.get('total_dead_neurons', 0)}, "
                       f"Saturated: {metrics.get('total_saturated_neurons', 0)}",
                confidence=analysis.confidence
            )
        
        return None
    
    def execute_growth_action(self, action: GrowthAction, context: NetworkContext) -> bool:
        """Execute the growth action."""
        try:
            if action.action_type == ActionType.ADD_PATCHES:
                return self._add_extrema_patches(context)
            elif action.action_type == ActionType.ADD_LAYER:
                return self._add_layer(context, action.position, action.size)
            else:
                return False
        except Exception as e:
            print(f"Failed to execute growth action: {e}")
            return False
    
    def _find_best_layer_position(self, context: NetworkContext) -> int:
        """Find the best position to add a new layer."""
        # Simple heuristic: add in the middle of the network
        network_stats = get_network_stats(context.network)
        architecture = network_stats['architecture']
        return len(architecture) // 2
    
    def _calculate_layer_size(self, context: NetworkContext) -> int:
        """Calculate appropriate size for new layer."""
        network_stats = get_network_stats(context.network)
        architecture = network_stats['architecture']
        
        if not architecture:
            return 64
        
        # Use average of existing layer sizes
        avg_size = sum(architecture) // len(architecture)
        return max(32, min(256, avg_size))
    
    def _add_extrema_patches(self, context: NetworkContext) -> bool:
        """Add patches to address extrema patterns."""
        extrema_result = context.metadata.get('extrema_analysis')
        if not extrema_result:
            return False
        
        extrema_metrics = extrema_result.metrics
        dead_neurons = extrema_metrics.get('dead_neurons', {})
        saturated_neurons = extrema_metrics.get('saturated_neurons', {})
        
        patches_added = 0
        sparse_layers = [layer for layer in context.network if isinstance(layer, StandardSparseLayer)]
        
        # Add patches for dead neurons
        for layer_idx, dead_list in dead_neurons.items():
            if isinstance(layer_idx, str):
                layer_idx = int(layer_idx)
            
            if layer_idx < len(sparse_layers) and len(dead_list) >= self.dead_neuron_threshold:
                layer = sparse_layers[layer_idx]
                patches_added += self._revive_dead_neurons(layer, dead_list[:self.patch_size])
        
        # Add patches for saturated neurons
        for layer_idx, saturated_list in saturated_neurons.items():
            if isinstance(layer_idx, str):
                layer_idx = int(layer_idx)
            
            if (layer_idx < len(sparse_layers) - 1 and 
                len(saturated_list) >= self.saturated_neuron_threshold):
                current_layer = sparse_layers[layer_idx]
                next_layer = sparse_layers[layer_idx + 1]
                patches_added += self._relieve_saturated_neurons(
                    current_layer, next_layer, saturated_list[:self.patch_size]
                )
        
        # Update metrics
        self._metrics['patches_added'] = self._metrics.get('patches_added', 0) + patches_added
        self._metrics['total_applications'] = self._metrics.get('total_applications', 0) + 1
        
        return patches_added > 0
    
    def _revive_dead_neurons(self, layer: StandardSparseLayer, dead_indices: List[int]) -> int:
        """Revive dead neurons by adding connections."""
        patches_added = 0
        
        with torch.no_grad():
            for dead_idx in dead_indices:
                if dead_idx < layer.mask.shape[0]:
                    current_connections = layer.mask[dead_idx, :].sum()
                    if current_connections < layer.mask.shape[1] * 0.1:
                        # Add connections to high-magnitude weights
                        weight_magnitudes = torch.abs(layer.linear.weight.data).mean(dim=0)
                        topk_inputs = torch.topk(weight_magnitudes, k=min(5, layer.mask.shape[1]))[1]
                        
                        for input_idx in topk_inputs[:3]:
                            layer.mask[dead_idx, input_idx] = 1.0
                            layer.linear.weight.data[dead_idx, input_idx] = torch.randn(1).item() * 0.1
                        
                        patches_added += 1
        
        return patches_added
    
    def _relieve_saturated_neurons(self, 
                                 current_layer: StandardSparseLayer,
                                 next_layer: StandardSparseLayer,
                                 saturated_indices: List[int]) -> int:
        """Relieve saturated neurons by adding output connections."""
        patches_added = 0
        
        with torch.no_grad():
            for sat_idx in saturated_indices:
                if sat_idx < current_layer.mask.shape[1]:
                    unused_outputs = torch.where(next_layer.mask[:, sat_idx] == 0)[0]
                    if len(unused_outputs) > 0:
                        for out_idx in unused_outputs[:2]:
                            next_layer.mask[out_idx, sat_idx] = 1.0
                            next_layer.linear.weight.data[out_idx, sat_idx] = torch.randn(1).item() * 0.1
                        
                        patches_added += 1
        
        return patches_added
    
    def _add_layer(self, context: NetworkContext, position: int, size: int) -> bool:
        """Add a new layer to the network."""
        try:
            # Get current architecture
            current_stats = get_network_stats(context.network)
            old_arch = current_stats['architecture']
            
            # Create new architecture
            new_arch = old_arch[:position] + [size] + old_arch[position:]
            
            # Create new network
            new_network = create_standard_network(
                architecture=new_arch,
                sparsity=0.02,  # Use default sparsity
                device=str(context.device)
            )
            
            # Copy weights (simplified - in practice would need more sophisticated transfer)
            self._copy_weights_simple(context.network, new_network, position)
            
            # Replace network in context
            context.network = new_network
            
            # Update metrics
            self._metrics['layers_added'] = self._metrics.get('layers_added', 0) + 1
            self._metrics['total_applications'] = self._metrics.get('total_applications', 0) + 1
            
            return True
            
        except Exception as e:
            print(f"Failed to add layer: {e}")
            return False
    
    def _copy_weights_simple(self, old_network: nn.Module, new_network: nn.Module, insert_position: int):
        """Simple weight copying (placeholder implementation)."""
        # This is a simplified version - in practice would need sophisticated weight transfer
        old_layers = [layer for layer in old_network if isinstance(layer, StandardSparseLayer)]
        new_layers = [layer for layer in new_network if isinstance(layer, StandardSparseLayer)]
        
        with torch.no_grad():
            # Copy layers before insertion point
            for i in range(min(insert_position, len(old_layers))):
                if i < len(new_layers):
                    new_layers[i].linear.weight.data.copy_(old_layers[i].linear.weight.data)
                    new_layers[i].linear.bias.data.copy_(old_layers[i].linear.bias.data)
                    new_layers[i].mask.data.copy_(old_layers[i].mask.data)
    
    def get_strategy_type(self) -> str:
        return "extrema_based"
    
    # Configurable interface
    def configure(self, config: Dict[str, Any]):
        """Configure the strategy."""
        self.extrema_threshold = config.get('extrema_threshold', self.extrema_threshold)
        self.dead_neuron_threshold = config.get('dead_neuron_threshold', self.dead_neuron_threshold)
        self.saturated_neuron_threshold = config.get('saturated_neuron_threshold', self.saturated_neuron_threshold)
        self.patch_size = config.get('patch_size', self.patch_size)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'extrema_threshold': self.extrema_threshold,
            'dead_neuron_threshold': self.dead_neuron_threshold,
            'saturated_neuron_threshold': self.saturated_neuron_threshold,
            'patch_size': self.patch_size
        }
    
    # Serializable interface
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'ExtremaGrowthStrategy',
            'config': self.get_configuration(),
            'metrics': self._metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtremaGrowthStrategy':
        """Deserialize from dictionary."""
        strategy = cls(**data.get('config', {}))
        strategy._metrics = data.get('metrics', {})
        return strategy
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset monitoring metrics."""
        self._metrics.clear()


class InformationFlowGrowthStrategy(GrowthStrategy, FullyConfigurableComponent):
    """
    Growth strategy based on information flow analysis.
    
    This strategy identifies information bottlenecks and adds layers
    or connections to improve information flow.
    """
    
    def __init__(self, 
                 bottleneck_threshold: float = 0.1,
                 efficiency_threshold: float = 0.7):
        self.bottleneck_threshold = bottleneck_threshold
        self.efficiency_threshold = efficiency_threshold
        self._metrics = {}
    
    def can_apply(self, context: NetworkContext) -> bool:
        """Check if information flow strategy can be applied."""
        return (self.validate_context(context) and
                'information_flow' in context.metadata)
    
    def analyze_growth_potential(self, context: NetworkContext) -> AnalysisResult:
        """Analyze growth potential based on information flow."""
        start_time = time.time()
        
        # Get information flow analysis from context
        info_flow_result = context.metadata.get('information_flow')
        if info_flow_result is None:
            return AnalysisResult(
                analyzer_name=self.get_name(),
                metrics={'error': 'No information flow analysis available'},
                confidence=0.0
            )
        
        info_metrics = info_flow_result.metrics
        
        bottlenecks = info_metrics.get('bottlenecks', [])
        efficiency = info_metrics.get('information_efficiency', 1.0)
        
        growth_potential = {
            'bottlenecks': bottlenecks,
            'efficiency': efficiency,
            'needs_layer': len(bottlenecks) > 0 and bottlenecks[0]['severity'] > self.bottleneck_threshold,
            'needs_skip_connections': efficiency < self.efficiency_threshold,
            'analysis_time': time.time() - start_time
        }
        
        recommendations = []
        if growth_potential['needs_layer']:
            recommendations.append("add_layer_for_information_flow")
        if growth_potential['needs_skip_connections']:
            recommendations.append("add_skip_connections")
        
        confidence = 0.7 if bottlenecks else 0.3
        
        return AnalysisResult(
            analyzer_name=self.get_name(),
            metrics=growth_potential,
            recommendations=recommendations,
            confidence=confidence,
            timestamp=start_time
        )
    
    def calculate_growth_action(self, 
                              analysis: AnalysisResult, 
                              context: NetworkContext) -> Optional[GrowthAction]:
        """Calculate growth action based on information flow analysis."""
        metrics = analysis.metrics
        
        if metrics.get('needs_layer', False):
            bottlenecks = metrics.get('bottlenecks', [])
            if bottlenecks:
                best_bottleneck = bottlenecks[0]
                return GrowthAction(
                    action_type=ActionType.ADD_LAYER,
                    position=best_bottleneck['position'],
                    size=self._calculate_bottleneck_layer_size(best_bottleneck, context),
                    reason=f"Information bottleneck at position {best_bottleneck['position']}, "
                           f"severity: {best_bottleneck['severity']:.3f}",
                    confidence=analysis.confidence
                )
        elif metrics.get('needs_skip_connections', False):
            return GrowthAction(
                action_type=ActionType.ADD_SKIP_CONNECTION,
                reason=f"Low information efficiency: {metrics.get('efficiency', 0):.3f}",
                confidence=analysis.confidence
            )
        
        return None
    
    def execute_growth_action(self, action: GrowthAction, context: NetworkContext) -> bool:
        """Execute the growth action."""
        try:
            if action.action_type == ActionType.ADD_LAYER:
                return self._add_layer_for_bottleneck(context, action.position, action.size)
            elif action.action_type == ActionType.ADD_SKIP_CONNECTION:
                return self._add_skip_connections(context)
            else:
                return False
        except Exception as e:
            print(f"Failed to execute information flow growth action: {e}")
            return False
    
    def _calculate_bottleneck_layer_size(self, bottleneck: Dict[str, Any], context: NetworkContext) -> int:
        """Calculate appropriate layer size for bottleneck relief."""
        info_loss = bottleneck.get('info_loss', 0.1)
        
        # Heuristic: larger loss requires larger layer
        base_size = 64
        size_multiplier = min(4.0, max(1.0, info_loss * 10))
        
        return int(base_size * size_multiplier)
    
    def _add_layer_for_bottleneck(self, context: NetworkContext, position: int, size: int) -> bool:
        """Add layer to relieve information bottleneck."""
        try:
            # Get current architecture
            current_stats = get_network_stats(context.network)
            old_arch = current_stats['architecture']
            
            # Create new architecture
            new_arch = old_arch[:position] + [size] + old_arch[position:]
            
            # Create new network
            new_network = create_standard_network(
                architecture=new_arch,
                sparsity=0.02,
                device=str(context.device)
            )
            
            # Replace network in context
            context.network = new_network
            
            # Update metrics
            self._metrics['layers_added'] = self._metrics.get('layers_added', 0) + 1
            self._metrics['total_applications'] = self._metrics.get('total_applications', 0) + 1
            
            return True
            
        except Exception as e:
            print(f"Failed to add layer for bottleneck: {e}")
            return False
    
    def _add_skip_connections(self, context: NetworkContext) -> bool:
        """Add skip connections to improve information flow."""
        # Placeholder implementation - skip connections are complex to implement
        # In practice, this would require significant architectural changes
        print("Skip connection addition not yet implemented")
        
        # Update metrics anyway to track attempts
        self._metrics['skip_connections_attempted'] = self._metrics.get('skip_connections_attempted', 0) + 1
        self._metrics['total_applications'] = self._metrics.get('total_applications', 0) + 1
        
        return False
    
    def get_strategy_type(self) -> str:
        return "information_flow_based"
    
    # Configurable interface
    def configure(self, config: Dict[str, Any]):
        """Configure the strategy."""
        self.bottleneck_threshold = config.get('bottleneck_threshold', self.bottleneck_threshold)
        self.efficiency_threshold = config.get('efficiency_threshold', self.efficiency_threshold)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'bottleneck_threshold': self.bottleneck_threshold,
            'efficiency_threshold': self.efficiency_threshold
        }
    
    # Serializable interface
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'InformationFlowGrowthStrategy',
            'config': self.get_configuration(),
            'metrics': self._metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InformationFlowGrowthStrategy':
        """Deserialize from dictionary."""
        strategy = cls(**data.get('config', {}))
        strategy._metrics = data.get('metrics', {})
        return strategy
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset monitoring metrics."""
        self._metrics.clear()


class ResidualBlockGrowthStrategy(GrowthStrategy, FullyConfigurableComponent):
    """
    Growth strategy that adds residual blocks to the network.
    """
    
    def __init__(self, 
                 num_layers: int = 2,
                 activation_threshold: float = 0.2):
        self.num_layers = num_layers
        self.activation_threshold = activation_threshold
        self._metrics = {}
    
    def can_apply(self, context: NetworkContext) -> bool:
        """Check if residual block strategy can be applied."""
        return self.validate_context(context)
    
    def analyze_growth_potential(self, context: NetworkContext) -> AnalysisResult:
        """Analyze potential for adding residual blocks."""
        start_time = time.time()
        
        # Simple heuristic: add residual blocks if network is deep enough
        network_stats = get_network_stats(context.network)
        architecture = network_stats['architecture']
        
        growth_potential = {
            'network_depth': len(architecture),
            'needs_residual_block': len(architecture) >= 4,  # Only for deeper networks
            'suggested_position': len(architecture) // 2,
            'analysis_time': time.time() - start_time
        }
        
        recommendations = []
        if growth_potential['needs_residual_block']:
            recommendations.append("add_residual_block")
        
        confidence = 0.8 if len(architecture) >= 4 else 0.2
        
        return AnalysisResult(
            analyzer_name=self.get_name(),
            metrics=growth_potential,
            recommendations=recommendations,
            confidence=confidence,
            timestamp=start_time
        )
    
    def calculate_growth_action(self, 
                              analysis: AnalysisResult, 
                              context: NetworkContext) -> Optional[GrowthAction]:
        """Calculate residual block growth action."""
        metrics = analysis.metrics
        
        if metrics.get('needs_residual_block', False):
            return GrowthAction(
                action_type=ActionType.ADD_RESIDUAL_BLOCK,
                position=metrics.get('suggested_position', 1),
                layer_count=self.num_layers,
                reason=f"Network depth: {metrics.get('network_depth', 0)} layers",
                confidence=analysis.confidence
            )
        
        return None
    
    def execute_growth_action(self, action: GrowthAction, context: NetworkContext) -> bool:
        """Execute residual block addition."""
        try:
            return self._add_residual_block(context, action.position, action.layer_count or self.num_layers)
        except Exception as e:
            print(f"Failed to add residual block: {e}")
            return False
    
    def _add_residual_block(self, context: NetworkContext, position: int, num_layers: int) -> bool:
        """Add a residual block to the network."""
        try:
            # Get network information
            sparse_layers = [layer for layer in context.network if isinstance(layer, StandardSparseLayer)]
            
            if position >= len(sparse_layers):
                return False
            
            # Determine dimensions
            if position == 0:
                in_features = sparse_layers[0].linear.in_features
                out_features = sparse_layers[0].linear.out_features
            else:
                in_features = sparse_layers[position - 1].linear.out_features
                out_features = sparse_layers[position].linear.in_features if position < len(sparse_layers) else in_features
            
            hidden_features = max(in_features // 2, 32)
            
            # Create residual block
            residual_block = SparseResidualBlock(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                sparsity=0.02,
                num_layers=num_layers,
                device=str(context.device)
            )
            
            # Insert into network (simplified)
            self._insert_residual_block(context.network, residual_block, position)
            
            # Update metrics
            self._metrics['residual_blocks_added'] = self._metrics.get('residual_blocks_added', 0) + 1
            self._metrics['total_applications'] = self._metrics.get('total_applications', 0) + 1
            
            return True
            
        except Exception as e:
            print(f"Failed to add residual block: {e}")
            return False
    
    def _insert_residual_block(self, network: nn.Module, residual_block: nn.Module, position: int):
        """Insert residual block into network (simplified implementation)."""
        # This is a placeholder - actual implementation would be more complex
        print(f"Would insert residual block at position {position}")
    
    def get_strategy_type(self) -> str:
        return "residual_block_based"
    
    # Configurable interface
    def configure(self, config: Dict[str, Any]):
        """Configure the strategy."""
        self.num_layers = config.get('num_layers', self.num_layers)
        self.activation_threshold = config.get('activation_threshold', self.activation_threshold)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'num_layers': self.num_layers,
            'activation_threshold': self.activation_threshold
        }
    
    # Serializable interface
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'ResidualBlockGrowthStrategy',
            'config': self.get_configuration(),
            'metrics': self._metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResidualBlockGrowthStrategy':
        """Deserialize from dictionary."""
        strategy = cls(**data.get('config', {}))
        strategy._metrics = data.get('metrics', {})
        return strategy
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset monitoring metrics."""
        self._metrics.clear()


class HybridGrowthStrategy(GrowthStrategy, FullyConfigurableComponent):
    """
    Hybrid strategy that combines multiple growth approaches.
    """
    
    def __init__(self, strategies: List[GrowthStrategy]):
        self.strategies = strategies
        self._metrics = {}
    
    def can_apply(self, context: NetworkContext) -> bool:
        """Check if any sub-strategy can be applied."""
        return any(strategy.can_apply(context) for strategy in self.strategies)
    
    def analyze_growth_potential(self, context: NetworkContext) -> AnalysisResult:
        """Analyze growth potential using all sub-strategies."""
        start_time = time.time()
        
        all_analyses = []
        for strategy in self.strategies:
            if strategy.can_apply(context):
                analysis = strategy.analyze_growth_potential(context)
                all_analyses.append(analysis)
        
        # Combine analyses
        combined_metrics = {
            'sub_analyses': [analysis.metrics for analysis in all_analyses],
            'num_applicable_strategies': len(all_analyses),
            'analysis_time': time.time() - start_time
        }
        
        # Combine recommendations
        all_recommendations = []
        for analysis in all_analyses:
            all_recommendations.extend(analysis.recommendations)
        
        # Average confidence
        avg_confidence = sum(analysis.confidence for analysis in all_analyses) / len(all_analyses) if all_analyses else 0.0
        
        return AnalysisResult(
            analyzer_name=self.get_name(),
            metrics=combined_metrics,
            recommendations=list(set(all_recommendations)),  # Remove duplicates
            confidence=avg_confidence,
            timestamp=start_time
        )
    
    def calculate_growth_action(self, 
                              analysis: AnalysisResult, 
                              context: NetworkContext) -> Optional[GrowthAction]:
        """Calculate growth action by selecting best sub-strategy."""
        best_action = None
        best_confidence = 0.0
        
        for strategy in self.strategies:
            if strategy.can_apply(context):
                strategy_analysis = strategy.analyze_growth_potential(context)
                action = strategy.calculate_growth_action(strategy_analysis, context)
                
                if action and action.confidence > best_confidence:
                    best_action = action
                    best_confidence = action.confidence
        
        if best_action:
            best_action.reason = f"Hybrid strategy selected: {best_action.reason}"
        
        return best_action
    
    def execute_growth_action(self, action: GrowthAction, context: NetworkContext) -> bool:
        """Execute growth action using appropriate sub-strategy."""
        # Find strategy that can handle this action type
        for strategy in self.strategies:
            if strategy.can_apply(context):
                try:
                    if strategy.execute_growth_action(action, context):
                        self._metrics['successful_applications'] = self._metrics.get('successful_applications', 0) + 1
                        return True
                except Exception as e:
                    print(f"Strategy {strategy.get_name()} failed: {e}")
                    continue
        
        self._metrics['failed_applications'] = self._metrics.get('failed_applications', 0) + 1
        return False
    
    def get_strategy_type(self) -> str:
        return "hybrid"
    
    # Configurable interface
    def configure(self, config: Dict[str, Any]):
        """Configure all sub-strategies."""
        for i, strategy in enumerate(self.strategies):
            if hasattr(strategy, 'configure'):
                strategy_config = config.get(f'strategy_{i}', {})
                strategy.configure(strategy_config)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get configuration of all sub-strategies."""
        config = {}
        for i, strategy in enumerate(self.strategies):
            if hasattr(strategy, 'get_configuration'):
                config[f'strategy_{i}'] = strategy.get_configuration()
        return config
    
    # Serializable interface
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'HybridGrowthStrategy',
            'strategies': [strategy.to_dict() if hasattr(strategy, 'to_dict') else str(strategy) 
                          for strategy in self.strategies],
            'metrics': self._metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HybridGrowthStrategy':
        """Deserialize from dictionary."""
        # This would need a strategy factory to recreate strategies
        # For now, return empty hybrid strategy
        strategy = cls([])
        strategy._metrics = data.get('metrics', {})
        return strategy
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        combined_metrics = self._metrics.copy()
        
        # Add sub-strategy metrics
        for i, strategy in enumerate(self.strategies):
            if hasattr(strategy, 'get_metrics'):
                sub_metrics = strategy.get_metrics()
                for key, value in sub_metrics.items():
                    combined_metrics[f'strategy_{i}_{key}'] = value
        
        return combined_metrics
    
    def reset_metrics(self):
        """Reset monitoring metrics."""
        self._metrics.clear()
        for strategy in self.strategies:
            if hasattr(strategy, 'reset_metrics'):
                strategy.reset_metrics()


# Export all strategies
__all__ = [
    'ExtremaGrowthStrategy',
    'InformationFlowGrowthStrategy',
    'ResidualBlockGrowthStrategy',
    'HybridGrowthStrategy'
]
