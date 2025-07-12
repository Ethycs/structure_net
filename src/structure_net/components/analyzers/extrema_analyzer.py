#!/usr/bin/env python3
"""
Extrema Analyzer Component

Migrated from evolution.extrema_analyzer to use the IAnalyzer interface.
Provides extrema detection and analysis capabilities for networks using the canonical standard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Set, List, Optional
from datetime import datetime

from ...core.interfaces import (
    IAnalyzer, IModel, AnalysisReport, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)
from ...core.layers import StandardSparseLayer


class ExtremaAnalyzer(IAnalyzer):
    """
    Analyzer component for detecting dead and saturated neurons in neural networks.
    
    This component analyzes activation patterns to identify extrema that may
    indicate training issues or opportunities for network optimization.
    """
    
    def __init__(self, 
                 dead_threshold: float = 0.01,
                 saturated_threshold: float = 0.99,
                 max_batches: int = 10,
                 name: str = None):
        super().__init__()
        self.dead_threshold = dead_threshold
        self.saturated_threshold = saturated_threshold
        self.max_batches = max_batches
        self._name = name or "ExtremaAnalyzer"
        
        # Component contract
        self._contract = ComponentContract(
            component_name=self._name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={'model', 'data_loader'},
            provided_outputs={'extrema_patterns', 'extrema_statistics'},
            optional_inputs={'device'},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True,
                estimated_runtime_seconds=5.0
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
    
    def analyze(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """
        Perform extrema analysis on the model.
        
        Args:
            model: Model to analyze
            report: Current analysis report
            context: Evolution context
            
        Returns:
            Dictionary containing extrema analysis results
        """
        self._track_execution(self._perform_analysis)
        return self._perform_analysis(model, report, context)
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """Internal analysis implementation."""
        try:
            # Get data loader from context
            data_loader = context.get('data_loader')
            if data_loader is None:
                raise ValueError("data_loader not found in context")
            
            device = context.get('device', 'cpu')
            
            # Detect extrema patterns
            extrema_patterns = self._detect_network_extrema(model, data_loader, device)
            
            # Calculate statistics
            extrema_stats = self._get_extrema_statistics(extrema_patterns)
            
            # Create analysis result
            result = {
                'extrema_patterns': extrema_patterns,
                'extrema_statistics': extrema_stats,
                'analysis_metadata': {
                    'dead_threshold': self.dead_threshold,
                    'saturated_threshold': self.saturated_threshold,
                    'max_batches_analyzed': self.max_batches,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.log(logging.INFO, f"Analyzed {len(extrema_patterns)} layers, found {extrema_stats['total_dead_neurons']} dead and {extrema_stats['total_saturated_neurons']} saturated neurons")
            
            return result
            
        except Exception as e:
            self.log(logging.ERROR, f"Analysis failed: {str(e)}")
            raise
    
    def get_required_metrics(self) -> Set[str]:
        """Metrics required for this analyzer."""
        return set()  # This analyzer works directly with model activations
    
    @torch.no_grad()
    def _analyze_layer_extrema(self, activations: torch.Tensor) -> Dict[str, List[int]]:
        """
        Identify dead and saturated neurons from activations.
        
        Args:
            activations: Tensor of activations [batch_size, num_neurons]
            
        Returns:
            Dictionary with 'low' (dead) and 'high' (saturated) neuron indices
        """
        mean_activations = activations.mean(dim=0)
        
        dead_neurons = torch.where(mean_activations < self.dead_threshold)[0].tolist()
        saturated_neurons = torch.where(mean_activations > self.saturated_threshold)[0].tolist()
        
        return {'low': dead_neurons, 'high': saturated_neurons}
    
    @torch.no_grad()
    def _detect_network_extrema(self, model: IModel, data_loader, device: str) -> List[Dict[str, List[int]]]:
        """
        Detect extrema patterns across all layers of the network.
        
        Args:
            model: Model to analyze
            data_loader: DataLoader for getting activations
            device: Device to run analysis on
            
        Returns:
            List of extrema patterns for each layer
        """
        model.eval()
        model = model.to(device)
        
        # Collect activations from all layers
        layer_activations = []
        
        def create_hook(layer_activations, layer_idx):
            def hook(module, input, output):
                if isinstance(module, StandardSparseLayer):
                    # Store post-activation values
                    layer_activations.append((layer_idx, output.detach().cpu()))
            return hook
        
        # Register hooks for all StandardSparseLayer modules
        hooks = []
        layer_idx = 0
        for module in model.modules():
            if isinstance(module, StandardSparseLayer):
                hook = create_hook(layer_activations, layer_idx)
                hooks.append(module.register_forward_hook(hook))
                layer_idx += 1
        
        # Run forward passes to collect activations
        all_layer_activations = [[] for _ in range(layer_idx)]
        
        try:
            for batch_idx, batch_data in enumerate(data_loader):
                if batch_idx >= self.max_batches:
                    break
                
                # Handle different batch formats
                if isinstance(batch_data, (list, tuple)):
                    data = batch_data[0]
                else:
                    data = batch_data
                    
                data = data.to(device)
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)  # Flatten
                
                # Clear previous activations
                layer_activations.clear()
                
                # Forward pass
                _ = model(data)
                
                # Store activations by layer
                for layer_idx, activation in layer_activations:
                    all_layer_activations[layer_idx].append(activation)
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Analyze extrema for each layer
        extrema_patterns = []
        for layer_idx in range(len(all_layer_activations)):
            if all_layer_activations[layer_idx]:
                # Concatenate all batches for this layer
                layer_acts = torch.cat(all_layer_activations[layer_idx], dim=0)
                
                # Apply ReLU if not the last layer
                if layer_idx < len(all_layer_activations) - 1:
                    layer_acts = F.relu(layer_acts)
                
                # Analyze extrema
                extrema = self._analyze_layer_extrema(layer_acts)
                extrema_patterns.append(extrema)
            else:
                extrema_patterns.append({'low': [], 'high': []})
        
        return extrema_patterns
    
    def _get_extrema_statistics(self, extrema_patterns: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        """
        Get summary statistics about extrema patterns.
        
        Args:
            extrema_patterns: List of extrema patterns
            
        Returns:
            Dictionary with extrema statistics
        """
        stats = {
            'total_dead_neurons': 0,
            'total_saturated_neurons': 0,
            'layers_with_dead': 0,
            'layers_with_saturated': 0,
            'worst_layer_dead': -1,
            'worst_layer_saturated': -1,
            'max_dead_in_layer': 0,
            'max_saturated_in_layer': 0,
            'layer_stats': []
        }
        
        for i, extrema in enumerate(extrema_patterns):
            dead_count = len(extrema['low'])
            saturated_count = len(extrema['high'])
            
            stats['total_dead_neurons'] += dead_count
            stats['total_saturated_neurons'] += saturated_count
            
            if dead_count > 0:
                stats['layers_with_dead'] += 1
                if dead_count > stats['max_dead_in_layer']:
                    stats['max_dead_in_layer'] = dead_count
                    stats['worst_layer_dead'] = i
            
            if saturated_count > 0:
                stats['layers_with_saturated'] += 1
                if saturated_count > stats['max_saturated_in_layer']:
                    stats['max_saturated_in_layer'] = saturated_count
                    stats['worst_layer_saturated'] = i
            
            stats['layer_stats'].append({
                'layer_index': i,
                'dead_neurons': dead_count,
                'saturated_neurons': saturated_count,
                'total_extrema': dead_count + saturated_count
            })
        
        return stats
    
    def can_apply(self, context: EvolutionContext) -> bool:
        """Check if this analyzer can be applied to the given context."""
        return (
            self.validate_context(context) and
            'model' in context and
            'data_loader' in context
        )
    
    def apply(self, context: EvolutionContext) -> bool:
        """Apply this analyzer (not typically used for analyzers)."""
        return self.can_apply(context)
    
    def get_analysis_type(self) -> str:
        """Get the type of analysis this analyzer performs."""
        return "extrema"
    
    def get_required_batches(self) -> int:
        """Get number of data batches required for analysis."""
        return self.max_batches
    
    def print_analysis(self, extrema_patterns: List[Dict[str, List[int]]], 
                      network_stats: Optional[Dict[str, Any]] = None):
        """
        Print a human-readable analysis of extrema patterns.
        
        Args:
            extrema_patterns: List of extrema patterns
            network_stats: Optional network statistics
        """
        stats = self._get_extrema_statistics(extrema_patterns)
        
        print("\n🔍 EXTREMA ANALYSIS")
        print("=" * 50)
        
        print(f"📊 Overall Statistics:")
        print(f"   Total dead neurons: {stats['total_dead_neurons']}")
        print(f"   Total saturated neurons: {stats['total_saturated_neurons']}")
        print(f"   Layers with dead neurons: {stats['layers_with_dead']}")
        print(f"   Layers with saturated neurons: {stats['layers_with_saturated']}")
        
        if stats['worst_layer_dead'] >= 0:
            print(f"   Worst layer (dead): Layer {stats['worst_layer_dead']} ({stats['max_dead_in_layer']} dead)")
        
        if stats['worst_layer_saturated'] >= 0:
            print(f"   Worst layer (saturated): Layer {stats['worst_layer_saturated']} ({stats['max_saturated_in_layer']} saturated)")
        
        print(f"\n📋 Layer-by-Layer Analysis:")
        for layer_stat in stats['layer_stats']:
            layer_idx = layer_stat['layer_index']
            dead = layer_stat['dead_neurons']
            saturated = layer_stat['saturated_neurons']
            total = layer_stat['total_extrema']
            
            # Get layer size if network stats available
            layer_size = "unknown"
            if network_stats and layer_idx < len(network_stats['layers']):
                layer_size = network_stats['layers'][layer_idx]['out_features']
            
            print(f"   Layer {layer_idx} (size: {layer_size}): {dead} dead, {saturated} saturated, {total} total extrema")
            
            # Show specific neuron indices for small numbers
            if dead > 0 and dead <= 10:
                dead_indices = extrema_patterns[layer_idx]['low']
                print(f"      Dead neurons: {dead_indices}")
            
            if saturated > 0 and saturated <= 10:
                saturated_indices = extrema_patterns[layer_idx]['high']
                print(f"      Saturated neurons: {saturated_indices}")