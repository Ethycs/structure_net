"""
Compactification metrics for analyzing sparse network architectures.

These metrics evaluate the effectiveness of homological compactification:
- Compression ratios and efficiency
- Patch effectiveness and distribution  
- Memory usage and hardware optimization
- Information preservation through sparsity
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass

from ...core import IMetric, ILayer, IModel, ComponentContract, EvolutionContext


@dataclass
class PatchAnalysis:
    """Analysis of dense patches in compactified network."""
    total_patches: int
    average_density: float
    patch_efficiency: float
    extrema_coverage: float
    memory_overhead: float


class CompressionRatioMetric(IMetric):
    """
    Measures compression effectiveness of compactified networks.
    
    Analyzes the ratio between sparse representation and original dense network,
    accounting for patch overhead and storage efficiency.
    """
    
    def __init__(self, name: str = "compression_ratio"):
        super().__init__(name)
        self._contract = ComponentContract(
            component_name="CompressionRatioMetric",
            version="1.0.0",
            required_inputs={'original_size', 'compressed_size'},
            optional_inputs={'patch_info', 'memory_layout'},
            provided_outputs={'compression_ratio', 'space_saved', 'efficiency_score',
                            'original_size', 'compressed_size'},
            maturity="production"
        )
    
    def analyze(self, target: Optional[Union[ILayer, IModel]], 
                context: EvolutionContext) -> Dict[str, Any]:
        """Analyze compression ratio of compactified network."""
        
        # Get size information
        original_size = context.get('original_size', 0)
        compressed_size = context.get('compressed_size', 0)
        
        if original_size == 0:
            raise ValueError("original_size must be provided in context")
        
        # Calculate basic compression metrics
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        space_saved = original_size - compressed_size
        space_saved_ratio = space_saved / original_size if original_size > 0 else 0
        
        # Account for patch overhead if available
        patch_info = context.get('patch_info', {})
        patch_overhead = 0
        if patch_info:
            patch_overhead = patch_info.get('metadata_size', 0)
            effective_compressed_size = compressed_size + patch_overhead
            effective_ratio = effective_compressed_size / original_size
        else:
            effective_ratio = compression_ratio
        
        # Calculate efficiency score (higher is better)
        efficiency_score = max(0, 1 - effective_ratio)
        
        # Memory layout efficiency
        memory_layout = context.get('memory_layout', {})
        layout_efficiency = memory_layout.get('cache_efficiency', 1.0)
        
        return {
            'compression_ratio': compression_ratio,
            'effective_compression_ratio': effective_ratio,
            'space_saved': space_saved,
            'space_saved_ratio': space_saved_ratio,
            'efficiency_score': efficiency_score * layout_efficiency,
            'patch_overhead': patch_overhead,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'memory_efficiency': layout_efficiency
        }


class PatchEffectivenessMetric(IMetric):
    """
    Evaluates effectiveness of dense patch placement.
    
    Analyzes how well patches capture important network regions
    and their contribution to overall network performance.
    """
    
    def __init__(self, name: str = "patch_effectiveness"):
        super().__init__(name)
        self._contract = ComponentContract(
            component_name="PatchEffectivenessMetric",
            version="1.0.0",
            required_inputs={'patches'},
            optional_inputs={'extrema_locations', 'importance_scores', 'activation_data'},
            provided_outputs={'patch_coverage', 'extrema_alignment', 'activation_capture',
                            'patch_density_distribution', 'effectiveness_score'},
            maturity="production"
        )
    
    def analyze(self, target: Optional[Union[ILayer, IModel]], 
                context: EvolutionContext) -> Dict[str, Any]:
        """Analyze effectiveness of patch placement."""
        
        patches = context.get('patches', [])
        if not patches:
            raise ValueError("patches must be provided in context")
        
        # Analyze patch coverage
        total_patches = len(patches)
        densities = [p.get('density', 0) for p in patches]
        average_density = np.mean(densities) if densities else 0
        density_std = np.std(densities) if densities else 0
        
        # Check extrema alignment
        extrema_locations = context.get('extrema_locations', [])
        extrema_alignment = 0.0
        if extrema_locations:
            # Calculate how well patches align with extrema
            aligned_patches = 0
            for patch in patches:
                patch_pos = patch.get('position', (0, 0))
                for extrema_pos in extrema_locations:
                    distance = np.sqrt(sum((a - b)**2 for a, b in zip(patch_pos, extrema_pos)))
                    if distance < patch.get('size', (8, 8))[0]:  # Within patch radius
                        aligned_patches += 1
                        break
            extrema_alignment = aligned_patches / len(patches) if patches else 0
        
        # Analyze activation capture
        activation_data = context.get('activation_data', {})
        activation_capture = 0.0
        if activation_data:
            # Measure how much important activation is captured by patches
            important_activations = activation_data.get('high_magnitude_locations', [])
            captured_activations = 0
            
            for patch in patches:
                patch_region = self._get_patch_region(patch)
                for act_pos in important_activations:
                    if self._is_in_region(act_pos, patch_region):
                        captured_activations += 1
            
            activation_capture = captured_activations / len(important_activations) if important_activations else 0
        
        # Calculate overall effectiveness score
        effectiveness_score = (
            0.4 * extrema_alignment +
            0.3 * activation_capture +
            0.2 * min(average_density / 0.2, 1.0) +  # Target 20% density
            0.1 * (1 - density_std / (average_density + 1e-8))  # Consistency bonus
        )
        
        return {
            'total_patches': total_patches,
            'average_density': average_density,
            'density_std': density_std,
            'patch_coverage': total_patches / context.get('target_patches', 1),
            'extrema_alignment': extrema_alignment,
            'activation_capture': activation_capture,
            'patch_density_distribution': {
                'mean': average_density,
                'std': density_std,
                'min': min(densities) if densities else 0,
                'max': max(densities) if densities else 0
            },
            'effectiveness_score': effectiveness_score
        }
    
    def _get_patch_region(self, patch: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get patch region as ((start_row, start_col), (end_row, end_col))."""
        pos = patch.get('position', (0, 0))
        size = patch.get('size', (8, 8))
        return (pos, (pos[0] + size[0], pos[1] + size[1]))
    
    def _is_in_region(self, point: Tuple[int, int], 
                     region: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """Check if point is within region."""
        (start_row, start_col), (end_row, end_col) = region
        row, col = point
        return start_row <= row < end_row and start_col <= col < end_col


class MemoryEfficiencyMetric(IMetric):
    """
    Analyzes memory usage patterns in compactified networks.
    
    Evaluates cache efficiency, memory access patterns, and
    hardware optimization of sparse representations.
    """
    
    def __init__(self, name: str = "memory_efficiency"):
        super().__init__(name)
        self._contract = ComponentContract(
            component_name="MemoryEfficiencyMetric",
            version="1.0.0",
            required_inputs={'memory_layout'},
            optional_inputs={'access_patterns', 'cache_metrics', 'hardware_specs'},
            provided_outputs={'cache_efficiency', 'memory_overhead', 'access_locality',
                            'bandwidth_utilization', 'memory_fragmentation'},
            maturity="production"
        )
    
    def analyze(self, target: Optional[Union[ILayer, IModel]], 
                context: EvolutionContext) -> Dict[str, Any]:
        """Analyze memory efficiency of compactified representation."""
        
        memory_layout = context.get('memory_layout', {})
        if not memory_layout:
            raise ValueError("memory_layout must be provided in context")
        
        # Cache efficiency analysis
        cache_hits = memory_layout.get('cache_hits', 0)
        cache_misses = memory_layout.get('cache_misses', 0)
        total_accesses = cache_hits + cache_misses
        cache_efficiency = cache_hits / total_accesses if total_accesses > 0 else 0
        
        # Memory overhead calculation
        sparse_data_size = memory_layout.get('sparse_data_size', 0)
        index_size = memory_layout.get('index_size', 0)
        metadata_size = memory_layout.get('metadata_size', 0)
        total_memory = sparse_data_size + index_size + metadata_size
        
        memory_overhead = (index_size + metadata_size) / total_memory if total_memory > 0 else 0
        
        # Access pattern analysis
        access_patterns = context.get('access_patterns', {})
        sequential_accesses = access_patterns.get('sequential', 0)
        random_accesses = access_patterns.get('random', 0)
        total_pattern_accesses = sequential_accesses + random_accesses
        access_locality = sequential_accesses / total_pattern_accesses if total_pattern_accesses > 0 else 0
        
        # Bandwidth utilization
        theoretical_bandwidth = context.get('hardware_specs', {}).get('memory_bandwidth', 1)
        actual_bandwidth = memory_layout.get('achieved_bandwidth', 0)
        bandwidth_utilization = actual_bandwidth / theoretical_bandwidth if theoretical_bandwidth > 0 else 0
        
        # Memory fragmentation
        allocated_blocks = memory_layout.get('allocated_blocks', 1)
        contiguous_blocks = memory_layout.get('contiguous_blocks', 1)
        fragmentation = 1 - (contiguous_blocks / allocated_blocks) if allocated_blocks > 0 else 0
        
        return {
            'cache_efficiency': cache_efficiency,
            'memory_overhead': memory_overhead,
            'access_locality': access_locality,
            'bandwidth_utilization': bandwidth_utilization,
            'memory_fragmentation': fragmentation,
            'total_memory_usage': total_memory,
            'sparse_data_ratio': sparse_data_size / total_memory if total_memory > 0 else 0,
            'index_overhead_ratio': index_size / total_memory if total_memory > 0 else 0,
            'cache_statistics': {
                'hits': cache_hits,
                'misses': cache_misses,
                'hit_rate': cache_efficiency
            }
        }


class ReconstructionQualityMetric(IMetric):
    """
    Measures quality of network reconstruction from sparse representation.
    
    Evaluates how well the compactified network preserves the original
    network's functionality and information content.
    """
    
    def __init__(self, name: str = "reconstruction_quality"):
        super().__init__(name)
        self._contract = ComponentContract(
            component_name="ReconstructionQualityMetric",
            version="1.0.0",
            required_inputs={'original_output', 'reconstructed_output'},
            optional_inputs={'intermediate_activations', 'gradient_comparison'},
            provided_outputs={'reconstruction_error', 'activation_similarity', 
                            'gradient_preservation', 'information_loss'},
            maturity="production"
        )
    
    def analyze(self, target: Optional[Union[ILayer, IModel]], 
                context: EvolutionContext) -> Dict[str, Any]:
        """Analyze quality of reconstruction from sparse representation."""
        
        original_output = context.get('original_output')
        reconstructed_output = context.get('reconstructed_output')
        
        if original_output is None or reconstructed_output is None:
            raise ValueError("Both original_output and reconstructed_output must be provided")
        
        # Convert to tensors if needed
        if not isinstance(original_output, torch.Tensor):
            original_output = torch.tensor(original_output)
        if not isinstance(reconstructed_output, torch.Tensor):
            reconstructed_output = torch.tensor(reconstructed_output)
        
        # Reconstruction error metrics
        mse_error = torch.mean((original_output - reconstructed_output) ** 2).item()
        mae_error = torch.mean(torch.abs(original_output - reconstructed_output)).item()
        
        # Relative error
        original_norm = torch.norm(original_output).item()
        if original_norm > 0:
            relative_error = torch.norm(original_output - reconstructed_output).item() / original_norm
        else:
            relative_error = 0.0
        
        # Activation similarity (cosine similarity)
        original_flat = original_output.flatten()
        reconstructed_flat = reconstructed_output.flatten()
        
        cosine_sim = torch.nn.functional.cosine_similarity(
            original_flat.unsqueeze(0), 
            reconstructed_flat.unsqueeze(0)
        ).item()
        
        # Correlation analysis
        if len(original_flat) > 1:
            correlation = torch.corrcoef(torch.stack([original_flat, reconstructed_flat]))[0, 1].item()
        else:
            correlation = 1.0
        
        # Gradient preservation (if available)
        gradient_comparison = context.get('gradient_comparison', {})
        gradient_preservation = gradient_comparison.get('similarity', 1.0)
        
        # Information loss estimation
        # Based on mutual information approximation
        original_entropy = self._estimate_entropy(original_output)
        reconstructed_entropy = self._estimate_entropy(reconstructed_output)
        information_loss = abs(original_entropy - reconstructed_entropy) / (original_entropy + 1e-8)
        
        # Overall quality score
        quality_score = (
            0.3 * (1 - min(relative_error, 1.0)) +
            0.3 * max(cosine_sim, 0) +
            0.2 * max(correlation, 0) +
            0.2 * gradient_preservation
        )
        
        return {
            'mse_error': mse_error,
            'mae_error': mae_error,
            'relative_error': relative_error,
            'cosine_similarity': cosine_sim,
            'correlation': correlation,
            'gradient_preservation': gradient_preservation,
            'information_loss': information_loss,
            'reconstruction_quality_score': quality_score,
            'activation_similarity': cosine_sim,
            'reconstruction_error': relative_error
        }
    
    def _estimate_entropy(self, tensor: torch.Tensor, bins: int = 50) -> float:
        """Estimate entropy of tensor values using histogram."""
        if tensor.numel() == 0:
            return 0.0
        
        # Convert to numpy for histogram
        values = tensor.detach().cpu().numpy().flatten()
        hist, _ = np.histogram(values, bins=bins, density=True)
        
        # Remove zero bins and normalize
        hist = hist[hist > 0]
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        return entropy