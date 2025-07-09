#!/usr/bin/env python3
"""
Compactification Metrics for Neural Networks

Provides metrics for analyzing network compactification effectiveness.
Extracted from the compactification module for reusability and modularity.

Key Features:
- Compression ratio analysis
- Patch effectiveness metrics
- Memory efficiency analysis
- Reconstruction error computation
- Storage optimization metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

from .base import BaseMetricAnalyzer, NetworkAnalyzerMixin, StatisticalUtilsMixin, MetricResult
from ...profiling import profile_component, profile_operation, ProfilerLevel


@dataclass
class CompressionStats:
    """Statistics about compression effectiveness."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    space_saved: int
    efficiency_score: float


@dataclass
class PatchEffectiveness:
    """Effectiveness metrics for patch-based compression."""
    patch_count: int
    avg_patch_density: float
    patch_coverage: float
    information_preservation: float
    reconstruction_quality: float


@dataclass
class MemoryProfile:
    """Memory usage profile for compactified networks."""
    total_memory: int
    patch_memory: int
    skeleton_memory: int
    metadata_memory: int
    memory_efficiency: float


@profile_component(component_name="compactification_analyzer", level=ProfilerLevel.DETAILED)
class CompactificationAnalyzer(BaseMetricAnalyzer, NetworkAnalyzerMixin, StatisticalUtilsMixin):
    """
    Specialized analyzer for network compactification effectiveness.
    
    Analyzes compression ratios, patch effectiveness, memory usage,
    and reconstruction quality for compactified neural networks.
    """
    
    def __init__(self, threshold_config=None):
        if threshold_config is None:
            from .base import ThresholdConfig
            threshold_config = ThresholdConfig()
        
        super().__init__(threshold_config)
        
    def compute_metrics(self, compact_data: Dict[str, Any], 
                       original_network: nn.Module = None) -> Dict[str, Any]:
        """
        Compute comprehensive compactification metrics.
        
        Args:
            compact_data: Compactified network data
            original_network: Original network for comparison
            
        Returns:
            Dictionary containing all compactification metrics
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._cache_key(compact_data, original_network)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        with profile_operation("compactification_analysis", "compression"):
            # Core compactification analysis
            compression_stats = self._analyze_compression(compact_data, original_network)
            patch_effectiveness = self._analyze_patch_effectiveness(compact_data)
            memory_profile = self._analyze_memory_usage(compact_data)
            reconstruction_metrics = self._analyze_reconstruction_quality(compact_data)
            
            # Compile metrics
            metrics = {
                'compression_stats': compression_stats,
                'patch_effectiveness': patch_effectiveness,
                'memory_profile': memory_profile,
                'reconstruction_metrics': reconstruction_metrics,
                'storage_efficiency': self._compute_storage_efficiency(compact_data),
                'parameter_reduction': self._compute_parameter_reduction(compact_data, original_network),
                'sparsity_analysis': self._analyze_sparsity_patterns(compact_data),
                'information_density': self._compute_information_density(compact_data),
                'compactification_quality': self._compute_overall_quality(compact_data)
            }
            
            # Update computation stats
            computation_time = time.time() - start_time
            self._computation_stats['total_calls'] += 1
            self._computation_stats['total_time'] += computation_time
            
            # Cache result
            self._cache_result(cache_key, metrics)
            
            return metrics
    
    def _analyze_compression(self, compact_data: Dict[str, Any], 
                           original_network: nn.Module = None) -> CompressionStats:
        """Analyze compression effectiveness."""
        
        # Calculate compressed size
        compressed_size = 0
        
        # Count patch data
        if 'patches' in compact_data:
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    compressed_size += patch_info['data'].numel()
        
        # Count skeleton data
        if 'skeleton' in compact_data:
            compressed_size += compact_data['skeleton'].numel()
        
        # Count metadata
        if 'metadata' in compact_data:
            metadata = compact_data['metadata']
            compressed_size += len(str(metadata))  # Rough estimate
        
        # Calculate original size
        original_size = 0
        if original_network is not None:
            original_size = sum(p.numel() for p in original_network.parameters())
        elif 'original_size' in compact_data:
            original_size = compact_data['original_size']
        else:
            # Estimate from compressed data
            original_size = compressed_size * 2  # Conservative estimate
        
        # Compression metrics
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        space_saved = original_size - compressed_size
        efficiency_score = 1.0 - compression_ratio
        
        return CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            space_saved=space_saved,
            efficiency_score=efficiency_score
        )
    
    def _analyze_patch_effectiveness(self, compact_data: Dict[str, Any]) -> PatchEffectiveness:
        """Analyze effectiveness of patch-based compression."""
        
        if 'patches' not in compact_data:
            return PatchEffectiveness(
                patch_count=0,
                avg_patch_density=0.0,
                patch_coverage=0.0,
                information_preservation=0.0,
                reconstruction_quality=0.0
            )
        
        patches = compact_data['patches']
        patch_count = len(patches)
        
        # Analyze patch densities
        densities = []
        total_patch_elements = 0
        total_nonzero_elements = 0
        
        for patch_info in patches:
            if 'data' in patch_info:
                patch_data = patch_info['data']
                total_elements = patch_data.numel()
                nonzero_elements = (patch_data.abs() > 1e-6).sum().item()
                
                density = nonzero_elements / total_elements if total_elements > 0 else 0
                densities.append(density)
                
                total_patch_elements += total_elements
                total_nonzero_elements += nonzero_elements
        
        avg_patch_density = np.mean(densities) if densities else 0.0
        
        # Patch coverage (how much of the network is covered by patches)
        if 'original_size' in compact_data:
            patch_coverage = total_patch_elements / compact_data['original_size']
        else:
            patch_coverage = 0.5  # Default estimate
        
        # Information preservation (based on non-zero elements)
        information_preservation = total_nonzero_elements / total_patch_elements if total_patch_elements > 0 else 0
        
        # Reconstruction quality (if available)
        reconstruction_quality = compact_data.get('reconstruction_error', 0.0)
        if reconstruction_quality > 0:
            reconstruction_quality = 1.0 / (1.0 + reconstruction_quality)  # Convert error to quality
        else:
            reconstruction_quality = 0.8  # Default estimate
        
        return PatchEffectiveness(
            patch_count=patch_count,
            avg_patch_density=avg_patch_density,
            patch_coverage=patch_coverage,
            information_preservation=information_preservation,
            reconstruction_quality=reconstruction_quality
        )
    
    def _analyze_memory_usage(self, compact_data: Dict[str, Any]) -> MemoryProfile:
        """Analyze memory usage patterns."""
        
        # Calculate memory usage for different components
        patch_memory = 0
        skeleton_memory = 0
        metadata_memory = 0
        
        # Patch memory
        if 'patches' in compact_data:
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    # Assume float32 (4 bytes per element)
                    patch_memory += patch_info['data'].numel() * 4
                # Add metadata overhead
                patch_memory += 64  # Rough estimate for position, size, etc.
        
        # Skeleton memory
        if 'skeleton' in compact_data:
            skeleton_memory = compact_data['skeleton'].numel() * 4
        
        # Metadata memory
        if 'metadata' in compact_data:
            metadata_memory = len(str(compact_data['metadata'])) * 2  # Rough estimate
        
        total_memory = patch_memory + skeleton_memory + metadata_memory
        
        # Memory efficiency
        if 'original_size' in compact_data:
            original_memory = compact_data['original_size'] * 4  # float32
            memory_efficiency = 1.0 - (total_memory / original_memory) if original_memory > 0 else 0
        else:
            memory_efficiency = 0.5  # Default estimate
        
        return MemoryProfile(
            total_memory=total_memory,
            patch_memory=patch_memory,
            skeleton_memory=skeleton_memory,
            metadata_memory=metadata_memory,
            memory_efficiency=memory_efficiency
        )
    
    def _analyze_reconstruction_quality(self, compact_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze reconstruction quality metrics."""
        
        metrics = {
            'reconstruction_error': 0.0,
            'information_loss': 0.0,
            'fidelity_score': 1.0,
            'structural_preservation': 1.0
        }
        
        # Direct reconstruction error if available
        if 'reconstruction_error' in compact_data:
            metrics['reconstruction_error'] = compact_data['reconstruction_error']
            metrics['fidelity_score'] = 1.0 / (1.0 + metrics['reconstruction_error'])
        
        # Information loss estimation
        if 'patches' in compact_data:
            total_preserved = 0
            total_original = 0
            
            for patch_info in compact_data['patches']:
                if 'data' in patch_info and 'original_data' in patch_info:
                    preserved = (patch_info['data'].abs() > 1e-6).sum().item()
                    original = (patch_info['original_data'].abs() > 1e-6).sum().item()
                    
                    total_preserved += preserved
                    total_original += original
            
            if total_original > 0:
                metrics['information_loss'] = 1.0 - (total_preserved / total_original)
        
        # Structural preservation (based on connectivity patterns)
        if 'skeleton' in compact_data:
            skeleton = compact_data['skeleton']
            nonzero_ratio = (skeleton.abs() > 1e-6).float().mean().item()
            metrics['structural_preservation'] = nonzero_ratio
        
        return metrics
    
    def _compute_storage_efficiency(self, compact_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute storage efficiency metrics."""
        
        # Calculate storage requirements
        patch_storage = 0
        skeleton_storage = 0
        index_storage = 0
        
        if 'patches' in compact_data:
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    # Data storage
                    patch_storage += patch_info['data'].numel() * 4
                    # Index storage (position, size)
                    index_storage += 16  # 4 ints * 4 bytes
        
        if 'skeleton' in compact_data:
            skeleton_storage = compact_data['skeleton'].numel() * 4
        
        total_storage = patch_storage + skeleton_storage + index_storage
        
        # Efficiency metrics
        if total_storage > 0:
            patch_efficiency = patch_storage / total_storage
            skeleton_efficiency = skeleton_storage / total_storage
            index_overhead = index_storage / total_storage
        else:
            patch_efficiency = skeleton_efficiency = index_overhead = 0.0
        
        return {
            'total_storage_bytes': total_storage,
            'patch_storage_ratio': patch_efficiency,
            'skeleton_storage_ratio': skeleton_efficiency,
            'index_overhead_ratio': index_overhead,
            'storage_density': self._compute_storage_density(compact_data)
        }
    
    def _compute_storage_density(self, compact_data: Dict[str, Any]) -> float:
        """Compute information density in storage."""
        
        total_stored_elements = 0
        total_nonzero_elements = 0
        
        # Count patch elements
        if 'patches' in compact_data:
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    data = patch_info['data']
                    total_stored_elements += data.numel()
                    total_nonzero_elements += (data.abs() > 1e-6).sum().item()
        
        # Count skeleton elements
        if 'skeleton' in compact_data:
            skeleton = compact_data['skeleton']
            total_stored_elements += skeleton.numel()
            total_nonzero_elements += (skeleton.abs() > 1e-6).sum().item()
        
        return total_nonzero_elements / total_stored_elements if total_stored_elements > 0 else 0.0
    
    def _compute_parameter_reduction(self, compact_data: Dict[str, Any], 
                                   original_network: nn.Module = None) -> Dict[str, Any]:
        """Compute parameter reduction metrics."""
        
        # Count compactified parameters
        compact_params = 0
        
        if 'patches' in compact_data:
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    compact_params += patch_info['data'].numel()
        
        if 'skeleton' in compact_data:
            compact_params += compact_data['skeleton'].numel()
        
        # Count original parameters
        original_params = 0
        if original_network is not None:
            original_params = sum(p.numel() for p in original_network.parameters())
        elif 'original_size' in compact_data:
            original_params = compact_data['original_size']
        
        # Reduction metrics
        if original_params > 0:
            reduction_ratio = 1.0 - (compact_params / original_params)
            reduction_factor = original_params / compact_params if compact_params > 0 else float('inf')
        else:
            reduction_ratio = 0.0
            reduction_factor = 1.0
        
        return {
            'original_parameters': original_params,
            'compact_parameters': compact_params,
            'reduction_ratio': reduction_ratio,
            'reduction_factor': reduction_factor,
            'parameters_saved': original_params - compact_params
        }
    
    def _analyze_sparsity_patterns(self, compact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sparsity patterns in compactified data."""
        
        sparsity_metrics = {
            'patch_sparsity': [],
            'skeleton_sparsity': 0.0,
            'overall_sparsity': 0.0,
            'sparsity_distribution': {}
        }
        
        total_elements = 0
        total_nonzero = 0
        
        # Analyze patch sparsity
        if 'patches' in compact_data:
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    data = patch_info['data']
                    elements = data.numel()
                    nonzero = (data.abs() > 1e-6).sum().item()
                    
                    patch_sparsity = 1.0 - (nonzero / elements) if elements > 0 else 1.0
                    sparsity_metrics['patch_sparsity'].append(patch_sparsity)
                    
                    total_elements += elements
                    total_nonzero += nonzero
        
        # Analyze skeleton sparsity
        if 'skeleton' in compact_data:
            skeleton = compact_data['skeleton']
            elements = skeleton.numel()
            nonzero = (skeleton.abs() > 1e-6).sum().item()
            
            sparsity_metrics['skeleton_sparsity'] = 1.0 - (nonzero / elements) if elements > 0 else 1.0
            
            total_elements += elements
            total_nonzero += nonzero
        
        # Overall sparsity
        sparsity_metrics['overall_sparsity'] = 1.0 - (total_nonzero / total_elements) if total_elements > 0 else 1.0
        
        # Sparsity distribution
        if sparsity_metrics['patch_sparsity']:
            sparsity_values = sparsity_metrics['patch_sparsity']
            sparsity_metrics['sparsity_distribution'] = {
                'mean': np.mean(sparsity_values),
                'std': np.std(sparsity_values),
                'min': np.min(sparsity_values),
                'max': np.max(sparsity_values),
                'median': np.median(sparsity_values)
            }
        
        return sparsity_metrics
    
    def _compute_information_density(self, compact_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute information density metrics."""
        
        # Information per byte
        total_bytes = 0
        total_information = 0
        
        if 'patches' in compact_data:
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    data = patch_info['data']
                    bytes_used = data.numel() * 4  # float32
                    information = (data.abs() > 1e-6).sum().item()
                    
                    total_bytes += bytes_used
                    total_information += information
        
        if 'skeleton' in compact_data:
            skeleton = compact_data['skeleton']
            bytes_used = skeleton.numel() * 4
            information = (skeleton.abs() > 1e-6).sum().item()
            
            total_bytes += bytes_used
            total_information += information
        
        # Density metrics
        info_per_byte = total_information / total_bytes if total_bytes > 0 else 0
        
        # Information entropy (simplified)
        entropy = 0.0
        if 'patches' in compact_data:
            all_values = []
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    values = patch_info['data'].flatten()
                    all_values.extend(values.tolist())
            
            if all_values:
                # Compute histogram-based entropy
                hist, _ = np.histogram(all_values, bins=50)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]  # Remove zeros
                entropy = -np.sum(probs * np.log2(probs))
        
        return {
            'information_per_byte': info_per_byte,
            'information_entropy': entropy,
            'compression_entropy': entropy / 8.0 if entropy > 0 else 0,  # Normalize to [0,1]
            'information_efficiency': min(1.0, info_per_byte / 0.25)  # Normalize assuming max 0.25 info/byte
        }
    
    def _compute_overall_quality(self, compact_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall compactification quality score."""
        
        # Get component scores
        compression_stats = self._analyze_compression(compact_data)
        patch_effectiveness = self._analyze_patch_effectiveness(compact_data)
        memory_profile = self._analyze_memory_usage(compact_data)
        
        # Component quality scores
        compression_quality = compression_stats.efficiency_score
        patch_quality = (patch_effectiveness.avg_patch_density + 
                        patch_effectiveness.information_preservation) / 2
        memory_quality = memory_profile.memory_efficiency
        
        # Weighted overall quality
        overall_quality = (0.4 * compression_quality + 
                          0.3 * patch_quality + 
                          0.3 * memory_quality)
        
        return {
            'compression_quality': compression_quality,
            'patch_quality': patch_quality,
            'memory_quality': memory_quality,
            'overall_quality': overall_quality,
            'quality_grade': self._grade_quality(overall_quality)
        }
    
    def _grade_quality(self, quality_score: float) -> str:
        """Convert quality score to letter grade."""
        if quality_score >= 0.9:
            return 'A+'
        elif quality_score >= 0.8:
            return 'A'
        elif quality_score >= 0.7:
            return 'B+'
        elif quality_score >= 0.6:
            return 'B'
        elif quality_score >= 0.5:
            return 'C+'
        elif quality_score >= 0.4:
            return 'C'
        else:
            return 'D'
    
    def analyze_compactification_trends(self, compact_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in compactification over time."""
        
        if not compact_history:
            return {}
        
        # Extract time series data
        compression_ratios = []
        memory_efficiencies = []
        patch_counts = []
        quality_scores = []
        
        for compact_data in compact_history:
            metrics = self.compute_metrics(compact_data)
            
            compression_ratios.append(metrics['compression_stats'].compression_ratio)
            memory_efficiencies.append(metrics['memory_profile'].memory_efficiency)
            patch_counts.append(metrics['patch_effectiveness'].patch_count)
            quality_scores.append(metrics['compactification_quality']['overall_quality'])
        
        # Trend analysis
        def compute_trend(values):
            if len(values) < 2:
                return 0.0
            return (values[-1] - values[0]) / len(values)
        
        return {
            'compression_trend': compute_trend(compression_ratios),
            'memory_efficiency_trend': compute_trend(memory_efficiencies),
            'patch_count_trend': compute_trend(patch_counts),
            'quality_trend': compute_trend(quality_scores),
            'stability_score': 1.0 - np.std(quality_scores) if quality_scores else 0.0,
            'improvement_rate': compute_trend(quality_scores),
            'total_iterations': len(compact_history)
        }


# Factory function for easy creation
def create_compactification_analyzer(**kwargs) -> CompactificationAnalyzer:
    """
    Factory function to create compactification analyzer.
    
    Args:
        **kwargs: Additional configuration options
        
    Returns:
        Configured CompactificationAnalyzer
    """
    return CompactificationAnalyzer(**kwargs)


# Export main classes
__all__ = [
    'CompressionStats',
    'PatchEffectiveness',
    'MemoryProfile',
    'CompactificationAnalyzer',
    'create_compactification_analyzer'
]
