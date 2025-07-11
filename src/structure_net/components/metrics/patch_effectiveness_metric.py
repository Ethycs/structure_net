"""
Patch effectiveness metric component.

This component analyzes the effectiveness of patch-based compression,
measuring patch density, coverage, and information preservation.
"""

from typing import Dict, Any, Union, Optional, List
import torch
import torch.nn as nn
import logging
import numpy as np

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class PatchEffectivenessMetric(BaseMetric):
    """
    Measures effectiveness of patch-based compression.
    
    Analyzes how well patches preserve information, their coverage
    of the network, and their density characteristics.
    """
    
    def __init__(self, density_threshold: float = 1e-6, name: str = None):
        """
        Initialize patch effectiveness metric.
        
        Args:
            density_threshold: Threshold for considering values non-zero
            name: Optional custom name
        """
        super().__init__(name or "PatchEffectivenessMetric")
        self.density_threshold = density_threshold
        self._measurement_schema = {
            "patch_count": int,
            "avg_patch_density": float,
            "patch_coverage": float,
            "information_preservation": float,
            "reconstruction_quality": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"compact_data"},
            provided_outputs={
                "metrics.patch_count",
                "metrics.avg_patch_density",
                "metrics.patch_coverage",
                "metrics.information_preservation",
                "metrics.reconstruction_quality"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute patch effectiveness metrics.
        
        Args:
            target: Not used directly
            context: Must contain 'compact_data'
            
        Returns:
            Dictionary containing patch effectiveness measurements
        """
        # Get compact data
        compact_data = context.get('compact_data')
        if compact_data is None:
            raise ValueError("PatchEffectivenessMetric requires 'compact_data' in context")
        
        # Check if patches exist
        if 'patches' not in compact_data:
            return self._empty_metrics()
        
        patches = compact_data['patches']
        patch_count = len(patches)
        
        if patch_count == 0:
            return self._empty_metrics()
        
        # Analyze patch densities
        densities = []
        total_patch_elements = 0
        total_nonzero_elements = 0
        
        for patch_info in patches:
            if 'data' in patch_info:
                patch_data = patch_info['data']
                total_elements = patch_data.numel()
                nonzero_elements = (patch_data.abs() > self.density_threshold).sum().item()
                
                if total_elements > 0:
                    density = nonzero_elements / total_elements
                    densities.append(density)
                
                total_patch_elements += total_elements
                total_nonzero_elements += nonzero_elements
        
        # Calculate metrics
        avg_patch_density = np.mean(densities) if densities else 0.0
        
        # Patch coverage
        if 'original_size' in compact_data and compact_data['original_size'] > 0:
            patch_coverage = total_patch_elements / compact_data['original_size']
        else:
            patch_coverage = 0.5  # Default estimate
        
        # Information preservation
        if total_patch_elements > 0:
            information_preservation = total_nonzero_elements / total_patch_elements
        else:
            information_preservation = 0.0
        
        # Reconstruction quality
        reconstruction_quality = self._compute_reconstruction_quality(compact_data)
        
        self.log(logging.DEBUG, 
                f"Patches: count={patch_count}, avg_density={avg_patch_density:.3f}, "
                f"coverage={patch_coverage:.3f}")
        
        return {
            "patch_count": patch_count,
            "avg_patch_density": avg_patch_density,
            "patch_coverage": patch_coverage,
            "information_preservation": information_preservation,
            "reconstruction_quality": reconstruction_quality
        }
    
    def _compute_reconstruction_quality(self, compact_data: Dict[str, Any]) -> float:
        """Compute reconstruction quality score."""
        # If reconstruction error is available, convert to quality
        if 'reconstruction_error' in compact_data:
            error = compact_data['reconstruction_error']
            if error > 0:
                return 1.0 / (1.0 + error)
            else:
                return 1.0
        else:
            # Default estimate based on other factors
            return 0.8
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when no patches exist."""
        return {
            "patch_count": 0,
            "avg_patch_density": 0.0,
            "patch_coverage": 0.0,
            "information_preservation": 0.0,
            "reconstruction_quality": 0.0
        }