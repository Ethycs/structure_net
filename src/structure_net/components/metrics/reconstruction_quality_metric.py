"""
Reconstruction quality metric component.

This component measures how well a compactified network can be
reconstructed, including error, information loss, and fidelity.
"""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import logging
import numpy as np

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class ReconstructionQualityMetric(BaseMetric):
    """
    Measures reconstruction quality of compactified networks.
    
    Evaluates reconstruction error, information loss, fidelity score,
    and structural preservation to determine quality of compression.
    """
    
    def __init__(self, threshold: float = 1e-6, name: str = None):
        """
        Initialize reconstruction quality metric.
        
        Args:
            threshold: Threshold for considering values significant
            name: Optional custom name
        """
        super().__init__(name or "ReconstructionQualityMetric")
        self.threshold = threshold
        self._measurement_schema = {
            "reconstruction_error": float,
            "information_loss": float,
            "fidelity_score": float,
            "structural_preservation": float
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
                "metrics.reconstruction_error",
                "metrics.information_loss",
                "metrics.fidelity_score",
                "metrics.structural_preservation"
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
        Compute reconstruction quality metrics.
        
        Args:
            target: Not used directly
            context: Must contain 'compact_data'
            
        Returns:
            Dictionary containing reconstruction quality measurements
        """
        # Get compact data
        compact_data = context.get('compact_data')
        if compact_data is None:
            raise ValueError("ReconstructionQualityMetric requires 'compact_data' in context")
        
        # Initialize metrics
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
        
        # Compute information loss
        information_loss = self._compute_information_loss(compact_data)
        if information_loss is not None:
            metrics['information_loss'] = information_loss
        
        # Compute structural preservation
        structural_preservation = self._compute_structural_preservation(compact_data)
        if structural_preservation is not None:
            metrics['structural_preservation'] = structural_preservation
        
        self.log(logging.DEBUG, 
                f"Reconstruction: error={metrics['reconstruction_error']:.3f}, "
                f"fidelity={metrics['fidelity_score']:.3f}, "
                f"info_loss={metrics['information_loss']:.3f}")
        
        return metrics
    
    def _compute_information_loss(self, compact_data: Dict[str, Any]) -> Optional[float]:
        """Compute information loss from compression."""
        if 'patches' not in compact_data:
            return None
        
        total_preserved = 0
        total_original = 0
        
        for patch_info in compact_data['patches']:
            if 'data' in patch_info:
                # Count preserved information
                preserved = (patch_info['data'].abs() > self.threshold).sum().item()
                total_preserved += preserved
                
                # If original data is available
                if 'original_data' in patch_info:
                    original = (patch_info['original_data'].abs() > self.threshold).sum().item()
                    total_original += original
                else:
                    # Estimate original from preserved
                    total_original += preserved * 1.2  # Assume 20% loss
        
        if total_original > 0:
            return 1.0 - (total_preserved / total_original)
        
        return None
    
    def _compute_structural_preservation(self, compact_data: Dict[str, Any]) -> Optional[float]:
        """Compute how well structure is preserved."""
        if 'skeleton' not in compact_data:
            return None
        
        skeleton = compact_data['skeleton']
        
        # Measure connectivity preservation
        nonzero_ratio = (skeleton.abs() > self.threshold).float().mean().item()
        
        # Additional structural metrics if available
        if 'structural_metrics' in compact_data:
            metrics = compact_data['structural_metrics']
            # Average various structural preservation indicators
            preservation_scores = []
            
            if 'connectivity_preserved' in metrics:
                preservation_scores.append(metrics['connectivity_preserved'])
            if 'layer_structure_preserved' in metrics:
                preservation_scores.append(metrics['layer_structure_preserved'])
            
            if preservation_scores:
                return np.mean(preservation_scores)
        
        return nonzero_ratio