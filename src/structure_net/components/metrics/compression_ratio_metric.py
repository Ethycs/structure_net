"""
Compression ratio metric component.

This component measures the effectiveness of network compression by
calculating compression ratios and space savings.
"""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class CompressionRatioMetric(BaseMetric):
    """
    Measures compression effectiveness through ratio and space calculations.
    
    Computes how much the network has been compressed relative to its
    original size, including compression ratio, space saved, and
    efficiency scores.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize compression ratio metric.
        
        Args:
            name: Optional custom name
        """
        super().__init__(name or "CompressionRatioMetric")
        self._measurement_schema = {
            "original_size": int,
            "compressed_size": int,
            "compression_ratio": float,
            "space_saved": int,
            "efficiency_score": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"compact_data"},
            optional_inputs={"original_network"},
            provided_outputs={
                "metrics.original_size",
                "metrics.compressed_size",
                "metrics.compression_ratio",
                "metrics.space_saved",
                "metrics.efficiency_score"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute compression ratio metrics.
        
        Args:
            target: Not used directly
            context: Must contain 'compact_data'
            
        Returns:
            Dictionary containing compression measurements
        """
        # Get compact data
        compact_data = context.get('compact_data')
        if compact_data is None:
            raise ValueError("CompressionRatioMetric requires 'compact_data' in context")
        
        original_network = context.get('original_network')
        
        # Calculate compressed size
        compressed_size = self._calculate_compressed_size(compact_data)
        
        # Calculate original size
        original_size = self._calculate_original_size(compact_data, original_network)
        
        # Compute metrics
        if original_size > 0:
            compression_ratio = compressed_size / original_size
            space_saved = original_size - compressed_size
            efficiency_score = 1.0 - compression_ratio
        else:
            compression_ratio = 1.0
            space_saved = 0
            efficiency_score = 0.0
        
        self.log(logging.DEBUG, 
                f"Compression: {original_size} -> {compressed_size} "
                f"(ratio: {compression_ratio:.3f})")
        
        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "space_saved": space_saved,
            "efficiency_score": efficiency_score
        }
    
    def _calculate_compressed_size(self, compact_data: Dict[str, Any]) -> int:
        """Calculate total compressed size."""
        compressed_size = 0
        
        # Count patch data
        if 'patches' in compact_data:
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    compressed_size += patch_info['data'].numel()
        
        # Count skeleton data
        if 'skeleton' in compact_data:
            compressed_size += compact_data['skeleton'].numel()
        
        # Count metadata (rough estimate)
        if 'metadata' in compact_data:
            metadata = compact_data['metadata']
            compressed_size += len(str(metadata))
        
        return compressed_size
    
    def _calculate_original_size(self, compact_data: Dict[str, Any],
                               original_network: Optional[nn.Module]) -> int:
        """Calculate original network size."""
        if original_network is not None:
            # Count actual parameters
            return sum(p.numel() for p in original_network.parameters())
        elif 'original_size' in compact_data:
            # Use stored original size
            return compact_data['original_size']
        else:
            # Conservative estimate
            compressed_size = self._calculate_compressed_size(compact_data)
            return compressed_size * 2