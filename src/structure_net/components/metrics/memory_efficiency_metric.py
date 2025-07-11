"""
Memory efficiency metric component.

This component analyzes memory usage patterns in compactified networks,
measuring efficiency across different storage components.
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


class MemoryEfficiencyMetric(BaseMetric):
    """
    Measures memory usage efficiency in compactified networks.
    
    Analyzes how memory is distributed across patches, skeleton,
    and metadata, computing overall memory efficiency.
    """
    
    def __init__(self, bytes_per_element: int = 4, name: str = None):
        """
        Initialize memory efficiency metric.
        
        Args:
            bytes_per_element: Bytes per tensor element (4 for float32)
            name: Optional custom name
        """
        super().__init__(name or "MemoryEfficiencyMetric")
        self.bytes_per_element = bytes_per_element
        self._measurement_schema = {
            "total_memory": int,
            "patch_memory": int,
            "skeleton_memory": int,
            "metadata_memory": int,
            "memory_efficiency": float
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
                "metrics.total_memory",
                "metrics.patch_memory",
                "metrics.skeleton_memory",
                "metrics.metadata_memory",
                "metrics.memory_efficiency"
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
        Compute memory efficiency metrics.
        
        Args:
            target: Not used directly
            context: Must contain 'compact_data'
            
        Returns:
            Dictionary containing memory efficiency measurements
        """
        # Get compact data
        compact_data = context.get('compact_data')
        if compact_data is None:
            raise ValueError("MemoryEfficiencyMetric requires 'compact_data' in context")
        
        # Calculate memory usage for different components
        patch_memory = self._calculate_patch_memory(compact_data)
        skeleton_memory = self._calculate_skeleton_memory(compact_data)
        metadata_memory = self._calculate_metadata_memory(compact_data)
        
        total_memory = patch_memory + skeleton_memory + metadata_memory
        
        # Calculate memory efficiency
        memory_efficiency = self._calculate_efficiency(compact_data, total_memory)
        
        self.log(logging.DEBUG, 
                f"Memory usage: total={total_memory}, patches={patch_memory}, "
                f"skeleton={skeleton_memory}, efficiency={memory_efficiency:.3f}")
        
        return {
            "total_memory": total_memory,
            "patch_memory": patch_memory,
            "skeleton_memory": skeleton_memory,
            "metadata_memory": metadata_memory,
            "memory_efficiency": memory_efficiency
        }
    
    def _calculate_patch_memory(self, compact_data: Dict[str, Any]) -> int:
        """Calculate memory used by patches."""
        patch_memory = 0
        
        if 'patches' in compact_data:
            for patch_info in compact_data['patches']:
                if 'data' in patch_info:
                    # Data memory
                    patch_memory += patch_info['data'].numel() * self.bytes_per_element
                    # Metadata overhead (position, size, etc.)
                    patch_memory += 64  # Rough estimate
        
        return patch_memory
    
    def _calculate_skeleton_memory(self, compact_data: Dict[str, Any]) -> int:
        """Calculate memory used by skeleton."""
        if 'skeleton' in compact_data:
            return compact_data['skeleton'].numel() * self.bytes_per_element
        return 0
    
    def _calculate_metadata_memory(self, compact_data: Dict[str, Any]) -> int:
        """Calculate memory used by metadata."""
        if 'metadata' in compact_data:
            # Rough estimate based on string representation
            return len(str(compact_data['metadata'])) * 2
        return 0
    
    def _calculate_efficiency(self, compact_data: Dict[str, Any], 
                            total_memory: int) -> float:
        """Calculate memory efficiency score."""
        if 'original_size' in compact_data:
            original_memory = compact_data['original_size'] * self.bytes_per_element
            if original_memory > 0:
                return 1.0 - (total_memory / original_memory)
        
        # Default estimate
        return 0.5