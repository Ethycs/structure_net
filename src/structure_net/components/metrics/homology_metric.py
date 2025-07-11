"""
Homology metric component.

This component computes homology groups as quotient spaces H = ker(∂) / im(∂_{+1}),
identifying the "true" information content that's not inherited from previous layers.
"""

from typing import Dict, Any, Union, Optional, Tuple
import torch
import torch.nn as nn
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class HomologyMetric(BaseMetric):
    """
    Computes homology groups for chain complexes in neural networks.
    
    Homology identifies information that is neither passed through (image)
    nor lost (kernel), representing the layer's unique contribution.
    """
    
    def __init__(self, tolerance: float = 1e-6, name: str = None):
        """
        Initialize homology metric.
        
        Args:
            tolerance: Numerical tolerance for computations
            name: Optional custom name
        """
        super().__init__(name or "HomologyMetric")
        self.tolerance = tolerance
        self._measurement_schema = {
            "homology_basis": torch.Tensor,
            "homology_dimension": int,
            "quotient_dimension": int,
            "exactness_measure": float,
            "information_persistence": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"kernel_basis", "prev_image_basis"},
            provided_outputs={
                "metrics.homology_basis",
                "metrics.homology_dimension",
                "metrics.quotient_dimension",
                "metrics.exactness_measure",
                "metrics.information_persistence"
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
        Compute homology metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'kernel_basis' and optionally 'prev_image_basis'
            
        Returns:
            Dictionary containing homology measurements
        """
        # Get required data from context
        kernel_basis = context.get('kernel_basis')
        if kernel_basis is None:
            raise ValueError("HomologyMetric requires 'kernel_basis' in context")
        
        # Previous layer's image basis (optional)
        prev_image_basis = context.get('prev_image_basis')
        
        # Chain data for sequence analysis (optional)
        chain_data = context.get('chain_data', {})
        
        return self._compute_homology(kernel_basis, prev_image_basis, chain_data)
    
    def _compute_homology(self, kernel_basis: torch.Tensor, 
                         prev_image_basis: Optional[torch.Tensor],
                         chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute homology as quotient space H = ker(∂) / im(∂_{+1}).
        
        Args:
            kernel_basis: Kernel (nullspace) of current layer
            prev_image_basis: Image (column space) of previous layer
            chain_data: Additional chain complex data
            
        Returns:
            Dictionary with homology metrics
        """
        # If no previous image, homology equals kernel
        if prev_image_basis is None or prev_image_basis.shape[1] == 0:
            homology_basis = kernel_basis
            homology_dim = kernel_basis.shape[1]
            quotient_dim = 0
            exactness = 0.0 if homology_dim > 0 else 1.0
            
            return {
                "homology_basis": homology_basis,
                "homology_dimension": homology_dim,
                "quotient_dimension": quotient_dim,
                "exactness_measure": exactness,
                "information_persistence": 1.0
            }
        
        # If kernel is empty, homology is trivial
        if kernel_basis.shape[1] == 0:
            return {
                "homology_basis": torch.zeros(kernel_basis.shape[0], 0, 
                                             device=kernel_basis.device),
                "homology_dimension": 0,
                "quotient_dimension": 0,
                "exactness_measure": 1.0,
                "information_persistence": 0.0
            }
        
        # Compute quotient space: ker ∩ (im)⊥
        try:
            # Orthogonalize previous image basis
            Q, R = torch.linalg.qr(prev_image_basis)
            
            # Project kernel onto orthogonal complement of image
            # P_orth = I - QQ^T
            proj_matrix = torch.eye(kernel_basis.shape[0], 
                                  device=kernel_basis.device) - Q @ Q.T
            
            # Homology basis = projection of kernel onto (im)⊥
            homology_basis_raw = proj_matrix @ kernel_basis
            
            # Remove near-zero vectors (numerical cleanup)
            norms = torch.norm(homology_basis_raw, dim=0)
            significant_mask = norms > self.tolerance
            homology_basis = homology_basis_raw[:, significant_mask]
            
            # Re-orthogonalize homology basis
            if homology_basis.shape[1] > 0:
                homology_basis, _ = torch.linalg.qr(homology_basis)
            
        except Exception as e:
            self.log(logging.WARNING, f"Homology computation failed: {e}")
            # Fallback to kernel
            homology_basis = kernel_basis
        
        # Compute dimensions
        homology_dim = homology_basis.shape[1]
        kernel_dim = kernel_basis.shape[1]
        image_dim = prev_image_basis.shape[1]
        
        # Quotient dimension: how much of kernel is "killed" by image
        quotient_dim = kernel_dim - homology_dim
        
        # Exactness measure: 1 if exact (H=0), 0 if maximally inexact
        max_homology = min(kernel_dim, kernel_basis.shape[0] - image_dim)
        exactness = 1.0 - (homology_dim / max_homology if max_homology > 0 else 0.0)
        
        # Information persistence: how much information survives
        persistence = homology_dim / kernel_dim if kernel_dim > 0 else 0.0
        
        self.log(logging.DEBUG, 
                f"Homology: dim={homology_dim}, exactness={exactness:.3f}, "
                f"persistence={persistence:.3f}")
        
        return {
            "homology_basis": homology_basis,
            "homology_dimension": homology_dim,
            "quotient_dimension": quotient_dim,
            "exactness_measure": exactness,
            "information_persistence": persistence
        }
    
    def compute_relative_homology(self, kernel_basis: torch.Tensor,
                                 image_basis: torch.Tensor,
                                 subspace: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Compute relative homology H(X, A) for a subspace A ⊂ X.
        
        This captures information in X that's not in A.
        
        Args:
            kernel_basis: Kernel of the main space
            image_basis: Image to quotient by
            subspace: Subspace for relative homology
            
        Returns:
            Dictionary with relative homology data
        """
        # Standard homology
        result = self._compute_homology(kernel_basis, image_basis, {})
        
        if subspace is None:
            return result
        
        # Compute relative homology
        # H(X, A) ≈ H(X/A)
        try:
            # Project out the subspace
            if subspace.shape[1] > 0:
                Q_sub, _ = torch.linalg.qr(subspace)
                proj = torch.eye(kernel_basis.shape[0], 
                               device=kernel_basis.device) - Q_sub @ Q_sub.T
                relative_kernel = proj @ kernel_basis
                
                # Compute homology of quotient
                relative_result = self._compute_homology(relative_kernel, image_basis, {})
                
                result.update({
                    "relative_homology_basis": relative_result["homology_basis"],
                    "relative_homology_dimension": relative_result["homology_dimension"],
                    "relative_persistence": relative_result["information_persistence"]
                })
            
        except Exception as e:
            self.log(logging.WARNING, f"Relative homology computation failed: {e}")
        
        return result
    
    def compute_homology_sequence(self, chain_sequence: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Compute homology for a sequence of chain complexes.
        
        Useful for analyzing information flow through multiple layers.
        
        Args:
            chain_sequence: List of dicts with 'kernel_basis' and 'image_basis'
            
        Returns:
            Dictionary with sequence homology analysis
        """
        if not chain_sequence:
            return {}
        
        homology_dimensions = []
        exactness_measures = []
        persistence_values = []
        
        # First layer (no previous image)
        first_result = self._compute_homology(
            chain_sequence[0].get('kernel_basis'),
            None,
            {}
        )
        homology_dimensions.append(first_result["homology_dimension"])
        exactness_measures.append(first_result["exactness_measure"])
        persistence_values.append(first_result["information_persistence"])
        
        # Subsequent layers
        for i in range(1, len(chain_sequence)):
            result = self._compute_homology(
                chain_sequence[i].get('kernel_basis'),
                chain_sequence[i-1].get('image_basis'),
                {}
            )
            homology_dimensions.append(result["homology_dimension"])
            exactness_measures.append(result["exactness_measure"])
            persistence_values.append(result["information_persistence"])
        
        # Analyze trends
        total_homology = sum(homology_dimensions)
        avg_exactness = sum(exactness_measures) / len(exactness_measures)
        avg_persistence = sum(persistence_values) / len(persistence_values)
        
        # Detect patterns
        increasing_homology = all(
            homology_dimensions[i] <= homology_dimensions[i+1] 
            for i in range(len(homology_dimensions)-1)
        )
        
        return {
            "homology_dimensions": homology_dimensions,
            "exactness_measures": exactness_measures,
            "persistence_values": persistence_values,
            "total_homology": total_homology,
            "average_exactness": avg_exactness,
            "average_persistence": avg_persistence,
            "increasing_homology": increasing_homology,
            "sequence_length": len(chain_sequence)
        }