"""
Chain complex metric component.

This component computes chain complex properties of neural network layers,
including kernel (nullspace) and image (column space) analysis.
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


class ChainComplexMetric(BaseMetric):
    """
    Computes chain complex properties of weight matrices.
    
    This metric analyzes the kernel (nullspace) and image (column space)
    of weight matrices, providing fundamental linear algebra properties
    that are essential for understanding information flow.
    """
    
    def __init__(self, tolerance: float = 1e-6, name: str = None):
        """
        Initialize chain complex metric.
        
        Args:
            tolerance: Numerical tolerance for rank determination
            name: Optional custom name
        """
        super().__init__(name or "ChainComplexMetric")
        self.tolerance = tolerance
        self._measurement_schema = {
            "kernel_basis": torch.Tensor,
            "image_basis": torch.Tensor,
            "kernel_dimension": int,
            "image_dimension": int,
            "cokernel_dimension": int,
            "numerical_rank": int
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"weight_matrix"},
            provided_outputs={
                "metrics.kernel_basis",
                "metrics.image_basis",
                "metrics.kernel_dimension",
                "metrics.image_dimension",
                "metrics.cokernel_dimension",
                "metrics.numerical_rank"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,  # SVD can be memory intensive
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute chain complex metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'weight_matrix' data
            
        Returns:
            Dictionary containing chain complex measurements
        """
        # Get weight matrix from context
        weight_matrix = context.get('weight_matrix')
        if weight_matrix is None:
            # Try to extract from layer
            if isinstance(target, ILayer):
                weight_matrix = self._extract_weight_matrix(target)
            else:
                raise ValueError("ChainComplexMetric requires 'weight_matrix' in context or a layer target")
        
        if weight_matrix.dim() != 2:
            raise ValueError("Weight matrix must be 2D")
        
        # Compute chain complex properties
        return self._analyze_chain_complex(weight_matrix)
    
    def _extract_weight_matrix(self, layer: ILayer) -> Optional[torch.Tensor]:
        """Extract weight matrix from a layer."""
        # Try common weight attribute names
        for attr_name in ['weight', 'linear.weight', 'W']:
            if hasattr(layer, attr_name):
                weight = getattr(layer, attr_name)
                if isinstance(weight, torch.Tensor) and weight.dim() == 2:
                    return weight
        
        # Try parameters
        for name, param in layer.named_parameters():
            if 'weight' in name and param.dim() == 2:
                return param
        
        raise ValueError(f"Could not extract weight matrix from layer {layer.name}")
    
    def _analyze_chain_complex(self, weight_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Perform chain complex analysis using SVD.
        
        Args:
            weight_matrix: Weight matrix to analyze (m x n)
            
        Returns:
            Dictionary with kernel basis, image basis, and dimensions
        """
        m, n = weight_matrix.shape
        
        # Singular Value Decomposition for numerical stability
        try:
            U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=True)
        except Exception as e:
            self.log(logging.WARNING, f"SVD failed: {e}, using fallback")
            # Fallback to economy SVD
            U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
            # Pad to full matrices
            if U.shape[1] < m:
                U = torch.cat([U, torch.zeros(m, m - U.shape[1], device=U.device)], dim=1)
            if Vt.shape[0] < n:
                Vt = torch.cat([Vt, torch.zeros(n - Vt.shape[0], n, device=Vt.device)], dim=0)
        
        # Determine numerical rank
        rank = torch.sum(S > self.tolerance).item()
        
        # Compute kernel basis (nullspace of A)
        # Kernel = span of right singular vectors with zero singular values
        if rank < n:
            kernel_basis = Vt[rank:, :].T  # Columns are basis vectors
        else:
            kernel_basis = torch.zeros(n, 0, device=weight_matrix.device)
        
        # Compute image basis (column space of A)
        # Image = span of left singular vectors with non-zero singular values
        if rank > 0:
            image_basis = U[:, :rank]
        else:
            image_basis = torch.zeros(m, 0, device=weight_matrix.device)
        
        # Dimensions
        kernel_dim = kernel_basis.shape[1]
        image_dim = image_basis.shape[1]
        cokernel_dim = m - image_dim  # Dimension of cokernel (quotient of codomain by image)
        
        self.log(logging.DEBUG, 
                f"Chain complex: rank={rank}, ker_dim={kernel_dim}, "
                f"im_dim={image_dim}, coker_dim={cokernel_dim}")
        
        return {
            "kernel_basis": kernel_basis,
            "image_basis": image_basis,
            "kernel_dimension": kernel_dim,
            "image_dimension": image_dim,
            "cokernel_dimension": cokernel_dim,
            "numerical_rank": rank
        }
    
    def compute_exact_sequence_properties(self, 
                                        weight_matrix: torch.Tensor,
                                        prev_weight_matrix: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Compute properties related to exact sequences.
        
        For a sequence ... → A → B → C → ...
        Exactness at B means im(A→B) = ker(B→C)
        
        Args:
            weight_matrix: Current layer's weight matrix (B→C)
            prev_weight_matrix: Previous layer's weight matrix (A→B)
            
        Returns:
            Dictionary with exactness measures
        """
        # Analyze current layer
        current_analysis = self._analyze_chain_complex(weight_matrix)
        
        if prev_weight_matrix is None:
            return {
                **current_analysis,
                "exactness": None,
                "homology_dimension": None
            }
        
        # Analyze previous layer
        prev_analysis = self._analyze_chain_complex(prev_weight_matrix)
        
        # Check exactness: im(prev) ?= ker(current)
        # This is measured by the dimension of homology H = ker/im
        prev_image = prev_analysis["image_basis"]
        current_kernel = current_analysis["kernel_basis"]
        
        # Project kernel onto orthogonal complement of image
        if prev_image.shape[1] > 0 and current_kernel.shape[1] > 0:
            # Ensure dimensions match
            if prev_image.shape[0] == current_kernel.shape[0]:
                Q, _ = torch.linalg.qr(prev_image)
                proj = torch.eye(current_kernel.shape[0], device=current_kernel.device) - Q @ Q.T
                homology = proj @ current_kernel
                
                # Count non-zero dimensions
                homology_norms = torch.norm(homology, dim=0)
                homology_dim = torch.sum(homology_norms > self.tolerance).item()
                
                # Exactness measure (0 = exact, 1 = maximally inexact)
                max_dim = min(prev_image.shape[1], current_kernel.shape[1])
                exactness = 1.0 - (homology_dim / max_dim if max_dim > 0 else 0.0)
            else:
                homology_dim = current_kernel.shape[1]
                exactness = 0.0  # Cannot be exact if dimensions don't match
        else:
            homology_dim = current_kernel.shape[1]
            exactness = 1.0 if homology_dim == 0 else 0.0
        
        return {
            **current_analysis,
            "exactness": exactness,
            "homology_dimension": homology_dim
        }