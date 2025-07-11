"""
Rank metric component.

This component measures the effective rank and rank-related properties
of neural network weight matrices.
"""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import numpy as np
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class RankMetric(BaseMetric):
    """
    Measures rank-related properties of weight matrices.
    
    Provides insights into the effective dimensionality of transformations,
    rank deficiency, and stable rank measurements.
    """
    
    def __init__(self, tolerance: float = 1e-6, stable_rank: bool = True, 
                 name: str = None):
        """
        Initialize rank metric.
        
        Args:
            tolerance: Singular value threshold for rank determination
            stable_rank: Whether to compute stable rank
            name: Optional custom name
        """
        super().__init__(name or "RankMetric")
        self.tolerance = tolerance
        self.compute_stable_rank = stable_rank
        self._measurement_schema = {
            "rank": int,
            "rank_ratio": float,
            "rank_deficiency": float,
            "stable_rank": float,
            "condition_number": float,
            "singular_values": list
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
                "metrics.rank",
                "metrics.rank_ratio",
                "metrics.rank_deficiency",
                "metrics.stable_rank",
                "metrics.condition_number",
                "metrics.singular_values"
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
        Compute rank metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'weight_matrix' data
            
        Returns:
            Dictionary containing rank measurements
        """
        # Get weight matrix from context
        weight_matrix = context.get('weight_matrix')
        if weight_matrix is None:
            # Try to extract from layer
            if isinstance(target, ILayer):
                weight_matrix = self._extract_weight_matrix(target)
            else:
                raise ValueError("RankMetric requires 'weight_matrix' in context or a layer target")
        
        if weight_matrix.dim() != 2:
            raise ValueError("Weight matrix must be 2D")
        
        return self._compute_rank_properties(weight_matrix)
    
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
    
    def _compute_rank_properties(self, weight_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Compute comprehensive rank properties.
        
        Args:
            weight_matrix: Weight matrix to analyze
            
        Returns:
            Dictionary with rank metrics
        """
        m, n = weight_matrix.shape
        min_dim = min(m, n)
        
        # Compute singular values
        try:
            _, S, _ = torch.linalg.svd(weight_matrix, full_matrices=False)
        except Exception as e:
            self.log(logging.WARNING, f"SVD failed: {e}, using alternative method")
            # Fallback: compute eigenvalues of A^T A
            if m >= n:
                ATA = weight_matrix.T @ weight_matrix
            else:
                ATA = weight_matrix @ weight_matrix.T
            eigenvalues = torch.linalg.eigvalsh(ATA)
            S = torch.sqrt(torch.abs(eigenvalues))
            S = torch.sort(S, descending=True)[0][:min_dim]
        
        # Numerical rank
        rank = torch.sum(S > self.tolerance).item()
        
        # Rank ratio (fraction of full rank)
        rank_ratio = rank / min_dim
        
        # Rank deficiency
        rank_deficiency = 1.0 - rank_ratio
        
        # Stable rank: ||A||_F^2 / ||A||_2^2
        # More robust to small perturbations than numerical rank
        if self.compute_stable_rank:
            frobenius_norm_sq = torch.sum(S ** 2).item()
            spectral_norm_sq = S[0].item() ** 2 if len(S) > 0 else 0.0
            stable_rank = frobenius_norm_sq / spectral_norm_sq if spectral_norm_sq > 0 else 0.0
        else:
            stable_rank = float(rank)
        
        # Condition number
        if rank > 0:
            # Use only non-zero singular values
            non_zero_S = S[:rank]
            condition_number = (non_zero_S[0] / non_zero_S[-1]).item()
        else:
            condition_number = float('inf')
        
        # Store top singular values for analysis
        num_singular_values = min(10, len(S))
        singular_values = S[:num_singular_values].tolist()
        
        self.log(logging.DEBUG, 
                f"Rank analysis: rank={rank}/{min_dim}, "
                f"stable_rank={stable_rank:.2f}, cond={condition_number:.2e}")
        
        return {
            "rank": rank,
            "rank_ratio": rank_ratio,
            "rank_deficiency": rank_deficiency,
            "stable_rank": stable_rank,
            "condition_number": condition_number,
            "singular_values": singular_values
        }
    
    def compute_rank_collapse_risk(self, weight_matrix: torch.Tensor,
                                  threshold_ratio: float = 0.1) -> Dict[str, Any]:
        """
        Assess risk of rank collapse during training.
        
        Rank collapse occurs when the effective rank drops significantly,
        limiting the model's expressiveness.
        
        Args:
            weight_matrix: Weight matrix to analyze
            threshold_ratio: Ratio of singular values considered at risk
            
        Returns:
            Dictionary with collapse risk metrics
        """
        result = self._compute_rank_properties(weight_matrix)
        
        # Analyze singular value distribution
        S = torch.tensor(result["singular_values"])
        if len(S) == 0:
            return {
                **result,
                "collapse_risk": 1.0,
                "at_risk_dimensions": 0,
                "gap_ratio": 0.0
            }
        
        # Count singular values close to collapse
        threshold = S[0] * threshold_ratio
        at_risk = torch.sum(S < threshold).item()
        at_risk_ratio = at_risk / len(S)
        
        # Compute gap ratio (largest gap in singular values)
        if len(S) > 1:
            gaps = S[:-1] - S[1:]
            max_gap_idx = torch.argmax(gaps)
            gap_ratio = gaps[max_gap_idx].item() / S[0].item()
        else:
            gap_ratio = 0.0
        
        # Overall collapse risk (0 = no risk, 1 = high risk)
        collapse_risk = at_risk_ratio * 0.5 + min(gap_ratio, 1.0) * 0.5
        
        return {
            **result,
            "collapse_risk": collapse_risk,
            "at_risk_dimensions": at_risk,
            "gap_ratio": gap_ratio
        }