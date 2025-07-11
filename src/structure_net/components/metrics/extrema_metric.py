"""
Extrema metric component.

This component detects and analyzes extrema (local maxima, minima, and saddle points)
in neural network weight matrices for topology-aware architecture design.
"""

from typing import Dict, Any, Union, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


@dataclass
class ExtremaPoint:
    """Information about a detected extremum."""
    position: Tuple[int, int]
    value: float
    type: str  # 'maximum', 'minimum', 'saddle'
    gradient_magnitude: float
    importance_score: float
    local_density: float


class ExtremaMetric(BaseMetric):
    """
    Detects and analyzes extrema in weight matrices.
    
    Extrema indicate important structural points in the weight landscape
    that can guide architectural decisions like patch placement.
    """
    
    def __init__(self, patch_size: int = 8, density_threshold: float = 0.15,
                 name: str = None):
        """
        Initialize extrema metric.
        
        Args:
            patch_size: Size of patches for local analysis
            density_threshold: Minimum density for significance
            name: Optional custom name
        """
        super().__init__(name or "ExtremaMetric")
        self.patch_size = patch_size
        self.density_threshold = density_threshold
        self._measurement_schema = {
            "extrema_points": list,
            "num_extrema": int,
            "extrema_density": float,
            "gradient_statistics": dict,
            "importance_distribution": dict
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
                "metrics.extrema_points",
                "metrics.num_extrema",
                "metrics.extrema_density",
                "metrics.gradient_statistics",
                "metrics.importance_distribution"
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
        Compute extrema metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'weight_matrix'
            
        Returns:
            Dictionary containing extrema measurements
        """
        # Get weight matrix
        weight_matrix = context.get('weight_matrix')
        if weight_matrix is None:
            if isinstance(target, ILayer):
                weight_matrix = self._extract_weight_matrix(target)
            else:
                raise ValueError("ExtremaMetric requires 'weight_matrix' in context or a layer target")
        
        if weight_matrix.dim() != 2:
            raise ValueError("Weight matrix must be 2D")
        
        # Detect extrema
        extrema_points = self._detect_extrema(weight_matrix)
        
        # Compute gradient statistics
        gradient_stats = self._compute_gradient_statistics(weight_matrix)
        
        # Analyze importance distribution
        importance_dist = self._analyze_importance_distribution(extrema_points)
        
        # Compute density
        num_extrema = len(extrema_points)
        extrema_density = num_extrema / weight_matrix.numel()
        
        self.log(logging.DEBUG, 
                f"Detected {num_extrema} extrema (density: {extrema_density:.4f})")
        
        return {
            "extrema_points": [self._extrema_to_dict(e) for e in extrema_points],
            "num_extrema": num_extrema,
            "extrema_density": extrema_density,
            "gradient_statistics": gradient_stats,
            "importance_distribution": importance_dist
        }
    
    def _extract_weight_matrix(self, layer: ILayer) -> torch.Tensor:
        """Extract weight matrix from layer."""
        for attr_name in ['weight', 'linear.weight', 'W']:
            if hasattr(layer, attr_name):
                weight = getattr(layer, attr_name)
                if isinstance(weight, torch.Tensor) and weight.dim() >= 2:
                    return weight.flatten(0, -2) if weight.dim() > 2 else weight
        
        for name, param in layer.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                return param.flatten(0, -2) if param.dim() > 2 else param
        
        raise ValueError(f"Could not extract weight matrix from layer {layer.name}")
    
    def _detect_extrema(self, weight_matrix: torch.Tensor) -> List[ExtremaPoint]:
        """Detect extrema in the weight matrix."""
        # Compute gradient magnitude
        grad_magnitude = self._compute_gradient_magnitude(weight_matrix)
        
        # Find local extrema
        extrema_positions = self._find_local_extrema(weight_matrix, grad_magnitude)
        
        # Create ExtremaPoint objects
        extrema_points = []
        for pos, extrema_type in extrema_positions:
            row, col = pos
            
            # Extract local patch for analysis
            patch = self._extract_patch(weight_matrix, row, col)
            
            # Compute properties
            value = weight_matrix[row, col].item()
            grad_mag = grad_magnitude[row, col].item()
            density = (patch.abs() > 1e-6).float().mean().item()
            importance = self._compute_importance_score(patch, grad_mag)
            
            if density >= self.density_threshold:
                extrema_points.append(ExtremaPoint(
                    position=pos,
                    value=value,
                    type=extrema_type,
                    gradient_magnitude=grad_mag,
                    importance_score=importance,
                    local_density=density
                ))
        
        # Sort by importance
        extrema_points.sort(key=lambda x: x.importance_score, reverse=True)
        
        return extrema_points
    
    def _compute_gradient_magnitude(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel operators."""
        # Sobel operators
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=weight_matrix.dtype, device=weight_matrix.device
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=weight_matrix.dtype, device=weight_matrix.device
        )
        
        # Pad weight matrix
        padded = F.pad(weight_matrix, (1, 1, 1, 1), mode='reflect')
        
        # Compute gradients
        grad_x = F.conv2d(
            padded.unsqueeze(0).unsqueeze(0),
            sobel_x.unsqueeze(0).unsqueeze(0),
            padding=0
        ).squeeze()
        
        grad_y = F.conv2d(
            padded.unsqueeze(0).unsqueeze(0),
            sobel_y.unsqueeze(0).unsqueeze(0),
            padding=0
        ).squeeze()
        
        # Magnitude
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return magnitude
    
    def _find_local_extrema(self, weight_matrix: torch.Tensor, 
                           grad_magnitude: torch.Tensor) -> List[Tuple[Tuple[int, int], str]]:
        """Find local extrema (maxima, minima, saddle points)."""
        h, w = weight_matrix.shape
        extrema = []
        
        # Use max pooling to find local maxima
        maxpool = F.max_pool2d(
            weight_matrix.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()
        
        # Use min pooling (via -maxpool(-x)) to find local minima
        minpool = -F.max_pool2d(
            (-weight_matrix).unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()
        
        # Detect extrema
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = weight_matrix[i, j]
                
                # Local maximum
                if center == maxpool[i, j] and grad_magnitude[i, j] > grad_magnitude.mean():
                    extrema.append(((i, j), 'maximum'))
                
                # Local minimum
                elif center == minpool[i, j] and grad_magnitude[i, j] > grad_magnitude.mean():
                    extrema.append(((i, j), 'minimum'))
                
                # Saddle point detection (simplified)
                elif grad_magnitude[i, j] < grad_magnitude.mean() * 0.1:
                    # Low gradient but not extremum - possible saddle
                    neighborhood = weight_matrix[i-1:i+2, j-1:j+2]
                    if self._is_saddle_point(neighborhood):
                        extrema.append(((i, j), 'saddle'))
        
        return extrema
    
    def _is_saddle_point(self, neighborhood: torch.Tensor) -> bool:
        """Check if 3x3 neighborhood indicates a saddle point."""
        center = neighborhood[1, 1]
        
        # Count values higher and lower than center
        higher = (neighborhood > center).sum().item() - 1  # Exclude center
        lower = (neighborhood < center).sum().item()
        
        # Saddle points have both higher and lower neighbors
        return higher >= 2 and lower >= 2
    
    def _extract_patch(self, weight_matrix: torch.Tensor, 
                      row: int, col: int) -> torch.Tensor:
        """Extract patch around given position."""
        h, w = weight_matrix.shape
        
        # Calculate patch boundaries
        row_start = max(0, row - self.patch_size // 2)
        row_end = min(h, row_start + self.patch_size)
        col_start = max(0, col - self.patch_size // 2)
        col_end = min(w, col_start + self.patch_size)
        
        # Adjust if at boundary
        if row_end - row_start < self.patch_size:
            row_start = max(0, row_end - self.patch_size)
        if col_end - col_start < self.patch_size:
            col_start = max(0, col_end - self.patch_size)
        
        return weight_matrix[row_start:row_end, col_start:col_end]
    
    def _compute_importance_score(self, patch: torch.Tensor, 
                                 gradient_magnitude: float) -> float:
        """Compute importance score for an extremum."""
        # Magnitude score
        magnitude_score = patch.abs().mean().item()
        
        # Variance score (indicates structure)
        variance_score = patch.var().item()
        
        # Connectivity score
        binary_patch = (patch.abs() > 1e-6).float()
        connectivity = 0.0
        h, w = binary_patch.shape
        
        for i in range(h):
            for j in range(w):
                if binary_patch[i, j] > 0:
                    # Count neighbors
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if (di != 0 or dj != 0) and 0 <= i+di < h and 0 <= j+dj < w:
                                if binary_patch[i+di, j+dj] > 0:
                                    neighbors += 1
                    connectivity += neighbors / 8.0
        
        connectivity_score = connectivity / binary_patch.sum().item() if binary_patch.sum() > 0 else 0
        
        # Combine scores
        importance = (
            0.3 * magnitude_score +
            0.3 * variance_score +
            0.2 * connectivity_score +
            0.2 * gradient_magnitude
        )
        
        return importance
    
    def _compute_gradient_statistics(self, weight_matrix: torch.Tensor) -> Dict[str, Any]:
        """Compute gradient-based statistics."""
        grad_magnitude = self._compute_gradient_magnitude(weight_matrix)
        
        return {
            'mean': grad_magnitude.mean().item(),
            'std': grad_magnitude.std().item(),
            'max': grad_magnitude.max().item(),
            'min': grad_magnitude.min().item(),
            'high_gradient_ratio': (grad_magnitude > grad_magnitude.mean()).float().mean().item()
        }
    
    def _analyze_importance_distribution(self, extrema_points: List[ExtremaPoint]) -> Dict[str, Any]:
        """Analyze the distribution of importance scores."""
        if not extrema_points:
            return {
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0,
                'top_10_percent_threshold': 0.0
            }
        
        importance_scores = [e.importance_score for e in extrema_points]
        importance_tensor = torch.tensor(importance_scores)
        
        return {
            'mean': importance_tensor.mean().item(),
            'std': importance_tensor.std().item() if len(importance_scores) > 1 else 0.0,
            'max': importance_tensor.max().item(),
            'min': importance_tensor.min().item(),
            'top_10_percent_threshold': torch.quantile(importance_tensor, 0.9).item()
        }
    
    def _extrema_to_dict(self, extrema: ExtremaPoint) -> Dict[str, Any]:
        """Convert ExtremaPoint to dictionary."""
        return {
            'position': extrema.position,
            'value': extrema.value,
            'type': extrema.type,
            'gradient_magnitude': extrema.gradient_magnitude,
            'importance_score': extrema.importance_score,
            'local_density': extrema.local_density
        }