"""
Persistence metric component.

This component computes persistence diagrams and related topological persistence
features for neural network weight matrices.
"""

from typing import Dict, Any, Union, Optional, List, Tuple
import torch
import torch.nn as nn
import logging
from dataclasses import dataclass

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


@dataclass
class PersistenceFeature:
    """A feature in the persistence diagram."""
    birth: float
    death: float
    dimension: int  # 0 for components, 1 for holes, etc.
    persistence: float  # death - birth
    feature_type: str  # 'component', 'hole', 'void'


class PersistenceMetric(BaseMetric):
    """
    Computes topological persistence for neural network weight matrices.
    
    Persistence diagrams track when topological features appear (birth)
    and disappear (death) as we filter the weight matrix at different thresholds.
    """
    
    def __init__(self, num_thresholds: int = 50, max_dimension: int = 1,
                 name: str = None):
        """
        Initialize persistence metric.
        
        Args:
            num_thresholds: Number of threshold levels for filtration
            max_dimension: Maximum homological dimension to compute
            name: Optional custom name
        """
        super().__init__(name or "PersistenceMetric")
        self.num_thresholds = num_thresholds
        self.max_dimension = max_dimension
        self._measurement_schema = {
            "persistence_features": list,
            "persistence_entropy": float,
            "total_persistence": float,
            "average_lifespan": float,
            "persistence_statistics": dict
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"weight_matrix"},
            provided_outputs={
                "metrics.persistence_features",
                "metrics.persistence_entropy",
                "metrics.total_persistence",
                "metrics.average_lifespan",
                "metrics.persistence_statistics"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute persistence metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'weight_matrix'
            
        Returns:
            Dictionary containing persistence measurements
        """
        # Get weight matrix
        weight_matrix = context.get('weight_matrix')
        if weight_matrix is None:
            if isinstance(target, ILayer):
                weight_matrix = self._extract_weight_matrix(target)
            else:
                raise ValueError("PersistenceMetric requires 'weight_matrix' in context or a layer target")
        
        if weight_matrix.dim() != 2:
            raise ValueError("Weight matrix must be 2D")
        
        # Compute persistence diagram
        persistence_features = self._compute_persistence_diagram(weight_matrix)
        
        # Compute persistence statistics
        persistence_stats = self._compute_persistence_statistics(persistence_features)
        
        # Compute persistence entropy
        persistence_entropy = self._compute_persistence_entropy(persistence_features)
        
        # Total persistence
        total_persistence = sum(f.persistence for f in persistence_features)
        
        # Average lifespan
        avg_lifespan = (total_persistence / len(persistence_features) 
                       if persistence_features else 0.0)
        
        self.log(logging.DEBUG, 
                f"Computed {len(persistence_features)} persistence features, "
                f"entropy: {persistence_entropy:.3f}")
        
        return {
            "persistence_features": [self._feature_to_dict(f) for f in persistence_features],
            "persistence_entropy": persistence_entropy,
            "total_persistence": total_persistence,
            "average_lifespan": avg_lifespan,
            "persistence_statistics": persistence_stats
        }
    
    def _extract_weight_matrix(self, layer: ILayer) -> torch.Tensor:
        """Extract weight matrix from layer."""
        for attr_name in ['weight', 'linear.weight', 'W']:
            if hasattr(layer, attr_name):
                weight = getattr(layer, attr_name)
                if isinstance(weight, torch.Tensor) and weight.dim() >= 2:
                    return weight.flatten(0, -2) if weight.dim() > 2 else weight
        
        raise ValueError(f"Could not extract weight matrix from layer")
    
    def _compute_persistence_diagram(self, weight_matrix: torch.Tensor) -> List[PersistenceFeature]:
        """
        Compute persistence diagram using threshold filtration.
        
        This is a simplified version that tracks connected components
        and holes as we vary the threshold.
        """
        features = []
        
        # Create threshold levels
        abs_weights = weight_matrix.abs()
        thresholds = torch.linspace(0, abs_weights.max().item(), self.num_thresholds)
        
        # Track features across thresholds
        prev_components = 0
        prev_holes = 0
        component_births = {}  # Track when components are born
        hole_births = {}  # Track when holes are born
        
        for i, threshold in enumerate(thresholds):
            # Create binary matrix at this threshold
            binary_matrix = (abs_weights > threshold).float()
            
            # Count topological features
            num_components = self._count_connected_components(binary_matrix)
            num_holes = self._estimate_holes(binary_matrix) if self.max_dimension >= 1 else 0
            
            # Track component births and deaths
            if i == 0:
                # All components born at threshold 0
                for j in range(num_components):
                    component_births[j] = threshold.item()
            else:
                # Components die when they merge
                if num_components < prev_components:
                    # Some components died (merged)
                    num_deaths = prev_components - num_components
                    for j in range(num_deaths):
                        if j in component_births:
                            features.append(PersistenceFeature(
                                birth=component_births[j],
                                death=threshold.item(),
                                dimension=0,
                                persistence=threshold.item() - component_births[j],
                                feature_type='component'
                            ))
                            del component_births[j]
                
                # New components born (splitting)
                elif num_components > prev_components:
                    num_births = num_components - prev_components
                    start_idx = len(component_births)
                    for j in range(num_births):
                        component_births[start_idx + j] = threshold.item()
            
            # Track hole births and deaths
            if self.max_dimension >= 1:
                if num_holes > prev_holes:
                    # New holes born
                    num_births = num_holes - prev_holes
                    start_idx = len(hole_births)
                    for j in range(num_births):
                        hole_births[start_idx + j] = threshold.item()
                
                elif num_holes < prev_holes:
                    # Holes died (filled)
                    num_deaths = prev_holes - num_holes
                    # Remove oldest holes first
                    sorted_births = sorted(hole_births.items(), key=lambda x: x[1])
                    for j in range(min(num_deaths, len(sorted_births))):
                        idx, birth_time = sorted_births[j]
                        features.append(PersistenceFeature(
                            birth=birth_time,
                            death=threshold.item(),
                            dimension=1,
                            persistence=threshold.item() - birth_time,
                            feature_type='hole'
                        ))
                        del hole_births[idx]
            
            prev_components = num_components
            prev_holes = num_holes
        
        # Add features that persist to infinity
        for birth_time in component_births.values():
            features.append(PersistenceFeature(
                birth=birth_time,
                death=float('inf'),
                dimension=0,
                persistence=float('inf'),
                feature_type='component'
            ))
        
        for birth_time in hole_births.values():
            features.append(PersistenceFeature(
                birth=birth_time,
                death=float('inf'),
                dimension=1,
                persistence=float('inf'),
                feature_type='hole'
            ))
        
        # Filter out features with very short persistence
        min_persistence = abs_weights.max().item() * 0.01
        features = [f for f in features if f.persistence > min_persistence or f.persistence == float('inf')]
        
        return features
    
    def _count_connected_components(self, binary_matrix: torch.Tensor) -> int:
        """Count connected components using flood fill."""
        if binary_matrix.sum() == 0:
            return 0
        
        visited = torch.zeros_like(binary_matrix, dtype=torch.bool)
        h, w = binary_matrix.shape
        components = 0
        
        def flood_fill(i, j):
            """Flood fill from position (i, j)."""
            stack = [(i, j)]
            while stack:
                ci, cj = stack.pop()
                if (0 <= ci < h and 0 <= cj < w and 
                    not visited[ci, cj] and binary_matrix[ci, cj] > 0):
                    visited[ci, cj] = True
                    # 8-connectivity
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di != 0 or dj != 0:
                                stack.append((ci + di, cj + dj))
        
        # Find all components
        for i in range(h):
            for j in range(w):
                if binary_matrix[i, j] > 0 and not visited[i, j]:
                    flood_fill(i, j)
                    components += 1
        
        return components
    
    def _estimate_holes(self, binary_matrix: torch.Tensor) -> int:
        """
        Estimate number of 1-dimensional holes using Euler characteristic.
        
        For 2D: χ = V - E + F, and β₁ = 1 - χ + β₀
        This is a simplified heuristic.
        """
        if binary_matrix.sum() == 0:
            return 0
        
        # Count vertices (non-zero entries)
        vertices = binary_matrix.sum().item()
        
        # Count edges (connections between vertices)
        h, w = binary_matrix.shape
        edges = 0
        
        for i in range(h):
            for j in range(w):
                if binary_matrix[i, j] > 0:
                    # Check 4-connectivity for edges
                    if j + 1 < w and binary_matrix[i, j + 1] > 0:
                        edges += 1
                    if i + 1 < h and binary_matrix[i + 1, j] > 0:
                        edges += 1
        
        # Estimate faces (2x2 blocks of all 1s)
        faces = 0
        for i in range(h - 1):
            for j in range(w - 1):
                if (binary_matrix[i, j] > 0 and binary_matrix[i+1, j] > 0 and
                    binary_matrix[i, j+1] > 0 and binary_matrix[i+1, j+1] > 0):
                    faces += 1
        
        # Euler characteristic
        euler_char = int(vertices - edges + faces)
        
        # Get number of components
        num_components = self._count_connected_components(binary_matrix)
        
        # Estimate holes: β₁ ≈ 1 - χ + β₀
        holes = max(0, 1 - euler_char + num_components - 1)
        
        return holes
    
    def _compute_persistence_statistics(self, 
                                      features: List[PersistenceFeature]) -> Dict[str, Any]:
        """Compute statistics about persistence features."""
        if not features:
            return {
                'num_features': 0,
                'by_dimension': {},
                'finite_features': 0,
                'infinite_features': 0
            }
        
        # Count by dimension
        by_dimension = {}
        for dim in range(self.max_dimension + 1):
            dim_features = [f for f in features if f.dimension == dim]
            if dim_features:
                persistences = [f.persistence for f in dim_features 
                              if f.persistence != float('inf')]
                by_dimension[f'dim_{dim}'] = {
                    'count': len(dim_features),
                    'avg_persistence': (sum(persistences) / len(persistences) 
                                      if persistences else 0.0),
                    'max_persistence': max(persistences) if persistences else 0.0
                }
        
        # Count finite vs infinite features
        finite_features = sum(1 for f in features if f.death != float('inf'))
        infinite_features = len(features) - finite_features
        
        return {
            'num_features': len(features),
            'by_dimension': by_dimension,
            'finite_features': finite_features,
            'infinite_features': infinite_features
        }
    
    def _compute_persistence_entropy(self, 
                                   features: List[PersistenceFeature]) -> float:
        """
        Compute persistence entropy.
        
        This measures the complexity of the persistence diagram.
        """
        if not features:
            return 0.0
        
        # Get finite persistences
        persistences = [f.persistence for f in features 
                       if f.persistence != float('inf') and f.persistence > 0]
        
        if not persistences:
            return 0.0
        
        # Normalize to probabilities
        total_persistence = sum(persistences)
        probabilities = [p / total_persistence for p in persistences]
        
        # Compute entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * torch.log2(torch.tensor(p)).item()
        
        return entropy
    
    def _feature_to_dict(self, feature: PersistenceFeature) -> Dict[str, Any]:
        """Convert PersistenceFeature to dictionary."""
        return {
            'birth': feature.birth,
            'death': feature.death if feature.death != float('inf') else 'inf',
            'dimension': feature.dimension,
            'persistence': feature.persistence if feature.persistence != float('inf') else 'inf',
            'feature_type': feature.feature_type
        }