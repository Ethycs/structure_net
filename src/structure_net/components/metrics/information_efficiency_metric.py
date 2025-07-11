"""
Information efficiency metric component.

This component measures how efficiently information flows through neural network
layers, identifying bottlenecks and redundancies.
"""

from typing import Dict, Any, Union, Optional, List
import torch
import torch.nn as nn
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class InformationEfficiencyMetric(BaseMetric):
    """
    Measures information flow efficiency in neural networks.
    
    Analyzes how much information is preserved, lost, or transformed
    as it flows through network layers.
    """
    
    def __init__(self, tolerance: float = 1e-6, name: str = None):
        """
        Initialize information efficiency metric.
        
        Args:
            tolerance: Numerical tolerance
            name: Optional custom name
        """
        super().__init__(name or "InformationEfficiencyMetric")
        self.tolerance = tolerance
        self._measurement_schema = {
            "flow_efficiency": float,
            "information_loss": float,
            "bottleneck_ratio": float,
            "redundancy_ratio": float,
            "capacity_utilization": float,
            "information_density": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"chain_data"},  # Requires chain complex analysis
            provided_outputs={
                "metrics.flow_efficiency",
                "metrics.information_loss",
                "metrics.bottleneck_ratio",
                "metrics.redundancy_ratio",
                "metrics.capacity_utilization",
                "metrics.information_density"
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
        Compute information efficiency metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'chain_data' with rank, kernel, image info
            
        Returns:
            Dictionary containing efficiency measurements
        """
        # Get chain data from context
        chain_data = context.get('chain_data')
        if chain_data is None:
            raise ValueError("InformationEfficiencyMetric requires 'chain_data' in context")
        
        # Extract or compute required values
        rank = chain_data.get('rank', 0)
        kernel_dim = chain_data.get('kernel_dimension', 0)
        image_dim = chain_data.get('image_dimension', 0)
        homology_dim = chain_data.get('homology_dimension', 0)
        
        # Total dimensions
        if hasattr(chain_data, 'kernel_basis'):
            total_dim = chain_data.kernel_basis.shape[0]
        else:
            total_dim = kernel_dim + rank  # Approximation
        
        # Additional context data
        input_dim = context.get('input_dimension', total_dim)
        output_dim = context.get('output_dimension', rank)
        
        return self._analyze_information_flow(
            rank, kernel_dim, image_dim, homology_dim,
            total_dim, input_dim, output_dim
        )
    
    def _analyze_information_flow(self, rank: int, kernel_dim: int, 
                                 image_dim: int, homology_dim: int,
                                 total_dim: int, input_dim: int, 
                                 output_dim: int) -> Dict[str, Any]:
        """
        Analyze information flow characteristics.
        
        Args:
            rank: Effective rank of transformation
            kernel_dim: Dimension of kernel (lost information)
            image_dim: Dimension of image (transmitted information)
            homology_dim: Dimension of homology (unique information)
            total_dim: Total dimension of the space
            input_dim: Input dimension
            output_dim: Output dimension
            
        Returns:
            Dictionary with efficiency metrics
        """
        # Prevent division by zero
        total_dim = max(total_dim, 1)
        input_dim = max(input_dim, 1)
        output_dim = max(output_dim, 1)
        
        # Flow efficiency: how much information passes through
        flow_efficiency = rank / total_dim
        
        # Information loss: fraction lost in kernel
        information_loss = kernel_dim / total_dim
        
        # Bottleneck ratio: how constrained is the flow
        bottleneck_ratio = kernel_dim / total_dim
        
        # Redundancy ratio: information that's not unique
        # High redundancy means low homology dimension
        if total_dim > kernel_dim:
            redundancy_ratio = 1.0 - (homology_dim / (total_dim - kernel_dim))
        else:
            redundancy_ratio = 1.0
        
        # Capacity utilization: how much of available capacity is used
        max_capacity = min(input_dim, output_dim)
        capacity_utilization = rank / max_capacity if max_capacity > 0 else 0.0
        
        # Information density: information per dimension
        information_density = rank / output_dim
        
        self.log(logging.DEBUG, 
                f"Information flow: efficiency={flow_efficiency:.3f}, "
                f"loss={information_loss:.3f}, utilization={capacity_utilization:.3f}")
        
        return {
            "flow_efficiency": flow_efficiency,
            "information_loss": information_loss,
            "bottleneck_ratio": bottleneck_ratio,
            "redundancy_ratio": max(0.0, redundancy_ratio),  # Ensure non-negative
            "capacity_utilization": capacity_utilization,
            "information_density": information_density
        }
    
    def compute_cascade_effects(self, chain_data: Dict[str, Any], 
                               threshold: float = 0.1) -> Dict[str, Any]:
        """
        Predict cascade effects of information bottlenecks.
        
        Identifies neurons that will be forced to zero due to
        information flow constraints.
        
        Args:
            chain_data: Chain complex analysis data
            threshold: Threshold for identifying affected neurons
            
        Returns:
            Dictionary with cascade predictions
        """
        result = self._compute_metric(None, EvolutionContext({'chain_data': chain_data}))
        
        # Predict cascade zeros based on kernel structure
        kernel_basis = chain_data.get('kernel_basis')
        if kernel_basis is None or not isinstance(kernel_basis, torch.Tensor):
            return {
                **result,
                "cascade_neurons": [],
                "cascade_severity": 0.0
            }
        
        # Neurons strongly aligned with kernel will cascade to zero
        kernel_alignment = torch.abs(kernel_basis).max(dim=1)[0]
        cascade_mask = kernel_alignment > threshold
        cascade_neurons = torch.where(cascade_mask)[0].tolist()
        
        # Cascade severity
        total_neurons = kernel_basis.shape[0]
        cascade_severity = len(cascade_neurons) / total_neurons if total_neurons > 0 else 0.0
        
        return {
            **result,
            "cascade_neurons": cascade_neurons,
            "cascade_severity": cascade_severity,
            "cascade_prediction_confidence": min(1.0, kernel_alignment.mean().item() * 2)
        }
    
    def compute_layer_efficiency_profile(self, 
                                       layer_chain_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute efficiency profile across multiple layers.
        
        Args:
            layer_chain_data: List of chain data for each layer
            
        Returns:
            Dictionary with efficiency trends and patterns
        """
        if not layer_chain_data:
            return {}
        
        # Collect efficiency metrics for each layer
        efficiencies = []
        losses = []
        utilizations = []
        bottlenecks = []
        
        for i, chain_data in enumerate(layer_chain_data):
            metrics = self._compute_metric(
                None, 
                EvolutionContext({'chain_data': chain_data})
            )
            efficiencies.append(metrics["flow_efficiency"])
            losses.append(metrics["information_loss"])
            utilizations.append(metrics["capacity_utilization"])
            bottlenecks.append(metrics["bottleneck_ratio"])
        
        # Analyze trends
        efficiency_trend = self._compute_trend(efficiencies)
        loss_trend = self._compute_trend(losses)
        
        # Find worst bottleneck
        max_bottleneck_idx = bottlenecks.index(max(bottlenecks))
        min_efficiency_idx = efficiencies.index(min(efficiencies))
        
        # Overall network efficiency (geometric mean)
        overall_efficiency = 1.0
        for eff in efficiencies:
            overall_efficiency *= eff
        overall_efficiency = overall_efficiency ** (1.0 / len(efficiencies))
        
        return {
            "layer_efficiencies": efficiencies,
            "layer_losses": losses,
            "layer_utilizations": utilizations,
            "layer_bottlenecks": bottlenecks,
            "overall_efficiency": overall_efficiency,
            "efficiency_trend": efficiency_trend,
            "loss_trend": loss_trend,
            "worst_bottleneck_layer": max_bottleneck_idx,
            "least_efficient_layer": min_efficiency_idx,
            "num_layers": len(layer_chain_data)
        }
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend in values (increasing, decreasing, stable)."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression
        x = torch.arange(len(values), dtype=torch.float32)
        y = torch.tensor(values, dtype=torch.float32)
        
        # Compute slope
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        
        if denominator > 0:
            slope = numerator / denominator
            
            # Determine trend based on slope magnitude
            if slope > 0.05:
                return "increasing"
            elif slope < -0.05:
                return "decreasing"
            else:
                return "stable"
        else:
            return "stable"