"""
Homological analyzer component.

This analyzer combines multiple homological and topological metrics to provide
comprehensive analysis of neural network information flow and structure.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import logging
from dataclasses import dataclass

from src.structure_net.core import (
    BaseAnalyzer, IModel, ILayer, EvolutionContext, AnalysisReport,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from src.structure_net.components.metrics import (
    ChainComplexMetric, RankMetric, BettiNumberMetric,
    HomologyMetric, InformationEfficiencyMetric
)


@dataclass
class ChainData:
    """Chain complex data for a layer."""
    kernel_basis: torch.Tensor
    image_basis: torch.Tensor
    homology_basis: torch.Tensor
    rank: int
    betti_numbers: List[int]
    information_efficiency: float
    cascade_zeros: Optional[torch.Tensor] = None


class HomologicalAnalyzer(BaseAnalyzer):
    """
    Analyzes homological properties of neural networks.
    
    Combines chain complex analysis, Betti numbers, homology groups,
    and information efficiency metrics to understand network topology
    and information flow.
    """
    
    def __init__(self, tolerance: float = 1e-6,
                 rank_threshold: float = 0.1,
                 bottleneck_threshold: float = 0.2,
                 track_history: bool = True,
                 name: str = None):
        """
        Initialize homological analyzer.
        
        Args:
            tolerance: Numerical tolerance for computations
            rank_threshold: Threshold for rank deficiency detection
            bottleneck_threshold: Threshold for bottleneck identification
            track_history: Whether to track analysis history
            name: Optional custom name
        """
        super().__init__(name or "HomologicalAnalyzer")
        self.tolerance = tolerance
        self.rank_threshold = rank_threshold
        self.bottleneck_threshold = bottleneck_threshold
        self.track_history = track_history
        
        # Initialize metrics
        self._chain_metric = ChainComplexMetric(tolerance=tolerance)
        self._rank_metric = RankMetric(tolerance=tolerance)
        self._betti_metric = BettiNumberMetric(tolerance=tolerance)
        self._homology_metric = HomologyMetric(tolerance=tolerance)
        self._efficiency_metric = InformationEfficiencyMetric(tolerance=tolerance)
        
        # History tracking
        self.chain_history = [] if track_history else None
        
        self._required_metrics = {
            "ChainComplexMetric",
            "RankMetric",
            "BettiNumberMetric",
            "HomologyMetric",
            "InformationEfficiencyMetric"
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(2, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "weight_matrices"},
            provided_outputs={
                "analysis.chain_complexes",
                "analysis.information_bottlenecks",
                "analysis.topological_summary",
                "analysis.layer_design_recommendations",
                "analysis.homological_complexity"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=False,
                parallel_safe=False  # Due to history tracking
            )
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, 
                         context: EvolutionContext) -> Dict[str, Any]:
        """
        Perform comprehensive homological analysis.
        
        Args:
            model: Model to analyze
            report: Analysis report containing metric data
            context: Evolution context with weight matrices
            
        Returns:
            Dictionary containing analysis results
        """
        # Get weight matrices
        weight_matrices = context.get('weight_matrices')
        if weight_matrices is None:
            weight_matrices = self._extract_weight_matrices(model)
            context['weight_matrices'] = weight_matrices
        
        # Analyze each layer's chain complex
        chain_complexes = self._analyze_chain_complexes(weight_matrices, report, context)
        
        # Detect information bottlenecks
        bottlenecks = self._detect_bottlenecks(chain_complexes)
        
        # Create topological summary
        topological_summary = self._create_topological_summary(chain_complexes)
        
        # Generate layer design recommendations
        design_recommendations = self._generate_design_recommendations(
            chain_complexes, bottlenecks
        )
        
        # Compute overall homological complexity
        homological_complexity = self._compute_homological_complexity(chain_complexes)
        
        return {
            "chain_complexes": chain_complexes,
            "information_bottlenecks": bottlenecks,
            "topological_summary": topological_summary,
            "layer_design_recommendations": design_recommendations,
            "homological_complexity": homological_complexity,
            "analysis_summary": self._create_analysis_summary(
                chain_complexes, bottlenecks, topological_summary
            )
        }
    
    def _extract_weight_matrices(self, model: IModel) -> List[torch.Tensor]:
        """Extract weight matrices from model."""
        weight_matrices = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                if module.weight.dim() >= 2:
                    # Flatten higher dimensional weights
                    weight = module.weight.flatten(0, -2) if module.weight.dim() > 2 else module.weight
                    weight_matrices.append(weight)
        
        return weight_matrices
    
    def _analyze_chain_complexes(self, weight_matrices: List[torch.Tensor],
                               report: AnalysisReport,
                               context: EvolutionContext) -> List[ChainData]:
        """Analyze chain complexes for all layers."""
        chain_complexes = []
        prev_chain = None
        
        for i, weight_matrix in enumerate(weight_matrices):
            # Create context for this layer
            layer_context = EvolutionContext({
                'weight_matrix': weight_matrix,
                'layer_index': i
            })
            
            # Run metrics if not in report
            metric_key = f"layer_{i}"
            
            # Chain complex analysis
            if f"ChainComplexMetric_{metric_key}" not in report.metrics:
                chain_result = self._chain_metric.analyze(None, layer_context)
                report.add_metric_data(f"ChainComplexMetric_{metric_key}", chain_result)
            else:
                chain_result = report.get(f"metrics.ChainComplexMetric_{metric_key}")
            
            # Rank analysis
            if f"RankMetric_{metric_key}" not in report.metrics:
                rank_result = self._rank_metric.analyze(None, layer_context)
                report.add_metric_data(f"RankMetric_{metric_key}", rank_result)
            else:
                rank_result = report.get(f"metrics.RankMetric_{metric_key}")
            
            # Betti numbers
            if f"BettiNumberMetric_{metric_key}" not in report.metrics:
                betti_result = self._betti_metric.analyze(None, layer_context)
                report.add_metric_data(f"BettiNumberMetric_{metric_key}", betti_result)
            else:
                betti_result = report.get(f"metrics.BettiNumberMetric_{metric_key}")
            
            # Homology analysis
            homology_context = EvolutionContext({
                'kernel_basis': chain_result['kernel_basis'],
                'prev_image_basis': prev_chain.image_basis if prev_chain else None
            })
            
            if f"HomologyMetric_{metric_key}" not in report.metrics:
                homology_result = self._homology_metric.analyze(None, homology_context)
                report.add_metric_data(f"HomologyMetric_{metric_key}", homology_result)
            else:
                homology_result = report.get(f"metrics.HomologyMetric_{metric_key}")
            
            # Information efficiency
            efficiency_context = EvolutionContext({
                'chain_data': {
                    'rank': rank_result['rank'],
                    'kernel_dimension': chain_result['kernel_dimension'],
                    'image_dimension': chain_result['image_dimension'],
                    'homology_dimension': homology_result['homology_dimension']
                }
            })
            
            if f"InformationEfficiencyMetric_{metric_key}" not in report.metrics:
                efficiency_result = self._efficiency_metric.analyze(None, efficiency_context)
                report.add_metric_data(f"InformationEfficiencyMetric_{metric_key}", efficiency_result)
            else:
                efficiency_result = report.get(f"metrics.InformationEfficiencyMetric_{metric_key}")
            
            # Create ChainData
            chain_data = ChainData(
                kernel_basis=chain_result['kernel_basis'],
                image_basis=chain_result['image_basis'],
                homology_basis=homology_result['homology_basis'],
                rank=rank_result['rank'],
                betti_numbers=betti_result['betti_numbers'],
                information_efficiency=efficiency_result['flow_efficiency'],
                cascade_zeros=self._predict_cascade_zeros(chain_result['kernel_basis'])
            )
            
            chain_complexes.append(chain_data)
            prev_chain = chain_data
            
            # Update history
            if self.track_history and self.chain_history is not None:
                self.chain_history.append(chain_data)
        
        return chain_complexes
    
    def _detect_bottlenecks(self, chain_complexes: List[ChainData]) -> List[Dict[str, Any]]:
        """Detect information bottlenecks using homological analysis."""
        bottlenecks = []
        
        for i, chain_data in enumerate(chain_complexes):
            layer_name = f"layer_{i}"
            
            # Check for rank deficiency
            input_dim = chain_data.kernel_basis.shape[0]
            if input_dim > 0:
                rank_ratio = chain_data.rank / input_dim
                
                if rank_ratio < self.rank_threshold:
                    bottlenecks.append({
                        'type': 'rank_deficiency',
                        'layer': layer_name,
                        'severity': 1.0 - rank_ratio,
                        'metric_value': rank_ratio,
                        'recommendation': 'increase_layer_capacity'
                    })
            
            # Check for large kernel (dead information)
            if input_dim > 0:
                kernel_ratio = chain_data.kernel_basis.shape[1] / input_dim
                if kernel_ratio > self.bottleneck_threshold:
                    bottlenecks.append({
                        'type': 'large_kernel',
                        'layer': layer_name,
                        'severity': kernel_ratio,
                        'metric_value': kernel_ratio,
                        'recommendation': 'add_skip_connections'
                    })
            
            # Check for low information efficiency
            if chain_data.information_efficiency < 0.5:
                bottlenecks.append({
                    'type': 'low_efficiency',
                    'layer': layer_name,
                    'severity': 1.0 - chain_data.information_efficiency,
                    'metric_value': chain_data.information_efficiency,
                    'recommendation': 'restructure_layer'
                })
            
            # Check for topological complexity
            if sum(chain_data.betti_numbers) > 5:
                bottlenecks.append({
                    'type': 'topological_complexity',
                    'layer': layer_name,
                    'severity': sum(chain_data.betti_numbers) / 10.0,
                    'metric_value': sum(chain_data.betti_numbers),
                    'recommendation': 'simplify_topology'
                })
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
        
        return bottlenecks
    
    def _create_topological_summary(self, chain_complexes: List[ChainData]) -> Dict[str, Any]:
        """Create summary of topological properties."""
        if not chain_complexes:
            return {}
        
        # Collect statistics
        ranks = [c.rank for c in chain_complexes]
        efficiencies = [c.information_efficiency for c in chain_complexes]
        betti_numbers = [c.betti_numbers for c in chain_complexes]
        kernel_dims = [c.kernel_basis.shape[1] for c in chain_complexes]
        
        # Overall complexity
        total_betti = sum(sum(betti) for betti in betti_numbers)
        
        # Information preservation
        total_input_dim = sum(c.kernel_basis.shape[0] for c in chain_complexes)
        total_preserved_dim = sum(ranks)
        preservation_ratio = total_preserved_dim / total_input_dim if total_input_dim > 0 else 0
        
        # Topological stability
        betti_variance = self._compute_betti_variance(betti_numbers)
        
        return {
            'layer_ranks': ranks,
            'information_efficiencies': efficiencies,
            'betti_numbers': betti_numbers,
            'kernel_dimensions': kernel_dims,
            'total_homological_complexity': total_betti,
            'information_preservation_ratio': preservation_ratio,
            'average_efficiency': sum(efficiencies) / len(efficiencies) if efficiencies else 0,
            'topological_stability': 1.0 / (1.0 + betti_variance),
            'num_layers': len(chain_complexes)
        }
    
    def _generate_design_recommendations(self, chain_complexes: List[ChainData],
                                       bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations for layer design based on homological analysis."""
        recommendations = []
        
        # Global recommendations based on bottlenecks
        bottleneck_types = set(b['type'] for b in bottlenecks)
        
        if 'rank_deficiency' in bottleneck_types:
            recommendations.append({
                'action': 'increase_model_capacity',
                'reason': 'Multiple layers have rank deficiency',
                'priority': 'high',
                'specific_layers': [b['layer'] for b in bottlenecks if b['type'] == 'rank_deficiency']
            })
        
        if 'large_kernel' in bottleneck_types:
            recommendations.append({
                'action': 'add_residual_connections',
                'reason': 'Large kernels indicate information loss',
                'priority': 'high',
                'specific_layers': [b['layer'] for b in bottlenecks if b['type'] == 'large_kernel']
            })
        
        # Layer-specific recommendations
        for i, chain_data in enumerate(chain_complexes):
            if i < len(chain_complexes) - 1:
                next_layer_design = self._design_next_layer_structure(
                    chain_data, 
                    target_dim=chain_complexes[i+1].kernel_basis.shape[0] if i+1 < len(chain_complexes) else None
                )
                
                recommendations.append({
                    'action': 'optimize_layer_connection',
                    'layer': f"layer_{i}_to_{i+1}",
                    'details': next_layer_design,
                    'priority': 'medium'
                })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
        
        return recommendations
    
    def _design_next_layer_structure(self, prev_chain: ChainData, 
                                   target_dim: Optional[int] = None) -> Dict[str, Any]:
        """Design optimal structure for next layer based on chain analysis."""
        effective_dim = prev_chain.rank
        
        # Avoid connecting from kernel (dead information)
        avoid_indices = prev_chain.cascade_zeros if prev_chain.cascade_zeros is not None else []
        
        # Find information-rich regions
        info_content = torch.norm(prev_chain.image_basis, dim=1)
        info_peaks = self._find_local_maxima(info_content)
        
        # Calculate optimal sparsity
        information_density = effective_dim / prev_chain.kernel_basis.shape[0]
        recommended_sparsity = 0.1 * (1.0 + sum(prev_chain.betti_numbers) / 10.0)
        
        return {
            'effective_input_dim': effective_dim,
            'avoid_connections_from': avoid_indices.tolist() if torch.is_tensor(avoid_indices) else avoid_indices,
            'information_peaks': info_peaks,
            'recommended_sparsity': recommended_sparsity,
            'information_density': information_density,
            'use_skip_connection': prev_chain.information_efficiency < 0.7
        }
    
    def _compute_homological_complexity(self, chain_complexes: List[ChainData]) -> Dict[str, Any]:
        """Compute overall homological complexity metrics."""
        if not chain_complexes:
            return {'complexity': 0.0, 'components': {}}
        
        # Betti complexity
        total_betti = sum(sum(c.betti_numbers) for c in chain_complexes)
        avg_betti = total_betti / len(chain_complexes)
        
        # Rank complexity (variance in ranks)
        ranks = [c.rank for c in chain_complexes]
        rank_variance = torch.var(torch.tensor(ranks, dtype=torch.float32)).item()
        
        # Information flow complexity
        efficiencies = [c.information_efficiency for c in chain_complexes]
        flow_variance = torch.var(torch.tensor(efficiencies, dtype=torch.float32)).item()
        
        # Overall complexity score
        complexity = (
            avg_betti / 5.0 * 0.4 +  # Normalized Betti complexity
            rank_variance / 100.0 * 0.3 +  # Normalized rank variance
            flow_variance * 0.3  # Flow variance
        )
        
        return {
            'complexity': min(1.0, complexity),
            'components': {
                'topological': avg_betti / 5.0,
                'rank_variation': rank_variance / 100.0,
                'flow_variation': flow_variance
            },
            'interpretation': self._interpret_complexity(complexity)
        }
    
    def _create_analysis_summary(self, chain_complexes: List[ChainData],
                               bottlenecks: List[Dict[str, Any]],
                               topological_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create concise summary of analysis."""
        num_severe_bottlenecks = sum(1 for b in bottlenecks if b['severity'] > 0.7)
        avg_efficiency = topological_summary.get('average_efficiency', 0)
        
        health_score = (
            avg_efficiency * 0.5 +
            (1.0 - num_severe_bottlenecks / max(len(chain_complexes), 1)) * 0.3 +
            topological_summary.get('topological_stability', 1.0) * 0.2
        )
        
        return {
            'overall_health': health_score,
            'num_bottlenecks': len(bottlenecks),
            'severe_bottlenecks': num_severe_bottlenecks,
            'average_efficiency': avg_efficiency,
            'health_assessment': self._assess_health(health_score),
            'key_issues': [b['type'] for b in bottlenecks[:3]]  # Top 3 issues
        }
    
    def _predict_cascade_zeros(self, kernel_basis: torch.Tensor) -> torch.Tensor:
        """Predict which neurons will cascade to zero."""
        if kernel_basis.shape[1] == 0:
            return torch.tensor([], device=kernel_basis.device, dtype=torch.long)
        
        # Neurons strongly aligned with kernel
        kernel_alignment = torch.abs(kernel_basis).max(dim=1)[0]
        cascade_threshold = kernel_alignment.mean() + kernel_alignment.std()
        
        return torch.where(kernel_alignment > cascade_threshold)[0]
    
    def _compute_betti_variance(self, betti_numbers: List[List[int]]) -> float:
        """Compute variance in Betti numbers across layers."""
        if not betti_numbers:
            return 0.0
        
        # Pad to same length
        max_len = max(len(b) for b in betti_numbers)
        padded = [b + [0] * (max_len - len(b)) for b in betti_numbers]
        
        # Compute variance for each Betti number
        variances = []
        for i in range(max_len):
            values = [b[i] for b in padded]
            if len(values) > 1:
                var = torch.var(torch.tensor(values, dtype=torch.float32)).item()
                variances.append(var)
        
        return sum(variances) / len(variances) if variances else 0.0
    
    def _find_local_maxima(self, values: torch.Tensor) -> List[int]:
        """Find indices of local maxima."""
        maxima = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                maxima.append(i)
        return maxima
    
    def _interpret_complexity(self, complexity: float) -> str:
        """Interpret complexity score."""
        if complexity < 0.3:
            return "simple"
        elif complexity < 0.6:
            return "moderate"
        elif complexity < 0.8:
            return "complex"
        else:
            return "highly_complex"
    
    def _assess_health(self, health_score: float) -> str:
        """Assess overall health."""
        if health_score > 0.8:
            return "excellent"
        elif health_score > 0.6:
            return "good"
        elif health_score > 0.4:
            return "fair"
        else:
            return "poor"