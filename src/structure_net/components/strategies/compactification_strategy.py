"""
Compactification Strategy Component.

This strategy determines when and how to apply compactification
based on network analysis and evolution goals.
"""

import torch
from typing import Dict, Any, Set

from ...core import (
    BaseStrategy, ComponentContract, ComponentVersion,
    Maturity, ResourceRequirements, ResourceLevel,
    AnalysisReport, EvolutionContext, EvolutionPlan
)


class CompactificationStrategy(BaseStrategy):
    """
    Strategy for network compactification decisions.
    
    Analyzes network state and proposes compactification plans
    based on:
    - Network size and complexity
    - Performance metrics
    - Memory constraints
    - Homological analysis
    """
    
    def __init__(self,
                 size_threshold: int = 1_000_000,
                 performance_threshold: float = 0.8,
                 memory_limit_mb: int = 1024,
                 enable_homological_guidance: bool = True,
                 name: str = None):
        """
        Initialize compactification strategy.
        
        Args:
            size_threshold: Minimum network size for compactification
            performance_threshold: Minimum performance to maintain
            memory_limit_mb: Memory limit triggering compactification
            enable_homological_guidance: Use homological analysis
            name: Optional custom name
        """
        super().__init__(name or "CompactificationStrategy")
        
        self.size_threshold = size_threshold
        self.performance_threshold = performance_threshold
        self.memory_limit_mb = memory_limit_mb
        self.enable_homological_guidance = enable_homological_guidance
        
        # Required analysis types
        self._required_analysis = {
            'model_stats',
            'performance_metrics',
            'memory_usage'
        }
        
        if enable_homological_guidance:
            self._required_analysis.add('homological_analysis')
        
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs=set(self._required_analysis),
            optional_inputs={'compression_history', 'gradient_analysis'},
            provided_outputs={'compactification_plan'},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Return the component contract."""
        return self._contract
    
    def get_strategy_type(self) -> str:
        """Return strategy type."""
        return 'compactification'
    
    def propose_plan(self, report: AnalysisReport, 
                    context: EvolutionContext) -> EvolutionPlan:
        """
        Propose compactification plan based on analysis.
        
        Args:
            report: Analysis report with network metrics
            context: Evolution context with current state
            
        Returns:
            Evolution plan for compactification
        """
        # Check if compactification is needed
        if not self._should_compactify(report, context):
            return self._create_no_op_plan()
        
        # Determine compactification parameters
        params = self._determine_parameters(report, context)
        
        # Select layers to compactify
        layer_selection = self._select_layers(report, context)
        
        # Create plan
        plan = EvolutionPlan({
            'type': 'compactification',
            'target_sparsity': params['sparsity'],
            'patch_density': params['patch_density'],
            'patch_size': params['patch_size'],
            'layer_names': layer_selection['layers'],
            'preserve_input_highway': params.get('preserve_highway', True)
        })
        
        # Add homological guidance if available
        if self.enable_homological_guidance and 'homological_analysis' in report:
            plan['extrema_data'] = self._extract_extrema_data(
                report['homological_analysis']
            )
        
        # Set plan metadata
        plan.priority = self._calculate_priority(report, context)
        plan.estimated_impact = self._estimate_impact(params, layer_selection)
        plan.created_by = self.name
        
        return plan
    
    def _should_compactify(self, report: AnalysisReport, 
                          context: EvolutionContext) -> bool:
        """Determine if compactification is needed."""
        # Check model size
        model_stats = report.get('model_stats', {})
        total_params = model_stats.get('total_parameters', 0)
        
        if total_params < self.size_threshold:
            return False
        
        # Check memory usage
        memory_usage = report.get('memory_usage', {})
        current_memory_mb = memory_usage.get('total_mb', 0)
        
        if current_memory_mb > self.memory_limit_mb:
            return True
        
        # Check performance vs size ratio
        performance = report.get('performance_metrics', {})
        accuracy = performance.get('accuracy', 1.0)
        
        # Size-adjusted performance metric
        size_factor = total_params / self.size_threshold
        adjusted_performance = accuracy / (1 + 0.1 * size_factor)
        
        if adjusted_performance < self.performance_threshold:
            return True
        
        # Check if explicitly requested
        if context.get('force_compactification', False):
            return True
        
        return False
    
    def _determine_parameters(self, report: AnalysisReport,
                            context: EvolutionContext) -> Dict[str, Any]:
        """Determine compactification parameters."""
        params = {}
        
        # Base parameters
        model_stats = report.get('model_stats', {})
        total_params = model_stats.get('total_parameters', 1)
        
        # Determine target sparsity based on size
        if total_params > 10_000_000:
            params['sparsity'] = 0.02  # 2% for very large networks
        elif total_params > 1_000_000:
            params['sparsity'] = 0.05  # 5% for large networks
        else:
            params['sparsity'] = 0.1   # 10% for smaller networks
        
        # Patch parameters
        params['patch_density'] = 0.2  # 20% density in patches
        params['patch_size'] = 8       # 8x8 patches
        
        # Adjust based on performance requirements
        performance = report.get('performance_metrics', {})
        if performance.get('accuracy', 1.0) < 0.9:
            # Less aggressive compactification for lower performance
            params['sparsity'] *= 2
            params['patch_density'] = 0.3
        
        # Homological adjustments
        if 'homological_analysis' in report:
            homology = report['homological_analysis']
            # Adjust patch size based on topological features
            avg_feature_size = homology.get('average_feature_size', 8)
            params['patch_size'] = max(4, min(16, int(avg_feature_size)))
        
        # Input highway preservation
        params['preserve_highway'] = context.get('preserve_input_highway', True)
        
        return params
    
    def _select_layers(self, report: AnalysisReport,
                      context: EvolutionContext) -> Dict[str, Any]:
        """Select which layers to compactify."""
        selection = {'layers': None, 'criteria': []}
        
        # Get layer information
        layer_stats = report.get('layer_statistics', {})
        if not layer_stats:
            # Compactify all eligible layers
            return selection
        
        # Rank layers by compactification benefit
        layer_scores = []
        
        for layer_name, stats in layer_stats.items():
            score = 0.0
            
            # Size factor
            param_count = stats.get('parameter_count', 0)
            if param_count > 0:
                score += param_count / 1000000  # Normalize by 1M
            
            # Redundancy factor
            if 'redundancy' in stats:
                score += stats['redundancy'] * 2
            
            # Activity factor (less active = better candidate)
            if 'average_activation' in stats:
                score += (1 - stats['average_activation']) * 1.5
            
            # Skip critical layers
            if stats.get('is_critical', False):
                score *= 0.1
            
            layer_scores.append((layer_name, score))
        
        # Sort by score and select top candidates
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 50% of layers or minimum 5 layers
        num_to_select = max(5, len(layer_scores) // 2)
        selected_layers = [name for name, _ in layer_scores[:num_to_select]]
        
        selection['layers'] = selected_layers
        selection['criteria'] = ['size', 'redundancy', 'activity']
        
        return selection
    
    def _extract_extrema_data(self, homological_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract extrema locations from homological analysis."""
        extrema_data = {}
        
        # Get critical points from homological analysis
        critical_points = homological_analysis.get('critical_points', {})
        
        for layer_name, points in critical_points.items():
            if 'extrema_locations' in points:
                extrema_data[layer_name] = points['extrema_locations']
        
        # Add persistence-based extrema
        persistence = homological_analysis.get('persistence_diagrams', {})
        for layer_name, diagram in persistence.items():
            if layer_name not in extrema_data and 'birth_locations' in diagram:
                # Use birth locations of persistent features
                extrema_data[layer_name] = diagram['birth_locations']
        
        return extrema_data
    
    def _calculate_priority(self, report: AnalysisReport,
                           context: EvolutionContext) -> float:
        """Calculate plan priority."""
        priority = 0.5  # Base priority
        
        # Memory pressure increases priority
        memory_usage = report.get('memory_usage', {})
        memory_ratio = memory_usage.get('total_mb', 0) / self.memory_limit_mb
        priority += min(0.3, memory_ratio * 0.3)
        
        # Size increases priority
        model_stats = report.get('model_stats', {})
        size_ratio = model_stats.get('total_parameters', 0) / (10 * self.size_threshold)
        priority += min(0.2, size_ratio * 0.2)
        
        # Performance decreases priority (don't compress well-performing models)
        performance = report.get('performance_metrics', {})
        accuracy = performance.get('accuracy', 1.0)
        priority -= (accuracy - 0.8) * 0.2
        
        return max(0.1, min(1.0, priority))
    
    def _estimate_impact(self, params: Dict[str, Any],
                        layer_selection: Dict[str, Any]) -> float:
        """Estimate impact of compactification."""
        # Base impact from sparsity
        sparsity = params['sparsity']
        impact = 1.0 - sparsity  # Compression impact
        
        # Adjust for layer selection
        if layer_selection['layers']:
            # Partial compactification has less impact
            impact *= 0.7
        
        # Adjust for patch density
        patch_density = params['patch_density']
        impact *= (1.0 - patch_density * 0.2)  # Patches reduce compression
        
        return impact
    
    def _create_no_op_plan(self) -> EvolutionPlan:
        """Create a no-operation plan."""
        plan = EvolutionPlan({
            'type': 'no_op',
            'reason': 'Compactification not needed'
        })
        plan.priority = 0.0
        plan.estimated_impact = 0.0
        plan.created_by = self.name
        return plan