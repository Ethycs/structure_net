"""
Catastrophe analyzer component.

This analyzer combines multiple stability metrics to detect and predict
catastrophic events like sudden performance drops or catastrophic forgetting.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import logging
import numpy as np

from src.structure_net.core import (
    BaseAnalyzer, IModel, ILayer, EvolutionContext, AnalysisReport,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from src.structure_net.components.metrics import (
    ActivationStabilityMetric, LyapunovMetric, TransitionEntropyMetric
)


class CatastropheAnalyzer(BaseAnalyzer):
    """
    Analyzes dynamical stability to predict catastrophic events.
    
    Combines activation stability, Lyapunov exponents, and transition
    entropy to detect instabilities that could lead to catastrophic
    forgetting or sudden performance degradation.
    """
    
    def __init__(self, n_lyapunov_directions: int = 10,
                 n_entropy_symbols: int = 64,
                 trajectory_length: int = 10,
                 name: str = None):
        """
        Initialize catastrophe analyzer.
        
        Args:
            n_lyapunov_directions: Number of directions for Lyapunov estimation
            n_entropy_symbols: Number of symbols for entropy computation
            trajectory_length: Length of trajectories to analyze
            name: Optional custom name
        """
        super().__init__(name or "CatastropheAnalyzer")
        self.trajectory_length = trajectory_length
        
        # Initialize metrics
        self._stability_metric = ActivationStabilityMetric()
        self._lyapunov_metric = LyapunovMetric(n_directions=n_lyapunov_directions)
        self._entropy_metric = TransitionEntropyMetric(n_symbols=n_entropy_symbols)
        
        self._required_metrics = {
            "ActivationStabilityMetric",
            "LyapunovMetric",
            "TransitionEntropyMetric"
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model", "test_data"},
            optional_inputs={"num_trajectories"},
            provided_outputs={
                "analysis.catastrophe_risk_score",
                "analysis.stability_analysis",
                "analysis.lyapunov_analysis",
                "analysis.dynamics_analysis",
                "analysis.risk_factors",
                "analysis.recommendations"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=True,
                parallel_safe=False
            )
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, 
                         context: EvolutionContext) -> Dict[str, Any]:
        """
        Perform comprehensive catastrophe risk analysis.
        
        Args:
            model: Model to analyze
            report: Analysis report containing metric data
            context: Evolution context with test data
            
        Returns:
            Dictionary containing analysis results
        """
        # Get test data
        test_data = context.get('test_data')
        if test_data is None:
            raise ValueError("CatastropheAnalyzer requires 'test_data' in context")
        
        num_trajectories = context.get('num_trajectories', 10)
        
        # Generate trajectories
        trajectories = self._generate_trajectories(model, test_data, num_trajectories)
        
        # Get activation trajectories for metrics
        activation_trajectories = self._collect_activation_trajectories(
            model, trajectories
        )
        
        # Run stability analysis
        stability_analysis = self._analyze_stability(
            activation_trajectories, report, context
        )
        
        # Run Lyapunov analysis
        lyapunov_analysis = self._analyze_lyapunov(
            model, test_data, report, context
        )
        
        # Run dynamics analysis
        dynamics_analysis = self._analyze_dynamics(
            activation_trajectories, report, context
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            stability_analysis, lyapunov_analysis, dynamics_analysis
        )
        
        # Compute overall risk score
        catastrophe_risk_score = self._compute_risk_score(
            stability_analysis, lyapunov_analysis, dynamics_analysis
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_factors, catastrophe_risk_score)
        
        return {
            "catastrophe_risk_score": catastrophe_risk_score,
            "stability_analysis": stability_analysis,
            "lyapunov_analysis": lyapunov_analysis,
            "dynamics_analysis": dynamics_analysis,
            "risk_factors": risk_factors,
            "recommendations": recommendations
        }
    
    def _generate_trajectories(self, model: nn.Module, test_data: torch.Tensor,
                             num_trajectories: int) -> List[List[torch.Tensor]]:
        """Generate input trajectories for analysis."""
        trajectories = []
        data_size = len(test_data)
        
        if data_size < self.trajectory_length:
            self.log(logging.WARNING, 
                    f"Test data too small ({data_size}) for trajectory length {self.trajectory_length}")
            return []
        
        for _ in range(num_trajectories):
            # Random starting point
            start_idx = np.random.randint(0, data_size - self.trajectory_length)
            trajectory = test_data[start_idx:start_idx + self.trajectory_length]
            trajectories.append(trajectory)
        
        return trajectories
    
    def _collect_activation_trajectories(self, model: nn.Module,
                                       input_trajectories: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        """Collect activation trajectories for input sequences."""
        activation_trajectories = []
        
        model.eval()
        with torch.no_grad():
            for trajectory in input_trajectories:
                activations = []
                
                # Hook to capture activations
                activation_buffer = []
                
                def hook_fn(module, input, output):
                    activation_buffer.append(output.detach().flatten())
                
                # Register hooks on all layers
                hooks = []
                for module in model.modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        hooks.append(module.register_forward_hook(hook_fn))
                
                # Process trajectory
                for x in trajectory:
                    activation_buffer.clear()
                    
                    # Forward pass
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    _ = model(x)
                    
                    # Concatenate all layer activations
                    if activation_buffer:
                        full_activation = torch.cat(activation_buffer)
                        activations.append(full_activation)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                if activations:
                    activation_trajectories.append(activations)
        
        return activation_trajectories
    
    def _analyze_stability(self, activation_trajectories: List[List[torch.Tensor]],
                          report: AnalysisReport,
                          context: EvolutionContext) -> Dict[str, Any]:
        """Analyze activation stability."""
        all_stability_results = []
        
        for traj in activation_trajectories:
            # Create context for stability metric
            stability_context = EvolutionContext({'activation_trajectory': traj})
            
            # Run stability metric
            stability_key = f"ActivationStabilityMetric_{len(all_stability_results)}"
            stability_result = self._stability_metric.analyze(None, stability_context)
            report.add_metric_data(stability_key, stability_result)
            
            all_stability_results.append(stability_result)
        
        if not all_stability_results:
            return {"error": "No stability results"}
        
        # Aggregate results
        mean_change_rates = [r['mean_change_rate'] for r in all_stability_results]
        max_change_rates = [r['max_change_rate'] for r in all_stability_results]
        stability_scores = [r['stability_score'] for r in all_stability_results]
        
        return {
            "avg_mean_change_rate": np.mean(mean_change_rates),
            "avg_max_change_rate": np.mean(max_change_rates),
            "min_stability_score": np.min(stability_scores),
            "avg_stability_score": np.mean(stability_scores),
            "high_change_trajectories": sum(1 for r in all_stability_results 
                                          if r['rapid_change_count'] > 0),
            "stability_variance": np.var(stability_scores)
        }
    
    def _analyze_lyapunov(self, model: nn.Module, test_data: torch.Tensor,
                         report: AnalysisReport,
                         context: EvolutionContext) -> Dict[str, Any]:
        """Analyze Lyapunov exponents."""
        # Sample inputs for Lyapunov analysis
        sample_size = min(100, len(test_data))
        input_samples = test_data[:sample_size]
        
        # Create context for Lyapunov metric
        lyapunov_context = EvolutionContext({
            'model': model,
            'input_samples': input_samples
        })
        
        # Run Lyapunov metric
        lyapunov_key = "LyapunovMetric"
        if lyapunov_key not in report.metrics:
            lyapunov_result = self._lyapunov_metric.analyze(model, lyapunov_context)
            report.add_metric_data(lyapunov_key, lyapunov_result)
        else:
            lyapunov_result = report.get(f"metrics.{lyapunov_key}")
        
        # Add interpretation
        chaos_risk = "high" if lyapunov_result['max_lyapunov'] > 1.0 else \
                    "medium" if lyapunov_result['max_lyapunov'] > 0.5 else "low"
        
        return {
            **lyapunov_result,
            "chaos_risk": chaos_risk,
            "instability_regions": lyapunov_result['positive_lyapunov_ratio']
        }
    
    def _analyze_dynamics(self, activation_trajectories: List[List[torch.Tensor]],
                         report: AnalysisReport,
                         context: EvolutionContext) -> Dict[str, Any]:
        """Analyze dynamical properties through transition entropy."""
        # Create context for entropy metric
        entropy_context = EvolutionContext({
            'activation_trajectories': activation_trajectories
        })
        
        # Run entropy metric
        entropy_key = "TransitionEntropyMetric"
        if entropy_key not in report.metrics:
            entropy_result = self._entropy_metric.analyze(None, entropy_context)
            report.add_metric_data(entropy_key, entropy_result)
        else:
            entropy_result = report.get(f"metrics.{entropy_key}")
        
        # Add interpretation
        complexity = "high" if entropy_result['entropy_rate'] > 0.8 else \
                    "medium" if entropy_result['entropy_rate'] > 0.5 else "low"
        
        return {
            **entropy_result,
            "dynamics_complexity": complexity,
            "unpredictability": 1 - entropy_result['predictability_score']
        }
    
    def _identify_risk_factors(self, stability_analysis: Dict[str, Any],
                             lyapunov_analysis: Dict[str, Any],
                             dynamics_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific risk factors."""
        risk_factors = []
        
        # Stability risks
        if stability_analysis.get('avg_mean_change_rate', 0) > 0.3:
            risk_factors.append({
                "factor": "high_activation_volatility",
                "severity": "high",
                "value": stability_analysis['avg_mean_change_rate'],
                "description": "Activation patterns are changing rapidly"
            })
        
        if stability_analysis.get('high_change_trajectories', 0) > 5:
            risk_factors.append({
                "factor": "frequent_rapid_changes",
                "severity": "medium",
                "value": stability_analysis['high_change_trajectories'],
                "description": "Multiple trajectories show rapid changes"
            })
        
        # Lyapunov risks
        if lyapunov_analysis.get('max_lyapunov', 0) > 1.0:
            risk_factors.append({
                "factor": "positive_lyapunov",
                "severity": "high",
                "value": lyapunov_analysis['max_lyapunov'],
                "description": "System shows chaotic behavior"
            })
        
        if lyapunov_analysis.get('instability_regions', 0) > 0.5:
            risk_factors.append({
                "factor": "widespread_instability",
                "severity": "high",
                "value": lyapunov_analysis['instability_regions'],
                "description": "Many regions show instability"
            })
        
        # Dynamics risks
        if dynamics_analysis.get('unpredictability', 0) > 0.7:
            risk_factors.append({
                "factor": "unpredictable_dynamics",
                "severity": "medium",
                "value": dynamics_analysis['unpredictability'],
                "description": "Network dynamics are highly unpredictable"
            })
        
        if dynamics_analysis.get('transition_diversity', 0) < 0.1:
            risk_factors.append({
                "factor": "rigid_dynamics",
                "severity": "low",
                "value": dynamics_analysis['transition_diversity'],
                "description": "Limited diversity in state transitions"
            })
        
        return risk_factors
    
    def _compute_risk_score(self, stability_analysis: Dict[str, Any],
                          lyapunov_analysis: Dict[str, Any],
                          dynamics_analysis: Dict[str, Any]) -> float:
        """Compute overall catastrophe risk score (0-1)."""
        risk = 0.0
        
        # Stability contribution (30%)
        stability_risk = stability_analysis.get('avg_mean_change_rate', 0) * 2  # Scale to 0-1
        stability_risk = min(1.0, stability_risk)
        risk += stability_risk * 0.3
        
        # Lyapunov contribution (40%)
        lyapunov_risk = lyapunov_analysis.get('stability_indicator', 0.5)
        risk += lyapunov_risk * 0.4
        
        # Dynamics contribution (30%)
        # Higher entropy can be protective (more diverse dynamics)
        entropy_protection = dynamics_analysis.get('predictability_score', 0.5)
        dynamics_risk = 1 - entropy_protection
        risk += dynamics_risk * 0.3
        
        return np.clip(risk, 0, 1)
    
    def _generate_recommendations(self, risk_factors: List[Dict[str, Any]],
                                risk_score: float) -> List[str]:
        """Generate recommendations based on risk analysis."""
        recommendations = []
        
        # Overall risk level
        if risk_score > 0.7:
            recommendations.append(
                "CRITICAL: High catastrophe risk detected. Consider immediate intervention."
            )
        elif risk_score > 0.5:
            recommendations.append(
                "WARNING: Moderate catastrophe risk. Monitor closely and prepare mitigation."
            )
        
        # Specific recommendations based on risk factors
        factor_types = {f['factor'] for f in risk_factors}
        
        if 'high_activation_volatility' in factor_types:
            recommendations.append(
                "Add batch normalization or layer normalization to stabilize activations"
            )
        
        if 'positive_lyapunov' in factor_types:
            recommendations.append(
                "Reduce learning rate or add gradient clipping to control chaos"
            )
        
        if 'widespread_instability' in factor_types:
            recommendations.append(
                "Consider architectural changes to improve stability (e.g., skip connections)"
            )
        
        if 'unpredictable_dynamics' in factor_types:
            recommendations.append(
                "Increase regularization to promote more predictable behavior"
            )
        
        if 'rigid_dynamics' in factor_types:
            recommendations.append(
                "Add noise or dropout to increase dynamics diversity"
            )
        
        # General recommendations
        if risk_score > 0.3:
            recommendations.append(
                "Implement checkpoint saving to recover from potential catastrophic events"
            )
        
        return recommendations