#!/usr/bin/env python3
"""
Hierarchical Bootstrapping Strategy Component

Migrated from evolution.hierarchical_bootstrapping to use the IStrategy interface.
Implements phase-based network evolution with knowledge transfer between phases.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Set, Optional
from datetime import datetime
import logging

from ...core.interfaces import (
    IStrategy, IModel, AnalysisReport, EvolutionContext, EvolutionPlan,
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)


class HierarchicalBootstrappingStrategy(IStrategy):
    """
    Strategy component for hierarchical, phase-based network evolution.
    
    This strategy manages network evolution through multiple phases (coarse, medium, fine),
    transferring knowledge from simpler phases to bootstrap more complex ones.
    
    Features:
    - Phase-based evolution with knowledge transfer
    - Progressive refinement stages
    - Residual learning between phases
    - Intelligent initialization based on previous phase analysis
    """
    
    def __init__(self, 
                 phases: List[str] = None,
                 refinement_threshold: float = 0.1,
                 knowledge_transfer_weight: float = 0.7,
                 name: str = None):
        super().__init__()
        self.phases = phases or ['coarse', 'medium', 'fine']
        self.refinement_threshold = refinement_threshold
        self.knowledge_transfer_weight = knowledge_transfer_weight
        self._name = name or "HierarchicalBootstrappingStrategy"
        
        # Phase management
        self.current_phase = 0
        self.phase_models = {}
        self.phase_knowledge = {}
        self.refinement_stages = []
        
        # Component contract
        self._contract = ComponentContract(
            component_name=self._name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={'model', 'performance_metrics', 'training_history'},
            provided_outputs={'phase_transition_plan', 'knowledge_transfer_plan', 'refinement_plan'},
            optional_inputs={'current_phase', 'target_phase'},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True,
                estimated_runtime_seconds=2.0
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Component contract declaration."""
        return self._contract
    
    @property
    def name(self) -> str:
        """Component name."""
        return self._name
    
    def propose_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """
        Generate evolution plan based on hierarchical bootstrapping analysis.
        
        Args:
            report: Analysis report containing model and performance metrics
            context: Evolution context
            
        Returns:
            Evolution plan with phase transition or refinement actions
        """
        self._track_execution(self._create_plan)
        return self._create_plan(report, context)
    
    def _create_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Internal plan creation implementation."""
        try:
            # Get current phase information
            current_phase = context.get('current_phase', self.current_phase)
            performance_metrics = report.get('performance_metrics', {})
            training_history = context.get('training_history', [])
            
            # Analyze phase transition needs
            phase_analysis = self._analyze_phase_transition_needs(
                current_phase, performance_metrics, training_history
            )
            
            # Create appropriate plan
            if phase_analysis['should_transition']:
                plan = self._create_phase_transition_plan(phase_analysis, context)
            elif phase_analysis['needs_refinement']:
                plan = self._create_refinement_plan(phase_analysis, context)
            else:
                plan = self._create_no_action_plan()
            
            self.log(logging.INFO, f"Created {plan['type']} plan for phase {current_phase}")
            
            return plan
            
        except Exception as e:
            self.log(logging.ERROR, f"Plan creation failed: {str(e)}")
            return self._create_no_action_plan()
    
    def _analyze_phase_transition_needs(self, current_phase: int, 
                                      performance_metrics: Dict[str, Any],
                                      training_history: List[Dict]) -> Dict[str, Any]:
        """
        Analyze whether the current phase should transition to the next.
        
        Args:
            current_phase: Current phase index
            performance_metrics: Current performance metrics
            training_history: Training history data
            
        Returns:
            Analysis results with transition recommendations
        """
        analysis = {
            'should_transition': False,
            'needs_refinement': False,
            'current_phase_name': self.phases[current_phase] if current_phase < len(self.phases) else 'final',
            'performance_plateau': False,
            'knowledge_ready': False,
            'next_phase_name': None
        }
        
        # Check if we're at the final phase
        if current_phase >= len(self.phases) - 1:
            analysis['needs_refinement'] = self._should_add_refinement_stage(
                performance_metrics, training_history
            )
            return analysis
        
        # Analyze performance plateau
        if len(training_history) >= 10:
            recent_performance = [h.get('accuracy', 0) for h in training_history[-10:]]
            performance_std = np.std(recent_performance) if recent_performance else 1.0
            analysis['performance_plateau'] = performance_std < self.refinement_threshold
        
        # Check if knowledge extraction is ready
        analysis['knowledge_ready'] = self._is_knowledge_extraction_ready(
            current_phase, performance_metrics
        )
        
        # Determine transition readiness
        analysis['should_transition'] = (
            analysis['performance_plateau'] and 
            analysis['knowledge_ready']
        )
        
        if analysis['should_transition']:
            analysis['next_phase_name'] = self.phases[current_phase + 1]
        
        return analysis
    
    def _should_add_refinement_stage(self, performance_metrics: Dict[str, Any],
                                   training_history: List[Dict]) -> bool:
        """
        Determine if a refinement stage should be added.
        
        Args:
            performance_metrics: Current performance metrics
            training_history: Training history
            
        Returns:
            True if refinement stage should be added
        """
        # Check for representation gaps
        accuracy = performance_metrics.get('accuracy', 0.0)
        loss = performance_metrics.get('loss', float('inf'))
        
        # Simple heuristic: add refinement if accuracy is below threshold
        # and loss is still decreasing
        if accuracy < 0.85 and len(training_history) >= 5:
            recent_losses = [h.get('loss', float('inf')) for h in training_history[-5:]]
            if len(recent_losses) >= 2:
                loss_trend = recent_losses[-1] - recent_losses[0]
                return loss_trend < 0  # Loss is decreasing
        
        return False
    
    def _is_knowledge_extraction_ready(self, current_phase: int,
                                     performance_metrics: Dict[str, Any]) -> bool:
        """
        Check if the current phase is ready for knowledge extraction.
        
        Args:
            current_phase: Current phase index
            performance_metrics: Performance metrics
            
        Returns:
            True if ready for knowledge extraction
        """
        # Simple heuristic: ready if performance is above minimum threshold
        accuracy = performance_metrics.get('accuracy', 0.0)
        min_accuracy = 0.5 + (current_phase * 0.15)  # Increasing threshold per phase
        
        return accuracy >= min_accuracy
    
    def _create_phase_transition_plan(self, analysis: Dict[str, Any],
                                    context: EvolutionContext) -> EvolutionPlan:
        """
        Create a plan for transitioning to the next phase.
        
        Args:
            analysis: Phase analysis results
            context: Evolution context
            
        Returns:
            Phase transition evolution plan
        """
        plan = EvolutionPlan({
            'type': 'hierarchical_bootstrap',
            'action': 'phase_transition',
            'current_phase': analysis['current_phase_name'],
            'next_phase': analysis['next_phase_name'],
            'knowledge_extraction_plan': self._create_knowledge_extraction_plan(context),
            'initialization_plan': self._create_phase_initialization_plan(
                analysis['current_phase_name'], analysis['next_phase_name']
            ),
            'metadata': {
                'strategy': self.name,
                'timestamp': datetime.now().isoformat(),
                'knowledge_transfer_weight': self.knowledge_transfer_weight
            }
        })
        
        plan.priority = 0.8  # High priority for phase transitions
        plan.estimated_impact = 0.6  # Significant impact expected
        plan.created_by = self.name
        
        return plan
    
    def _create_refinement_plan(self, analysis: Dict[str, Any],
                              context: EvolutionContext) -> EvolutionPlan:
        """
        Create a plan for adding a refinement stage.
        
        Args:
            analysis: Phase analysis results
            context: Evolution context
            
        Returns:
            Refinement evolution plan
        """
        plan = EvolutionPlan({
            'type': 'hierarchical_bootstrap',
            'action': 'add_refinement',
            'current_phase': analysis['current_phase_name'],
            'refinement_stage': len(self.refinement_stages),
            'gap_analysis': self._analyze_representation_gaps(context),
            'refinement_strategy': self._create_refinement_strategy(),
            'metadata': {
                'strategy': self.name,
                'timestamp': datetime.now().isoformat()
            }
        })
        
        plan.priority = 0.6  # Medium priority for refinement
        plan.estimated_impact = 0.3  # Moderate impact expected
        plan.created_by = self.name
        
        return plan
    
    def _create_no_action_plan(self) -> EvolutionPlan:
        """Create a no-action plan."""
        plan = EvolutionPlan({
            'type': 'hierarchical_bootstrap',
            'action': 'continue_training',
            'reason': 'Current phase not ready for transition or refinement',
            'metadata': {
                'strategy': self.name,
                'timestamp': datetime.now().isoformat()
            }
        })
        
        plan.priority = 0.0
        plan.estimated_impact = 0.0
        plan.created_by = self.name
        
        return plan
    
    def _create_knowledge_extraction_plan(self, context: EvolutionContext) -> Dict[str, Any]:
        """
        Create a plan for extracting knowledge from the current phase.
        
        Args:
            context: Evolution context
            
        Returns:
            Knowledge extraction plan
        """
        model = context.get('model')
        if model is None:
            return {'method': 'none', 'reason': 'No model available'}
        
        return {
            'method': 'weight_analysis',
            'extract_dominant_paths': True,
            'extract_extrema_patterns': True,
            'extract_weight_statistics': True,
            'analyze_activations': True,
            'target_layers': 'all'
        }
    
    def _create_phase_initialization_plan(self, current_phase: str,
                                        next_phase: str) -> Dict[str, Any]:
        """
        Create initialization plan for the next phase.
        
        Args:
            current_phase: Current phase name
            next_phase: Next phase name
            
        Returns:
            Initialization plan
        """
        return {
            'initialization_method': 'knowledge_bootstrap',
            'transfer_weights': True,
            'scale_architecture': True,
            'preserve_learned_features': True,
            'initialization_noise_std': 0.01,
            'knowledge_weight': self.knowledge_transfer_weight,
            'target_complexity_increase': 1.5  # 50% more complex than current phase
        }
    
    def _analyze_representation_gaps(self, context: EvolutionContext) -> List[Dict[str, Any]]:
        """
        Analyze representation gaps in the current model.
        
        Args:
            context: Evolution context
            
        Returns:
            List of identified gaps
        """
        # Simplified gap analysis - in practice this would be more sophisticated
        gaps = [
            {
                'type': 'high_frequency_details',
                'severity': np.random.rand(),
                'location': 'final_layers',
                'suggested_fix': 'add_detail_refinement_branch'
            },
            {
                'type': 'feature_correlation',
                'severity': np.random.rand(),
                'location': 'intermediate_layers',
                'suggested_fix': 'add_correlation_correction_layer'
            }
        ]
        
        return gaps
    
    def _create_refinement_strategy(self) -> Dict[str, Any]:
        """
        Create strategy for refinement stage.
        
        Returns:
            Refinement strategy configuration
        """
        return {
            'refinement_type': 'residual_correction',
            'architecture_modification': 'add_branches',
            'learning_strategy': 'fine_tune_residuals',
            'target_improvement': 0.05  # 5% improvement target
        }
    
    def get_strategy_type(self) -> str:
        """Get the type of strategy."""
        return "hierarchical_bootstrap"
    
    def can_apply(self, context: EvolutionContext) -> bool:
        """Check if this strategy can be applied to the given context."""
        return (
            self.validate_context(context) and
            'model' in context and
            ('performance_metrics' in context or 'training_history' in context)
        )
    
    def apply(self, context: EvolutionContext) -> bool:
        """Apply this strategy (creates plan via propose_plan)."""
        return self.can_apply(context)
    
    def get_required_metrics(self) -> Set[str]:
        """Get metrics required by this strategy."""
        return {
            'performance_metrics',
            'training_history',
            'model_complexity',
            'phase_information'
        }
    
    # Phase management methods
    
    def extract_knowledge(self, model: nn.Module, phase_name: str) -> Dict[str, Any]:
        """
        Extract knowledge from a trained model phase.
        
        Args:
            model: Trained model
            phase_name: Name of the current phase
            
        Returns:
            Extracted knowledge dictionary
        """
        knowledge = {
            'phase': phase_name,
            'extraction_timestamp': datetime.now().isoformat(),
            'dominant_paths': self._extract_dominant_paths(model),
            'extrema_patterns': self._extract_extrema_patterns(model),
            'weight_patterns': self._extract_weight_statistics(model),
            'architectural_insights': self._extract_architectural_insights(model)
        }
        
        self.phase_knowledge[phase_name] = knowledge
        return knowledge
    
    def _extract_dominant_paths(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Extract dominant activation paths from the model."""
        # Simplified implementation - would use actual activation analysis
        paths = []
        for i, module in enumerate(model.modules()):
            if isinstance(module, nn.Linear):
                # Analyze weight magnitudes to find dominant connections
                weights = module.weight.data
                top_connections = torch.topk(weights.abs().flatten(), k=10)
                paths.append({
                    'layer': i,
                    'type': 'linear',
                    'dominant_weights': top_connections.values.tolist(),
                    'connection_indices': top_connections.indices.tolist()
                })
        return paths
    
    def _extract_extrema_patterns(self, model: nn.Module) -> Dict[str, Any]:
        """Extract extrema patterns from model activations."""
        # Simplified implementation
        return {
            'dead_neurons_estimate': np.random.randint(0, 20),
            'saturated_neurons_estimate': np.random.randint(0, 15),
            'activation_ranges': {
                'min': -2.0,
                'max': 5.0,
                'mean': 0.5,
                'std': 1.2
            }
        }
    
    def _extract_weight_statistics(self, model: nn.Module) -> Dict[str, Any]:
        """Extract statistical properties of model weights."""
        all_weights = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                all_weights.extend(module.weight.data.flatten().tolist())
        
        if all_weights:
            weights_tensor = torch.tensor(all_weights)
            return {
                'mean': weights_tensor.mean().item(),
                'std': weights_tensor.std().item(),
                'min': weights_tensor.min().item(),
                'max': weights_tensor.max().item(),
                'sparsity': (weights_tensor.abs() < 1e-6).float().mean().item()
            }
        
        return {'mean': 0.0, 'std': 0.1, 'min': 0.0, 'max': 0.0, 'sparsity': 0.0}
    
    def _extract_architectural_insights(self, model: nn.Module) -> Dict[str, Any]:
        """Extract insights about model architecture."""
        layer_count = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            'layer_count': layer_count,
            'total_parameters': total_params,
            'architecture_complexity': layer_count * np.log(total_params + 1),
            'depth_to_width_ratio': layer_count / max(1, total_params // layer_count)
        }