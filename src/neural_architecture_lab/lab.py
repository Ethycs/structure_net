"""
Main Neural Architecture Lab implementation.
"""

import asyncio
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .core import (
    Hypothesis, HypothesisResult, Experiment, ExperimentResult,
    LabConfig, ExperimentStatus, HypothesisCategory
)
from .runners import AsyncExperimentRunner
from .analyzers import InsightExtractor, StatisticalAnalyzer


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyJSONEncoder, self).default(obj)

class NeuralArchitectureLab:
    """
    The main lab for running systematic neural architecture experiments.
    
    This lab provides a scientific framework for testing hypotheses about
    neural network architectures, training strategies, and growth patterns.
    """
    
    def __init__(self, config: LabConfig):
        """
        Initialize the Neural Architecture Lab.
        
        Args:
            config: Lab configuration
        """
        self.config = config
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.results: Dict[str, HypothesisResult] = {}
        
        # Create results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.runner = AsyncExperimentRunner(config)
        self.insight_extractor = InsightExtractor()
        self.statistical_analyzer = StatisticalAnalyzer(
            significance_level=config.significance_level
        )
        
        # Lab state
        self.lab_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = None
        self.total_experiments_run = 0
        
        # Hypothesis tracking
        self.hypothesis_tree = {}  # Track hypothesis relationships
        self.pending_hypotheses = []  # Queue of hypotheses to test
        
    def register_hypothesis(self, hypothesis: Hypothesis) -> str:
        """
        Register a new hypothesis for testing.
        
        Args:
            hypothesis: The hypothesis to register
            
        Returns:
            Hypothesis ID
        """
        if hypothesis.id in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis.id} already registered")
        
        self.hypotheses[hypothesis.id] = hypothesis
        self.pending_hypotheses.append(hypothesis.id)
        
        if self.config.verbose:
            print(f"📊 Registered hypothesis: {hypothesis.name}")
            print(f"   Category: {hypothesis.category.value}")
            print(f"   Question: {hypothesis.question}")
        
        return hypothesis.id
    
    def register_hypothesis_batch(self, hypotheses: List[Hypothesis]):
        """Register multiple hypotheses at once."""
        for hypothesis in hypotheses:
            self.register_hypothesis(hypothesis)
    
    def generate_experiments(self, hypothesis: Hypothesis) -> List[Experiment]:
        """
        Generate experiments to test a hypothesis.
        
        Args:
            hypothesis: The hypothesis to test
            
        Returns:
            List of experiments
        """
        experiments = []
        
        # Generate parameter combinations
        param_space = hypothesis.parameter_space
        
        # For now, simple grid search (can be enhanced with more sophisticated methods)
        param_combinations = self._generate_parameter_grid(param_space)
        
        # Ensure minimum experiments
        if len(param_combinations) < self.config.min_experiments_per_hypothesis:
            # Duplicate with different seeds
            original_combinations = param_combinations.copy()
            seeds_needed = self.config.min_experiments_per_hypothesis // len(original_combinations) + 1
            param_combinations = []
            for seed in range(seeds_needed):
                for params in original_combinations:
                    param_combinations.append({**params, 'seed': seed})
        
        # Create experiments
        for i, params in enumerate(param_combinations[:self.config.min_experiments_per_hypothesis * 2]):
            # Merge with control parameters
            full_params = {**hypothesis.control_parameters, **params}
            
            experiment = Experiment(
                id=f"{hypothesis.id}_exp_{i:03d}",
                hypothesis_id=hypothesis.id,
                name=f"{hypothesis.name} - Experiment {i+1}",
                parameters=full_params,
                seed=params.get('seed', i)
            )
            experiments.append(experiment)
            self.experiments[experiment.id] = experiment
        
        return experiments
    
    def _generate_parameter_grid(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter combinations from parameter space."""
        combinations = [{}]
        
        for param_name, param_spec in param_space.items():
            new_combinations = []
            
            if isinstance(param_spec, list):
                # List of discrete values
                for value in param_spec:
                    for combo in combinations:
                        new_combinations.append({**combo, param_name: value})
            elif isinstance(param_spec, dict):
                # Range specification
                if 'min' in param_spec and 'max' in param_spec:
                    # Numeric range
                    n_samples = param_spec.get('n_samples', 3)
                    if param_spec.get('log_scale', False):
                        values = np.logspace(
                            np.log10(param_spec['min']),
                            np.log10(param_spec['max']),
                            n_samples
                        )
                    else:
                        values = np.linspace(
                            param_spec['min'],
                            param_spec['max'],
                            n_samples
                        )
                    
                    for value in values:
                        for combo in combinations:
                            new_combinations.append({**combo, param_name: float(value)})
                elif 'values' in param_spec:
                    # Explicit values
                    for value in param_spec['values']:
                        for combo in combinations:
                            new_combinations.append({**combo, param_name: value})
            else:
                # Single value
                for combo in combinations:
                    new_combinations.append({**combo, param_name: param_spec})
            
            combinations = new_combinations
        
        return combinations
    
    async def test_hypothesis(self, hypothesis_id: str) -> HypothesisResult:
        """
        Test a hypothesis by running all its experiments.
        
        Args:
            hypothesis_id: ID of hypothesis to test
            
        Returns:
            Hypothesis test results
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        if self.config.verbose:
            print(f"\n🧪 Testing hypothesis: {hypothesis.name}")
            print(f"   Prediction: {hypothesis.prediction}")
        
        # Generate experiments
        experiments = self.generate_experiments(hypothesis)
        
        if self.config.verbose:
            print(f"   Generated {len(experiments)} experiments")
        
        # Run experiments
        experiment_results = await self.runner.run_experiments(experiments)
        
        # Update total count
        self.total_experiments_run += len(experiment_results)
        
        # Analyze results
        hypothesis_result = self._analyze_hypothesis_results(
            hypothesis, experiment_results
        )
        
        # Store results
        self.results[hypothesis_id] = hypothesis_result
        hypothesis.tested = True
        hypothesis.results.append(hypothesis_result)
        
        # Save results
        self._save_hypothesis_results(hypothesis, hypothesis_result)
        
        # Generate follow-up hypotheses if enabled
        if self.config.enable_adaptive_hypotheses and hypothesis_result.confirmed:
            follow_ups = self._generate_follow_up_hypotheses(
                hypothesis, hypothesis_result
            )
            for follow_up in follow_ups:
                self.register_hypothesis(follow_up)
        
        if hypothesis_id in self.pending_hypotheses:
            self.pending_hypotheses.remove(hypothesis_id)

        return hypothesis_result
    
    def _analyze_hypothesis_results(
        self, 
        hypothesis: Hypothesis, 
        experiment_results: List[ExperimentResult]
    ) -> HypothesisResult:
        """Analyze experiment results to determine if hypothesis is confirmed."""
        
        # Separate successful and failed experiments
        successful_results = [r for r in experiment_results if r.error is None]
        failed_results = [r for r in experiment_results if r.error is not None]
        
        if not successful_results:
            # All experiments failed
            return HypothesisResult(
                hypothesis_id=hypothesis.id,
                num_experiments=len(experiment_results),
                successful_experiments=0,
                failed_experiments=len(failed_results),
                confirmed=False,
                confidence=0.0,
                effect_size=0.0,
                best_parameters={},
                best_metrics={},
                key_insights=["All experiments failed"],
                unexpected_findings=[],
                suggested_hypotheses=[],
                experiment_results=experiment_results,
                statistical_summary={}
            )
        
        # Extract primary metrics
        primary_metrics = [r.primary_metric for r in successful_results]
        
        # Statistical analysis
        statistical_summary = self.statistical_analyzer.analyze_results(
            successful_results, hypothesis.success_metrics
        )
        
        # Determine if hypothesis is confirmed
        confirmed = statistical_summary['meets_success_criteria']
        if self.config.require_statistical_significance:
            confirmed = confirmed and statistical_summary['statistically_significant']
        
        # Find best configuration
        best_result = max(successful_results, key=lambda r: r.primary_metric)
        
        # Extract insights
        insights = self.insight_extractor.extract_insights(
            hypothesis, successful_results, statistical_summary
        )
        
        # Calculate effect size
        if 'baseline' in hypothesis.control_parameters:
            baseline = hypothesis.control_parameters['baseline']
            effect_size = (np.mean(primary_metrics) - baseline) / baseline
        else:
            effect_size = statistical_summary.get('effect_size', 0.0)
        
        return HypothesisResult(
            hypothesis_id=hypothesis.id,
            num_experiments=len(experiment_results),
            successful_experiments=len(successful_results),
            failed_experiments=len(failed_results),
            confirmed=confirmed,
            confidence=statistical_summary['confidence'],
            effect_size=effect_size,
            best_parameters=best_result.experiment_id,
            best_metrics=best_result.metrics,
            key_insights=insights['key_insights'],
            unexpected_findings=insights['unexpected_findings'],
            suggested_hypotheses=insights['suggested_hypotheses'],
            experiment_results=experiment_results,
            statistical_summary=statistical_summary
        )
    
    def _generate_follow_up_hypotheses(
        self, 
        parent: Hypothesis, 
        result: HypothesisResult
    ) -> List[Hypothesis]:
        """Generate follow-up hypotheses based on results."""
        follow_ups = []
        
        # Get current depth in hypothesis tree
        current_depth = self._get_hypothesis_depth(parent.id)
        if current_depth >= self.config.max_hypothesis_depth:
            return follow_ups
        
        # Generate based on unexpected findings
        for finding in result.unexpected_findings[:2]:  # Limit to 2 follow-ups
            follow_up = Hypothesis(
                id=f"{parent.id}_followup_{len(follow_ups)}",
                name=f"Follow-up: {finding[:50]}...",
                description=f"Investigating: {finding}",
                category=parent.category,
                question=f"Why does {finding}?",
                prediction="Further investigation needed",
                test_function=parent.test_function,
                parameter_space=self._refine_parameter_space(
                    parent.parameter_space, result
                ),
                control_parameters=parent.control_parameters,
                success_metrics=parent.success_metrics,
                tags=parent.tags + ["follow_up"],
                references=[parent.id]
            )
            follow_ups.append(follow_up)
        
        # Track in hypothesis tree
        self.hypothesis_tree[parent.id] = [h.id for h in follow_ups]
        
        return follow_ups
    
    def _refine_parameter_space(
        self, 
        original_space: Dict[str, Any], 
        result: HypothesisResult
    ) -> Dict[str, Any]:
        """Refine parameter space based on results."""
        # This is a simple implementation - can be made more sophisticated
        refined_space = {}
        
        best_params = result.best_parameters
        if isinstance(best_params, str):
            # Find the actual parameters from experiment
            for exp_result in result.experiment_results:
                if exp_result.experiment_id == best_params:
                    best_params = self.experiments[exp_result.experiment_id].parameters
                    break
        
        for param, spec in original_space.items():
            if param in best_params:
                best_value = best_params[param]
                
                if isinstance(spec, dict) and 'min' in spec and 'max' in spec:
                    # Narrow the range around the best value
                    range_width = spec['max'] - spec['min']
                    new_width = range_width * 0.5
                    refined_spec = {
                        'min': max(spec['min'], best_value - new_width/2),
                        'max': min(spec['max'], best_value + new_width/2),
                        'n_samples': spec.get('n_samples', 3),
                        'log_scale': spec.get('log_scale', False)
                    }
                    refined_space[param] = refined_spec
                else:
                    refined_space[param] = spec
            else:
                refined_space[param] = spec
        
        return refined_space
    
    def _get_hypothesis_depth(self, hypothesis_id: str) -> int:
        """Get the depth of a hypothesis in the tree."""
        depth = 0
        current_id = hypothesis_id
        
        # Traverse up the tree
        for _ in range(self.config.max_hypothesis_depth):
            found_parent = False
            for parent_id, children in self.hypothesis_tree.items():
                if current_id in children:
                    depth += 1
                    current_id = parent_id
                    found_parent = True
                    break
            
            if not found_parent:
                break
        
        return depth
    
    def _save_hypothesis_results(self, hypothesis: Hypothesis, result: HypothesisResult):
        """Save hypothesis results to disk."""
        results_file = self.results_dir / f"{hypothesis.id}_results.json"
        
        # Convert to serializable format
        data = {
            'hypothesis': {
                'id': hypothesis.id,
                'name': hypothesis.name,
                'description': hypothesis.description,
                'category': hypothesis.category.value,
                'question': hypothesis.question,
                'prediction': hypothesis.prediction,
                'created_at': hypothesis.created_at.isoformat(),
            },
            'result': {
                'confirmed': result.confirmed,
                'confidence': result.confidence,
                'effect_size': result.effect_size,
                'num_experiments': result.num_experiments,
                'successful_experiments': result.successful_experiments,
                'failed_experiments': result.failed_experiments,
                'best_metrics': result.best_metrics,
                'key_insights': result.key_insights,
                'unexpected_findings': result.unexpected_findings,
                'statistical_summary': result.statistical_summary,
                'completed_at': result.completed_at.isoformat()
            },
            'experiments': [
                {
                    'id': exp_result.experiment_id,
                    'metrics': exp_result.metrics,
                    'primary_metric': exp_result.primary_metric,
                    'model_parameters': exp_result.model_parameters,
                    'training_time': exp_result.training_time,
                    'error': exp_result.error
                }
                for exp_result in result.experiment_results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyJSONEncoder)
    
    async def run_all_hypotheses(self) -> Dict[str, HypothesisResult]:
        """
        Run all pending hypotheses.
        
        Returns:
            Dictionary of hypothesis results
        """
        self.start_time = time.time()
        
        if self.config.verbose:
            print(f"\n🔬 Neural Architecture Lab Session {self.lab_id}")
            print(f"   Hypotheses to test: {len(self.pending_hypotheses)}")
            print(f"   Max parallel experiments: {self.config.max_parallel_experiments}")
        
        results = {}
        
        while self.pending_hypotheses:
            hypothesis_id = self.pending_hypotheses.pop(0)
            
            try:
                result = await self.test_hypothesis(hypothesis_id)
                results[hypothesis_id] = result
                
                if self.config.verbose:
                    print(f"\n✅ Hypothesis {hypothesis_id}: {'CONFIRMED' if result.confirmed else 'REJECTED'}")
                    print(f"   Confidence: {result.confidence:.2%}")
                    print(f"   Effect size: {result.effect_size:.3f}")
                
            except Exception as e:
                print(f"\n❌ Failed to test hypothesis {hypothesis_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate lab report
        self._generate_lab_report(results)
        
        return results
    
    def _generate_lab_report(self, results: Dict[str, HypothesisResult]):
        """Generate a comprehensive lab report."""
        elapsed_time = time.time() - self.start_time
        
        report = {
            'lab_id': self.lab_id,
            'total_hypotheses': len(results),
            'confirmed_hypotheses': sum(1 for r in results.values() if r.confirmed),
            'total_experiments': self.total_experiments_run,
            'elapsed_time': elapsed_time,
            'hypotheses_per_category': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Categorize results
        for hypothesis_id, result in results.items():
            hypothesis = self.hypotheses[hypothesis_id]
            category = hypothesis.category.value
            
            if category not in report['hypotheses_per_category']:
                report['hypotheses_per_category'][category] = {
                    'total': 0,
                    'confirmed': 0
                }
            
            report['hypotheses_per_category'][category]['total'] += 1
            if result.confirmed:
                report['hypotheses_per_category'][category]['confirmed'] += 1
            
            # Collect key findings
            if result.confirmed and result.effect_size > 0.1:
                report['key_findings'].append({
                    'hypothesis': hypothesis.name,
                    'effect_size': result.effect_size,
                    'insight': result.key_insights[0] if result.key_insights else ""
                })
        
        # Sort findings by effect size
        report['key_findings'].sort(key=lambda x: x['effect_size'], reverse=True)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(results)
        
        # Save report
        report_file = self.results_dir / f"lab_report_{self.lab_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        if self.config.verbose:
            print(f"\n📊 LAB REPORT")
            print(f"   Total hypotheses tested: {report['total_hypotheses']}")
            print(f"   Confirmed: {report['confirmed_hypotheses']} ({report['confirmed_hypotheses']/report['total_hypotheses']*100:.1f}%)")
            print(f"   Total experiments: {report['total_experiments']}")
            print(f"   Time elapsed: {elapsed_time:.1f}s")
            
            if report['key_findings']:
                print(f"\n🔍 Top Findings:")
                for finding in report['key_findings'][:3]:
                    print(f"   • {finding['hypothesis']}")
                    print(f"     Effect size: {finding['effect_size']:.3f}")
                    if finding['insight']:
                        print(f"     {finding['insight']}")
    
    def _generate_recommendations(self, results: Dict[str, HypothesisResult]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Architecture recommendations
        architecture_results = [
            (h, r) for h, r in results.items() 
            if self.hypotheses[h].category == HypothesisCategory.ARCHITECTURE
        ]
        
        if architecture_results:
            confirmed_archs = [h for h, r in architecture_results if r.confirmed]
            if confirmed_archs:
                recommendations.append(
                    f"Consider using architectures from hypotheses: {', '.join(confirmed_archs[:3])}"
                )
        
        # Training recommendations
        training_results = [
            (h, r) for h, r in results.items()
            if self.hypotheses[h].category == HypothesisCategory.TRAINING
        ]
        
        if training_results:
            best_training = max(
                [r for _, r in training_results if r.confirmed],
                key=lambda r: r.effect_size,
                default=None
            )
            if best_training:
                recommendations.append(
                    f"Training strategy from {best_training.hypothesis_id} showed {best_training.effect_size:.1%} improvement"
                )
        
        return recommendations
    
    def get_hypothesis_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all hypotheses."""
        status = {}
        
        for hypothesis_id, hypothesis in self.hypotheses.items():
            status[hypothesis_id] = {
                'name': hypothesis.name,
                'category': hypothesis.category.value,
                'tested': hypothesis.tested,
                'pending': hypothesis_id in self.pending_hypotheses,
                'results': len(hypothesis.results),
                'confirmed': any(r.confirmed for r in hypothesis.results) if hypothesis.results else None
            }
        
        return status