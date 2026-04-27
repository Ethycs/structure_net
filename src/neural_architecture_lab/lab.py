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
from dataclasses import asdict
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .core import (
    Hypothesis, HypothesisResult, Experiment, ExperimentResult,
    LabConfig, ExperimentStatus, HypothesisCategory
)
from .runners import AsyncExperimentRunner
from .analyzers import InsightExtractor, StatisticalAnalyzer
from src.structure_net.logging.standardized_logging import StandardizedLogger, LoggingConfig
import logging

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
        
        # Create results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Logging
        self.setup_logging()

        # Initialize components
        self.runner = AsyncExperimentRunner(config)
        self.runner.logger = self.logger  # Pass logger to runner
        self.insight_extractor = InsightExtractor()
        self.statistical_analyzer = StatisticalAnalyzer(
            significance_level=config.significance_level
        )
        
        # Lab state
        self.lab_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = None
        self.total_experiments_run = 0
        
        # Hypothesis tracking
        self.hypothesis_tree = {}
        self.pending_hypotheses = []

    def setup_logging(self):
        """Configures the logging for the lab based on its own config object."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        handlers = [logging.StreamHandler()]
        if self.config.log_file:
            log_file_path = self.results_dir / self.config.log_file
            handlers.append(logging.FileHandler(log_file_path))
            
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # Override any existing basicConfig
        )
        
        if self.config.module_log_levels:
            for module, level in self.config.module_log_levels.items():
                level_val = getattr(logging, level.upper(), logging.INFO)
                logging.getLogger(module).setLevel(level_val)

        # The StandardizedLogger for experiments is configured separately
        # and doesn't rely on the root logger.
        logging_config = LoggingConfig(
            project_name=self.config.project_name,
            queue_dir=str(self.results_dir / "experiment_queue"),
            sent_dir=str(self.results_dir / "experiment_sent"),
            rejected_dir=str(self.results_dir / "experiment_rejected"),
            enable_wandb=self.config.enable_wandb
        )
        self.logger = StandardizedLogger(logging_config)

        if self.config.verbose:
            logging.info(f"NAL logging initialized for project: {self.config.project_name}")
            logging.info(f"Log level set to {self.config.log_level}")
            if self.config.log_file:
                logging.info(f"Logging to file: {log_file_path}")
            if self.config.enable_wandb:
                logging.info("Weights & Biases logging is enabled.")

        
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
        
        # Log hypothesis to ChromaDB
        hyp_dict = {
            'id': hypothesis.id,
            'name': hypothesis.name,
            'description': hypothesis.description,
            'category': hypothesis.category.value,
            'question': hypothesis.question,
            'prediction': hypothesis.prediction,
            'created_at': hypothesis.created_at.isoformat(),
            'tested': hypothesis.tested
        }
        self.logger.log_hypothesis(hyp_dict)
        
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
        
        param_combinations = self._generate_parameter_grid(hypothesis.parameter_space)
        
        if len(param_combinations) < self.config.min_experiments_per_hypothesis:
            original_combinations = param_combinations.copy()
            seeds_needed = self.config.min_experiments_per_hypothesis // len(original_combinations) + 1
            param_combinations = []
            for seed in range(seeds_needed):
                for params in original_combinations:
                    param_combinations.append({**params, 'seed': seed})
        
        for i, params in enumerate(param_combinations[:self.config.min_experiments_per_hypothesis * 2]):
            full_params = {**hypothesis.control_parameters, **params}
            
            experiment = Experiment(
                id=f"{hypothesis.id}_exp_{i:03d}",
                hypothesis_id=hypothesis.id,
                name=f"{hypothesis.name} - Experiment {i+1}",
                parameters=full_params,
                seed=params.get('seed', i)
            )
            experiments.append(experiment)
        
        return experiments
    
    def _generate_parameter_grid(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter combinations from parameter space."""
        combinations = [{}]
        
        for param_name, param_spec in param_space.items():
            new_combinations = []
            
            if isinstance(param_spec, list):
                values = param_spec
            elif isinstance(param_spec, dict):
                if 'min' in param_spec and 'max' in param_spec:
                    n_samples = param_spec.get('n_samples', 3)
                    if param_spec.get('log_scale', False):
                        values = np.logspace(np.log10(param_spec['min']), np.log10(param_spec['max']), n_samples)
                    else:
                        values = np.linspace(param_spec['min'], param_spec['max'], n_samples)
                elif 'values' in param_spec:
                    values = param_spec['values']
            else:
                values = [param_spec]

            for value in values:
                for combo in combinations:
                    new_combinations.append({**combo, param_name: value})
            
            combinations = new_combinations
        
        return combinations
    
    async def test_hypothesis(self, hypothesis_id: str) -> HypothesisResult:
        """
        Test a hypothesis by running all its experiments.
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        if self.config.verbose:
            print(f"\n🧪 Testing hypothesis: {hypothesis.name}")
        
        experiments = self.generate_experiments(hypothesis)
        
        if self.config.verbose:
            print(f"   Generated {len(experiments)} experiments")
        
        experiment_results = await self.runner.run_experiments(experiments, hypothesis.test_function)
        
        for result in experiment_results:
            self.logger.log_experiment_result(result)

        self.total_experiments_run += len(experiment_results)
        
        # Pass the actual experiment results for analysis
        hypothesis_result = self._analyze_hypothesis_results(
            hypothesis, experiment_results
        )
        
        hypothesis.tested = True
        
        self._save_hypothesis_results(hypothesis, hypothesis_result)
        
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
        
        successful_results = [r for r in experiment_results if r.error is None]
        
        if not successful_results:
            return HypothesisResult(
                hypothesis_id=hypothesis.id,
                num_experiments=len(experiment_results),
                successful_experiments=0,
                failed_experiments=len(experiment_results),
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
        
        statistical_summary = self.statistical_analyzer.analyze_results(
            successful_results, hypothesis.success_metrics
        )
        
        confirmed = statistical_summary['meets_success_criteria']
        if self.config.require_statistical_significance:
            confirmed = confirmed and statistical_summary['statistically_significant']
        
        best_result = max(successful_results, key=lambda r: r.primary_metric)
        
        insights = self.insight_extractor.extract_insights(
            hypothesis, successful_results, statistical_summary
        )
        
        return HypothesisResult(
            hypothesis_id=hypothesis.id,
            num_experiments=len(experiment_results),
            successful_experiments=len(successful_results),
            failed_experiments=len(experiment_results) - len(successful_results),
            confirmed=confirmed,
            confidence=statistical_summary['confidence'],
            effect_size=statistical_summary.get('effect_size', 0.0),
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
        # TODO: Implement follow-up hypothesis generation
        return []
    
    def _refine_parameter_space(
        self,
        original_space: Dict[str, Any],
        result: HypothesisResult
    ) -> Dict[str, Any]:
        # ... (implementation remains the same)
        pass
    
    def _get_hypothesis_depth(self, hypothesis_id: str) -> int:
        # ... (implementation remains the same)
        pass
    
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
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyJSONEncoder)
    
    async def run_all_hypotheses(self) -> Dict[str, HypothesisResult]:
        """
        Run all pending hypotheses.
        """
        self.start_time = time.time()
        
        if self.config.verbose:
            print(f"\n🔬 Neural Architecture Lab Session {self.lab_id}")
        
        results = {}
        
        while self.pending_hypotheses:
            hypothesis_id = self.pending_hypotheses.pop(0)
            
            try:
                result = await self.test_hypothesis(hypothesis_id)
                results[hypothesis_id] = result
                
            except Exception as e:
                print(f"\n❌ Failed to test hypothesis {hypothesis_id}: {e}")
                import traceback
                traceback.print_exc()
        
        self._generate_lab_report(results)
        
        return results
    
    def _generate_lab_report(self, results: Dict[str, HypothesisResult]):
        # ... (implementation remains the same)
        pass
    
    def _generate_recommendations(self, results: Dict[str, HypothesisResult]) -> List[str]:
        # ... (implementation remains the same)
        pass
    
    def get_hypothesis_status(self) -> Dict[str, Dict[str, Any]]:
        # ... (implementation remains the same)
        pass