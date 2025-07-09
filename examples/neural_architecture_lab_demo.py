#!/usr/bin/env python3
"""
Demonstration of the Neural Architecture Lab for testing structure_net hypotheses.

This example shows how to:
1. Create hypotheses about neural network architectures
2. Run systematic experiments to test them
3. Analyze results and extract insights
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import asyncio
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis,
    HypothesisCategory,
    get_all_hypotheses
)
from src.neural_architecture_lab.hypothesis_library import (
    ArchitectureHypotheses,
    GrowthHypotheses,
    SparsityHypotheses,
    TrainingHypotheses
)


def create_custom_hypotheses():
    """Create custom hypotheses specific to structure_net features."""
    
    hypotheses = []
    
    # Hypothesis 1: Adaptive Learning Rates
    def test_adaptive_lr(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Test function will be handled by the runner."""
        # The runner will handle the actual implementation
        # This is just a placeholder to define the hypothesis
        return None, {}
    
    adaptive_lr_hypothesis = Hypothesis(
        id="structure_net_adaptive_lr",
        name="Structure Net Adaptive Learning Rate Strategies",
        description="Compare all four adaptive learning rate strategies in structure_net",
        category=HypothesisCategory.TRAINING,
        question="Which adaptive learning rate strategy (basic/advanced/comprehensive/ultimate) performs best?",
        prediction="The 'ultimate' strategy will provide best accuracy with stable convergence",
        test_function=test_adaptive_lr,
        parameter_space={
            'lr_strategy': ['basic', 'advanced', 'comprehensive', 'ultimate'],
            'architecture': [[784, 256, 128, 10], [784, 512, 256, 128, 10]],
            'base_lr': {'min': 0.0001, 'max': 0.01, 'n_samples': 3, 'log_scale': True}
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 30,
            'batch_size': 128,
            'enable_metrics': True,
            'primary_metric_type': 'accuracy'
        },
        success_metrics={
            'accuracy': 0.5,
            'stability': 0.9  # Low variance in learning rate
        },
        tags=['structure_net', 'adaptive_lr', 'optimization']
    )
    hypotheses.append(adaptive_lr_hypothesis)
    
    # Hypothesis 2: Growth System Effectiveness
    def test_growth_system(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    growth_hypothesis = Hypothesis(
        id="structure_net_growth_v2",
        name="Integrated Growth System v2 Effectiveness",
        description="Test the IntegratedGrowthSystem v2 with different growth strategies",
        category=HypothesisCategory.GROWTH,
        question="Does the v2 growth system improve final accuracy while maintaining efficiency?",
        prediction="Growth will improve accuracy by 10-15% with <2x parameter increase",
        test_function=test_growth_system,
        parameter_space={
            'enable_growth': [True, False],
            'growth_interval': [5, 10, 20],
            'neurons_per_growth': [16, 32, 64],
            'architecture': [[784, 64, 10], [784, 128, 64, 10]]  # Start small to show growth
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 50,
            'batch_size': 128,
            'max_neurons': 2048,
            'lr_strategy': 'advanced',
            'primary_metric_type': 'efficiency'  # accuracy per parameter
        },
        success_metrics={
            'accuracy': 0.55,
            'efficiency': 0.01  # accuracy per million parameters
        },
        tags=['structure_net', 'growth', 'efficiency']
    )
    hypotheses.append(growth_hypothesis)
    
    # Hypothesis 3: Sparse Initialization Benefits
    def test_sparse_init(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    sparse_hypothesis = Hypothesis(
        id="structure_net_sparse_init",
        name="Sparse Initialization Performance",
        description="Test if sparse initialization improves training dynamics",
        category=HypothesisCategory.SPARSITY,
        question="Does sparse initialization lead to better final models?",
        prediction="5-10% sparsity will improve generalization without hurting accuracy",
        test_function=test_sparse_init,
        parameter_space={
            'sparsity': [0.0, 0.02, 0.05, 0.1, 0.2],
            'architecture': [[784, 256, 128, 10], [784, 512, 256, 10]]
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 40,
            'batch_size': 128,
            'lr_strategy': 'advanced',
            'primary_metric_type': 'accuracy'
        },
        success_metrics={
            'accuracy': 0.5,
            'generalization': 0.95  # val_acc / train_acc
        },
        tags=['structure_net', 'sparsity', 'initialization']
    )
    hypotheses.append(sparse_hypothesis)
    
    # Hypothesis 4: Residual Blocks with Growth
    def test_residual_growth(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    residual_hypothesis = Hypothesis(
        id="structure_net_residual_growth",
        name="Residual Blocks During Growth",
        description="Test if adding residual connections during growth improves training",
        category=HypothesisCategory.ARCHITECTURE,
        question="Do residual connections help when growing networks dynamically?",
        prediction="Residual connections will improve gradient flow in grown networks",
        test_function=test_residual_growth,
        parameter_space={
            'use_residual': [True, False],
            'skip_frequency': [2, 3],
            'enable_growth': [True],
            'growth_interval': [10, 15]
        },
        control_parameters={
            'architecture': [784, 128, 128, 128, 10],
            'dataset': 'cifar10',
            'epochs': 50,
            'batch_size': 128,
            'lr_strategy': 'comprehensive',
            'primary_metric_type': 'convergence_speed'
        },
        success_metrics={
            'accuracy': 0.55,
            'convergence_speed': 1.5  # Relative to non-residual
        },
        tags=['structure_net', 'residual', 'growth', 'architecture']
    )
    hypotheses.append(residual_hypothesis)
    
    # Hypothesis 5: Complete Metrics System
    def test_metrics_overhead(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    metrics_hypothesis = Hypothesis(
        id="structure_net_metrics_overhead",
        name="Metrics System Performance Impact",
        description="Measure the overhead of the complete metrics system",
        category=HypothesisCategory.OPTIMIZATION,
        question="What is the performance cost of comprehensive metrics collection?",
        prediction="Metrics will add <10% training time overhead while providing valuable insights",
        test_function=test_metrics_overhead,
        parameter_space={
            'enable_metrics': [True, False],
            'architecture': [[784, 256, 128, 10]],
            'sparsity': [0.02]
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 20,
            'batch_size': 256,
            'lr_strategy': 'basic',
            'quick_test': True,  # Use subset of data
            'primary_metric_type': 'accuracy'
        },
        success_metrics={
            'accuracy': 0.45,  # Lower bar due to quick test
            'overhead_ratio': 1.1  # Max 10% overhead
        },
        tags=['structure_net', 'metrics', 'performance']
    )
    hypotheses.append(metrics_hypothesis)
    
    return hypotheses


async def main():
    """Run the Neural Architecture Lab demonstration."""
    
    print("🔬 Neural Architecture Lab - Structure Net Testing")
    print("=" * 60)
    
    # Configure the lab
    config = LabConfig(
        max_parallel_experiments=4,  # Run 4 experiments in parallel
        experiment_timeout=1800,  # 30 minutes per experiment
        device_ids=[0, 1] if torch.cuda.device_count() > 1 else [0],
        min_experiments_per_hypothesis=5,
        require_statistical_significance=True,
        significance_level=0.05,
        results_dir="nal_structure_net_results",
        save_best_models=True,
        verbose=True,
        enable_adaptive_hypotheses=True,
        max_hypothesis_depth=2
    )
    
    # Create the lab
    lab = NeuralArchitectureLab(config)
    
    # Register hypotheses
    print("\n📋 Registering hypotheses...")
    
    # Add custom structure_net hypotheses
    custom_hypotheses = create_custom_hypotheses()
    for hypothesis in custom_hypotheses:
        lab.register_hypothesis(hypothesis)
    
    # Add some standard hypotheses from the library
    standard_hypotheses = [
        ArchitectureHypotheses.depth_vs_width(),
        GrowthHypotheses.growth_timing(),
        TrainingHypotheses.batch_size_scaling()
    ]
    
    for hypothesis in standard_hypotheses:
        lab.register_hypothesis(hypothesis)
    
    print(f"\nTotal hypotheses registered: {len(lab.hypotheses)}")
    
    # Show hypothesis categories
    categories = {}
    for h in lab.hypotheses.values():
        cat = h.category.value
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nHypotheses by category:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    
    # Run specific hypothesis or all
    print("\n🚀 Starting experiments...")
    
    # Option 1: Test a specific hypothesis
    # result = await lab.test_hypothesis("structure_net_adaptive_lr")
    
    # Option 2: Test all hypotheses
    results = await lab.run_all_hypotheses()
    
    print("\n✅ Experiments completed!")
    
    # Show results summary
    print("\n📊 Results Summary:")
    confirmed_count = sum(1 for r in results.values() if r.confirmed)
    print(f"  Confirmed hypotheses: {confirmed_count}/{len(results)}")
    
    # Show top findings
    print("\n🏆 Top Findings:")
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].effect_size,
        reverse=True
    )
    
    for hypothesis_id, result in sorted_results[:3]:
        hypothesis = lab.hypotheses[hypothesis_id]
        print(f"\n  {hypothesis.name}")
        print(f"    Confirmed: {'Yes' if result.confirmed else 'No'}")
        print(f"    Effect size: {result.effect_size:.3f}")
        print(f"    Best accuracy: {result.best_metrics.get('accuracy', 0):.3f}")
        if result.key_insights:
            print(f"    Key insight: {result.key_insights[0]}")
    
    # Show unexpected findings
    unexpected = []
    for hypothesis_id, result in results.items():
        if result.unexpected_findings:
            unexpected.extend([
                (lab.hypotheses[hypothesis_id].name, finding)
                for finding in result.unexpected_findings
            ])
    
    if unexpected:
        print("\n🤔 Unexpected Findings:")
        for name, finding in unexpected[:5]:
            print(f"  {name}: {finding}")
    
    # Show follow-up suggestions
    print("\n💡 Suggested Follow-up Experiments:")
    for hypothesis_id, result in results.items():
        for suggestion in result.suggested_hypotheses[:2]:
            print(f"  - {suggestion}")
    
    print(f"\n📁 Detailed results saved to: {config.results_dir}/")
    
    return results


if __name__ == "__main__":
    # Run the async main function
    results = asyncio.run(main())
    
    print("\n🎉 Neural Architecture Lab demonstration complete!")