#!/usr/bin/env python3
"""
Neural Architecture Lab Stress Test with Seed Model Support

This refactored version uses the NAL framework to run systematic experiments
with seed models from promising_models directory. It provides:
- Scientific hypothesis testing approach
- Comprehensive metrics integration
- Seed model evolution experiments
- Statistical analysis of results
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import json
from datetime import datetime

from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis,
    HypothesisCategory
)
from src.structure_net.core.io_operations import load_model_seed


def create_seed_evolution_hypotheses(seed_models: List[str]) -> List[Hypothesis]:
    """Create hypotheses for testing seed model evolution strategies."""
    hypotheses = []
    
    # Hypothesis 1: Seed Model Fine-tuning
    def test_seed_finetuning(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Test function for seed fine-tuning - handled by runner."""
        return None, {}
    
    seed_finetune_hypothesis = Hypothesis(
        id="seed_model_finetuning",
        name="Seed Model Fine-tuning Effectiveness",
        description="Test if fine-tuning pre-trained seed models outperforms training from scratch",
        category=HypothesisCategory.TRAINING,
        question="Do seed models provide better starting points than random initialization?",
        prediction="Seed models will converge faster and achieve 5-10% higher accuracy",
        test_function=test_seed_finetuning,
        parameter_space={
            'use_seed': [True, False],
            'seed_path': seed_models[:3] if len(seed_models) > 3 else seed_models,  # Test top 3 seeds
            'lr_strategy': ['advanced', 'comprehensive', 'ultimate'],
            'fine_tune_epochs': [10, 20, 30]
        },
        control_parameters={
            'dataset': 'cifar10',
            'batch_size': 128,
            'base_lr': 0.001,
            'primary_metric_type': 'accuracy'
        },
        success_metrics={
            'accuracy': 0.5,
            'convergence_speed': 1.5  # 50% faster than baseline
        },
        tags=['seed_models', 'fine_tuning', 'transfer_learning']
    )
    hypotheses.append(seed_finetune_hypothesis)
    
    # Hypothesis 2: Architecture Mutation from Seeds
    def test_seed_mutations(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    seed_mutation_hypothesis = Hypothesis(
        id="seed_architecture_mutations",
        name="Seed Architecture Evolution",
        description="Test different mutation strategies for evolving seed architectures",
        category=HypothesisCategory.ARCHITECTURE,
        question="Which mutation strategy produces the best evolved architectures from seeds?",
        prediction="Layer size mutations will outperform depth mutations",
        test_function=test_seed_mutations,
        parameter_space={
            'seed_path': seed_models[0] if seed_models else None,  # Use best seed
            'mutation_type': ['layer_size', 'add_layer', 'remove_layer', 'mixed'],
            'mutation_rate': [0.1, 0.2, 0.3],
            'num_mutations': [1, 2, 3]
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 30,
            'batch_size': 128,
            'lr_strategy': 'comprehensive',
            'primary_metric_type': 'efficiency'
        },
        success_metrics={
            'accuracy': 0.5,
            'efficiency': 0.01  # accuracy per million parameters
        },
        tags=['seed_models', 'architecture_search', 'evolution']
    )
    if seed_models:  # Only add if we have seed models
        hypotheses.append(seed_mutation_hypothesis)
    
    # Hypothesis 3: Growth from Seed Models
    def test_seed_growth(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    seed_growth_hypothesis = Hypothesis(
        id="seed_model_growth",
        name="Growing Pre-trained Networks",
        description="Test if growing pre-trained networks is more effective than growing from scratch",
        category=HypothesisCategory.GROWTH,
        question="Does starting from a pre-trained seed improve growth effectiveness?",
        prediction="Pre-trained seeds will show 20% better growth efficiency",
        test_function=test_seed_growth,
        parameter_space={
            'use_seed': [True, False],
            'seed_path': seed_models[0] if seed_models else None,
            'enable_growth': [True],
            'growth_interval': [5, 10],
            'neurons_per_growth': [16, 32],
            'growth_strategy': ['uniform', 'targeted', 'gradient_based']
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 50,
            'batch_size': 128,
            'max_neurons': 2048,
            'lr_strategy': 'ultimate',
            'primary_metric_type': 'efficiency'
        },
        success_metrics={
            'accuracy': 0.55,
            'growth_efficiency': 0.02  # accuracy gain per growth event
        },
        tags=['seed_models', 'growth', 'network_expansion']
    )
    if seed_models:
        hypotheses.append(seed_growth_hypothesis)
    
    # Hypothesis 4: Ensemble of Seeds
    def test_seed_ensemble(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    seed_ensemble_hypothesis = Hypothesis(
        id="seed_model_ensemble",
        name="Seed Model Ensembling",
        description="Test if ensembling multiple seed models improves performance",
        category=HypothesisCategory.ARCHITECTURE,
        question="Does combining predictions from multiple seeds outperform single models?",
        prediction="Ensemble of 3+ seeds will achieve 3-5% accuracy improvement",
        test_function=test_seed_ensemble,
        parameter_space={
            'ensemble_size': [2, 3, 5] if len(seed_models) >= 5 else [2, len(seed_models)],
            'ensemble_method': ['average', 'weighted', 'stacking'],
            'seed_paths': [seed_models]  # Will be subsampled by ensemble_size
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 20,  # Just for ensemble training
            'batch_size': 128,
            'primary_metric_type': 'accuracy'
        },
        success_metrics={
            'accuracy': 0.55,
            'ensemble_gain': 1.03  # 3% improvement over best single model
        },
        tags=['seed_models', 'ensemble', 'model_combination']
    )
    if len(seed_models) >= 2:
        hypotheses.append(seed_ensemble_hypothesis)
    
    # Hypothesis 5: Sparsity Evolution from Seeds
    def test_seed_sparsity(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    seed_sparsity_hypothesis = Hypothesis(
        id="seed_sparsity_evolution",
        name="Sparsity Evolution in Seed Models",
        description="Test progressive sparsification of pre-trained models",
        category=HypothesisCategory.SPARSITY,
        question="Can we increase sparsity in seed models without losing accuracy?",
        prediction="Progressive pruning will maintain 95% accuracy at 50% sparsity",
        test_function=test_seed_sparsity,
        parameter_space={
            'seed_path': seed_models[0] if seed_models else None,
            'target_sparsity': [0.3, 0.5, 0.7],
            'pruning_schedule': ['linear', 'exponential', 'magnitude'],
            'fine_tune_after_prune': [True, False]
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 30,
            'batch_size': 128,
            'lr_strategy': 'advanced',
            'primary_metric_type': 'efficiency'
        },
        success_metrics={
            'accuracy_retention': 0.95,  # Maintain 95% of original accuracy
            'sparsity': 0.5  # Achieve 50% sparsity
        },
        tags=['seed_models', 'sparsity', 'pruning', 'compression']
    )
    if seed_models:
        hypotheses.append(seed_sparsity_hypothesis)
    
    return hypotheses


def create_stress_test_hypotheses() -> List[Hypothesis]:
    """Create comprehensive stress test hypotheses for all structure_net features."""
    hypotheses = []
    
    # Hypothesis 1: Ultimate Learning Rate Strategy
    def test_ultimate_lr(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    ultimate_lr_hypothesis = Hypothesis(
        id="ultimate_lr_stress_test",
        name="Ultimate Learning Rate Strategy Stress Test",
        description="Stress test the ultimate learning rate strategy with complex architectures",
        category=HypothesisCategory.TRAINING,
        question="Does the ultimate LR strategy handle complex training scenarios effectively?",
        prediction="Ultimate strategy will show superior stability and convergence",
        test_function=test_ultimate_lr,
        parameter_space={
            'architecture': [
                [784, 512, 256, 128, 64, 10],
                [784, 1024, 512, 256, 128, 64, 32, 10],
                [784, 256, 512, 1024, 512, 256, 10]  # Hourglass
            ],
            'lr_strategy': ['basic', 'advanced', 'comprehensive', 'ultimate'],
            'batch_size': [64, 128, 256],
            'enable_growth': [True, False]
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 50,
            'sparsity': 0.02,
            'enable_metrics': True,
            'primary_metric_type': 'convergence_speed'
        },
        success_metrics={
            'accuracy': 0.5,
            'convergence_speed': 1.0,
            'stability': 0.9
        },
        tags=['stress_test', 'learning_rate', 'ultimate', 'complex_architectures']
    )
    hypotheses.append(ultimate_lr_hypothesis)
    
    # Hypothesis 2: Integrated Growth System v2
    def test_growth_v2(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    growth_v2_hypothesis = Hypothesis(
        id="growth_system_v2_stress",
        name="Integrated Growth System v2 Stress Test",
        description="Test growth system with extreme growth scenarios",
        category=HypothesisCategory.GROWTH,
        question="How does the growth system handle rapid expansion and complex triggers?",
        prediction="Gradient-based growth will outperform fixed-interval growth",
        test_function=test_growth_v2,
        parameter_space={
            'initial_architecture': [[784, 32, 10], [784, 64, 32, 10]],
            'growth_trigger': ['epoch_based', 'loss_plateau', 'gradient_based'],
            'growth_interval': [3, 5, 10],
            'neurons_per_growth': [32, 64, 128],
            'max_growth_events': [10, 15, 20]
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs': 100,
            'batch_size': 128,
            'lr_strategy': 'ultimate',
            'enable_residual': True,
            'primary_metric_type': 'efficiency'
        },
        success_metrics={
            'final_accuracy': 0.6,
            'growth_efficiency': 0.01,
            'parameter_efficiency': 0.01
        },
        tags=['stress_test', 'growth', 'extreme_scenarios']
    )
    hypotheses.append(growth_v2_hypothesis)
    
    # Hypothesis 3: Complete Metrics System
    def test_metrics_system(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    metrics_hypothesis = Hypothesis(
        id="complete_metrics_stress",
        name="Complete Metrics System Under Load",
        description="Test all metrics analyzers with large networks and long training",
        category=HypothesisCategory.OPTIMIZATION,
        question="What is the performance impact of comprehensive metrics collection?",
        prediction="Full metrics will add <15% overhead while providing valuable insights",
        test_function=test_metrics_system,
        parameter_space={
            'enable_metrics': [True, False],
            'metrics_config': [
                {'minimal': True},
                {'standard': True},
                {'comprehensive': True},
                {'everything': True}
            ],
            'architecture': [[784, 512, 256, 128, 10]],
            'epochs': [20, 50]
        },
        control_parameters={
            'dataset': 'cifar10',
            'batch_size': 256,
            'lr_strategy': 'comprehensive',
            'primary_metric_type': 'accuracy'
        },
        success_metrics={
            'accuracy': 0.45,
            'overhead_ratio': 1.15  # Max 15% overhead
        },
        tags=['stress_test', 'metrics', 'performance', 'profiling']
    )
    hypotheses.append(metrics_hypothesis)
    
    return hypotheses


def find_best_seeds(top_n: int = 5) -> List[str]:
    """Find the best seed models from promising_models directory."""
    models_dir = Path("data/promising_models")
    
    if not models_dir.exists():
        print("❌ No promising_models directory found")
        return []
    
    # Find all model files
    model_files = list(models_dir.glob("**/model_cifar10_*.pt"))
    
    if not model_files:
        print("❌ No models found in promising_models")
        return []
    
    # Parse and rank models
    models_info = []
    for model_path in model_files:
        filename = model_path.name
        if '_acc' in filename:
            try:
                acc_str = filename.split('_acc')[1].split('_')[0]
                accuracy = float(acc_str)
                models_info.append((str(model_path), accuracy))
            except:
                continue
    
    # Sort by accuracy
    models_info.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top {top_n} seed models by accuracy:")
    for i, (path, acc) in enumerate(models_info[:top_n]):
        print(f"   {i+1}. {Path(path).name}")
        print(f"      Accuracy: {acc:.2%}")
    
    return [path for path, _ in models_info[:top_n]]


async def main():
    """Run NAL-based stress test with seed models."""
    parser = argparse.ArgumentParser(
        description="Neural Architecture Lab Stress Test with Seed Models"
    )
    parser.add_argument("--seeds", nargs='+', help="Paths to seed models")
    parser.add_argument("--auto-seeds", type=int, default=0, 
                       help="Automatically select top N seed models")
    parser.add_argument("--hypotheses", nargs='+', 
                       choices=['seed', 'stress', 'all'], default=['all'],
                       help="Which hypothesis sets to test")
    parser.add_argument("--max-parallel", type=int, default=4,
                       help="Maximum parallel experiments")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with reduced parameters")
    parser.add_argument("--save-models", action="store_true",
                       help="Save best models from experiments")
    
    args = parser.parse_args()
    
    print("🔬 Neural Architecture Lab - Stress Test with Seed Models")
    print("=" * 70)
    
    # Gather seed models
    seed_models = []
    if args.seeds:
        seed_models.extend(args.seeds)
    
    if args.auto_seeds > 0:
        auto_seeds = find_best_seeds(args.auto_seeds)
        seed_models.extend(auto_seeds)
    
    # Remove duplicates
    seed_models = list(dict.fromkeys(seed_models))
    
    if seed_models:
        print(f"\n🌱 Using {len(seed_models)} seed models")
    else:
        print("\n📝 No seed models specified")
    
    # Configure the lab
    config = LabConfig(
        max_parallel_experiments=args.max_parallel,
        experiment_timeout=3600 if not args.quick else 600,  # 1 hour or 10 min
        device_ids=[0, 1] if torch.cuda.device_count() > 1 else [0],
        min_experiments_per_hypothesis=3 if args.quick else 5,
        require_statistical_significance=not args.quick,
        significance_level=0.05,
        results_dir=f"nal_stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_best_models=args.save_models,
        verbose=True,
        enable_adaptive_hypotheses=True,
        max_hypothesis_depth=2 if not args.quick else 1
    )
    
    # Create the lab
    lab = NeuralArchitectureLab(config)
    
    # Register hypotheses
    print("\n📋 Registering hypotheses...")
    
    hypothesis_sets = []
    if 'seed' in args.hypotheses or 'all' in args.hypotheses:
        if seed_models:
            seed_hypotheses = create_seed_evolution_hypotheses(seed_models)
            hypothesis_sets.extend(seed_hypotheses)
            print(f"   Added {len(seed_hypotheses)} seed evolution hypotheses")
        else:
            print("   ⚠️  No seed models available for seed hypotheses")
    
    if 'stress' in args.hypotheses or 'all' in args.hypotheses:
        stress_hypotheses = create_stress_test_hypotheses()
        hypothesis_sets.extend(stress_hypotheses)
        print(f"   Added {len(stress_hypotheses)} stress test hypotheses")
    
    # Register all hypotheses
    for hypothesis in hypothesis_sets:
        lab.register_hypothesis(hypothesis)
    
    print(f"\nTotal hypotheses registered: {len(lab.hypotheses)}")
    
    # Run experiments
    print("\n🚀 Starting experiments...")
    start_time = time.time()
    
    results = await lab.run_all_hypotheses()
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Experiments completed in {elapsed_time:.1f} seconds!")
    
    # Generate comprehensive report
    report = {
        'configuration': {
            'seed_models': seed_models,
            'hypotheses_tested': list(results.keys()),
            'total_experiments': lab.total_experiments_run,
            'elapsed_time': elapsed_time
        },
        'results_summary': {},
        'best_configurations': {},
        'key_insights': []
    }
    
    # Analyze results
    print("\n📊 Results Summary:")
    confirmed_count = sum(1 for r in results.values() if r.confirmed)
    print(f"  Confirmed hypotheses: {confirmed_count}/{len(results)}")
    
    # Find best results by category
    categories = {}
    for hypothesis_id, result in results.items():
        hypothesis = lab.hypotheses[hypothesis_id]
        category = hypothesis.category.value
        
        if category not in categories:
            categories[category] = []
        
        categories[category].append({
            'hypothesis': hypothesis.name,
            'confirmed': result.confirmed,
            'effect_size': result.effect_size,
            'best_accuracy': result.best_metrics.get('accuracy', 0)
        })
    
    # Report by category
    for category, cat_results in categories.items():
        print(f"\n  {category.upper()}:")
        cat_results.sort(key=lambda x: x['effect_size'], reverse=True)
        for res in cat_results[:2]:
            print(f"    - {res['hypothesis']}")
            print(f"      Confirmed: {'✓' if res['confirmed'] else '✗'}")
            print(f"      Effect size: {res['effect_size']:.3f}")
            print(f"      Best accuracy: {res['best_accuracy']:.3f}")
    
    # Special seed model results
    if seed_models:
        print("\n🌱 Seed Model Performance:")
        seed_results = [
            (h_id, r) for h_id, r in results.items() 
            if 'seed' in h_id
        ]
        
        if seed_results:
            for h_id, result in seed_results[:3]:
                hypothesis = lab.hypotheses[h_id]
                print(f"  {hypothesis.name}:")
                if result.confirmed:
                    print(f"    ✓ Hypothesis confirmed!")
                if result.key_insights:
                    print(f"    Key insight: {result.key_insights[0]}")
    
    # Save detailed report
    report_file = Path(config.results_dir) / "stress_test_report.json"
    
    # Add detailed results to report
    for hypothesis_id, result in results.items():
        hypothesis = lab.hypotheses[hypothesis_id]
        report['results_summary'][hypothesis_id] = {
            'name': hypothesis.name,
            'category': hypothesis.category.value,
            'confirmed': result.confirmed,
            'confidence': result.confidence,
            'effect_size': result.effect_size,
            'best_metrics': result.best_metrics,
            'num_experiments': result.num_experiments,
            'insights': result.key_insights[:3] if result.key_insights else []
        }
    
    # Extract best configurations
    for hypothesis_id, result in results.items():
        if result.confirmed and result.best_parameters:
            report['best_configurations'][hypothesis_id] = {
                'parameters': result.best_parameters,
                'metrics': result.best_metrics
            }
    
    # Compile key insights
    all_insights = []
    for result in results.values():
        all_insights.extend(result.key_insights)
    
    # Deduplicate and take top insights
    unique_insights = list(dict.fromkeys(all_insights))
    report['key_insights'] = unique_insights[:10]
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {config.results_dir}/")
    print(f"📄 Report saved to: {report_file}")
    
    # Show top recommendations
    print("\n💡 Top Recommendations:")
    for insight in report['key_insights'][:5]:
        print(f"  • {insight}")
    
    return results


if __name__ == "__main__":
    # Run the async main function
    import time
    start = time.time()
    results = asyncio.run(main())
    total_time = time.time() - start
    
    print(f"\n🎉 NAL stress test completed in {total_time:.1f} seconds!")
    print("🔬 Use the detailed results to guide further experiments and optimizations.")