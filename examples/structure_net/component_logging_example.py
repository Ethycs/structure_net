#!/usr/bin/env python3
"""
Example: Component-Based Logging with Photoshop-like Composability

This demonstrates how to use the component-based logging system to:
1. Create experiments from templates (like Photoshop templates)
2. Compose custom experiments from individual components
3. Log iterations with strict schema validation
4. Create WandB artifacts with proper structure
"""

import torch
import torch.nn as nn
from typing import List
import numpy as np

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logging import (
    ComponentLogger,
    create_evolution_experiment,
    MetricSchema,
    EvolverSchema,
    ModelSchema,
    TrainerSchema,
    NALSchema
)


def example_1_template_based():
    """Example 1: Using a pre-built template (like opening a PSD file)."""
    print("=" * 60)
    print("Example 1: Template-Based Experiment")
    print("=" * 60)
    
    # Create an evolution experiment from template
    logger, execution = create_evolution_experiment(
        hypothesis="Sparse networks can match dense network performance",
        architecture=[784, 512, 256, 128, 10],  # MNIST architecture
        # Customize template parameters
        metric__config__efficiency_weight=0.5,  # More weight on efficiency
        evolver__config__mutation_rate=0.2,     # Higher mutation rate
        trainer__config__epochs=20              # More epochs
    )
    
    print(f"Created experiment: {execution.execution_id}")
    print(f"Composition hash: {execution.composition.generate_hash()}")
    print(f"Template: {execution.composition.template_name}")
    
    # Simulate training iterations
    for iteration in range(5):
        # Simulate metrics from each component
        accuracy = 0.85 + iteration * 0.02 + np.random.uniform(-0.01, 0.01)
        loss = 0.5 - iteration * 0.08
        
        logger.log_iteration(
            execution_id=execution.execution_id,
            iteration=iteration,
            metric_outputs={
                "accuracy": accuracy,
                "efficiency": 0.95,  # High efficiency due to sparsity
                "fitness_score": accuracy * 0.5 + 0.95 * 0.5  # Weighted combination
            },
            trainer_metrics={
                "learning_rate": 0.001 * (0.9 ** iteration),  # Decay
                "gradient_norm": 2.5 - iteration * 0.3
            },
            accuracy=accuracy,
            loss=loss,
            evolver_actions=["mutate_weights"] if iteration % 2 == 0 else [],
            model_changes={"sparsity": 0.95 + iteration * 0.01} if iteration % 2 == 0 else None
        )
        
        print(f"  Iteration {iteration}: accuracy={accuracy:.3f}, loss={loss:.3f}")
    
    # Finalize and create artifact
    final_metrics = {
        "accuracy": 0.93,
        "efficiency": 0.95,
        "total_parameters": 550000,
        "active_parameters": 27500,  # 5% of total
        "inference_time_ms": 1.2
    }
    
    artifact_hash = logger.finalize_experiment(
        execution_id=execution.execution_id,
        final_metrics=final_metrics,
        status="completed"
    )
    
    print(f"\n✅ Experiment completed!")
    print(f"   Artifact hash: {artifact_hash}")
    print(f"   Final accuracy: {final_metrics['accuracy']}")
    print(f"   Efficiency gain: {final_metrics['efficiency']}")


def example_2_custom_composition():
    """Example 2: Custom component composition (like creating layers from scratch)."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Component Composition")
    print("=" * 60)
    
    logger = ComponentLogger()
    
    # Create custom components (like adding Photoshop layers)
    
    # Layer 1: Geometric metric analyzer
    metric = MetricSchema(
        component_id="metric_geometric_001",
        metric_name="geometric_curvature",
        outputs=["curvature", "geodesic_distance", "manifold_dimension"],
        requires_gradients=True,
        config={
            "n_samples": 1000,
            "epsilon": 0.01
        }
    )
    
    # Layer 2: Curvature-guided evolver
    evolver = EvolverSchema(
        component_id="evolver_curvature_001",
        evolver_name="curvature_minimizer",
        inputs=["curvature", "geodesic_distance"],
        outputs=["smooth_connections", "reduce_curvature"],
        preserves_function=True,
        config={
            "smoothing_rate": 0.1,
            "target_curvature": 0.5
        }
    )
    
    # Layer 3: Fiber bundle model
    model = ModelSchema(
        component_id="model_fiber_001",
        model_name="fiber_bundle_network",
        architecture=[784, 1024, 512, 256, 10],
        total_parameters=1200000,
        sparsity=0.0,  # Dense initially
        supports_growth=False,  # Fixed topology
        config={
            "bundle_dim": 4,
            "base_manifold": "hyperbolic"
        }
    )
    
    # Layer 4: Geometry-aware trainer
    trainer = TrainerSchema(
        component_id="trainer_geometric_001",
        trainer_name="riemannian_sgd",
        optimizer="custom",
        learning_rate=0.01,
        batch_size=64,
        config={
            "metric_tensor": "fisher_information",
            "natural_gradient": True
        }
    )
    
    # Layer 5: Geometric hypothesis testing
    nal = NALSchema(
        component_id="nal_geometric_001",
        hypothesis="Geometric constraints improve generalization",
        statistical_tests=["manifold_t_test", "curvature_correlation"],
        success_criteria={
            "generalization_gap": 0.05,  # Max 5% gap
            "curvature_reduction": 0.3   # 30% reduction
        },
        config={
            "significance_level": 0.01,
            "multiple_comparison_correction": "bonferroni"
        }
    )
    
    # Compose the experiment (stack the layers)
    execution = logger.create_experiment_from_components(
        execution_id="geometric_exp_001",
        metric=metric,
        evolver=evolver,
        model=model,
        trainer=trainer,
        nal=nal,
        name="Geometric Deep Learning Experiment"
    )
    
    print(f"Created custom experiment: {execution.execution_id}")
    print(f"Composition: {execution.composition.name}")
    print("\nComponent stack:")
    print(f"  1. Metric:  {metric.metric_name}")
    print(f"  2. Evolver: {evolver.evolver_name}")
    print(f"  3. Model:   {model.model_name}")
    print(f"  4. Trainer: {trainer.trainer_name}")
    print(f"  5. NAL:     {nal.hypothesis[:30]}...")
    
    # Simulate geometric training
    for iteration in range(3):
        curvature = 2.0 - iteration * 0.5  # Decreasing curvature
        accuracy = 0.80 + iteration * 0.05
        
        logger.log_iteration(
            execution_id=execution.execution_id,
            iteration=iteration,
            metric_outputs={
                "curvature": curvature,
                "geodesic_distance": 1.5 - iteration * 0.2,
                "manifold_dimension": 3.8 - iteration * 0.1
            },
            trainer_metrics={
                "riemannian_norm": 0.8 - iteration * 0.1,
                "metric_tensor_trace": 4.5 + iteration * 0.3
            },
            accuracy=accuracy,
            loss=1.0 - accuracy,
            evolver_actions=["smooth_connections", "reduce_curvature"],
            model_changes={"curvature": curvature},
            nal_decisions={"significance": 0.001 if iteration == 2 else 0.1}
        )
        
        print(f"  Iteration {iteration}: curvature={curvature:.2f}, accuracy={accuracy:.3f}")
    
    # Finalize with geometric metrics
    final_metrics = {
        "accuracy": 0.95,
        "generalization_gap": 0.03,  # 3% gap - SUCCESS!
        "curvature_reduction": 0.5,   # 50% reduction - SUCCESS!
        "final_curvature": 0.5,
        "manifold_stability": 0.98
    }
    
    artifact_hash = logger.finalize_experiment(
        execution_id=execution.execution_id,
        final_metrics=final_metrics,
        status="completed"
    )
    
    print(f"\n✅ Geometric experiment completed!")
    print(f"   Hypothesis confirmed: YES")
    print(f"   Generalization gap: {final_metrics['generalization_gap']:.1%}")
    print(f"   Curvature reduction: {final_metrics['curvature_reduction']:.1%}")


def example_3_component_reuse():
    """Example 3: Reusing components (like Smart Objects)."""
    print("\n" + "=" * 60)
    print("Example 3: Component Reuse")
    print("=" * 60)
    
    # Create a reusable metric component
    shared_metric = MetricSchema(
        component_id="metric_shared_001",
        metric_name="multi_scale_analysis",
        outputs=["local_features", "global_features", "scale_ratio"],
        config={"scales": [1, 4, 16, 64]}
    )
    
    print("Created shared metric component: multi_scale_analysis")
    print("This component can be reused across multiple experiments\n")
    
    # Use the same metric in two different experiments
    experiments = []
    
    for i, evolver_type in enumerate(["adaptive_growth", "pruning_only"]):
        logger = ComponentLogger()
        
        # Reuse the shared metric
        evolver = EvolverSchema(
            component_id=f"evolver_{i}",
            evolver_name=evolver_type,
            inputs=["scale_ratio"],  # Uses output from shared metric
            outputs=["modify_architecture"],
            config={}
        )
        
        # Standard components for comparison
        model = ModelSchema(
            component_id=f"model_{i}",
            model_name="standard_mlp",
            architecture=[784, 256, 10],
            total_parameters=201000,
            sparsity=0.5,
            config={}
        )
        
        trainer = TrainerSchema(
            component_id=f"trainer_{i}",
            trainer_name="standard",
            optimizer="adam",
            learning_rate=0.001,
            batch_size=128,
            config={}
        )
        
        nal = NALSchema(
            component_id=f"nal_{i}",
            hypothesis=f"{evolver_type} improves multi-scale performance",
            success_criteria={"scale_ratio_improvement": 0.2},
            config={}
        )
        
        execution = logger.create_experiment_from_components(
            execution_id=f"reuse_exp_{i}",
            metric=shared_metric,  # REUSE!
            evolver=evolver,
            model=model,
            trainer=trainer,
            nal=nal,
            name=f"Component reuse: {evolver_type}"
        )
        
        experiments.append((evolver_type, execution))
        print(f"Experiment {i+1}: {evolver_type} + shared multi_scale_analysis")
    
    print("\nBoth experiments use the SAME metric component (like a Smart Object)")
    print("This ensures consistent analysis across different evolvers")


if __name__ == "__main__":
    # Run all examples
    example_1_template_based()
    example_2_custom_composition()
    example_3_component_reuse()
    
    print("\n" + "=" * 60)
    print("Component-based logging demonstration complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Templates provide quick experiment setup (like PSD templates)")
    print("2. Custom compositions allow full flexibility (like building from layers)")
    print("3. Components are reusable across experiments (like Smart Objects)")
    print("4. All data is strictly validated before creating WandB artifacts")
    print("5. The 5-layer system ensures consistent experiment structure")