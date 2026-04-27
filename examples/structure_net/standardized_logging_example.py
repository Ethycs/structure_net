#!/usr/bin/env python3
"""
Example: Standardized Logging System Usage

Demonstrates how to use the new standardized logging system with
WandB artifact integration and Pydantic validation.
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import time

# Import the standardized logging system
from src.logging.standardized_logging import (
    StandardizedLogger,
    LoggingConfig,
    ExperimentResult,
    ExperimentConfig,
    MetricsData,
    GrowthEvent,
    HomologicalMetrics,
    TopologicalMetrics,
    CompactificationMetrics,
    initialize_logging,
    log_experiment,
    log_metrics,
    log_growth_event
)

# Import metrics analyzers
from src.structure_net.evolution.metrics import (
    HomologicalAnalyzer,
    TopologicalAnalyzer,
    CompactificationAnalyzer,
    create_homological_analyzer,
    create_topological_analyzer,
    create_compactification_analyzer
)


def create_sample_network():
    """Create a sample network for demonstration."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def simulate_training_metrics():
    """Simulate training metrics for demonstration."""
    return {
        "accuracy": np.random.uniform(0.7, 0.95),
        "loss": np.random.uniform(0.1, 0.5),
        "epoch": np.random.randint(1, 100),
        "iteration": np.random.randint(1, 1000),
        "learning_rate": np.random.uniform(0.0001, 0.01),
        "total_parameters": 100000,
        "active_connections": 80000,
        "sparsity": 0.2,
        "growth_occurred": np.random.choice([True, False]),
        "architecture": [784, 256, 128, 10],
        "extrema_ratio": np.random.uniform(0.1, 0.3)
    }


def demonstrate_basic_logging():
    """Demonstrate basic logging functionality."""
    print("🔧 Demonstrating Basic Logging")
    print("=" * 50)
    
    # Initialize logging system
    config = LoggingConfig(
        project_name="structure_net_demo",
        enable_wandb=False,  # Disable for demo
        auto_upload=False
    )
    
    logger = initialize_logging(config)
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        experiment_id="demo_experiment_001",
        experiment_type="growth_tracking",
        dataset="MNIST",
        model_type="sparse_mlp",
        batch_size=64,
        learning_rate=0.001,
        epochs=50,
        seed_architecture=[784, 256, 128, 10],
        sparsity=0.02,
        growth_enabled=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_seed=42
    )
    
    # Create metrics data
    metrics_data = MetricsData(**simulate_training_metrics())
    
    # Create complete experiment result
    experiment_result = ExperimentResult(
        experiment_id="demo_experiment_001",
        config=experiment_config,
        metrics=metrics_data
    )
    
    # Log the experiment
    result_hash = log_experiment(experiment_result)
    print(f"✅ Logged experiment result with hash: {result_hash}")
    
    # Check queue status
    status = logger.get_queue_status()
    print(f"📊 Queue status: {status}")
    
    return logger


def demonstrate_metrics_integration():
    """Demonstrate integration with metrics analyzers."""
    print("\n🧮 Demonstrating Metrics Integration")
    print("=" * 50)
    
    # Create sample weight matrix
    weight_matrix = torch.randn(128, 256) * 0.1
    weight_matrix = weight_matrix * (torch.rand_like(weight_matrix) > 0.8)  # Make sparse
    
    # Create analyzers
    homological_analyzer = create_homological_analyzer()
    topological_analyzer = create_topological_analyzer()
    
    # Analyze the weight matrix
    print("🔍 Analyzing weight matrix...")
    
    # Homological analysis
    homological_results = homological_analyzer.compute_metrics(weight_matrix)
    homological_metrics = HomologicalMetrics(
        rank=homological_results['rank'],
        betti_numbers=homological_results['betti_numbers'],
        information_efficiency=homological_results['information_efficiency'],
        kernel_dimension=homological_results['kernel_dimension'],
        image_dimension=homological_results['image_dimension'],
        bottleneck_severity=homological_results['bottleneck_severity']
    )
    
    # Topological analysis
    topological_results = topological_analyzer.compute_metrics(weight_matrix)
    topological_metrics = TopologicalMetrics(
        extrema_count=topological_results['extrema_count'],
        extrema_density=topological_results['extrema_density'],
        persistence_entropy=topological_results['topological_signature'].persistence_entropy,
        connectivity_density=topological_results['topological_signature'].connectivity_density,
        topological_complexity=topological_results['topological_signature'].topological_complexity
    )
    
    # Create experiment with advanced metrics
    config = ExperimentConfig(
        experiment_id="demo_metrics_002",
        experiment_type="homological_analysis",
        dataset="MNIST",
        model_type="sparse_mlp",
        batch_size=64,
        learning_rate=0.001,
        epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    metrics = MetricsData(**simulate_training_metrics())
    
    experiment_result = ExperimentResult(
        experiment_id="demo_metrics_002",
        config=config,
        metrics=metrics,
        homological_metrics=homological_metrics,
        topological_metrics=topological_metrics
    )
    
    # Log the experiment
    result_hash = log_experiment(experiment_result)
    print(f"✅ Logged experiment with advanced metrics: {result_hash}")
    
    # Print some results
    print(f"📈 Homological rank: {homological_metrics.rank}")
    print(f"📈 Betti numbers: {homological_metrics.betti_numbers}")
    print(f"📈 Extrema count: {topological_metrics.extrema_count}")
    print(f"📈 Topological complexity: {topological_metrics.topological_complexity:.3f}")


def demonstrate_growth_tracking():
    """Demonstrate growth event tracking."""
    print("\n🌱 Demonstrating Growth Event Tracking")
    print("=" * 50)
    
    # Simulate a growth event
    growth_event = GrowthEvent(
        epoch=25,
        iteration=500,
        growth_type="add_connections",
        growth_location="layer_2",
        connections_added=150,
        accuracy_before=0.82,
        accuracy_after=0.85,
        performance_delta=0.03,
        architecture_before=[784, 256, 128, 10],
        architecture_after=[784, 256, 128, 10]  # Same architecture, more connections
    )
    
    # Log the growth event
    event_hash = log_growth_event("demo_growth_003", growth_event)
    print(f"✅ Logged growth event: {event_hash}")
    
    # Create a series of growth events
    growth_events = []
    for i in range(5):
        event = GrowthEvent(
            epoch=10 + i * 10,
            growth_type=np.random.choice(["add_connections", "add_layer", "prune_connections"]),
            connections_added=np.random.randint(50, 200),
            accuracy_before=0.7 + i * 0.05,
            accuracy_after=0.72 + i * 0.05,
            architecture_before=[784, 256, 128, 10],
            architecture_after=[784, 256, 128, 10]
        )
        growth_events.append(event)
    
    # Create experiment with multiple growth events
    config = ExperimentConfig(
        experiment_id="demo_growth_series_004",
        experiment_type="growth_series",
        dataset="MNIST",
        model_type="growing_mlp",
        batch_size=64,
        learning_rate=0.001,
        epochs=100,
        growth_enabled=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    metrics = MetricsData(**simulate_training_metrics())
    
    experiment_result = ExperimentResult(
        experiment_id="demo_growth_series_004",
        config=config,
        metrics=metrics,
        growth_events=growth_events
    )
    
    result_hash = log_experiment(experiment_result)
    print(f"✅ Logged experiment with {len(growth_events)} growth events: {result_hash}")


def demonstrate_compactification_metrics():
    """Demonstrate compactification metrics logging."""
    print("\n📦 Demonstrating Compactification Metrics")
    print("=" * 50)
    
    # Create sample compactification data
    compact_data = {
        'patches': [
            {'data': torch.randn(8, 8) * 0.1, 'position': (0, 0)},
            {'data': torch.randn(8, 8) * 0.1, 'position': (8, 8)},
            {'data': torch.randn(8, 8) * 0.1, 'position': (16, 16)}
        ],
        'skeleton': torch.randn(64, 128) * 0.05,
        'original_size': 10000,
        'reconstruction_error': 0.02
    }
    
    # Analyze compactification
    compactification_analyzer = create_compactification_analyzer()
    compactification_results = compactification_analyzer.compute_metrics(compact_data)
    
    compactification_metrics = CompactificationMetrics(
        compression_ratio=compactification_results['compression_stats'].compression_ratio,
        patch_count=compactification_results['patch_effectiveness'].patch_count,
        memory_efficiency=compactification_results['memory_profile'].memory_efficiency,
        reconstruction_error=compactification_results['reconstruction_metrics']['reconstruction_error'],
        information_preservation=compactification_results['patch_effectiveness'].information_preservation
    )
    
    # Create experiment with compactification metrics
    config = ExperimentConfig(
        experiment_id="demo_compactification_005",
        experiment_type="compactification_analysis",
        dataset="MNIST",
        model_type="compactified_mlp",
        batch_size=64,
        learning_rate=0.001,
        epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    metrics = MetricsData(**simulate_training_metrics())
    
    experiment_result = ExperimentResult(
        experiment_id="demo_compactification_005",
        config=config,
        metrics=metrics,
        compactification_metrics=compactification_metrics
    )
    
    result_hash = log_experiment(experiment_result)
    print(f"✅ Logged compactification experiment: {result_hash}")
    
    print(f"📈 Compression ratio: {compactification_metrics.compression_ratio:.3f}")
    print(f"📈 Patch count: {compactification_metrics.patch_count}")
    print(f"📈 Memory efficiency: {compactification_metrics.memory_efficiency:.3f}")


def demonstrate_validation_and_error_handling():
    """Demonstrate validation and error handling."""
    print("\n🛡️ Demonstrating Validation and Error Handling")
    print("=" * 50)
    
    logger = initialize_logging(LoggingConfig(enable_wandb=False))
    
    # Test 1: Invalid accuracy (> 1.0)
    print("🧪 Test 1: Invalid accuracy value")
    try:
        invalid_metrics = {
            "accuracy": 1.5,  # Invalid: > 1.0
            "loss": 0.2,
            "epoch": 10
        }
        log_metrics("test_invalid_001", invalid_metrics)
    except ValueError as e:
        print(f"✅ Caught expected validation error: {e}")
    
    # Test 2: Missing required fields
    print("\n🧪 Test 2: Missing required fields")
    try:
        incomplete_config = {
            "experiment_id": "test_002",
            "experiment_type": "test",
            # Missing required fields like dataset, model_type, etc.
        }
        ExperimentConfig(**incomplete_config)
    except Exception as e:
        print(f"✅ Caught expected validation error: {e}")
    
    # Test 3: Invalid experiment_id
    print("\n🧪 Test 3: Invalid experiment_id")
    try:
        invalid_config = ExperimentConfig(
            experiment_id="ab",  # Too short
            experiment_type="test",
            dataset="MNIST",
            model_type="mlp",
            batch_size=32,
            learning_rate=0.001,
            epochs=10,
            device="cpu"
        )
    except Exception as e:
        print(f"✅ Caught expected validation error: {e}")
    
    # Test 4: Valid data should work
    print("\n🧪 Test 4: Valid data")
    try:
        valid_metrics = {
            "accuracy": 0.85,
            "loss": 0.2,
            "epoch": 10,
            "learning_rate": 0.001
        }
        result_hash = log_metrics("test_valid_004", valid_metrics)
        print(f"✅ Successfully logged valid metrics: {result_hash}")
    except Exception as e:
        print(f"❌ Unexpected error with valid data: {e}")
    
    # Check queue status
    status = logger.get_queue_status()
    print(f"📊 Final queue status: {status}")


def demonstrate_schema_migration():
    """Demonstrate schema migration capabilities."""
    print("\n🔄 Demonstrating Schema Migration")
    print("=" * 50)
    
    logger = initialize_logging(LoggingConfig(enable_wandb=False))
    
    # Simulate old data format (missing schema_version)
    old_data = {
        "experiment_id": "legacy_experiment_001",
        "config": {
            "experiment_id": "legacy_experiment_001",
            "experiment_type": "legacy_test",
            "dataset": "MNIST",
            "model_type": "old_mlp",
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "device": "cpu"
        },
        "metrics": {
            "accuracy": 0.85,
            "loss": 0.2,
            "epoch": 10
        }
        # Note: missing schema_version
    }
    
    print("📄 Original data (missing schema_version):")
    print(f"   Keys: {list(old_data.keys())}")
    
    # Migrate the data
    migrated_data = logger.migrate_schema(old_data)
    print("📄 Migrated data:")
    print(f"   Keys: {list(migrated_data.keys())}")
    print(f"   Schema version: {migrated_data.get('schema_version')}")
    
    # Validate migrated data
    is_valid = logger.validate_schema(migrated_data)
    print(f"✅ Migrated data is valid: {is_valid}")


def main():
    """Run all demonstrations."""
    print("🚀 Structure Net Standardized Logging System Demo")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        logger = demonstrate_basic_logging()
        demonstrate_metrics_integration()
        demonstrate_growth_tracking()
        demonstrate_compactification_metrics()
        demonstrate_validation_and_error_handling()
        demonstrate_schema_migration()
        
        # Final summary
        print("\n📊 Final Summary")
        print("=" * 50)
        status = logger.get_queue_status()
        print(f"Total experiments queued: {status['queued']}")
        print(f"Total experiments sent: {status['sent']}")
        print(f"Total experiments rejected: {status['rejected']}")
        
        print("\n✅ All demonstrations completed successfully!")
        print("\n💡 Key Benefits:")
        print("   • Pydantic validation ensures data quality")
        print("   • WandB artifact integration for versioning")
        print("   • Local queue system for offline resilience")
        print("   • Automatic deduplication via content hashing")
        print("   • Schema migration support for evolution")
        print("   • Comprehensive experiment tracking")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
