#!/usr/bin/env python3
"""
Fiber Bundle Network Example

Demonstrates the fiber bundle neural network with geometric principles:
- Curvature-guided growth
- Holonomy measurement
- Multi-class neuron analysis
- Catastrophe detection
- Integration with standardized logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

# Import fiber bundle network
from src.structure_net.models.fiber_bundle_network import (
    FiberBundle,
    FiberBundleConfig,
    FiberBundleBuilder
)

# Import standardized logging
from src.structure_net.logging.standardized_logging import (
    initialize_logging,
    LoggingConfig,
    ExperimentResult,
    ExperimentConfig,
    MetricsData,
    GrowthEvent,
    HomologicalMetrics,
    TopologicalMetrics,
    log_experiment,
    log_growth_event
)


def create_synthetic_data(num_samples: int = 1000, input_dim: int = 784, num_classes: int = 10):
    """Create synthetic data for demonstration."""
    # Generate synthetic data with some structure
    X = torch.randn(num_samples, input_dim)
    
    # Create class-dependent structure
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Add class-specific patterns
    for class_idx in range(num_classes):
        mask = (y == class_idx)
        if mask.sum() > 0:
            # Add class-specific signal
            signal = torch.randn(input_dim) * 0.5
            X[mask] += signal.unsqueeze(0)
    
    return X, y


def create_data_loaders(batch_size: int = 64):
    """Create train and test data loaders."""
    # Create synthetic datasets
    X_train, y_train = create_synthetic_data(num_samples=2000)
    X_test, y_test = create_synthetic_data(num_samples=500)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def demonstrate_basic_fiber_bundle():
    """Demonstrate basic fiber bundle network functionality."""
    print("🧮 Demonstrating Basic Fiber Bundle Network")
    print("=" * 50)
    
    # Create network
    config = FiberBundleConfig(
        base_dim=4,  # 4 layers
        fiber_dim=128,  # 128 neurons per layer
        initial_sparsity=0.95,
        growth_rate=0.1,
        growth_strategy="curvature_guided",
        use_homological_metrics=True,
        use_topological_metrics=True
    )
    
    network = FiberBundle(config)
    print(f"✅ Created fiber bundle network: {config.base_dim} layers, {config.fiber_dim} fiber dimension")
    
    # Test forward pass
    test_input = torch.randn(32, 128)
    output = network(test_input)
    print(f"✅ Forward pass successful: {test_input.shape} → {output.shape}")
    
    # Test with activation tracking
    output, activations = network(test_input, return_activations=True)
    print(f"✅ Activation tracking: {len(activations)} activation tensors")
    
    # Get initial metrics
    metrics = network.get_metrics()
    print(f"✅ Initial metrics computed: {len(metrics)} metrics")
    print(f"   - Total curvature: {metrics.get('curvature/total', 0):.4f}")
    print(f"   - Global sparsity: {metrics.get('sparsity/global', 0):.4f}")
    print(f"   - Total parameters: {metrics.get('network/total_parameters', 0)}")
    
    return network


def demonstrate_geometric_analysis():
    """Demonstrate geometric analysis capabilities."""
    print("\n🔍 Demonstrating Geometric Analysis")
    print("=" * 50)
    
    # Create network
    network = FiberBundleBuilder.create_mnist_bundle()
    
    # Compute curvature for each layer
    print("📐 Computing curvature for each connection:")
    for idx in range(len(network.connections)):
        curvature = network.compute_connection_curvature(idx)
        print(f"   Layer {idx}: curvature = {curvature.item():.6f}")
    
    # Measure holonomy
    test_vectors = torch.randn(50, network.config.fiber_dim)
    holonomy = network.measure_holonomy(test_vectors)
    print(f"📏 Holonomy measurement: {holonomy:.6f}")
    
    # Get homological metrics
    homological_metrics = network.get_homological_metrics()
    if homological_metrics:
        print(f"🧮 Homological analysis:")
        print(f"   - Rank: {homological_metrics['rank']}")
        print(f"   - Betti numbers: {homological_metrics['betti_numbers']}")
        print(f"   - Information efficiency: {homological_metrics['information_efficiency']:.4f}")
    
    # Get topological metrics
    topological_metrics = network.get_topological_metrics()
    if topological_metrics:
        print(f"🔍 Topological analysis:")
        print(f"   - Extrema count: {topological_metrics['extrema_count']}")
        print(f"   - Extrema density: {topological_metrics['extrema_density']:.6f}")
        print(f"   - Topological complexity: {topological_metrics['topological_signature'].topological_complexity:.4f}")
    
    return network


def demonstrate_network_growth():
    """Demonstrate network growth based on geometric principles."""
    print("\n🌱 Demonstrating Network Growth")
    print("=" * 50)
    
    # Create network
    network = FiberBundleBuilder.create_cifar10_bundle()
    
    # Record initial state
    initial_params = sum(p.numel() for p in network.parameters())
    initial_metrics = network.get_metrics()
    
    print(f"📊 Initial state:")
    print(f"   - Parameters: {initial_params}")
    print(f"   - Total curvature: {initial_metrics.get('curvature/total', 0):.4f}")
    
    # Perform growth
    growth_data = {
        'performance_metrics': {'accuracy': 0.75, 'loss': 0.5},
        'geometric_metrics': initial_metrics
    }
    
    growth_event = network.grow_network(growth_data)
    
    # Record final state
    final_params = sum(p.numel() for p in network.parameters())
    final_metrics = network.get_metrics()
    
    print(f"📈 Growth completed:")
    print(f"   - Strategy: {growth_event['growth_strategy']}")
    print(f"   - Actions taken: {len(growth_event['actions'])}")
    print(f"   - Parameters added: {growth_event['parameters_added']}")
    print(f"   - Final parameters: {final_params}")
    print(f"   - Curvature change: {growth_event['curvature_change']:.6f}")
    
    # Show growth actions
    print(f"🔧 Growth actions:")
    for i, action in enumerate(growth_event['actions'][:3]):  # Show first 3
        print(f"   {i+1}. {action['action']} at layer {action['layer']}: "
              f"{action['connections_added']} connections ({action['reason']})")
    
    return network, growth_event


def demonstrate_catastrophe_detection():
    """Demonstrate catastrophe point detection."""
    print("\n⚠️  Demonstrating Catastrophe Detection")
    print("=" * 50)
    
    # Create network and data
    network = FiberBundleBuilder.create_mnist_bundle()
    train_loader, test_loader = create_data_loaders(batch_size=32)
    
    # Get a batch of test data
    test_inputs, test_labels = next(iter(test_loader))
    
    # Detect catastrophe points
    catastrophic_indices = network.detect_catastrophe_points(test_inputs, epsilon=0.01)
    
    print(f"🔍 Catastrophe analysis:")
    print(f"   - Test samples: {test_inputs.shape[0]}")
    print(f"   - Catastrophic samples: {len(catastrophic_indices)}")
    print(f"   - Catastrophe rate: {len(catastrophic_indices) / test_inputs.shape[0]:.2%}")
    
    if catastrophic_indices:
        print(f"   - Catastrophic indices: {catastrophic_indices[:5]}...")  # Show first 5
    
    return network, catastrophic_indices


def demonstrate_multiclass_analysis():
    """Demonstrate multi-class neuron analysis."""
    print("\n🎯 Demonstrating Multi-Class Neuron Analysis")
    print("=" * 50)
    
    # Create network and data
    network = FiberBundleBuilder.create_cifar10_bundle()
    train_loader, test_loader = create_data_loaders(batch_size=64)
    
    # Analyze multi-class neurons
    analysis = network.analyze_multiclass_neurons(test_loader, layer_idx=-2)
    
    if analysis:
        print(f"🧠 Multi-class neuron analysis:")
        print(f"   - Total neurons analyzed: {network.config.fiber_dim}")
        print(f"   - Multi-class neurons: {analysis['multi_class_count']}")
        print(f"   - Highly selective neurons: {analysis['highly_selective_count']}")
        print(f"   - Dead neurons: {analysis['dead_neurons']}")
        print(f"   - Promiscuous neurons: {analysis['promiscuous_neurons']}")
        print(f"   - Mean classes per neuron: {analysis['mean_classes_per_neuron']:.2f}")
        
        # Calculate specialization metrics
        total_neurons = network.config.fiber_dim
        specialization_ratio = analysis['highly_selective_count'] / total_neurons
        multi_class_ratio = analysis['multi_class_count'] / total_neurons
        
        print(f"📊 Specialization metrics:")
        print(f"   - Specialization ratio: {specialization_ratio:.2%}")
        print(f"   - Multi-class ratio: {multi_class_ratio:.2%}")
    
    return network, analysis


def demonstrate_training_integration():
    """Demonstrate training with geometric regularization."""
    print("\n🏋️ Demonstrating Training with Geometric Regularization")
    print("=" * 50)
    
    # Create network and data
    network = FiberBundleBuilder.create_mnist_bundle()
    train_loader, test_loader = create_data_loaders(batch_size=64)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop (simplified)
    network.train()
    total_loss = 0
    correct = 0
    total = 0
    
    print("🚀 Starting training...")
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if batch_idx >= 5:  # Limit for demo
            break
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        
        # Add geometric regularization
        reg_loss = 0
        for idx in range(len(network.connections)):
            curv = network.compute_connection_curvature(idx)
            reg_loss += network.config.gauge_regularization * torch.relu(
                curv - network.config.max_curvature
            )
        
        total_loss_step = loss + reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_step.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 2 == 0:
            print(f"   Batch {batch_idx}: loss = {loss.item():.4f}, "
                  f"reg_loss = {reg_loss.item():.6f}")
    
    # Final metrics
    train_acc = correct / total
    avg_loss = total_loss / min(5, len(train_loader))
    
    print(f"✅ Training completed:")
    print(f"   - Average loss: {avg_loss:.4f}")
    print(f"   - Training accuracy: {train_acc:.2%}")
    
    # Get final geometric metrics
    final_metrics = network.get_metrics()
    print(f"📊 Final geometric metrics:")
    print(f"   - Total curvature: {final_metrics.get('curvature/total', 0):.4f}")
    print(f"   - Global sparsity: {final_metrics.get('sparsity/global', 0):.4f}")
    
    return network, {'loss': avg_loss, 'accuracy': train_acc}


def demonstrate_logging_integration():
    """Demonstrate integration with standardized logging system."""
    print("\n📝 Demonstrating Logging Integration")
    print("=" * 50)
    
    # Initialize logging
    config = LoggingConfig(
        project_name="fiber_bundle_demo",
        enable_wandb=False,  # Disable for demo
        auto_upload=False
    )
    
    logger = initialize_logging(config)
    
    # Create network and get metrics
    network = FiberBundleBuilder.create_cifar10_bundle()
    metrics = network.get_metrics()
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        experiment_id="fiber_bundle_demo_001",
        experiment_type="fiber_bundle_geometric",
        dataset="synthetic",
        model_type="fiber_bundle",
        batch_size=64,
        learning_rate=0.001,
        epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create metrics data
    metrics_data = MetricsData(
        accuracy=0.85,
        loss=0.25,
        epoch=5,
        total_parameters=metrics.get('network/total_parameters', 0),
        sparsity=metrics.get('sparsity/global', 0)
    )
    
    # Get specialized metrics
    homological_metrics = network.get_homological_metrics()
    topological_metrics = network.get_topological_metrics()
    
    # Convert to schema format
    hom_metrics = None
    if homological_metrics:
        hom_metrics = HomologicalMetrics(
            rank=homological_metrics['rank'],
            betti_numbers=homological_metrics['betti_numbers'],
            information_efficiency=homological_metrics['information_efficiency'],
            kernel_dimension=homological_metrics['kernel_dimension'],
            image_dimension=homological_metrics['image_dimension'],
            bottleneck_severity=homological_metrics['bottleneck_severity']
        )
    
    topo_metrics = None
    if topological_metrics:
        topo_metrics = TopologicalMetrics(
            extrema_count=topological_metrics['extrema_count'],
            extrema_density=topological_metrics['extrema_density'],
            persistence_entropy=topological_metrics['topological_signature'].persistence_entropy,
            connectivity_density=topological_metrics['topological_signature'].connectivity_density,
            topological_complexity=topological_metrics['topological_signature'].topological_complexity
        )
    
    # Create complete experiment result
    experiment_result = ExperimentResult(
        experiment_id="fiber_bundle_demo_001",
        config=experiment_config,
        metrics=metrics_data,
        homological_metrics=hom_metrics,
        topological_metrics=topo_metrics,
        custom_metrics={
            'curvature_total': metrics.get('curvature/total', 0),
            'curvature_mean': metrics.get('curvature/mean', 0),
            'network_layers': metrics.get('network/layers', 0),
            'growth_strategy': network.config.growth_strategy
        }
    )
    
    # Log the experiment
    result_hash = log_experiment(experiment_result)
    print(f"✅ Logged fiber bundle experiment: {result_hash}")
    
    # Simulate and log a growth event
    growth_event = GrowthEvent(
        epoch=5,
        growth_type="curvature_guided",
        connections_added=150,
        accuracy_before=0.82,
        accuracy_after=0.85,
        architecture_before=[784, 512, 512, 512, 10],
        architecture_after=[784, 512, 512, 512, 10]
    )
    
    event_hash = log_growth_event("fiber_bundle_demo_001", growth_event)
    print(f"✅ Logged growth event: {event_hash}")
    
    # Check queue status
    status = logger.get_queue_status()
    print(f"📊 Queue status: {status}")
    
    return logger


def main():
    """Run all fiber bundle demonstrations."""
    print("🚀 Fiber Bundle Network Demonstration")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        network1 = demonstrate_basic_fiber_bundle()
        network2 = demonstrate_geometric_analysis()
        network3, growth_event = demonstrate_network_growth()
        network4, catastrophic_indices = demonstrate_catastrophe_detection()
        network5, multiclass_analysis = demonstrate_multiclass_analysis()
        network6, training_results = demonstrate_training_integration()
        logger = demonstrate_logging_integration()
        
        # Final summary
        print("\n📊 Final Summary")
        print("=" * 50)
        print("✅ All demonstrations completed successfully!")
        
        print("\n💡 Key Features Demonstrated:")
        print("   • Fiber bundle network architecture with geometric structure")
        print("   • Curvature computation and holonomy measurement")
        print("   • Geometric-guided network growth strategies")
        print("   • Catastrophe point detection for robustness analysis")
        print("   • Multi-class neuron analysis for specialization tracking")
        print("   • Training with geometric regularization")
        print("   • Integration with standardized logging system")
        print("   • Homological and topological metrics integration")
        
        print("\n🎯 Geometric Insights:")
        print("   • Curvature guides optimal growth locations")
        print("   • Holonomy measures information transport quality")
        print("   • Multi-class neurons reveal network specialization")
        print("   • Catastrophe detection identifies fragile regions")
        print("   • Gauge invariance preserves network symmetries")
        
        print("\n🔬 Research Applications:")
        print("   • Geometric deep learning with explicit structure")
        print("   • Robust network architectures via catastrophe avoidance")
        print("   • Interpretable growth through geometric principles")
        print("   • Advanced network analysis via topological methods")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
