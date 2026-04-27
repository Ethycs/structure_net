#!/usr/bin/env python3
"""
Dataset Switching Demo

Demonstrates how to use the new data factory to easily switch between datasets
in Structure Net experiments.
"""

import torch
from data_factory import (
    create_dataset, 
    get_dataset_config, 
    list_available_datasets,
    get_dataset_metadata
)
from neural_architecture_lab.hypothesis_library import SeedSearchHypotheses
from structure_net.core.network_factory import create_standard_network
from seed_search.architecture_generator import ArchitectureGenerator


def demonstrate_dataset_configs():
    """Show available datasets and their configurations."""
    print("=" * 60)
    print("Available Datasets:")
    print("=" * 60)
    
    for dataset_name in list_available_datasets():
        config = get_dataset_config(dataset_name)
        print(f"\n{dataset_name.upper()}:")
        print(f"  Full name: {config.full_name}")
        print(f"  Input shape: {config.input_shape}")
        print(f"  Input size (flattened): {config.input_size}")
        print(f"  Number of classes: {config.num_classes}")
        print(f"  Training samples: {config.num_train_samples:,}")
        print(f"  Test samples: {config.num_test_samples:,}")


def demonstrate_dataset_loading():
    """Show how to load different datasets."""
    print("\n" + "=" * 60)
    print("Loading Datasets:")
    print("=" * 60)
    
    # Load MNIST
    print("\nLoading MNIST...")
    mnist_data = create_dataset(
        dataset_name="mnist",
        batch_size=128,
        subset_fraction=0.1,  # Use 10% for quick demo
        experiment_id="demo_mnist"
    )
    print(f"  Train batches: {len(mnist_data['train_loader'])}")
    print(f"  Test batches: {len(mnist_data['test_loader'])}")
    print(f"  Input size: {mnist_data['input_size']}")
    print(f"  Num classes: {mnist_data['num_classes']}")
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    cifar_data = create_dataset(
        dataset_name="cifar10",
        batch_size=256,
        subset_fraction=0.1,
        experiment_id="demo_cifar10"
    )
    print(f"  Train batches: {len(cifar_data['train_loader'])}")
    print(f"  Test batches: {len(cifar_data['test_loader'])}")
    print(f"  Input size: {cifar_data['input_size']}")
    print(f"  Num classes: {cifar_data['num_classes']}")


def demonstrate_architecture_generation():
    """Show how architecture generation adapts to different datasets."""
    print("\n" + "=" * 60)
    print("Architecture Generation for Different Datasets:")
    print("=" * 60)
    
    for dataset_name in ["mnist", "cifar10"]:
        print(f"\n{dataset_name.upper()} Architectures:")
        
        # Create generator for dataset
        gen = ArchitectureGenerator.from_dataset(dataset_name)
        architectures = gen.generate_systematic_batch(num_architectures=5)
        
        for i, arch in enumerate(architectures[:3]):  # Show first 3
            print(f"  Architecture {i+1}: {arch}")


def demonstrate_hypothesis_creation():
    """Show how hypotheses adapt to different datasets."""
    print("\n" + "=" * 60)
    print("Creating Hypotheses for Different Datasets:")
    print("=" * 60)
    
    # Create seed search hypothesis for MNIST
    mnist_hypothesis = SeedSearchHypotheses.find_optimal_seeds("mnist")
    print(f"\nMNIST Hypothesis:")
    print(f"  ID: {mnist_hypothesis.id}")
    print(f"  Name: {mnist_hypothesis.name}")
    print(f"  Dataset: {mnist_hypothesis.control_parameters['dataset']}")
    print(f"  Architecture count: {len(mnist_hypothesis.parameter_space['architecture'])}")
    
    # Create seed search hypothesis for CIFAR-10
    cifar_hypothesis = SeedSearchHypotheses.find_optimal_seeds("cifar10")
    print(f"\nCIFAR-10 Hypothesis:")
    print(f"  ID: {cifar_hypothesis.id}")
    print(f"  Name: {cifar_hypothesis.name}")
    print(f"  Dataset: {cifar_hypothesis.control_parameters['dataset']}")
    print(f"  Architecture count: {len(cifar_hypothesis.parameter_space['architecture'])}")


def demonstrate_network_creation():
    """Show how to create networks for different datasets."""
    print("\n" + "=" * 60)
    print("Creating Networks for Different Datasets:")
    print("=" * 60)
    
    for dataset_name in ["mnist", "cifar10"]:
        config = get_dataset_config(dataset_name)
        
        # Create a simple network
        architecture = [config.input_size, 256, 128, config.num_classes]
        network = create_standard_network(
            architecture=architecture,
            sparsity=0.02
        )
        
        # Count parameters
        params = sum(p.numel() for p in network.parameters())
        
        print(f"\n{dataset_name.upper()} Network:")
        print(f"  Architecture: {architecture}")
        print(f"  Parameters: {params:,}")
        
        # Test forward pass with dummy data
        if dataset_name == "mnist":
            dummy_input = torch.randn(8, config.input_size)
        else:  # CIFAR-10
            dummy_input = torch.randn(8, config.input_size)
        
        output = network(dummy_input)
        print(f"  Output shape: {output.shape}")


def main():
    """Run all demonstrations."""
    print("\n🚀 Structure Net Dataset Switching Demo")
    print("=" * 80)
    
    # 1. Show available datasets
    demonstrate_dataset_configs()
    
    # 2. Show dataset loading
    demonstrate_dataset_loading()
    
    # 3. Show architecture generation
    demonstrate_architecture_generation()
    
    # 4. Show hypothesis creation
    demonstrate_hypothesis_creation()
    
    # 5. Show network creation
    demonstrate_network_creation()
    
    print("\n" + "=" * 80)
    print("✅ Dataset switching demo complete!")
    print("\nKey Benefits:")
    print("  • Easy dataset switching with a single parameter")
    print("  • Automatic architecture adaptation")
    print("  • Consistent API across all datasets")
    print("  • Metadata tracking for reproducibility")
    print("  • Extensible to custom datasets")


if __name__ == "__main__":
    main()