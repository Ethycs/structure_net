#!/usr/bin/env python3
"""
Adaptive Learning Rates Example

This example demonstrates how to use the new adaptive learning rate strategies
in structure_net for sophisticated training with phase-based, layer-wise,
and connection-age-aware learning rate adaptation.

Key strategies demonstrated:
1. Exponential Backoff for Loss
2. Layer-wise Adaptive Growth Rates  
3. Soft Clamping (Gradual Freezing)
4. Scale-Dependent Learning Rates
5. Growth Phase-Based Adjustment
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import structure_net with new adaptive learning rate capabilities
from src.structure_net import (
    create_standard_network,
    ExponentialBackoffScheduler,
    LayerwiseAdaptiveRates,
    SoftClampingScheduler,
    ScaleDependentRates,
    GrowthPhaseScheduler,
    AdaptiveLearningRateManager,
    create_adaptive_training_loop
)


def load_mnist_data(batch_size=64):
    """Load MNIST dataset for demonstration."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def example_1_exponential_backoff():
    """Example 1: Exponential Backoff Scheduler"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 1: EXPONENTIAL BACKOFF SCHEDULER")
    print("="*60)
    
    # Create scheduler with aggressive early → gentle late learning
    scheduler = ExponentialBackoffScheduler(initial_lr=1.0, decay_rate=0.95)
    
    print("Exponential backoff progression:")
    for epoch in range(0, 50, 5):
        weight = scheduler.get_loss_weight(epoch)
        print(f"  Epoch {epoch:2d}: Loss weight = {weight:.4f}")
    
    print("\n💡 Use case: Natural curriculum from finding major highways to refining local roads")


def example_2_layerwise_adaptive():
    """Example 2: Layer-wise Adaptive Rates"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 2: LAYER-WISE ADAPTIVE RATES")
    print("="*60)
    
    # Create layer-wise adaptive rates
    layerwise = LayerwiseAdaptiveRates(
        early_rate=0.02,    # Fast growth for feature extraction
        middle_rate=0.01,   # Medium growth for feature combination
        late_rate=0.005     # Slow growth for sparse bridges
    )
    
    n_layers = 5
    rates = layerwise.get_layer_rates(n_layers)
    
    print("Layer-wise learning rates:")
    for i, rate in enumerate(rates):
        layer_type = "Early" if i < n_layers//3 else "Late" if i > 2*n_layers//3 else "Middle"
        print(f"  Layer {i}: {rate:.3f} ({layer_type})")
    
    print("\n💡 Use case: Early layers learn features fast, late layers form sparse bridges slowly")


def example_3_soft_clamping():
    """Example 3: Soft Clamping (Gradual Freezing)"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 3: SOFT CLAMPING (GRADUAL FREEZING)")
    print("="*60)
    
    # Create soft clamping scheduler
    soft_clamp = SoftClampingScheduler(max_age=100, min_clamp_factor=0.1)
    
    print("Connection aging progression:")
    connection_id = "layer_0_conn_5_10"
    
    for age in [0, 10, 25, 50, 75, 100, 150]:
        clamp_factor = soft_clamp.soft_clamping(connection_id, age)
        print(f"  Age {age:3d}: Clamp factor = {clamp_factor:.3f}")
    
    print("\n💡 Use case: Old connections adapt slowly but don't freeze completely")


def example_4_scale_dependent():
    """Example 4: Scale-Dependent Learning Rates"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 4: SCALE-DEPENDENT LEARNING RATES")
    print("="*60)
    
    # Create scale-dependent rates
    scale_rates = ScaleDependentRates(
        coarse_scale_lr=0.001,   # Slow for major pathways
        medium_scale_lr=0.01,    # Moderate for features
        fine_scale_lr=0.1        # Fast for details
    )
    
    print("Scale-dependent rates for different connection types:")
    
    # Test different connection scenarios
    scenarios = [
        (0, 5, 0.8, "Early layer, strong connection"),
        (2, 5, 0.3, "Middle layer, medium connection"),
        (4, 5, 0.05, "Late layer, weak connection"),
        (1, 5, 0.9, "Early layer, very strong connection")
    ]
    
    for layer_idx, n_layers, strength, description in scenarios:
        rate = scale_rates.get_connection_rate(layer_idx, n_layers, strength)
        scale = scale_rates.determine_connection_scale(layer_idx, n_layers, strength)
        print(f"  {description}: {rate:.3f} ({scale})")
    
    print("\n💡 Use case: Major pathways learn slowly, details learn fast")


def example_5_growth_phases():
    """Example 5: Growth Phase-Based Adjustment"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 5: GROWTH PHASE-BASED ADJUSTMENT")
    print("="*60)
    
    # Create growth phase scheduler
    phase_scheduler = GrowthPhaseScheduler(
        early_lr=0.1,           # Aggressive for structure discovery
        middle_lr=0.01,         # Moderate for feature development
        late_lr=0.001,          # Gentle for fine-tuning
        early_phase_end=20,
        middle_phase_end=50
    )
    
    print("Growth phase progression:")
    for epoch in [5, 15, 25, 35, 45, 55, 75]:
        lr = phase_scheduler.phase_based_lr(epoch)
        phase = phase_scheduler.get_current_phase(epoch)
        print(f"  Epoch {epoch:2d}: LR = {lr:.3f} ({phase} phase)")
    
    print("\n💡 Use case: Aggressive early → moderate middle → gentle late learning")


def example_6_unified_manager():
    """Example 6: Unified Adaptive Learning Rate Manager"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 6: UNIFIED ADAPTIVE MANAGER")
    print("="*60)
    
    # Create a sample network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = create_standard_network(
        architecture=[784, 128, 64, 10],
        sparsity=0.02,
        device=str(device)
    )
    
    # Create unified manager with all strategies enabled
    lr_manager = AdaptiveLearningRateManager(
        network=network,
        base_lr=0.001,
        enable_exponential_backoff=True,
        enable_layerwise_rates=True,
        enable_soft_clamping=True,
        enable_scale_dependent=True,
        enable_phase_based=True
    )
    
    # Create adaptive optimizer
    optimizer = lr_manager.create_adaptive_optimizer()
    
    print(f"Created adaptive optimizer with {len(optimizer.param_groups)} parameter groups")
    
    # Simulate training progression
    print("\nLearning rate progression over epochs:")
    for epoch in [0, 10, 25, 50]:
        lr_manager.update_learning_rates(optimizer, epoch)
        summary = lr_manager.get_current_rates_summary()
        
        print(f"\nEpoch {epoch}:")
        if 'exponential_backoff' in summary['strategies']:
            weight = summary['strategies']['exponential_backoff']['current_weight']
            print(f"  Exponential backoff weight: {weight:.4f}")
        
        if 'phase_based' in summary['strategies']:
            phase = summary['strategies']['phase_based']['current_phase']
            phase_lr = summary['strategies']['phase_based']['current_lr']
            print(f"  Current phase: {phase} (LR: {phase_lr:.4f})")
        
        # Show first few layer rates
        for i, group in enumerate(optimizer.param_groups[:3]):
            if 'layer_idx' in group:
                print(f"  Layer {group['layer_idx']} LR: {group['lr']:.6f}")
    
    print("\n💡 Use case: All strategies working together for sophisticated adaptation")


def example_7_complete_training():
    """Example 7: Complete Training Loop with Adaptive Rates"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 7: COMPLETE ADAPTIVE TRAINING")
    print("="*60)
    
    # Load data
    print("Loading MNIST data...")
    train_loader, test_loader = load_mnist_data(batch_size=128)
    
    # Create network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = create_standard_network(
        architecture=[784, 256, 128, 10],
        sparsity=0.02,
        device=str(device)
    )
    
    print(f"Created network on {device}")
    print(f"Network architecture: [784, 256, 128, 10]")
    print(f"Network sparsity: 2%")
    
    # Run adaptive training
    print("\nStarting adaptive training with all strategies...")
    
    trained_network, history = create_adaptive_training_loop(
        network=network,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=30,
        base_lr=0.001,
        enable_exponential_backoff=True,
        enable_layerwise_rates=True,
        enable_soft_clamping=True,
        enable_scale_dependent=True,
        enable_phase_based=True
    )
    
    # Show results
    final_acc = history[-1]['val_acc']
    print(f"\n✅ Training complete!")
    print(f"Final validation accuracy: {final_acc:.2%}")
    
    # Show learning rate evolution
    print("\nLearning rate strategy evolution:")
    for epoch_data in history[::10]:  # Every 10th epoch
        epoch = epoch_data['epoch']
        val_acc = epoch_data['val_acc']
        lr_data = epoch_data['learning_rates']
        
        print(f"  Epoch {epoch:2d}: Val Acc = {val_acc:.2%}")
        
        if 'exponential_backoff' in lr_data['strategies']:
            weight = lr_data['strategies']['exponential_backoff']['current_weight']
            print(f"    Backoff weight: {weight:.4f}")
        
        if 'phase_based' in lr_data['strategies']:
            phase = lr_data['strategies']['phase_based']['current_phase']
            print(f"    Phase: {phase}")
    
    print("\n💡 Complete example showing all adaptive strategies in action!")


def main():
    """Run all adaptive learning rate examples."""
    print("🚀 ADAPTIVE LEARNING RATES EXAMPLES")
    print("=" * 80)
    print("Demonstrating sophisticated differential learning rate strategies")
    print("for structure_net neural networks")
    
    # Run all examples
    example_1_exponential_backoff()
    example_2_layerwise_adaptive()
    example_3_soft_clamping()
    example_4_scale_dependent()
    example_5_growth_phases()
    example_6_unified_manager()
    
    # Ask user if they want to run the complete training example
    print("\n" + "="*60)
    response = input("Run complete training example? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        example_7_complete_training()
    else:
        print("Skipping complete training example.")
    
    print("\n" + "="*80)
    print("✅ ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("These adaptive learning rate strategies provide:")
    print("  🎯 Exponential Backoff: Aggressive early → gentle late")
    print("  🏗️  Layer-wise Rates: Different rates for different layer types")
    print("  🔒 Soft Clamping: Gradual freezing instead of hard stops")
    print("  📏 Scale-Dependent: Different rates for different connection scales")
    print("  📈 Phase-Based: Learning rates that adapt to training phase")
    print("  🎛️  Unified Manager: All strategies working together")
    print("\nThese create a natural curriculum: aggressive early learning")
    print("that gradually becomes gentler as the network matures!")


if __name__ == "__main__":
    main()
