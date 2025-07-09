#!/usr/bin/env python3
"""
Stress Test Runner with Pre-trained Seed Model

This script loads a pre-trained model from the promising_models directory
and runs stress tests on it, including growth experiments and performance evaluation.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime
import torch.cuda as cuda

# Add structure_net to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.structure_net.core.io_operations import load_model_seed
from src.structure_net.evolution.integrated_growth_system import IntegratedGrowthSystem
from src.structure_net.evolution.advanced_layers import ThresholdConfig, MetricsConfig
from src.structure_net.core.network_analysis import get_network_stats
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_cifar10_data(batch_size=128, subset_size=None):
    """Load CIFAR-10 dataset with optional subset for faster testing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    # Use subset if specified for faster testing
    if subset_size:
        train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        test_dataset = Subset(test_dataset, range(min(subset_size // 5, len(test_dataset))))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).view(data.size(0), -1)  # Flatten for FC network
            target = target.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy


def stress_test_growth(model, train_loader, val_loader, device='cuda', growth_iterations=3):
    """Run growth stress test on the model."""
    print("\n🌱 Starting Growth Stress Test")
    print("=" * 60)
    
    # Configure growth system
    threshold_config = ThresholdConfig()
    metrics_config = MetricsConfig()
    
    # Initialize growth system with only network and configs
    growth_system = IntegratedGrowthSystem(
        network=model,
        config=threshold_config,
        metrics_config=metrics_config
    )
    
    # Run growth using grow_network method with data loaders
    results = growth_system.grow_network(
        train_loader=train_loader,
        val_loader=val_loader,
        growth_iterations=growth_iterations,
        epochs_per_iteration=10,
        tournament_epochs=5
    )
    
    return results


def stress_test_robustness(model, test_loader, device='cuda'):
    """Test model robustness to various perturbations."""
    print("\n🛡️ Starting Robustness Stress Test")
    print("=" * 60)
    
    model.eval()
    results = {}
    
    # Test 1: Gaussian noise
    print("Testing with Gaussian noise...")
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    noise_results = []
    
    for noise_level in noise_levels:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device).view(data.size(0), -1)
                target = target.to(device)
                
                # Add Gaussian noise
                noisy_data = data + torch.randn_like(data) * noise_level
                
                outputs = model(noisy_data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        noise_results.append({'noise_level': noise_level, 'accuracy': accuracy})
        print(f"  Noise level {noise_level}: {accuracy:.2%}")
    
    results['gaussian_noise'] = noise_results
    
    # Test 2: Input dropout
    print("\nTesting with input dropout...")
    dropout_rates = [0.1, 0.2, 0.3, 0.5]
    dropout_results = []
    
    for dropout_rate in dropout_rates:
        correct = 0
        total = 0
        dropout = torch.nn.Dropout(p=dropout_rate)
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device).view(data.size(0), -1)
                target = target.to(device)
                
                # Apply dropout to inputs
                dropped_data = dropout(data)
                
                outputs = model(dropped_data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        dropout_results.append({'dropout_rate': dropout_rate, 'accuracy': accuracy})
        print(f"  Dropout rate {dropout_rate}: {accuracy:.2%}")
    
    results['input_dropout'] = dropout_results
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run stress test with pre-trained seed model")
    parser.add_argument("model", type=str, help="Path to seed model")
    parser.add_argument("--quick", action="store_true", help="Run quick test with smaller dataset")
    parser.add_argument("--growth-iterations", type=int, default=3, help="Number of growth iterations")
    parser.add_argument("--skip-growth", action="store_true", help="Skip growth stress test")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness stress test")
    
    args = parser.parse_args()
    
    print("🚀 Stress Test with Pre-trained Seed Model")
    print("=" * 60)
    
    # Use the provided model path
    model_path = args.model
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📂 Loading model: {model_path}")
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, checkpoint_data = load_model_seed(model_path, device=str(device))
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Print model info
    stats = get_network_stats(model)
    print(f"\n📊 Model Statistics:")
    print(f"   Architecture: {checkpoint_data.get('architecture', 'Unknown')}")
    print(f"   Total parameters: {stats['total_parameters']:,}")
    print(f"   Total connections: {stats['total_connections']:,}")
    print(f"   Overall sparsity: {stats['overall_sparsity']:.1%}")
    print(f"   Original accuracy: {checkpoint_data.get('accuracy', 0.0):.2%}")
    
    # Load data
    print("\n📊 Loading CIFAR-10 data...")
    subset_size = 5000 if args.quick else None
    train_loader, test_loader = load_cifar10_data(subset_size=subset_size)
    
    # Re-evaluate model on current data
    print("\n🎯 Re-evaluating model accuracy...")
    current_accuracy = evaluate_model(model, test_loader, device)
    print(f"   Current accuracy: {current_accuracy:.2%}")
    
    # Run stress tests
    all_results = {
        'model_path': model_path,
        'original_stats': stats,
        'original_accuracy': checkpoint_data.get('accuracy', 0.0),
        'current_accuracy': current_accuracy,
        'tests': {}
    }
    
    # Growth stress test
    if not args.skip_growth:
        try:
            growth_results = stress_test_growth(
                model, train_loader, test_loader, device, 
                growth_iterations=args.growth_iterations
            )
            all_results['tests']['growth'] = growth_results
            
            # Re-evaluate after growth
            final_accuracy = evaluate_model(model, test_loader, device)
            print(f"\n📈 Final accuracy after growth: {final_accuracy:.2%}")
            print(f"   Improvement: {(final_accuracy - current_accuracy)*100:+.1f}%")
            all_results['final_accuracy_after_growth'] = final_accuracy
            
        except Exception as e:
            print(f"⚠️ Growth test failed: {e}")
            all_results['tests']['growth'] = {'error': str(e)}
    
    # Robustness stress test
    if not args.skip_robustness:
        try:
            robustness_results = stress_test_robustness(model, test_loader, device)
            all_results['tests']['robustness'] = robustness_results
        except Exception as e:
            print(f"⚠️ Robustness test failed: {e}")
            all_results['tests']['robustness'] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"stress_test_results_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Stress test completed!")


if __name__ == "__main__":
    main()