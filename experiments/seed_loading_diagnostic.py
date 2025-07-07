#!/usr/bin/env python3
"""
Seed Loading Diagnostic Test

This script tests the exact difference between GPU seed hunter and hybrid growth
data preprocessing to identify why seeds load with 4.69% instead of 47% accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

def gpu_seed_hunter_preprocessing():
    """Exact GPU seed hunter CIFAR-10 preprocessing"""
    print("🔍 GPU Seed Hunter Preprocessing:")
    
    # Step 1: Load with transforms (like GPU seed hunter does)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    # Step 2: Convert to tensors and move to GPU (like GPU seed hunter)
    train_x = torch.from_numpy(train_dataset.data).float().reshape(-1, 3072) / 255.0
    train_y = torch.tensor(train_dataset.targets)
    test_x = torch.from_numpy(test_dataset.data).float().reshape(-1, 3072) / 255.0
    test_y = torch.tensor(test_dataset.targets)
    
    # Step 3: Normalize using dataset statistics
    train_mean = train_x.mean(dim=0)
    train_std = train_x.std(dim=0)
    train_x = (train_x - train_mean) / (train_std + 1e-8)
    test_x = (test_x - train_mean) / (train_std + 1e-8)
    
    print(f"   Train data shape: {train_x.shape}")
    print(f"   Train mean: {train_x.mean():.6f}")
    print(f"   Train std: {train_x.std():.6f}")
    print(f"   Sample values: {train_x[0][:5]}")
    
    return train_x, train_y, test_x, test_y

def hybrid_growth_preprocessing():
    """Current hybrid growth CIFAR-10 preprocessing"""
    print("\n🔍 Hybrid Growth Preprocessing:")
    
    # Load raw datasets without any transforms first
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=None)
    
    # Convert to tensors and preprocess
    train_x = torch.from_numpy(train_dataset.data).float().reshape(-1, 3072) / 255.0
    train_y = torch.tensor(train_dataset.targets)
    test_x = torch.from_numpy(test_dataset.data).float().reshape(-1, 3072) / 255.0
    test_y = torch.tensor(test_dataset.targets)
    
    # Normalize using dataset statistics
    train_mean = train_x.mean(dim=0)
    train_std = train_x.std(dim=0)
    train_x = (train_x - train_mean) / (train_std + 1e-8)
    test_x = (test_x - train_mean) / (train_std + 1e-8)
    
    print(f"   Train data shape: {train_x.shape}")
    print(f"   Train mean: {train_x.mean():.6f}")
    print(f"   Train std: {train_x.std():.6f}")
    print(f"   Sample values: {train_x[0][:5]}")
    
    return train_x, train_y, test_x, test_y

def load_and_test_model(checkpoint_path, test_x, test_y, preprocessing_name):
    """Load model and test accuracy with given preprocessing"""
    print(f"\n🧪 Testing model with {preprocessing_name} preprocessing:")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model (EXACT version of PersistentSparseLayer)
    class TestSparseLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.mask = None
            
        def forward(self, x):
            # CRITICAL: Enforce sparsity on every forward pass like GPU seed hunter
            if self.mask is not None:
                return torch.nn.functional.linear(x, self.linear.weight * self.mask, self.linear.bias)
            else:
                return self.linear(x)
    
    # Build model matching checkpoint architecture
    architecture = checkpoint['architecture']
    model = nn.Sequential()
    
    for i in range(len(architecture) - 1):
        layer = TestSparseLayer(architecture[i], architecture[i+1])
        model.add_module(str(i*2), layer)
        if i < len(architecture) - 2:
            model.add_module(str(i*2+1), nn.ReLU())
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    for i in range(len(architecture) - 1):
        layer_key = str(i * 2)
        weight_key = f'{layer_key}.linear.weight'
        bias_key = f'{layer_key}.linear.bias'
        mask_key = f'{layer_key}.mask'
        
        if weight_key in state_dict:
            model[i*2].linear.weight.data = state_dict[weight_key]
            model[i*2].linear.bias.data = state_dict[bias_key]
            
            # Store mask for forward pass (don't multiply weights!)
            if mask_key in state_dict:
                model[i*2].mask = state_dict[mask_key]
                print(f"   Loaded mask for layer {i}: {model[i*2].mask.sum().item()}/{model[i*2].mask.numel()} connections")
            else:
                print(f"   No mask found for layer {i}")
    
    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        batch_size = 1000
        for i in range(0, len(test_x), batch_size):
            batch_x = test_x[i:i+batch_size]
            batch_y = test_y[i:i+batch_size]
            
            output = model(batch_x)
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    print(f"   Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    return accuracy

def main():
    print("🔬 SEED LOADING DIAGNOSTIC TEST")
    print("="*50)
    
    # Test both preprocessing methods
    gpu_train_x, gpu_train_y, gpu_test_x, gpu_test_y = gpu_seed_hunter_preprocessing()
    hybrid_train_x, hybrid_train_y, hybrid_test_x, hybrid_test_y = hybrid_growth_preprocessing()
    
    # Compare preprocessing results
    print(f"\n📊 Preprocessing Comparison:")
    print(f"   Data shapes match: {gpu_train_x.shape == hybrid_train_x.shape}")
    print(f"   Data values match: {torch.allclose(gpu_train_x, hybrid_train_x, atol=1e-6)}")
    print(f"   Mean difference: {(gpu_train_x.mean() - hybrid_train_x.mean()).abs():.8f}")
    print(f"   Std difference: {(gpu_train_x.std() - hybrid_train_x.std()).abs():.8f}")
    print(f"   Max absolute difference: {(gpu_train_x - hybrid_train_x).abs().max():.8f}")
    
    # Test model with both preprocessing methods
    checkpoint_path = "data/promising_models/20250707_020152/model_cifar10_3layers_seed9_acc0.47_patch0.345_sparse0.050_BEST_ACCURACY_GLOBAL.pt"
    
    gpu_accuracy = load_and_test_model(checkpoint_path, gpu_test_x, gpu_test_y, "GPU Seed Hunter")
    hybrid_accuracy = load_and_test_model(checkpoint_path, hybrid_test_x, hybrid_test_y, "Hybrid Growth")
    
    print(f"\n🎯 RESULTS:")
    print(f"   GPU Seed Hunter accuracy: {gpu_accuracy:.4f} ({gpu_accuracy:.2%})")
    print(f"   Hybrid Growth accuracy: {hybrid_accuracy:.4f} ({hybrid_accuracy:.2%})")
    print(f"   Accuracy difference: {abs(gpu_accuracy - hybrid_accuracy):.4f}")
    
    if abs(gpu_accuracy - hybrid_accuracy) < 0.01:
        print("   ✅ Preprocessing methods are equivalent!")
    else:
        print("   ❌ Preprocessing methods differ significantly!")
        print("   🔍 Need to investigate further...")

if __name__ == "__main__":
    main()
