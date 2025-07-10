#!/usr/bin/env python3
"""
Test Seed Validation using model_io

Use the canonical model_io validation mechanism to check our CIFAR-10 seeds.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
from src.structure_net.core.validation import validate_model_quality
from src.structure_net.core.io_operations import load_model_seed

def test_cifar10_seed():
    """Test loading and validating a CIFAR-10 seed."""
    
    # Test our best accuracy model
    model_path = "data/promising_models/20250707_065048/model_cifar10_3layers_seed7_acc0.35_patch0.640_sparse0.010_BEST_ACCURACY_GLOBAL.pt"
    
    print("🔬 TESTING CIFAR-10 SEED VALIDATION")
    print("=" * 50)
    print(f"Model: {model_path}")
    
    # Use canonical validation function
    print("\n📋 Running canonical model validation...")
    
    try:
        result = validate_model_quality(
            filepath=model_path,
            dataset='cifar10',
            device='cuda',
            test_accuracy=True,
            test_forward_pass=True,
            verbose=True
        )
        
        print(f"\n✅ Validation Result: {result}")
        
        if result['valid']:
            print("🎯 Model validation PASSED")
            print(f"   Accuracy: {result.get('accuracy', 'N/A')}")
            print(f"   Architecture: {result.get('architecture', 'N/A')}")
            print(f"   Forward pass: {'✅' if result.get('forward_pass_ok', False) else '❌'}")
        else:
            print("❌ Model validation FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Validation failed with exception: {e}")
        
        # Try manual loading to see what happens
        print("\n🔧 Attempting manual load for debugging...")
        try:
            model, metadata = load_model_seed(model_path, device='cuda')
            print(f"   ✅ Manual load successful")
            print(f"   Architecture: {metadata.get('architecture', 'N/A')}")
            print(f"   Accuracy: {metadata.get('accuracy', 'N/A')}")
            
            # Test forward pass with dummy data
            print("\n🧪 Testing forward pass with dummy CIFAR-10 data...")
            dummy_data = torch.randn(64, 3072).cuda()  # CIFAR-10 flattened
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_data)
                print(f"   Input shape: {dummy_data.shape}")
                print(f"   Output shape: {output.shape}")
                print(f"   ✅ Forward pass successful")
                
        except Exception as e2:
            print(f"   ❌ Manual load failed: {e2}")

if __name__ == "__main__":
    test_cifar10_seed()
