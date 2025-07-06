#!/usr/bin/env python3
"""
Basic test to verify the multi-scale network implementation works.
"""

import sys
import os
import torch

# Add current directory to path
sys.path.append('.')

def test_basic_functionality():
    """Test basic network functionality."""
    print("Testing multi-scale network implementation...")
    
    try:
        # Test imports
        from src.structure_net import create_multi_scale_network
        print("✓ Import successful!")
        
        # Test network creation with CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = create_multi_scale_network(784, [256, 128], 10, device=device)
        print("✓ Network created successfully!")
        print(f"  Architecture: {network.network.layer_sizes}")
        print(f"  Device: {device}")
        
        # Test connectivity stats
        stats = network.network.get_connectivity_stats()
        print(f"✓ Initial connectivity: {stats['connectivity_ratio']:.6f}")
        print(f"  Total connections: {stats['total_active_connections']}")
        
        # Test forward pass
        x = torch.randn(32, 784).to(device)
        output = network(x)
        print(f"✓ Forward pass successful! Output shape: {output.shape}")
        
        # Test extrema detection
        extrema = network.network.detect_extrema()
        print(f"✓ Extrema detection: {len(extrema)} layers processed")
        
        # Test growth scheduler
        scheduler_stats = network.growth_scheduler.get_stats()
        print(f"✓ Growth scheduler initialized: {scheduler_stats['current_phase']} phase")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
