#!/usr/bin/env python3
"""
Modern Indefinite Growth Network Creation Test

This script tests the network creation and architecture handling to debug
the layer dropping issue in the modern indefinite growth experiment.
"""

import sys
import os
sys.path.append('.')

from src.structure_net import create_standard_network, get_network_stats
import torch

print("🔬 MODERN INDEFINITE GROWTH NETWORK CREATION TEST")
print("=" * 60)

# Test network creation with the exact same parameters as the experiment
seed_arch = [784, 128, 10]
sparsity = 0.02
device = 'cpu'

print(f"🧪 Creating network with architecture: {seed_arch}")
print(f"   Sparsity: {sparsity:.1%}")
print(f"   Device: {device}")

# Create network using canonical standard
network = create_standard_network(
    architecture=seed_arch,
    sparsity=sparsity,
    device=device
)

print(f"\n🔍 Network layers:")
for i, layer in enumerate(network):
    print(f"  {i}: {layer}")
    if hasattr(layer, 'linear'):
        print(f"     Weight shape: {layer.linear.weight.shape}")
        print(f"     Bias shape: {layer.linear.bias.shape}")
        print(f"     Mask shape: {layer.mask.shape}")

# Get network statistics
stats = get_network_stats(network)
print(f"\n📊 Network statistics:")
print(f"   Architecture from stats: {stats['architecture']}")
print(f"   Total layers: {len(stats['layers'])}")
print(f"   Total parameters: {stats['total_parameters']:,}")
print(f"   Total connections: {stats['total_connections']:,}")
print(f"   Overall sparsity: {stats['overall_sparsity']:.1%}")

# Test forward pass
print(f"\n🧪 Testing forward pass:")
x = torch.randn(1, 784)
print(f"   Input shape: {x.shape}")

y = network(x)
print(f"   Output shape: {y.shape}")
print(f"   Output sample: {y[0, :5]}")

# Test activation collection (like in the experiment)
print(f"\n🔍 Testing activation collection:")
from src.structure_net.core.model_io import StandardSparseLayer

x = torch.randn(5, 784)  # Batch of 5
batch_activations = []

for i, layer in enumerate(network):
    if isinstance(layer, StandardSparseLayer):
        x = layer(x)
        batch_activations.append(x.detach())
        print(f"   After StandardSparseLayer {i}: {x.shape}")
    elif isinstance(layer, torch.nn.ReLU):
        x = layer(x)
        if batch_activations:
            batch_activations[-1] = x.detach()
        print(f"   After ReLU {i}: {x.shape}")

print(f"\n📊 Collected activations:")
for i, acts in enumerate(batch_activations):
    print(f"   Layer {i}: {acts.shape}")

print(f"\n✅ Test complete!")
print(f"   Expected layers: {len(seed_arch) - 1} (sparse layers)")
print(f"   Collected activations: {len(batch_activations)}")
print(f"   Architecture match: {stats['architecture'] == seed_arch}")

if stats['architecture'] != seed_arch:
    print(f"\n❌ ARCHITECTURE MISMATCH!")
    print(f"   Expected: {seed_arch}")
    print(f"   Got: {stats['architecture']}")
else:
    print(f"\n✅ Architecture matches perfectly!")
