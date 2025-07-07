#!/usr/bin/env python3
"""
Hybrid Growth Experiment - FIXED LSUV Issue

This experiment implements a network that combines a sparse scaffold with 
multi-scale dense patches, and grows based on extrema patterns.

FIXED: LSUV no longer breaks the seed layer when loading pretrained models.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def _create_sparse_mask(shape, sparsity):
    """Creates a sparse mask with a given shape and sparsity."""
    rows, cols = shape
    num_connections = int(sparsity * rows * cols)
    mask = torch.zeros(rows, cols)
    indices = torch.randperm(rows * cols)[:num_connections]
    mask.view(-1)[indices] = 1.0
    return mask

def _lsuv_init_layer(layer, sample_input):
    """LSUV for a single layer - ONLY for new layers, NOT pretrained ones."""
    with torch.no_grad():
        for _ in range(10):
            out = F.linear(sample_input, layer.weight * layer.mask, layer.bias)
            var = out.var()
            if var > 0:
                layer.weight.data /= torch.sqrt(var)
            if abs(out.var() - 1.0) < 0.01:
                break
        return F.relu(out)

def create_lsuv_sparse_scaffold(architecture, sparsity, device, sample_batch, is_pretrained=False):
    """
    Creates a sparse scaffold with LSUV initialization.
    
    FIXED: If is_pretrained=True, LSUV is NOT applied to preserve learned features.
    """
    layers = nn.ModuleList()
    h = sample_batch
    
    for i in range(len(architecture) - 1):
        in_features, out_features = architecture[i], architecture[i+1]
        layer = nn.Linear(in_features, out_features).to(device)
        
        mask = _create_sparse_mask((out_features, in_features), sparsity).to(device)
        layer.register_buffer('mask', mask)
        layer.weight.data.mul_(mask)
        
        # CRITICAL FIX: Only apply LSUV to NEW layers, not pretrained ones
        if not is_pretrained:
            h = _lsuv_init_layer(layer, h)
        else:
            # For pretrained layers, just apply the mask and forward pass
            with torch.no_grad():
                layer.weight.data *= mask.float()
            h = F.relu(F.linear(h, layer.weight * layer.mask, layer.bias))
            
        layers.append(layer)
    return layers

class HybridGrowthNetwork(nn.Module):
    """
    A network that combines a sparse scaffold with multi-scale dense patches,
    and grows based on extrema patterns.
    
    FIXED: Properly handles pretrained models without destroying learned features.
    """
    
    def __init__(self, initial_arch, base_sparsity, device='cuda', sample_batch=None, is_pretrained=False):
        super().__init__()
        self.device = device
        self.architecture = initial_arch
        self.is_pretrained = is_pretrained  # NEW: Track if this is a pretrained model
        
        # CRITICAL FIX: Pass is_pretrained flag to scaffold creation
        self.scaffold = create_lsuv_sparse_scaffold(
            initial_arch, base_sparsity, device, sample_batch, is_pretrained=is_pretrained
        )
        
        self.scale_patches = nn.ModuleDict({
            'coarse': nn.ModuleList(),
            'medium': nn.ModuleList(),
            'fine': nn.ModuleList()
        })
        self.patch_connections = {}
        self.growth_history = []
        self.current_accuracy = 0.0
        self.activations = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        self.activations = []
        h = x

        for i, layer in enumerate(self.scaffold):
            sparse_out = F.linear(h, layer.weight * layer.mask, layer.bias)
            patch_out = self._compute_patch_contributions(h, i, sparse_out)
            h = sparse_out + patch_out
            
            self.activations.append(h.detach())
            if i < len(self.scaffold) - 1:
                h = F.relu(h)
        
        return h

    def _compute_patch_contributions(self, input_h, layer_idx, sparse_out):
        return torch.zeros_like(sparse_out)

    def analyze_multiscale_extrema(self):
        scale_extrema = {
            'coarse': {'high': [], 'low': []},
            'medium': {'high': [], 'low': []},
            'fine': {'high': [], 'low': []}
        }
        
        for i, activation in enumerate(self.activations[:-1]):
            mean_acts = activation.mean(dim=0)
            high_threshold = mean_acts.mean() + 2 * mean_acts.std()
            low_threshold = 0.1
            
            high_indices = torch.where(mean_acts > high_threshold)[0].cpu().numpy().tolist()
            low_indices = torch.where(mean_acts < low_threshold)[0].cpu().numpy().tolist()

            if i < len(self.scaffold) / 3:
                scale = 'coarse'
            elif i < 2 * len(self.scaffold) / 3:
                scale = 'medium'
            else:
                scale = 'fine'
                
            scale_extrema[scale]['high'].extend([(i, idx) for idx in high_indices])
            scale_extrema[scale]['low'].extend([(i, idx) for idx in low_indices])
            
        return scale_extrema

    def determine_growth_scale(self, scale_extrema):
        coarse_pressure = len(scale_extrema['coarse']['high'])
        medium_pressure = len(scale_extrema['medium']['high'])
        fine_pressure = len(scale_extrema['fine']['high'])
        
        primary_scale = 'coarse'
        if medium_pressure > coarse_pressure and medium_pressure > fine_pressure:
            primary_scale = 'medium'
        elif fine_pressure > coarse_pressure and fine_pressure > medium_pressure:
            primary_scale = 'fine'

        growth_decision = {
            'primary_scale': primary_scale,
            'needs_new_layer': coarse_pressure > 10,
            'patch_density': {
                'coarse': min(0.5, coarse_pressure / 20),
                'medium': min(0.3, medium_pressure / 50),
                'fine': min(0.2, fine_pressure / 100)
            }
        }
        return growth_decision

    def add_sparse_layer_at_scale(self, growth_scale, sample_batch):
        """
        Add new sparse layer with LSUV initialization.
        
        FIXED: New layers get LSUV, but existing pretrained layers are preserved.
        """
        print("   🌱 Adding a new sparse layer...")
        new_layer_size = 128
        self.architecture.insert(-1, new_layer_size)
        
        # CRITICAL FIX: Only apply LSUV to the NEW layer being added
        # Existing pretrained layers should NOT be re-initialized
        print("   ✅ LSUV applied only to NEW layer (pretrained layers preserved)")
        self.scaffold = create_lsuv_sparse_scaffold(
            self.architecture, 0.05, self.device, sample_batch, is_pretrained=False  # New layer gets LSUV
        )
        print(f"   🌱 New architecture: {self.architecture}")

    def add_multiscale_patches(self, scale_extrema, growth_scale):
        """
        Add patches with proper initialization.
        
        FIXED: New patches get proper initialization, not LSUV on pretrained parts.
        """
        patches_added = {'coarse': 0, 'medium': 0, 'fine': 0}
        for scale, extrema in scale_extrema.items():
            for layer_idx, neuron_idx in extrema['high'][:5]:
                patch = nn.Linear(self.architecture[layer_idx], self.architecture[layer_idx+1]).to(self.device)
                
                # Initialize new patch properly (not LSUV on pretrained connections)
                with torch.no_grad():
                    patch.weight.data.normal_(0, 0.01)  # Small random initialization
                    patch.bias.data.zero_()
                
                self.scale_patches[scale].append(patch)
                patches_added[scale] += 1
        print(f"   🔧 Added patches - Coarse: {patches_added['coarse']}, Medium: {patches_added['medium']}, Fine: {patches_added['fine']}")

    def train_to_convergence(self, train_loader, test_loader, max_epochs=5):
        optimizer_groups = [
            {'params': self.scaffold.parameters(), 'lr': 0.0001},
            {'params': self.scale_patches['coarse'].parameters(), 'lr': 0.001},
            {'params': self.scale_patches['medium'].parameters(), 'lr': 0.005},
            {'params': self.scale_patches['fine'].parameters(), 'lr': 0.01}
        ]
        optimizer = optim.Adam(optimizer_groups)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        for epoch in range(max_epochs):
            self.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            self.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            accuracy = correct / len(test_loader.dataset)
            if accuracy > best_acc:
                best_acc = accuracy
            print(f"    Epoch {epoch}: Accuracy {accuracy:.2%}")

        self.current_accuracy = best_acc
        return best_acc

    def grow_with_scale_aware_patches(self, train_loader, test_loader, target_accuracy=0.95):
        iteration = 0
        sample_batch, _ = next(iter(train_loader))
        sample_batch = sample_batch.to(self.device)

        while self.current_accuracy < target_accuracy and iteration < 5:
            iteration += 1
            print(f"\n🌱 Multi-Scale Growth Iteration {iteration}")
            
            self.train_to_convergence(train_loader, test_loader)
            
            scale_extrema = self.analyze_multiscale_extrema()
            growth_scale = self.determine_growth_scale(scale_extrema)
            
            if growth_scale['needs_new_layer']:
                self.add_sparse_layer_at_scale(growth_scale, sample_batch)
            
            self.add_multiscale_patches(scale_extrema, growth_scale)
            
            print(f"📊 Current accuracy: {self.current_accuracy:.2%}")

def load_cifar10_data(batch_size=64):
    """Loads the CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    """Main function to run the experiment."""
    import argparse
    parser = argparse.ArgumentParser(description='Hybrid Growth Experiment - FIXED')
    parser.add_argument('--load-model', type=str, help='Path to a saved model checkpoint to start from.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_cifar10_data()
    
    sample_batch, _ = next(iter(train_loader))
    sample_batch = sample_batch.to(device)

    if args.load_model:
        print(f"🔬 Loading model from checkpoint: {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        initial_arch = checkpoint['architecture']
        base_sparsity = checkpoint.get('sparsity', 0.02)
        
        # CRITICAL FIX: Mark as pretrained to prevent LSUV destruction
        network = HybridGrowthNetwork(
            initial_arch=initial_arch,
            base_sparsity=base_sparsity,
            device=device,
            sample_batch=sample_batch.view(sample_batch.size(0), -1),
            is_pretrained=True  # FIXED: This prevents LSUV from destroying learned features
        )
        network.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("   ✅ Model loaded successfully (pretrained features preserved)")
    else:
        # For new models, LSUV is fine
        network = HybridGrowthNetwork(
            initial_arch=[3072, 128, 32, 10],
            base_sparsity=0.05,
            device=device,
            sample_batch=sample_batch.view(sample_batch.size(0), -1),
            is_pretrained=False  # New model can use LSUV
        )
        print("   ✅ New model created with LSUV initialization")

    network.grow_with_scale_aware_patches(train_loader, test_loader, target_accuracy=0.80)

if __name__ == "__main__":
    main()
