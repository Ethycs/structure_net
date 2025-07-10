# patched_density_experiment.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

class PatchedDensityNetwork(nn.Module):
    """Network with sparse scaffold + dense patches at extrema"""
    
    def __init__(self, architecture=[784, 256, 128, 10], scaffold_sparsity=0.02, device='cuda'):
        super().__init__()
        self.device = device
        self.architecture = architecture
        
        # Create sparse scaffold
        self.scaffold = self._create_sparse_scaffold(architecture, scaffold_sparsity)
        
        # These will be our dense patches
        self.patches = nn.ModuleDict()
        self.patch_connections = {}  # Track where patches connect
        
    def _create_sparse_scaffold(self, architecture, sparsity):
        """Create the initial sparse network"""
        layers = nn.ModuleList()
        
        for i in range(len(architecture) - 1):
            in_features = architecture[i]
            out_features = architecture[i + 1]
            
            # Create sparse layer
            layer = nn.Linear(in_features, out_features)
            
            # Apply sparsity mask
            num_connections = int(sparsity * in_features * out_features)
            mask = torch.zeros(out_features, in_features)
            
            # Random sparse connections
            indices = torch.randperm(in_features * out_features)[:num_connections]
            mask.view(-1)[indices] = 1.0
            
            # Register mask as buffer
            layer.register_buffer('mask', mask)
            
            # Apply mask to weights
            with torch.no_grad():
                layer.weight.mul_(mask)
            
            layers.append(layer)
            
        return layers
    
    def forward(self, x):
        """Forward pass through scaffold + patches"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        self.activations = []
        h = x
        
        for i, layer in enumerate(self.scaffold):
            # Apply sparse weights
            sparse_out = F.linear(h, layer.weight * layer.mask, layer.bias)
            
            # Add patch contributions for this layer
            patch_out = self._compute_patch_contributions(h, i, sparse_out)
            
            # Combine sparse and patch outputs
            h = sparse_out + patch_out
            
            # Store activation for extrema detection
            self.activations.append(h.detach())
            
            # Apply activation (except last layer)
            if i < len(self.scaffold) - 1:
                h = F.relu(h)
        
        return h

    def _compute_patch_contributions(self, input_h, layer_idx, sparse_out):
        """Compute contributions from all patches at this layer"""
        batch_size = input_h.size(0)
        output_size = sparse_out.size(1)
        
        patch_contribution = torch.zeros(batch_size, output_size, device=self.device)
        
        # Process patches that affect this layer
        for patch_name, patch_info in self.patch_connections.items():
            if patch_info['target_layer'] == layer_idx:
                patch = self.patches[patch_name]
                
                if patch_info['type'] == 'high_extrema':
                    # High extrema patch: takes saturated neuron, creates alternative paths
                    source_neuron = patch_info['source_neuron']
                    if layer_idx > 0 and source_neuron < self.activations[layer_idx-1].size(1):
                        # Get the saturated neuron's activation from previous layer
                        neuron_act = self.activations[layer_idx-1][:, source_neuron:source_neuron+1]
                        
                        # Process through patch
                        patch_out = patch(neuron_act)
                        
                        # Add to specific output neurons
                        target_neurons = patch_info['target_neurons']
                        for j, target in enumerate(target_neurons):
                            if j < patch_out.size(1) and target < output_size:
                                patch_contribution[:, target] += patch_out[:, j] * 0.3
                
                elif patch_info['type'] == 'low_extrema':
                    # Low extrema patch: provides additional input to dead neurons
                    target_neuron = patch_info['target_neuron']
                    if target_neuron < output_size:
                        # Take broader input from previous layer
                        if layer_idx > 0:
                            prev_size = input_h.size(1)
                            sample_size = min(20, prev_size)
                            sample_indices = torch.randperm(prev_size)[:sample_size]
                            sampled_input = input_h[:, sample_indices]
                            
                            # Process through patch
                            patch_out = patch(sampled_input)
                            
                            # Add to dead neuron
                            patch_contribution[:, target_neuron] += patch_out.squeeze(-1) * 0.5
        
        return patch_contribution
    
    def detect_extrema(self):
        """Detect extrema in the network"""
        extrema = {'high': {}, 'low': {}}
        
        if not hasattr(self, 'activations') or not self.activations:
            print("⚠️  Activations not found. Run a forward pass before detecting extrema.")
            return extrema

        for i, activations in enumerate(self.activations[:-1]):  # Skip output layer
            # Average across batch
            mean_acts = activations.mean(dim=0)
            
            # High extrema (saturated neurons)
            high_threshold = mean_acts.mean() + 2 * mean_acts.std()
            high_indices = torch.where(mean_acts > high_threshold)[0]
            if len(high_indices) > 0:
                extrema['high'][i] = high_indices.cpu().numpy().tolist()
            
            # Low extrema (dead neurons)
            low_threshold = 0.1
            low_indices = torch.where(mean_acts < low_threshold)[0]
            if len(low_indices) > 0:
                extrema['low'][i] = low_indices.cpu().numpy().tolist()
        
        return extrema
    
    def create_dense_patches(self, extrema, patch_density=0.5):
        """Create dense patches around extrema"""
        patches_created = 0
        
        # Create patches for high extrema
        for layer_idx, neuron_indices in extrema.get('high', {}).items():
            next_layer_size = self.scaffold[layer_idx + 1].out_features if layer_idx < len(self.scaffold) - 1 else 10
            
            for j, neuron_idx in enumerate(neuron_indices[:5]):  # Limit to 5 patches per layer
                patch_name = f"high_{layer_idx}_{neuron_idx}"
                
                # Create a small dense network patch
                patch = nn.Sequential(
                    nn.Linear(1, 8),
                    nn.ReLU(),
                    nn.Linear(8, min(4, next_layer_size))
                ).to(self.device)
                
                self.patches[patch_name] = patch
                
                # Track connections
                target_neurons = list(range(min(4, next_layer_size)))
                self.patch_connections[patch_name] = {
                    'type': 'high_extrema',
                    'source_layer': layer_idx,
                    'source_neuron': int(neuron_idx),
                    'target_layer': layer_idx + 1,
                    'target_neurons': target_neurons
                }
                
                patches_created += 1
        
        # Create patches for low extrema
        for layer_idx, neuron_indices in extrema.get('low', {}).items():
            if layer_idx > 0:  # Can't add input to first layer
                
                for j, neuron_idx in enumerate(neuron_indices[:5]):
                    patch_name = f"low_{layer_idx}_{neuron_idx}"
                    
                    # Create input patch for dead neuron
                    patch = nn.Sequential(
                        nn.Linear(20, 8),  # Sample 20 inputs
                        nn.ReLU(),
                        nn.Linear(8, 1)
                    ).to(self.device)
                    
                    self.patches[patch_name] = patch
                    
                    self.patch_connections[patch_name] = {
                        'type': 'low_extrema',
                        'source_layer': layer_idx - 1,
                        'target_layer': layer_idx,
                        'target_neuron': int(neuron_idx)
                    }
                    
                    patches_created += 1
        
        return patches_created
    
    def analyze_patch_effectiveness(self):
        """Analyze how effective patches are"""
        stats = {
            'active_count': 0,
            'avg_contribution': 0.0,
            'weight_norms': {}
        }
        
        for name, patch in self.patches.items():
            weight_norm = sum(p.norm().item() for p in patch.parameters())
            stats['weight_norms'][name] = weight_norm
            
            if weight_norm > 0.1:  # Active threshold
                stats['active_count'] += 1
                stats['avg_contribution'] += weight_norm
        
        if stats['active_count'] > 0:
            stats['avg_contribution'] /= stats['active_count']
            
        return stats


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load MNIST
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Create patched density network
    model = PatchedDensityNetwork(device=device).to(device)
    
    # Phase 1: Train sparse scaffold
    print("\n🏗️  Phase 1: Training sparse scaffold (2% sparsity)")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"  Epoch {epoch+1}/20, Test Acc: {test_acc:.2f}%")
    
    # Detect extrema
    print("\nDetecting extrema on test set...")
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        _ = model(data.to(device))
    extrema = model.detect_extrema()
    
    total_extrema = sum(len(v) for v in extrema.get('high', {}).values()) + \
                   sum(len(v) for v in extrema.get('low', {}).values())
    print(f"\nFound {sum(len(v) for v in extrema.get('high', {}).values())} high extrema, "
          f"{sum(len(v) for v in extrema.get('low', {}).values())} low extrema")
    
    # Phase 2: Create dense patches
    print(f"\n🔧 Phase 2: Creating dense patches (density=0.5)")
    patches_created = model.create_dense_patches(extrema, patch_density=0.5)
    print(f"✅ Created {patches_created} dense patches")
    
    # Phase 3: Train with dual learning rates
    print(f"\n🎯 Phase 3: Training with dual learning rates")
    print(f"   Scaffold LR: 0.0001 (slow)")
    print(f"   Patch LR: 0.0005 (faster)")
    
    # Create parameter groups
    param_groups = [
        {'params': model.scaffold.parameters(), 'lr': 0.0001, 'name': 'scaffold'}
    ]
    
    for patch_name, patch in model.patches.items():
        param_groups.append({
            'params': patch.parameters(),
            'lr': 0.0005,
            'name': f'patch_{patch_name}'
        })
    
    optimizer = optim.Adam(param_groups)
    
    best_accuracy = 0
    for epoch in range(30):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        
        if epoch % 5 == 0:
            patch_stats = model.analyze_patch_effectiveness()
            print(f"  Epoch {epoch}: Test Acc: {test_acc:.2f}%")
            print(f"    Active patches: {patch_stats['active_count']}/{len(model.patches)}")
            print(f"    Avg patch weight norm: {patch_stats['avg_contribution']:.4f}")
        
        best_accuracy = max(best_accuracy, test_acc)
    
    print(f"\n🎉 Experiment completed. Best accuracy: {best_accuracy:.2f}%")
    
    # Final analysis
    print("\n📊 Final Analysis:")
    print(f"  Sparse scaffold parameters: {sum(p.numel() for p in model.scaffold.parameters())}")
    print(f"  Patch parameters: {sum(p.numel() for p in model.patches.parameters())}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Compare to dense network
    dense_params = sum((model.architecture[i] * model.architecture[i+1]) for i in range(len(model.architecture)-1))
    sparsity = 1 - (sum(p.numel() for p in model.parameters()) / dense_params)
    print(f"  Effective sparsity: {sparsity:.2%}")


if __name__ == "__main__":
    main()
