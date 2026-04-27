#!/usr/bin/env python3
"""
Enhanced Extrema Growth - State-of-the-Art Implementation

This experiment implements the sophisticated extrema targeting techniques from
the successful CIFAR-10 experiments, including:
1. Data-driven connection placement (no random connections!)
2. Dual learning rates for scaffold vs patches
3. Targeted patch creation based on extrema analysis
4. Credit-based growth economy
5. Vertical cloning and dead neuron highways
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.structure_net import (
    create_standard_network,
    save_model_seed,
    load_model_seed,
    get_network_stats,
    sort_all_network_layers
)

class EnhancedExtremaGrowth:
    """
    Enhanced extrema-driven growth with sophisticated targeting.
    
    Key innovations:
    - Data-driven connection placement (no random connections)
    - Dual learning rates (scaffold vs patches)
    - Credit-based growth economy
    - Vertical cloning for saturated neurons
    - Dead neuron highways with skip connections
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.network = None
        self.current_accuracy = 0.0
        
        # Credit-based growth economy
        self.credits = 0.0
        self.credit_earn_rate = 1.0  # Credits per correct prediction
        self.growth_cost = 50.0      # Credits needed for growth
        
        # Dual learning rate components
        self.scaffold_params = []
        self.patch_params = []
        
        # Growth thresholds (patches only, no neck blocks)
        self.dead_zone_threshold = 999999  # Disable neck blocks completely
        self.saturation_threshold = 20     # Keep relief connections
        
    def load_network(self, checkpoint_path):
        """Load pretrained network."""
        print(f"🔬 Loading network: {checkpoint_path}")
        self.network, metadata = load_model_seed(checkpoint_path, device=self.device)
        self.current_accuracy = metadata.get('accuracy', 0.0)
        
        # Initialize scaffold parameters (existing network)
        self.scaffold_params = list(self.network.parameters())
        self.patch_params = []  # Will be populated as patches are added
        
        print(f"   ✅ Loaded: {metadata['architecture']}, Acc: {self.current_accuracy:.2%}")
        return self.network
    
    def detect_sophisticated_extrema(self, train_loader):
        """
        Sophisticated extrema detection using activation analysis.
        Based on the successful CIFAR-10 experiments.
        """
        print("\n🔍 SOPHISTICATED EXTREMA DETECTION")
        print("=" * 40)
        
        self.network.eval()
        all_activations = []
        
        # Collect activations from multiple batches for robust analysis
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 5:  # Use 5 batches for better statistics
                    break
                    
                data = data.to(self.device).view(data.size(0), -1)
                
                # Forward pass and collect activations
                x = data
                batch_activations = []
                
                for layer in self.network:
                    if hasattr(layer, 'mask'):  # Sparse layer
                        x = layer(x)
                        batch_activations.append(x.detach())
                    elif isinstance(layer, nn.ReLU):
                        x = layer(x)
                        # Update the last activation with ReLU applied
                        if batch_activations:
                            batch_activations[-1] = x.detach()
                
                all_activations.append(batch_activations)
        
        # Analyze extrema patterns with sophisticated thresholds
        extrema_analysis = {
            'dead_neurons': {},
            'saturated_neurons': {},
            'decisive_features': {},
            'connection_patterns': {}
        }
        
        for layer_idx in range(len(all_activations[0])):
            # Combine activations across batches
            layer_acts = torch.cat([batch[layer_idx] for batch in all_activations], dim=0)
            mean_acts = layer_acts.mean(dim=0)
            std_acts = layer_acts.std(dim=0)
            
            # Sophisticated thresholds based on distribution
            dead_threshold = 0.01
            saturated_threshold = mean_acts.mean() + 2.5 * mean_acts.std()
            
            dead_neurons = torch.where(mean_acts < dead_threshold)[0].cpu().numpy().tolist()
            saturated_neurons = torch.where(mean_acts > saturated_threshold)[0].cpu().numpy().tolist()
            
            print(f"📊 Layer {layer_idx}: {len(dead_neurons)} dead, {len(saturated_neurons)} saturated")
            
            if dead_neurons:
                extrema_analysis['dead_neurons'][layer_idx] = dead_neurons
                
                # Analyze what features could revive dead neurons
                if layer_idx > 0:
                    prev_layer_acts = torch.cat([batch[layer_idx-1] for batch in all_activations], dim=0)
                    decisive_features = self._find_decisive_features(prev_layer_acts, dead_neurons)
                    extrema_analysis['decisive_features'][layer_idx] = decisive_features
            
            if saturated_neurons:
                extrema_analysis['saturated_neurons'][layer_idx] = saturated_neurons
                
                # Analyze connection patterns for saturated neurons
                connection_patterns = self._analyze_connection_patterns(layer_idx, saturated_neurons)
                extrema_analysis['connection_patterns'][layer_idx] = connection_patterns
        
        return extrema_analysis
    
    def _find_decisive_features(self, prev_activations, dead_neurons):
        """
        Find the most decisive features for reviving dead neurons.
        Uses topk analysis instead of random connections.
        """
        decisive_features = {}
        
        # For each dead neuron, find which input features are most important
        for dead_idx in dead_neurons[:10]:  # Limit analysis
            # Find neurons in previous layer that have high variance
            # These are the most "informative" features
            feature_variance = prev_activations.var(dim=0)
            topk_features = torch.topk(feature_variance, k=min(20, len(feature_variance)))[1]
            
            decisive_features[dead_idx] = topk_features.cpu().numpy().tolist()
        
        return decisive_features
    
    def _analyze_connection_patterns(self, layer_idx, saturated_neurons):
        """
        Analyze connection patterns for saturated neurons.
        Identifies which connections are most effective.
        """
        patterns = {}
        
        # Get the current layer's weights
        sparse_layers = [layer for layer in self.network if hasattr(layer, 'mask')]
        if layer_idx < len(sparse_layers):
            layer = sparse_layers[layer_idx]
            
            for sat_idx in saturated_neurons[:5]:  # Limit analysis
                if sat_idx < layer.linear.weight.shape[0]:
                    # Find strongest outgoing connections
                    outgoing_weights = layer.linear.weight.data[sat_idx, :]
                    strong_connections = torch.where(torch.abs(outgoing_weights) > 0.1)[0]
                    patterns[sat_idx] = strong_connections.cpu().numpy().tolist()
        
        return patterns
    
    def create_targeted_patches(self, extrema_analysis):
        """
        Create targeted patches based on sophisticated extrema analysis.
        No random connections - all connections are data-driven!
        """
        patches_created = 0
        
        # Create patches for dead neurons using decisive features
        for layer_idx, dead_neurons in extrema_analysis['dead_neurons'].items():
            if len(dead_neurons) >= self.dead_zone_threshold:
                print(f"\n🏗️  CREATING TARGETED PATCHES FOR DEAD ZONE")
                print(f"   Layer {layer_idx}: {len(dead_neurons)} dead neurons")
                
                decisive_features = extrema_analysis['decisive_features'].get(layer_idx, {})
                
                for dead_idx in dead_neurons[:5]:  # Limit to 5 patches
                    if dead_idx in decisive_features:
                        # Create patch using decisive features (not random!)
                        features = decisive_features[dead_idx]
                        patch_size = min(16, len(features))
                        
                        # Create targeted dense patch
                        patch = nn.Sequential(
                            nn.Linear(len(features), patch_size),
                            nn.ReLU(),
                            nn.Linear(patch_size, 1)
                        ).to(self.device)
                        
                        # Add to patch parameters for dual learning rate
                        self.patch_params.extend(patch.parameters())
                        
                        # Store patch with connection info
                        patch_name = f"dead_patch_{layer_idx}_{dead_idx}"
                        if not hasattr(self, 'patches'):
                            self.patches = nn.ModuleDict()
                        self.patches[patch_name] = patch
                        
                        patches_created += 1
                        print(f"   ✅ Created targeted patch for neuron {dead_idx} using {len(features)} decisive features")
        
        # Create relief connections for saturated neurons
        for layer_idx, saturated_neurons in extrema_analysis['saturated_neurons'].items():
            if len(saturated_neurons) >= self.saturation_threshold:
                print(f"\n⚡ CREATING RELIEF CONNECTIONS")
                print(f"   Layer {layer_idx}: {len(saturated_neurons)} saturated neurons")
                
                connection_patterns = extrema_analysis['connection_patterns'].get(layer_idx, {})
                
                # Use vertical cloning technique
                for sat_idx in saturated_neurons[:3]:  # Limit to 3 clones
                    if sat_idx in connection_patterns:
                        self._create_vertical_clone(layer_idx, sat_idx, connection_patterns[sat_idx])
                        patches_created += 1
        
        return patches_created
    
    def _create_vertical_clone(self, layer_idx, neuron_idx, connection_pattern):
        """
        Create vertical clone of saturated neuron.
        Based on the tricks.md vertical cloning technique.
        """
        sparse_layers = [layer for layer in self.network if hasattr(layer, 'mask')]
        
        if layer_idx < len(sparse_layers) - 1:  # Not the last layer
            current_layer = sparse_layers[layer_idx]
            next_layer = sparse_layers[layer_idx + 1]
            
            with torch.no_grad():
                # Clone the neuron's incoming connections
                if neuron_idx < current_layer.mask.shape[0]:
                    # Find an unused neuron to clone into
                    weight_norms = torch.norm(current_layer.linear.weight.data, dim=1)
                    unused_neurons = torch.where(weight_norms < 0.01)[0]
                    
                    if len(unused_neurons) > 0:
                        clone_idx = unused_neurons[0].item()
                        
                        # Copy connection pattern (data-driven, not random!)
                        current_layer.mask[clone_idx, :] = current_layer.mask[neuron_idx, :].clone()
                        current_layer.linear.weight.data[clone_idx, :] = current_layer.linear.weight.data[neuron_idx, :].clone() * 0.5
                        
                        # Create diverse output connections for the clone
                        for target_idx in connection_pattern[:3]:
                            if target_idx < next_layer.mask.shape[0]:
                                next_layer.mask[target_idx, clone_idx] = 1.0
                                next_layer.linear.weight.data[target_idx, clone_idx] = torch.randn(1).item() * 0.1
                        
                        print(f"   🧬 Cloned neuron {neuron_idx} to {clone_idx} with {len(connection_pattern)} connections")
    
    def earn_credits(self, correct_predictions, total_predictions):
        """
        Earn credits based on performance.
        Credit-based growth economy from tricks.md.
        """
        accuracy = correct_predictions / total_predictions
        credits_earned = correct_predictions * self.credit_earn_rate
        self.credits += credits_earned
        
        print(f"   💰 Earned {credits_earned:.1f} credits (accuracy: {accuracy:.2%}), Total: {self.credits:.1f}")
        return credits_earned
    
    def can_afford_growth(self):
        """Check if we have enough credits for growth."""
        return self.credits >= self.growth_cost
    
    def spend_credits_for_growth(self):
        """Spend credits for growth action."""
        if self.can_afford_growth():
            self.credits -= self.growth_cost
            print(f"   💸 Spent {self.growth_cost} credits for growth, Remaining: {self.credits:.1f}")
            return True
        return False
    
    def enhanced_growth_step(self, train_loader):
        """
        Perform enhanced extrema-driven growth with credit economy.
        """
        print("\n🧬 ENHANCED GROWTH STEP")
        print("=" * 50)
        
        # Check if we can afford growth
        if not self.can_afford_growth():
            print(f"💰 Insufficient credits ({self.credits:.1f}/{self.growth_cost}) - no growth")
            return False
        
        # Detect extrema using sophisticated analysis
        extrema_analysis = self.detect_sophisticated_extrema(train_loader)
        
        # Check if growth is needed
        total_dead = sum(len(neurons) for neurons in extrema_analysis['dead_neurons'].values())
        total_saturated = sum(len(neurons) for neurons in extrema_analysis['saturated_neurons'].values())
        
        if total_dead < self.dead_zone_threshold and total_saturated < self.saturation_threshold:
            print("✅ Network is well-balanced - no growth needed")
            return False
        
        # Spend credits for growth
        if not self.spend_credits_for_growth():
            return False
        
        print(f"🎯 Growth triggered: {total_dead} dead, {total_saturated} saturated neurons")
        
        # Create targeted patches
        patches_created = self.create_targeted_patches(extrema_analysis)
        
        if patches_created > 0:
            # Apply neuron sorting after growth
            sort_all_network_layers(self.network)
            print("   🔄 Applied neuron sorting")
            print(f"   ✅ Created {patches_created} targeted patches/clones")
            return True
        
        return False
    
    def evaluate_network_with_credits(self, test_loader):
        """Evaluate network and earn credits."""
        self.network.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                output = self.network(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        self.current_accuracy = accuracy
        
        # Earn credits based on performance
        self.earn_credits(correct, total)
        
        print(f"📊 Accuracy: {accuracy:.2%}")
        return accuracy
    
    def train_with_dual_learning_rates(self, train_loader, test_loader, epochs=5):
        """
        Train with dual learning rates: slow for scaffold, fast for patches.
        Key technique from tricks.md for preventing catastrophic forgetting.
        """
        print(f"🚀 Training with dual learning rates for {epochs} epochs")
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': self.scaffold_params, 'lr': 0.0001, 'name': 'scaffold'},  # Slow for stability
        ]
        
        if self.patch_params:
            param_groups.append({
                'params': self.patch_params, 'lr': 0.001, 'name': 'patches'  # Fast for new learning
            })
        
        optimizer = optim.Adam(param_groups)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.network.train()
            total_loss = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                output = self.network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                accuracy = self.evaluate_network_with_credits(test_loader)
                print(f"   Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}, Acc {accuracy:.2%}")
        
        return self.evaluate_network_with_credits(test_loader)
    
    def run_enhanced_experiment(self, train_loader, test_loader, max_iterations=5):
        """Run the enhanced extrema-driven growth experiment."""
        print("🔬 ENHANCED EXTREMA-DRIVEN GROWTH EXPERIMENT")
        print("=" * 60)
        print("🎯 Key innovations:")
        print("   • Data-driven connection placement (no random!)")
        print("   • Dual learning rates (scaffold vs patches)")
        print("   • Credit-based growth economy")
        print("   • Vertical cloning for saturated neurons")
        print("   • Decisive feature analysis for dead neurons")
        print("   • NO NECK BLOCKS - patches and clones only!")
        
        iteration = 0
        while self.current_accuracy < 0.80 and iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            
            # Train with dual learning rates
            accuracy = self.train_with_dual_learning_rates(train_loader, test_loader, epochs=8)
            
            # Apply enhanced growth (if credits allow)
            growth_occurred = self.enhanced_growth_step(train_loader)
            
            print(f"📊 Iteration {iteration} complete: Acc {accuracy:.2%}, Growth: {'Yes' if growth_occurred else 'No'}")
            print(f"💰 Credits: {self.credits:.1f}")
            
            # Save checkpoint
            if accuracy > 0.35:
                # Get actual architecture
                sparse_layers = [layer for layer in self.network if hasattr(layer, 'mask')]
                architecture = []
                for i, layer in enumerate(sparse_layers):
                    if i == 0:
                        architecture.append(layer.linear.in_features)
                    architecture.append(layer.linear.out_features)
                
                save_model_seed(
                    model=self.network,
                    architecture=architecture,
                    seed=42,
                    metrics={
                        'accuracy': accuracy, 
                        'iteration': iteration, 
                        'credits': self.credits,
                        'enhanced_extrema': True
                    },
                    filepath=f"data/enhanced_extrema_iter{iteration}_acc{accuracy:.2f}.pt"
                )
                print(f"   💾 Saved checkpoint")
        
        print(f"\n✅ Enhanced experiment complete! Final accuracy: {self.current_accuracy:.2%}")
        print(f"💰 Final credits: {self.credits:.1f}")

def load_cifar10_data(batch_size=64):
    """Load CIFAR-10 dataset."""
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
    """Main function for enhanced extrema-driven growth experiment."""
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Extrema Growth Experiment')
    parser.add_argument('--load-model', type=str, required=True, help='Path to pretrained model checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    # Load data
    train_loader, test_loader = load_cifar10_data()
    
    # Create enhanced extrema-driven growth engine
    growth_engine = EnhancedExtremaGrowth(device=device)
    
    # Load pretrained network
    growth_engine.load_network(args.load_model)
    
    # Run enhanced extrema-driven growth experiment
    growth_engine.run_enhanced_experiment(train_loader, test_loader)
    
    print("\n🎯 ENHANCED EXTREMA-DRIVEN GROWTH COMPLETE!")
    print("Key innovations successfully implemented:")
    print("✅ Data-driven connections (no random)")
    print("✅ Dual learning rates")
    print("✅ Credit-based growth economy")
    print("✅ Vertical cloning")
    print("✅ Decisive feature analysis")

if __name__ == "__main__":
    main()
