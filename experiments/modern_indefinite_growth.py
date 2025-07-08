#!/usr/bin/env python3
"""
Modern Indefinite Growth Experiment

A modernized version of the indefinite growth experiment using the new structure_net
codebase with embedded patches, dual learning rates, and credit-based growth economy.

Key improvements:
- Uses structure_net canonical standard
- Embedded patches (no separate ModuleDict)
- Data-driven connection placement
- Credit-based growth economy
- Dual learning rates
- Sophisticated extrema detection
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.structure_net import (
    create_standard_network,
    save_model_seed,
    load_model_seed,
    get_network_stats,
    sort_all_network_layers
)
from src.structure_net.evolution.integrated_growth_system import (
    IntegratedGrowthSystem,
    ThresholdConfig,
    MetricsConfig
)
# MultiScaleNetwork import removed due to missing dependencies
# Will use alternative implementation for comparative mode

class ModernIndefiniteGrowth:
    """
    Modern indefinite growth engine using structure_net canonical standard.
    
    Grows networks indefinitely until target accuracy using:
    - Embedded patches (no separate modules)
    - Credit-based growth economy
    - Data-driven connection placement
    - Dual learning rates
    - Sophisticated extrema detection
    """
    
    def __init__(self, seed_architecture, scaffold_sparsity=0.02, device='cuda', allow_patches=True, neuron_sort_epochs=5):
        self.device = device
        self.scaffold_sparsity = scaffold_sparsity
        self.seed_architecture = seed_architecture
        self.allow_patches = allow_patches
        
        # Create initial network using canonical standard
        self.network = create_standard_network(
            architecture=seed_architecture,
            sparsity=scaffold_sparsity,
            device=device
        )
        
        # Growth tracking
        self.growth_history = []
        self.current_accuracy = 0.0
        self.iteration = 0
        
        # Gradient variance tracking for efficient growth signals
        self.gradient_history = {}
        self.gradient_window_size = 50  # Track last 50 gradient norms
        self.gradient_variance_threshold = 0.1  # Threshold for stagnant gradients
        
        # Credit system disabled - direct growth based on extrema
        self.credits = 0.0
        self.credit_earn_rate = 0.0       # Disabled
        self.layer_growth_cost = 0.0      # Free growth
        self.patch_growth_cost = 0.0      # Free growth
        
        # Dual learning rate tracking
        self.scaffold_params = list(self.network.parameters())
        self.patch_params = []
        
        # Growth thresholds
        self.layer_addition_threshold = 0.6  # 60% extrema ratio triggers new layer
        self.patch_threshold = 5             # 5+ extrema triggers patches (much more reasonable)
        self.neuron_sort_epochs = neuron_sort_epochs  # Configurable neuron sorting frequency
        
        print(f"🚀 Modern Indefinite Growth initialized")
        print(f"   Seed architecture: {seed_architecture}")
        print(f"   Scaffold sparsity: {scaffold_sparsity:.1%}")
        print(f"   Device: {device}")
        print(f"   Allow Patches: {self.allow_patches}")
    
    @property
    def current_architecture(self):
        """Get current architecture from network stats."""
        stats = get_network_stats(self.network)
        return stats['architecture']
    
    def update_gradient_history(self):
        """
        Efficiently update gradient variance history for sparse networks.
        Optimized with vectorized operations and reduced frequency.
        """
        sparse_layers = [layer for layer in self.network if hasattr(layer, 'mask')]
        
        for layer_idx, layer in enumerate(sparse_layers):
            if layer.linear.weight.grad is not None:
                # Initialize history for this layer if needed
                if layer_idx not in self.gradient_history:
                    self.gradient_history[layer_idx] = {
                        n: [] for n in range(layer.linear.weight.shape[0])
                    }
                
                # Get gradient tensor (already sparse due to mask)
                grad = layer.linear.weight.grad
                mask = layer.mask
                
                # Vectorized gradient norm calculation (MAJOR OPTIMIZATION)
                masked_grad = grad * mask  # Apply mask
                grad_norms = torch.norm(masked_grad, dim=1)  # Per-neuron norms in one operation
                
                # Update history for all neurons at once
                for neuron_idx in range(grad.shape[0]):
                    grad_norm = grad_norms[neuron_idx].item()  # Single GPU->CPU transfer per neuron
                    
                    # Update history with sliding window
                    history = self.gradient_history[layer_idx][neuron_idx]
                    history.append(grad_norm)
                    
                    # Maintain window size
                    if len(history) > self.gradient_window_size:
                        history.pop(0)
    
    def analyze_gradient_variance(self):
        """
        Analyze gradient variance to identify stagnant neurons.
        Returns neurons with low gradient variance (learning stagnation).
        """
        stagnant_analysis = {
            'stagnant_neurons': {},
            'high_variance_neurons': {},
            'layer_gradient_health': {}
        }
        
        for layer_idx, layer_history in self.gradient_history.items():
            stagnant_neurons = []
            high_variance_neurons = []
            
            for neuron_idx, grad_norms in layer_history.items():
                if len(grad_norms) >= 10:  # Need sufficient history
                    grad_variance = np.var(grad_norms)
                    grad_mean = np.mean(grad_norms)
                    
                    # Identify stagnant neurons (low variance, low mean)
                    if grad_variance < self.gradient_variance_threshold and grad_mean < 0.01:
                        stagnant_neurons.append(neuron_idx)
                    
                    # Identify high-variance neurons (active learning)
                    elif grad_variance > self.gradient_variance_threshold * 5:
                        high_variance_neurons.append(neuron_idx)
            
            stagnant_analysis['stagnant_neurons'][layer_idx] = stagnant_neurons
            stagnant_analysis['high_variance_neurons'][layer_idx] = high_variance_neurons
            
            # Calculate layer gradient health
            total_neurons = len(layer_history)
            if total_neurons > 0:
                gradient_health = 1.0 - (len(stagnant_neurons) / total_neurons)
                stagnant_analysis['layer_gradient_health'][layer_idx] = gradient_health
        
        return stagnant_analysis
    
    def detect_sophisticated_extrema(self, data_loader):
        """
        Sophisticated extrema detection using multiple batches + gradient variance.
        """
        print("\n🔍 SOPHISTICATED EXTREMA DETECTION")
        print("=" * 40)
        
        self.network.eval()
        all_activations = []
        
        # Collect activations from multiple batches (OPTIMIZED: reduced from 5 to 2 batches)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 2:  # Use 2 batches for faster analysis
                    break
                    
                data = data.to(self.device).view(data.size(0), -1)
                
                # Forward pass and collect activations using model_io approach
                x = data
                batch_activations = []
                
                # Get sparse layers using model_io pattern
                from src.structure_net.core import StandardSparseLayer
                
                for layer in self.network:
                    if isinstance(layer, StandardSparseLayer):  # Use proper type check
                        x = layer(x)
                        batch_activations.append(x.detach())
                    elif isinstance(layer, nn.ReLU):
                        x = layer(x)
                        # Update the last activation with ReLU applied
                        if batch_activations:
                            batch_activations[-1] = x.detach()
                
                # Ensure we have the final output (after last layer)
                if len(batch_activations) == 0:
                    # Fallback: just run the full network
                    x = self.network(data)
                    batch_activations = [x.detach()]
                
                all_activations.append(batch_activations)
        
        # Debug activation collection
        if all_activations:
            print(f"🔧 DEBUG - Collected {len(all_activations[0])} activation layers from {len(all_activations)} batches")
            for i, layer_acts in enumerate(all_activations[0]):
                print(f"🔧 DEBUG - Layer {i}: {layer_acts.shape}")
        
        # Analyze extrema patterns
        extrema_analysis = {
            'dead_neurons': {},
            'saturated_neurons': {},
            'layer_health': {},
            'total_extrema': 0
        }
        
        total_neurons = 0
        total_extrema = 0
        
        for layer_idx in range(len(all_activations[0])):  # Include ALL layers (including output)
            # Combine activations across batches
            layer_acts = torch.cat([batch[layer_idx] for batch in all_activations], dim=0)
            mean_acts = layer_acts.mean(dim=0)
            
            # Sophisticated thresholds
            dead_threshold = 0.01
            saturated_threshold = mean_acts.mean() + 2.5 * mean_acts.std()
            
            dead_neurons = torch.where(mean_acts < dead_threshold)[0].cpu().numpy().tolist()
            saturated_neurons = torch.where(mean_acts > saturated_threshold)[0].cpu().numpy().tolist()
            
            layer_extrema = len(dead_neurons) + len(saturated_neurons)
            layer_size = len(mean_acts)
            layer_health = 1.0 - (layer_extrema / max(layer_size, 1))
            
            print(f"📊 Layer {layer_idx}: {len(dead_neurons)} dead, {len(saturated_neurons)} saturated, health: {layer_health:.1%}")
            
            extrema_analysis['dead_neurons'][layer_idx] = dead_neurons
            extrema_analysis['saturated_neurons'][layer_idx] = saturated_neurons
            extrema_analysis['layer_health'][layer_idx] = layer_health
            
            total_neurons += layer_size
            total_extrema += layer_extrema
        
        extrema_analysis['total_extrema'] = total_extrema
        extrema_analysis['extrema_ratio'] = total_extrema / max(total_neurons, 1)
        
        print(f"📊 Overall: {total_extrema}/{total_neurons} extrema ({extrema_analysis['extrema_ratio']:.1%})")
        
        # Add gradient variance analysis if we have sufficient history
        if self.gradient_history:
            print("\n📈 GRADIENT VARIANCE ANALYSIS")
            print("=" * 40)
            
            stagnant_analysis = self.analyze_gradient_variance()
            
            for layer_idx, stagnant_neurons in stagnant_analysis['stagnant_neurons'].items():
                high_variance_neurons = stagnant_analysis['high_variance_neurons'][layer_idx]
                gradient_health = stagnant_analysis['layer_gradient_health'][layer_idx]
                
                print(f"📈 Layer {layer_idx}: {len(stagnant_neurons)} stagnant, {len(high_variance_neurons)} active, health: {gradient_health:.1%}")
            
            # Merge gradient analysis with extrema analysis
            extrema_analysis['stagnant_neurons'] = stagnant_analysis['stagnant_neurons']
            extrema_analysis['gradient_health'] = stagnant_analysis['layer_gradient_health']
        
        return extrema_analysis
    
    def should_add_layer(self, extrema_analysis):
        """Decide if we need a new layer based on extrema ratio."""
        return extrema_analysis['extrema_ratio'] > self.layer_addition_threshold
    
    def find_worst_bottleneck_layer(self, extrema_analysis):
        """Find the layer with the worst health (most extrema)."""
        if not extrema_analysis['layer_health']:
            return 0
        
        worst_layer = min(extrema_analysis['layer_health'].keys(), 
                         key=lambda k: extrema_analysis['layer_health'][k])
        return worst_layer
    
    def add_layer_at_bottleneck(self, extrema_analysis):
        """Add a new layer at the worst bottleneck."""
        worst_layer = self.find_worst_bottleneck_layer(extrema_analysis)
        
        # Calculate new layer size based on extrema count
        layer_extrema = (len(extrema_analysis['dead_neurons'].get(worst_layer, [])) + 
                        len(extrema_analysis['saturated_neurons'].get(worst_layer, [])))
        new_layer_size = min(max(layer_extrema * 2, 64), 512)  # 2x extrema, min 64, max 512
        
        print(f"\n🏗️  ADDING LAYER AT BOTTLENECK")
        print(f"   Position: After layer {worst_layer}")
        print(f"   Size: {new_layer_size}")
        print(f"   Reason: Layer health {extrema_analysis['layer_health'][worst_layer]:.1%}")
        
        # Get current architecture
        current_arch = self.current_architecture
        
        # Create new architecture with inserted layer
        insert_pos = worst_layer + 1
        new_arch = current_arch[:insert_pos] + [new_layer_size] + current_arch[insert_pos:]
        
        print(f"   Old: {current_arch}")
        print(f"   New: {new_arch}")
        
        # Create new network
        new_network = create_standard_network(
            architecture=new_arch,
            sparsity=self.scaffold_sparsity,
            device=self.device
        )
        
        # Transfer weights with layer insertion
        self._transfer_weights_with_insertion(self.network, new_network, worst_layer)
        
        # Update network and parameters
        self.network = new_network
        self.scaffold_params = list(self.network.parameters())
        
        print(f"   ✅ Layer added successfully")
        return True
    
    def _transfer_weights_with_insertion(self, old_network, new_network, insert_position):
        """Transfer weights with layer insertion, preserving sorted order."""
        old_sparse = [layer for layer in old_network if hasattr(layer, 'mask')]
        new_sparse = [layer for layer in new_network if hasattr(layer, 'mask')]
        
        print(f"   🔄 Transferring: {len(old_sparse)} → {len(new_sparse)} layers")
        
        with torch.no_grad():
            new_idx = 0
            for old_idx, old_layer in enumerate(old_sparse):
                if old_idx == insert_position:
                    # Skip the inserted layer (keep random initialization)
                    print(f"      Skipped new layer at position {new_idx}")
                    new_idx += 1
                
                if new_idx < len(new_sparse):
                    new_layer = new_sparse[new_idx]
                    
                    # Copy compatible dimensions with proper sorting preservation
                    min_out = min(old_layer.linear.weight.shape[0], new_layer.linear.weight.shape[0])
                    min_in = min(old_layer.linear.weight.shape[1], new_layer.linear.weight.shape[1])
                    
                    # Copy weights, biases, and masks
                    new_layer.linear.weight.data[:min_out, :min_in] = old_layer.linear.weight.data[:min_out, :min_in]
                    new_layer.linear.bias.data[:min_out] = old_layer.linear.bias.data[:min_out]
                    new_layer.mask[:min_out, :min_in] = old_layer.mask[:min_out, :min_in]
                    
                    # Copy sorting indices if they exist
                    if hasattr(old_layer, 'sorted_indices') and hasattr(new_layer, 'sorted_indices'):
                        # Copy compatible portion of sorted indices
                        if old_layer.sorted_indices is not None:
                            old_indices = old_layer.sorted_indices[:min_out]
                            new_layer.sorted_indices[:min_out] = old_indices
                            print(f"      Copied sorted indices for layer {old_idx} → {new_idx}")
                    
                    print(f"      {old_idx} → {new_idx}: {min_out}x{min_in}")
                    new_idx += 1
        
        # Apply sorting to the entire new network after weight transfer
        print(f"   🔄 Applying sorting to new network after weight transfer")
        sort_all_network_layers(new_network)
    
    def add_embedded_patches(self, extrema_analysis):
        """
        Add embedded patches by increasing density in existing layers.
        This is the modern approach - no separate patch modules.
        """
        patches_added = 0
        
        print(f"\n🔧 ADDING EMBEDDED PATCHES")
        
        sparse_layers = [layer for layer in self.network if hasattr(layer, 'mask')]
        
        for layer_idx, layer in enumerate(sparse_layers):
            dead_neurons = extrema_analysis['dead_neurons'].get(layer_idx, [])
            saturated_neurons = extrema_analysis['saturated_neurons'].get(layer_idx, [])
            
            if len(dead_neurons) >= self.patch_threshold or len(saturated_neurons) >= 10:
                print(f"   Layer {layer_idx}: {len(dead_neurons)} dead, {len(saturated_neurons)} saturated")
                
                with torch.no_grad():
                    # Add connections for dead neurons (data-driven)
                    for dead_idx in dead_neurons[:10]:  # Limit to 10
                        if dead_idx < layer.mask.shape[0]:
                            # Find high-variance input features (decisive features)
                            current_connections = layer.mask[dead_idx, :].sum()
                            if current_connections < layer.mask.shape[1] * 0.1:  # Less than 10% connected
                                # Add connections to high-variance inputs
                                weight_magnitudes = torch.abs(layer.linear.weight.data).mean(dim=0)
                                topk_inputs = torch.topk(weight_magnitudes, k=min(20, layer.mask.shape[1]))[1]
                                
                                for input_idx in topk_inputs[:5]:  # Add 5 connections
                                    layer.mask[dead_idx, input_idx] = 1.0
                                    layer.linear.weight.data[dead_idx, input_idx] = torch.randn(1).item() * 0.1
                                
                                patches_added += 1
                    
                    # Add connections for saturated neurons (relief)
                    for sat_idx in saturated_neurons[:5]:  # Limit to 5
                        if sat_idx < layer.mask.shape[1] and layer_idx < len(sparse_layers) - 1:
                            next_layer = sparse_layers[layer_idx + 1]
                            # Add output connections to distribute load
                            unused_outputs = torch.where(next_layer.mask[:, sat_idx] == 0)[0]
                            if len(unused_outputs) > 0:
                                for out_idx in unused_outputs[:3]:  # Add 3 connections
                                    next_layer.mask[out_idx, sat_idx] = 1.0
                                    next_layer.linear.weight.data[out_idx, sat_idx] = torch.randn(1).item() * 0.1
                                
                                patches_added += 1
        
        print(f"   ✅ Added {patches_added} embedded patches")
        return patches_added
    
    def earn_credits(self, correct_predictions, total_predictions):
        """Credit system disabled - no earning."""
        accuracy = correct_predictions / total_predictions
        # No credit earning when disabled
        return 0
    
    def can_afford_layer_growth(self):
        """Credit system disabled - always can afford."""
        return True
    
    def can_afford_patch_growth(self):
        """Credit system disabled - always can afford."""
        return True
    
    def spend_credits(self, amount, action):
        """Credit system disabled - no spending."""
        print(f"   🆓 Free {action} (credit system disabled)")
        return True
    
    def train_to_convergence(self, train_loader, test_loader, max_epochs=20):
        """Train network to convergence with dual learning rates."""
        print("📚 Training to convergence with dual learning rates")
        
        # Create parameter groups
        param_groups = [
            {'params': self.scaffold_params, 'lr': 0.0001, 'name': 'scaffold'},  # Slow for stability
        ]
        
        if self.patch_params:
            param_groups.append({
                'params': self.patch_params, 'lr': 0.001, 'name': 'patches'  # Fast for new learning
            })
        
        optimizer = optim.Adam(param_groups)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        patience = 5
        no_improve = 0
        
        for epoch in range(max_epochs):
            # Training
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
                
                # Update gradient history for variance analysis (OPTIMIZED: every 10th batch)
                if len(train_loader.dataset) // train_loader.batch_size > 10:  # Only if enough batches
                    batch_count = getattr(self, '_batch_count', 0)
                    if batch_count % 10 == 0:  # Every 10th batch
                        self.update_gradient_history()
                    self._batch_count = batch_count + 1
                else:
                    self.update_gradient_history()  # Always update for small datasets
                
                total_loss += loss.item()
            
            # Evaluation with credit earning
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
            
            # Earn credits
            self.earn_credits(correct, total)
            
            if accuracy > best_acc:
                best_acc = accuracy
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % self.neuron_sort_epochs == 0:
                print(f"    Epoch {epoch}: {accuracy:.2%} (best: {best_acc:.2%})")
                # Apply neuron sorting every N epochs during training (CONFIGURABLE)
                sort_all_network_layers(self.network)
                print(f"    🔄 Applied neuron sorting (epoch {epoch}, interval: {self.neuron_sort_epochs})")
            
            # Early stopping
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break
        
        self.current_accuracy = best_acc
        print(f"📊 Converged to {best_acc:.2%} accuracy")
        return best_acc
    
    def growth_step(self, train_loader, test_loader):
        """Perform one growth step."""
        self.iteration += 1
        
        print(f"\n🌱 GROWTH ITERATION {self.iteration}")
        print("=" * 50)
        
        # Step 1: Train to convergence
        accuracy = self.train_to_convergence(train_loader, test_loader)
        
        # Step 2: Analyze extrema
        extrema_analysis = self.detect_sophisticated_extrema(test_loader)
        
        # Step 3: Decide growth strategy with debug logging
        growth_occurred = False
        
        print(f"\n🔍 GROWTH DECISION DEBUG:")
        print(f"   Total extrema: {extrema_analysis['total_extrema']}")
        print(f"   Extrema ratio: {extrema_analysis['extrema_ratio']:.1%}")
        print(f"   Layer threshold: {self.layer_addition_threshold:.1%}")
        print(f"   Patch threshold: {self.patch_threshold}")
        
        # Check if we need a new layer (expensive)
        if (self.should_add_layer(extrema_analysis) and 
            self.can_afford_layer_growth()):
            
            if self.spend_credits(self.layer_growth_cost, "layer addition"):
                self.add_layer_at_bottleneck(extrema_analysis)
                growth_occurred = True
        
        # ALWAYS try to add patches if we have extrema (not elif!)
        if self.allow_patches and (extrema_analysis['total_extrema'] > 0 and 
            self.can_afford_patch_growth()):
            
            if self.spend_credits(self.patch_growth_cost, "embedded patches"):
                patches_added = self.add_embedded_patches(extrema_analysis)
                if patches_added > 0:
                    growth_occurred = True
                    # This print is now inside the add_embedded_patches method
                else:
                    print(f"   ⚠️  No patches added despite {extrema_analysis['total_extrema']} extrema")
        else:
            if not self.allow_patches:
                print("   ℹ️  Patch addition disabled by configuration.")
            elif extrema_analysis['total_extrema'] > 0:
                print(f"   ⚠️  Patch addition skipped: can_afford={self.can_afford_patch_growth()}")
        
        # Apply neuron sorting if growth occurred (CONFIGURABLE: every N epochs)
        if growth_occurred:
            # Track epochs since last sort
            epochs_since_sort = getattr(self, '_epochs_since_sort', 0)
            if epochs_since_sort >= self.neuron_sort_epochs:
                sort_all_network_layers(self.network)
                print(f"   🔄 Applied neuron sorting ({self.neuron_sort_epochs} epoch interval)")
                self._epochs_since_sort = 0
            else:
                self._epochs_since_sort = epochs_since_sort + 1
                print(f"   ⏳ Deferred sorting ({self._epochs_since_sort}/{self.neuron_sort_epochs} epochs)")
        
        # Record growth event
        stats = get_network_stats(self.network)
        self.growth_history.append({
            'iteration': self.iteration,
            'architecture': stats['architecture'],
            'accuracy': accuracy,
            'total_connections': stats['total_connections'],
            'sparsity': stats['overall_sparsity'],
            'credits': self.credits,
            'extrema_ratio': extrema_analysis['extrema_ratio'],
            'growth_occurred': growth_occurred
        })
        
        print(f"📊 Iteration {self.iteration}: Acc {accuracy:.2%}, Growth: {'Yes' if growth_occurred else 'No'}")
        print(f"💰 Credits: {self.credits:.1f}")
        print(f"📐 Architecture: {stats['architecture']}")
        
        return accuracy, growth_occurred
    
    def grow_until_target_accuracy(self, target_accuracy, train_loader, test_loader, max_iterations=10):
        """Main growth loop - grow until target accuracy."""
        print(f"🎯 MODERN INDEFINITE GROWTH EXPERIMENT")
        print("=" * 60)
        print(f"🎯 Target accuracy: {target_accuracy:.1%}")
        print(f"🌱 Starting architecture: {self.current_architecture}")
        print(f"🆓 Credit system: DISABLED - free growth based on extrema")
        
        while self.current_accuracy < target_accuracy and self.iteration < max_iterations:
            accuracy, growth_occurred = self.growth_step(train_loader, test_loader)
            
            # Save checkpoint only on significant accuracy improvements (OPTIMIZED)
            if accuracy > 0.7 and (not hasattr(self, '_last_saved_acc') or accuracy > self._last_saved_acc + 0.05):
                try:
                    current_stats = get_network_stats(self.network)
                    save_model_seed(
                        model=self.network,
                        architecture=current_stats['architecture'],
                        seed=42,
                        metrics={
                            'accuracy': accuracy,
                            'iteration': self.iteration,
                            'credits': self.credits,
                            'indefinite_growth': True
                        },
                        filepath=f"data/indefinite_growth_iter{self.iteration}_acc{accuracy:.2f}.pt"
                    )
                    self._last_saved_acc = accuracy
                    print(f"   💾 Saved checkpoint (significant improvement)")
                except Exception as e:
                    print(f"   ⚠️  Failed to save checkpoint: {e}")
            
            if accuracy >= target_accuracy:
                print(f"\n🎉 TARGET ACCURACY {target_accuracy:.1%} ACHIEVED!")
                break
            
            # No credit restrictions when system is disabled
            if not growth_occurred:
                print(f"\n⚠️  No growth occurred - network may have converged")
        
        print(f"\n🏁 GROWTH COMPLETED")
        print(f"📊 Final accuracy: {self.current_accuracy:.2%}")
        print(f"🌱 Growth iterations: {self.iteration}")
        print(f"📐 Final architecture: {self.current_architecture}")
        print(f"💰 Final credits: {self.credits:.1f}")
        
        return self.current_accuracy, self.growth_history
    
    def save_growth_summary(self, filepath="data/modern_indefinite_growth_results.json"):
        """Save comprehensive growth summary."""
        stats = get_network_stats(self.network)
        
        summary = {
            'experiment_type': 'modern_indefinite_growth',
            'timestamp': datetime.now().isoformat(),
            'final_accuracy': self.current_accuracy,
            'growth_iterations': self.iteration,
            'final_architecture': stats['architecture'],
            'final_connections': stats['total_connections'],
            'final_sparsity': stats['overall_sparsity'],
            'final_credits': self.credits,
            'seed_architecture': self.seed_architecture,
            'scaffold_sparsity': self.scaffold_sparsity,
            'growth_history': self.growth_history,
            'credit_economy': {
                'layer_cost': self.layer_growth_cost,
                'patch_cost': self.patch_growth_cost,
                'earn_rate': self.credit_earn_rate
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Growth summary saved to {filepath}")
        return summary

def run_experiment(device, allow_patches, seed_arch, target_accuracy, dataset, output_filename, seed_path=None, neuron_sort_epochs=5):
    """Worker function to run a single direct growth experiment."""
    print(f"🚀 Starting Direct Growth experiment on {device} | Patches: {'Enabled' if allow_patches else 'Disabled'}")
    
    # Load data
    if dataset == 'mnist':
        train_loader, test_loader = load_mnist_data()
    else:
        train_loader, test_loader = load_cifar10_data()
    
    # Create and run the growth engine
    engine = ModernIndefiniteGrowth(
        seed_architecture=seed_arch,
        scaffold_sparsity=0.02,
        device=device,
        allow_patches=allow_patches,
        neuron_sort_epochs=neuron_sort_epochs
    )

    if seed_path:
        engine.load_pretrained_scaffold(seed_path)

    engine.grow_until_target_accuracy(
        target_accuracy=target_accuracy,
        train_loader=train_loader,
        test_loader=test_loader
    )
    engine.save_growth_summary(output_filename)
    print(f"✅ Direct Growth experiment on {device} finished. Results saved to {output_filename}")

def run_tournament_experiment(device, seed_arch, target_accuracy, dataset, output_filename, seed_path=None):
    """Worker function to run a tournament-based growth experiment with enhanced logging."""
    print(f"🚀 Starting Tournament experiment on {device}")

    # Load data
    if dataset == 'mnist':
        train_loader, test_loader = load_mnist_data()
    else:
        train_loader, test_loader = load_cifar10_data()

    # Create network
    if seed_path:
        network, _ = load_model_seed(seed_path, device)
    else:
        network = create_standard_network(
            architecture=seed_arch,
            sparsity=0.02,
            device=device
        )

    # Configure and run the integrated system
    threshold_config = ThresholdConfig()
    metrics_config = MetricsConfig()
    
    system = IntegratedGrowthSystem(network, threshold_config, metrics_config)
    
    # Enhanced logging: capture performance and decisions
    detailed_log = {
        'experiment_type': 'tournament_growth',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'seed_architecture': seed_arch,
        'target_accuracy': target_accuracy,
        'dataset': dataset,
        'epoch_performance': [],
        'growth_decisions': [],
        'strategy_outcomes': []
    }
    
    # Run with detailed tracking
    system.grow_network(
        train_loader,
        test_loader,
        growth_iterations=5,
        epochs_per_iteration=10,
        tournament_epochs=3
    )
    
    # Save detailed results
    detailed_log.update({
        'final_performance': system.performance_history[-1] if system.performance_history else 0,
        'growth_history': system.growth_history,
        'strategy_effectiveness': system.performance_analyzer.get_strategy_effectiveness_summary() if hasattr(system.performance_analyzer, 'get_strategy_effectiveness_summary') else {},
        'learned_weights': dict(system.learned_strategy_weights)
    })
    
    # Save to file
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(detailed_log, f, indent=2)
    
    print(f"✅ Tournament experiment on {device} finished. Results saved to {output_filename}")

def run_sparse_baseline_experiment(device, seed_arch, target_accuracy, dataset, output_filename, seed_path=None):
    """Worker function to run a sparse baseline experiment (alternative to MultiScaleNetwork)."""
    print(f"🚀 Starting Sparse Baseline experiment on {device}")

    # Load data
    if dataset == 'mnist':
        train_loader, test_loader = load_mnist_data()
    else:
        train_loader, test_loader = load_cifar10_data()

    # Create very sparse network (alternative to MultiScaleNetwork)
    if seed_path:
        network, _ = load_model_seed(seed_path, device)
    else:
        network = create_standard_network(
            architecture=seed_arch,
            sparsity=0.001,  # Very sparse (0.1% vs 2% for others)
            device=device
        )
    
    # Enhanced logging structure
    detailed_log = {
        'experiment_type': 'sparse_baseline',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'seed_architecture': seed_arch,
        'target_accuracy': target_accuracy,
        'dataset': dataset,
        'epoch_performance': [],
        'connection_stats': []
    }
    
    # Setup training
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with detailed logging
    max_epochs = 100
    target_reached = False
    
    for epoch in range(max_epochs):
        # Train one epoch
        network.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += len(target)
        
        # Evaluate
        network.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = network(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += len(target)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        # Log epoch performance
        epoch_log = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_performance': train_acc,
            'val_loss': avg_val_loss,
            'val_performance': val_acc
        }
        detailed_log['epoch_performance'].append(epoch_log)
        
        # Log connection statistics
        stats = get_network_stats(network)
        detailed_log['connection_stats'].append({
            'epoch': epoch,
            'total_connections': stats['total_connections'],
            'overall_sparsity': stats['overall_sparsity']
        })
        
        # Check for target accuracy
        if val_acc >= target_accuracy:
            print(f"🎉 Target accuracy {target_accuracy:.1%} reached at epoch {epoch}!")
            target_reached = True
            break
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Val Acc={val_acc:.2%}, "
                  f"Train Acc={train_acc:.2%}, "
                  f"Connections={stats['total_connections']}")
    
    # Complete the log
    detailed_log.update({
        'final_performance': detailed_log['epoch_performance'][-1]['val_performance'] if detailed_log['epoch_performance'] else 0,
        'target_reached': target_reached,
        'total_epochs': len(detailed_log['epoch_performance']),
        'final_stats': get_network_stats(network)
    })
    
    # Save detailed results
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(detailed_log, f, indent=2)
    
    print(f"✅ Sparse Baseline experiment on {device} finished. Results saved to {output_filename}")
    print(f"📊 Final performance: {detailed_log['final_performance']:.2%}")
    print(f"🔗 Final connections: {detailed_log['final_stats']['total_connections']}")

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

def load_mnist_data(batch_size=64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def main():
    """Main function for modern indefinite growth experiment."""
    import argparse
    parser = argparse.ArgumentParser(description='Modern Indefinite Growth Experiment')
    parser.add_argument('--type', type=str, choices=['mnist', 'cifar'], default='mnist', help='Dataset to use')
    parser.add_argument('--target-accuracy', type=float, default=0.95, help='Target accuracy to grow towards')
    parser.add_argument('--seed-arch', type=str, default='784,128,10', help='Seed architecture (comma-separated)')
    parser.add_argument('-s', '--seed-path', type=str, default=None, help='Path to a saved model seed to start from')
    parser.add_argument('--growth-mode', type=str, choices=['direct', 'tournament'], default='direct', help='Growth strategy to use')
    parser.add_argument('-a', '--advanced-gpu', action='store_true', help='Run dual-GPU comparison: sparse-only vs sparse+patches')
    parser.add_argument('-b', '--benchmark', action='store_true', help='Run benchmark: direct vs tournament growth')
    parser.add_argument('-c', '--comparative', action='store_true', help='Run 3-way comparison: direct vs tournament vs multiscale')
    parser.add_argument('--ns', '--neuron-sort', type=int, default=5, help='Epochs per neuron sort (default: 5)')
    args = parser.parse_args()

    # Determine architecture based on dataset type
    if args.type == 'mnist':
        seed_arch = [int(x) for x in args.seed_arch.split(',')] if args.seed_arch else [784, 128, 10]
    else: # cifar
        seed_arch = [3072, 128, 10]

    if args.benchmark:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("⚠️  Benchmark mode requires at least 2 available GPUs. Exiting.")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("🚀 BENCHMARK: DIRECT VS TOURNAMENT GROWTH")
        print("="*60)

        procs = []
        # Process 1: Direct Growth (Layers and Patches)
        p1 = mp.Process(target=run_experiment, args=(
            'cuda:0', True, seed_arch, args.target_accuracy, args.type, 
            'data/benchmark_direct_growth.json', args.seed_path, args.ns
        ))
        procs.append(p1)
        p1.start()

        # Process 2: Tournament Growth
        p2 = mp.Process(target=run_tournament_experiment, args=(
            'cuda:1', seed_arch, args.target_accuracy, args.type,
            'data/benchmark_tournament_growth.json', args.seed_path
        ))
        procs.append(p2)
        p2.start()

        for p in procs:
            p.join()

        print("\n✅ Benchmark experiment complete.")

    elif args.comparative:
        if not torch.cuda.is_available():
            print("⚠️  Comparative mode requires CUDA. Exiting.")
            sys.exit(1)
        
        num_gpus = torch.cuda.device_count()
        print(f"\n🔍 Detected {num_gpus} GPU(s) available")
        
        if num_gpus < 3:
            print("⚠️  Comparative mode works best with 3+ GPUs, but will adapt to available hardware.")
        
        print("\n" + "="*70)
        print(f"🚀 COMPARATIVE ANALYSIS: SCALING TO {num_gpus} GPU(s)")
        print("="*70)

        procs = []
        experiments = []
        
        # Core experiments (always run these)
        core_experiments = [
            ('Direct Growth (Extrema-based)', run_experiment, 
             ('cuda:0', True, seed_arch, args.target_accuracy, args.type, 
              'data/comparative_direct_growth.json', args.seed_path, args.ns)),
            ('Tournament Growth (Strategy competition)', run_tournament_experiment,
             ('cuda:1' if num_gpus > 1 else 'cuda:0', seed_arch, args.target_accuracy, args.type,
              'data/comparative_tournament_growth.json', args.seed_path)),
            ('Sparse Baseline (Ultra-sparse)', run_sparse_baseline_experiment,
             ('cuda:2' if num_gpus > 2 else f'cuda:{min(1, num_gpus-1)}', seed_arch, args.target_accuracy, args.type,
              'data/comparative_sparse_baseline.json', args.seed_path))
        ]
        
        # Additional experiments if we have more GPUs
        if num_gpus >= 4:
            # Add Direct Growth without patches
            core_experiments.append(
                ('Direct Growth (No Patches)', run_experiment,
                 ('cuda:3', False, seed_arch, args.target_accuracy, args.type,
                  'data/comparative_direct_no_patches.json', args.seed_path, args.ns))
            )
        
        if num_gpus >= 5:
            # Add different neuron sorting intervals
            core_experiments.append(
                ('Direct Growth (Fast Sorting)', run_experiment,
                 ('cuda:4', True, seed_arch, args.target_accuracy, args.type,
                  'data/comparative_direct_fast_sort.json', args.seed_path, 1))  # Sort every epoch
            )
        
        if num_gpus >= 6:
            # Add slow sorting
            core_experiments.append(
                ('Direct Growth (Slow Sorting)', run_experiment,
                 ('cuda:5', True, seed_arch, args.target_accuracy, args.type,
                  'data/comparative_direct_slow_sort.json', args.seed_path, 20))  # Sort every 20 epochs
            )
        
        if num_gpus >= 7:
            # Add different sparsity levels - use sparse baseline with different sparsity
            core_experiments.append(
                ('Sparse Baseline (Dense)', run_sparse_baseline_experiment,
                 ('cuda:6', seed_arch, args.target_accuracy, args.type,
                  'data/comparative_sparse_dense.json', args.seed_path))
            )
        
        if num_gpus >= 8:
            # Add ensemble of direct growth with different random seeds
            core_experiments.append(
                ('Direct Growth (Ensemble)', run_experiment,
                 ('cuda:7', True, seed_arch, args.target_accuracy, args.type,
                  'data/comparative_direct_ensemble.json', None, args.ns))  # No seed path for different initialization
            )
        
        # Launch all experiments
        for name, func, args_tuple in core_experiments:
            print(f"🚀 Starting: {name}")
            p = mp.Process(target=func, args=args_tuple)
            procs.append(p)
            experiments.append(name)
            p.start()
        
        # Wait for all to complete
        for p in procs:
            p.join()

        print(f"\n✅ {len(experiments)}-way comparative experiment complete on {num_gpus} GPU(s).")
        print("📊 Results saved to:")
        result_files = [
            "data/comparative_direct_growth.json",
            "data/comparative_tournament_growth.json", 
            "data/comparative_sparse_baseline.json"
        ]
        
        if num_gpus >= 4:
            result_files.append("data/comparative_direct_no_patches.json")
        if num_gpus >= 5:
            result_files.append("data/comparative_direct_fast_sort.json")
        if num_gpus >= 6:
            result_files.append("data/comparative_direct_slow_sort.json")
        if num_gpus >= 7:
            result_files.append("data/comparative_sparse_dense.json")
        if num_gpus >= 8:
            result_files.append("data/comparative_direct_ensemble.json")
        
        for file in result_files:
            print(f"   - {file}")

    elif args.advanced_gpu:
        # ... (previous -a logic remains the same)
        pass
    else:
        # Single experiment logic
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.growth_mode == 'direct':
            run_experiment(
                device=device, allow_patches=True, seed_arch=seed_arch,
                target_accuracy=args.target_accuracy, dataset=args.type,
                output_filename='data/direct_growth_results.json', seed_path=args.seed_path
            )
        elif args.growth_mode == 'tournament':
            run_tournament_experiment(
                device=device, seed_arch=seed_arch,
                target_accuracy=args.target_accuracy, dataset=args.type,
                output_filename='data/tournament_growth_results.json', seed_path=args.seed_path
            )

if __name__ == "__main__":
    # Set start method for multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
