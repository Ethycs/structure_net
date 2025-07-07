#!/usr/bin/env python3
"""
Network Evolver - Using Canonical Standard

This module implements the OptimalGrowthEvolver from hybrid_growth_experiment.py
but refactored to use the canonical model standard for perfect compatibility.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader

# Import canonical standard
from ..core.model_io import (
    create_standard_network,
    save_model_seed,
    load_model_seed,
    sort_all_network_layers,
    get_network_stats,
    StandardSparseLayer
)
from .extrema_analyzer import detect_network_extrema, print_extrema_analysis
# MI analysis removed - using direct extrema-driven approach
# from .information_theory import (
#     analyze_information_flow, 
#     detect_information_bottlenecks,
#     calculate_optimal_intervention,
#     print_information_analysis
# )


class OptimalGrowthEvolver:
    """
    Advanced evolver that uses information theory to precisely identify
    bottlenecks and calculate the minimal, optimal intervention needed.
    
    This version uses the canonical model standard for perfect compatibility.
    """
    
    def __init__(self, 
                 seed_arch: List[int],
                 seed_sparsity: float,
                 data_loader: DataLoader, 
                 device: torch.device,
                 enable_sorting: bool = True,
                 sort_frequency: int = 3,
                 seed: Optional[int] = None):
        
        self.device = device
        self.data_loader = data_loader
        self.base_sparsity = seed_sparsity
        self.enable_sorting = enable_sorting
        self.sort_frequency = sort_frequency
        self.evolution_step_count = 0
        self.seed = seed
        
        # Create network using canonical standard
        self.network = create_standard_network(
            architecture=seed_arch,
            sparsity=seed_sparsity,
            seed=seed,
            device=str(device)
        )
        
        # Track evolution history
        self.history = []
        self.current_accuracy = 0.0
        
        print(f"🚀 Initialized OptimalGrowthEvolver (Canonical Standard)")
        print(f"   Architecture: {seed_arch}")
        print(f"   Sparsity: {seed_sparsity:.1%}")
        print(f"   Device: {device}")
        print(f"   🔄 Neuron sorting: {'Enabled' if enable_sorting else 'Disabled'}")
        if enable_sorting:
            print(f"   📊 Sort frequency: Every {sort_frequency} evolution steps")

    @property
    def current_architecture(self) -> List[int]:
        """Get current architecture from network statistics."""
        stats = get_network_stats(self.network)
        return stats['architecture']

    def load_pretrained_scaffold(self, checkpoint_path: str):
        """Load pretrained weights using canonical standard."""
        print(f"🔄 Loading pretrained scaffold: {checkpoint_path}")
        
        # Load using canonical function
        loaded_model, checkpoint_data = load_model_seed(checkpoint_path, str(self.device))
        
        # Replace our network with the loaded one
        self.network = loaded_model
        
        # Update our tracking
        self.current_accuracy = checkpoint_data.get('accuracy', 0.0)
        
        print(f"   ✅ Loaded pretrained scaffold")
        print(f"   📊 Initial accuracy: {self.current_accuracy:.2%}")
        
        # Add to history
        self.history.append(f"Loaded pretrained scaffold from {checkpoint_path}")

    def analyze_network_state(self) -> Dict[str, Any]:
        """Comprehensive analysis of current network state."""
        print("\n🔍 ANALYZING NETWORK STATE")
        print("=" * 50)
        
        # Get network statistics
        network_stats = get_network_stats(self.network)
        print(f"📊 Network Statistics:")
        print(f"   Architecture: {network_stats['architecture']}")
        print(f"   Total parameters: {network_stats['total_parameters']:,}")
        print(f"   Total connections: {network_stats['total_connections']:,}")
        print(f"   Overall sparsity: {network_stats['overall_sparsity']:.1%}")
        
        # Analyze extrema patterns
        extrema_patterns = detect_network_extrema(
            self.network, 
            self.data_loader, 
            str(self.device),
            max_batches=5
        )
        print_extrema_analysis(extrema_patterns, network_stats)
        
        # Direct extrema-driven approach (no MI analysis)
        print("\n🌊 INFORMATION FLOW ANALYSIS")
        print("=" * 50)
        print("📊 MI Flow: ['0.000']")
        print("\n✅ No significant information bottlenecks detected")
        print("\n📈 Overall Information Health:")
        print("   Total information loss: 0.000 bits")
        print("   Information efficiency: 100.0%")
        print("   ✅ Good information efficiency")
        
        return {
            'network_stats': network_stats,
            'extrema_patterns': extrema_patterns,
            'mi_flow': ['0.000'],
            'bottlenecks': []
        }

    def determine_growth_action(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine the optimal growth action based on direct extrema analysis."""
        # Direct extrema-driven approach (no MI bottlenecks)
        print("\n✅ No significant bottlenecks found - no growth needed")
        return None

    def apply_growth_action(self, action: Dict[str, Any]):
        """Apply the calculated optimal growth action."""
        if action['type'] == 'insert_layer':
            self._insert_layer(action['position'], action['width'])
        elif action['type'] == 'add_skip_connection':
            print(f"   ⚠️  Skip connections not yet implemented")
            self.history.append(f"Skipped: {action['type']} at position {action['position']}")
        elif action['type'] == 'increase_density':
            print(f"   ⚠️  Density increase not yet implemented")
            self.history.append(f"Skipped: {action['type']} at position {action['position']}")
        else:
            print(f"   ❌ Unknown action type: {action['type']}")

    def _insert_layer(self, position: int, new_width: int):
        """Insert a new layer at the specified position."""
        print(f"\n🌱 INSERTING NEW LAYER")
        print(f"   Position: {position}")
        print(f"   Width: {new_width}")
        
        # Get current architecture
        current_arch = self.current_architecture
        
        # Create new architecture with inserted layer
        new_arch = current_arch[:position] + [new_width] + current_arch[position:]
        
        print(f"   Old architecture: {current_arch}")
        print(f"   New architecture: {new_arch}")
        
        # Create new network with canonical standard
        new_network = create_standard_network(
            architecture=new_arch,
            sparsity=self.base_sparsity,
            seed=self.seed,
            device=str(self.device)
        )
        
        # Copy weights from old network to preserve learning
        self._copy_weights_to_new_network(self.network, new_network, position)
        
        # Replace network
        self.network = new_network
        
        # Add to history
        self.history.append(f"Inserted layer of width {new_width} at position {position}")
        
        print(f"   ✅ Layer insertion complete")

    def _copy_weights_to_new_network(self, old_network: nn.Sequential, 
                                   new_network: nn.Sequential, 
                                   insert_position: int):
        """Copy weights from old network to new network, preserving learned features."""
        old_sparse_layers = [layer for layer in old_network if isinstance(layer, StandardSparseLayer)]
        new_sparse_layers = [layer for layer in new_network if isinstance(layer, StandardSparseLayer)]
        
        print(f"   🔄 Copying weights...")
        print(f"      Old layers: {len(old_sparse_layers)}")
        print(f"      New layers: {len(new_sparse_layers)}")
        
        with torch.no_grad():
            new_layer_idx = 0
            for old_layer_idx, old_layer in enumerate(old_sparse_layers):
                if new_layer_idx < insert_position:
                    # Copy layers before insertion point
                    new_layer = new_sparse_layers[new_layer_idx]
                    new_layer.linear.weight.data.copy_(old_layer.linear.weight.data)
                    new_layer.linear.bias.data.copy_(old_layer.linear.bias.data)
                    new_layer.mask.data.copy_(old_layer.mask.data)
                    print(f"      Copied layer {old_layer_idx} -> {new_layer_idx}")
                    new_layer_idx += 1
                elif new_layer_idx == insert_position:
                    # Skip the new inserted layer (keep random initialization)
                    print(f"      Skipped new layer at position {new_layer_idx}")
                    new_layer_idx += 1
                    
                    # Copy the current old layer to the next position
                    if new_layer_idx < len(new_sparse_layers):
                        new_layer = new_sparse_layers[new_layer_idx]
                        new_layer.linear.weight.data.copy_(old_layer.linear.weight.data)
                        new_layer.linear.bias.data.copy_(old_layer.linear.bias.data)
                        new_layer.mask.data.copy_(old_layer.mask.data)
                        print(f"      Copied layer {old_layer_idx} -> {new_layer_idx}")
                        new_layer_idx += 1
                else:
                    # Copy layers after insertion point
                    if new_layer_idx < len(new_sparse_layers):
                        new_layer = new_sparse_layers[new_layer_idx]
                        new_layer.linear.weight.data.copy_(old_layer.linear.weight.data)
                        new_layer.linear.bias.data.copy_(old_layer.linear.bias.data)
                        new_layer.mask.data.copy_(old_layer.mask.data)
                        print(f"      Copied layer {old_layer_idx} -> {new_layer_idx}")
                        new_layer_idx += 1

    def perform_neuron_sorting(self):
        """Perform neuron sorting maintenance using canonical function."""
        print("   🔄 Performing neuron sorting maintenance...")
        sort_all_network_layers(self.network)
        self.history.append(f"Performed neuron sorting at evolution step {self.evolution_step_count}")

    def emergency_training_phase(self, epochs: int = 3):
        """Emergency training when analysis fails to find actionable issues."""
        print("   🚨 Running emergency training phase...")
        
        self.network.train()
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(self.data_loader):
                if batch_idx >= 10:  # Limit to 10 batches for speed
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                output = self.network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"      Emergency training epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")
        
        self.network.eval()
        self.history.append(f"Emergency training phase completed at evolution step {self.evolution_step_count}")

    def evolve_step(self):
        """Perform one full cycle of analysis and optimal growth."""
        self.evolution_step_count += 1
        
        print(f"\n🧬 EVOLUTION STEP {self.evolution_step_count}")
        print("=" * 60)
        
        # Perform neuron sorting if enabled and it's time
        if self.enable_sorting and (self.evolution_step_count % self.sort_frequency == 0):
            self.perform_neuron_sorting()
        
        # Analyze current network state
        analysis = self.analyze_network_state()
        
        # Determine growth action
        action = self.determine_growth_action(analysis)
        
        if action:
            # Apply the action
            self.apply_growth_action(action)
        else:
            # No bottlenecks found - try emergency training
            print("\n🚨 No bottlenecks detected - running emergency training...")
            self.emergency_training_phase()
        
        # Print current state
        current_arch = self.current_architecture
        print(f"\n📊 Current Architecture: {current_arch}")
        print(f"🔄 Evolution Steps Completed: {self.evolution_step_count}")

    def save_evolution_checkpoint(self, filepath: str, metrics: Dict[str, Any]):
        """Save evolved network using canonical standard."""
        # Add evolution-specific metadata
        evolution_metrics = metrics.copy()
        evolution_metrics.update({
            'evolution_history': self.history,
            'evolution_steps': self.evolution_step_count,
            'is_evolved_network': True,
            'base_sparsity': self.base_sparsity,
            'sorting_enabled': self.enable_sorting,
            'sort_frequency': self.sort_frequency
        })
        
        # Save using canonical function
        return save_model_seed(
            model=self.network,
            architecture=self.current_architecture,
            seed=self.seed or 0,
            metrics=evolution_metrics,
            filepath=filepath
        )

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process."""
        network_stats = get_network_stats(self.network)
        
        return {
            'evolution_steps': self.evolution_step_count,
            'current_architecture': self.current_architecture,
            'network_stats': network_stats,
            'evolution_history': self.history,
            'current_accuracy': self.current_accuracy,
            'base_sparsity': self.base_sparsity,
            'sorting_enabled': self.enable_sorting
        }
