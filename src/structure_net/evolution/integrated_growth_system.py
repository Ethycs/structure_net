#!/usr/bin/env python3
"""
Integrated Growth System with Tournament-Based Strategy Selection

This module implements a sophisticated, tournament-based growth system.
It evaluates multiple growth strategies in parallel and selects the most
promising one to apply to the base network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import logging
from collections import defaultdict
from typing import Dict, List, Any

# Import from the existing structure_net library
from ..core.layers import StandardSparseLayer
from .complete_metrics_system import CompleteMetricsSystem, CompleteGraphAnalyzer
from ..core.network_factory import create_standard_network

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThresholdConfig:
    """Configuration for thresholds used in growth and analysis."""
    def __init__(self):
        self.activation_threshold = 0.01
        self.weight_threshold = 0.01
        self.gradient_threshold = 1e-4
        self.persistence_ratio = 0.8
        self.adaptive = True
        self.min_active_ratio = 0.05
        self.max_active_ratio = 0.5

class MetricsConfig:
    """Configuration for which metrics to compute."""
    def __init__(self):
        self.compute_mi = True
        self.compute_activity = True
        self.compute_sensli = True
        self.compute_graph = True
        self.compute_betweenness = False
        self.compute_spectral = False

# ============================================================================
# PART 6: PARALLEL GROWTH TOURNAMENT
# ============================================================================

class ParallelGrowthTournament:
    """
    Run a tournament of growth strategies in parallel to find the best one.
    """
    def __init__(self, base_network, threshold_config, metrics_config):
        self.base_network = base_network
        self.threshold_config = threshold_config
        self.metrics_config = metrics_config
        self.metrics_system = CompleteMetricsSystem(
            base_network, threshold_config, metrics_config
        )

    def run_tournament(self, train_loader, val_loader, growth_iterations=1, epochs_per_iteration=5):
        """
        Run a tournament of growth strategies.
        """
        logger.info("🏆 Starting Growth Tournament...")
        
        # Define strategies
        strategies = self._get_growth_strategies()
        
        results = []
        
        for strategy_name, strategy_fn in strategies.items():
            logger.info(f"  Testing strategy: {strategy_name}")
            
            # Create a copy of the network for this strategy
            candidate_network = copy.deepcopy(self.base_network)
            
            # Apply the growth strategy
            actions_taken = strategy_fn(candidate_network, train_loader)
            
            # Train the candidate network for a few epochs
            initial_acc = self._evaluate_network(candidate_network, val_loader)
            self._train_candidate(candidate_network, train_loader, epochs_per_iteration)
            final_acc = self._evaluate_network(candidate_network, val_loader)
            
            improvement = final_acc - initial_acc
            
            results.append({
                'strategy': strategy_name,
                'network': candidate_network,
                'improvement': improvement,
                'actions': actions_taken,
                'final_accuracy': final_acc
            })
            
            logger.info(f"    Improvement: {improvement:+.2%}")

        # Select the winner
        winner = max(results, key=lambda x: x['improvement'])
        logger.info(f"🎉 Winning Strategy: {winner['strategy']} ({winner['improvement']:+.2%})")
        
        return {'winner': winner, 'all_results': results}

    def _get_growth_strategies(self):
        """Define the set of growth strategies to compete."""
        return {
            "Add Layer at Bottleneck": self._strategy_add_layer,
            "Add Patches to Extrema": self._strategy_add_patches,
            "Prune Weak Connections": self._strategy_prune_weak,
            "Hybrid: Add Layer & Patches": self._strategy_hybrid_growth,
        }

    def _strategy_add_layer(self, network, data_loader):
        """Strategy: Add a new layer at the point of lowest health."""
        metrics = self.metrics_system.compute_all_metrics(data_loader, num_batches=3)
        health_scores = {k: v['layer_health_score'] for k, v in metrics['activity_metrics'].items()}
        
        if not health_scores:
            return [{'action': 'add_layer', 'reason': 'No health scores available', 'success': False}]
            
        worst_layer_key = min(health_scores, key=health_scores.get)
        worst_layer_idx = int(worst_layer_key.split('_')[-1])
        
        current_arch = [l.in_features for l in network.layers] + [network.layers[-1].out_features]
        new_layer_size = max(64, current_arch[worst_layer_idx] // 2)
        
        new_arch = current_arch[:worst_layer_idx+1] + [new_layer_size] + current_arch[worst_layer_idx+1:]
        
        # This is a placeholder for the actual network modification logic
        # In a real implementation, you would replace the network object.
        logger.info(f"    Action: Add layer of size {new_layer_size} after layer {worst_layer_idx}")
        
        return [{'action': 'add_layer', 'position': worst_layer_idx, 'size': new_layer_size}]

    def _strategy_add_patches(self, network, data_loader):
        """Strategy: Add connections to fix dead or saturated neurons."""
        # This is a placeholder for the actual patching logic
        logger.info("    Action: Add patches to extrema neurons")
        return [{'action': 'add_patches', 'reason': 'Fixing extrema'}]

    def _strategy_prune_weak(self, network, data_loader):
        """Strategy: Prune the weakest connections to enforce sparsity."""
        # This is a placeholder for the actual pruning logic
        logger.info("    Action: Prune weak connections")
        return [{'action': 'prune_weak', 'reason': 'Enforcing sparsity'}]

    def _strategy_hybrid_growth(self, network, data_loader):
        """Strategy: A mix of adding a layer and patching."""
        actions = self._strategy_add_layer(network, data_loader)
        actions.extend(self._strategy_add_patches(network, data_loader))
        return actions

    def _train_candidate(self, network, train_loader, epochs):
        """Train a candidate network for a few epochs."""
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        network.train()
        
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = network(data.view(data.size(0), -1))
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def _evaluate_network(self, network, val_loader):
        """Evaluate a network's accuracy."""
        network.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = network(data.view(data.size(0), -1))
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += len(target)
        return correct / total

# ============================================================================
# PART 7: ADAPTIVE THRESHOLD MANAGEMENT
# ============================================================================

class AdaptiveThresholdManager:
   """Dynamically adjust thresholds based on network behavior."""
   
   def __init__(self, initial_config: ThresholdConfig):
       self.config = initial_config
       self.history = defaultdict(list)
       self.adjustment_patience = 3
       
   def update_thresholds(self, network_stats: Dict):
       """Update thresholds based on network statistics."""
       
       # Track history
       for key, value in network_stats.items():
           self.history[key].append(value)
       
       # Check if adjustment needed
       if len(self.history['active_ratio']) >= self.adjustment_patience:
           recent_active = np.mean(self.history['active_ratio'][-self.adjustment_patience:])
           
           if recent_active < self.config.min_active_ratio:
               # Too few active neurons - lower threshold
               old_threshold = self.config.activation_threshold
               self.config.activation_threshold *= 0.9
               logger.info(f"Lowering activation threshold: {old_threshold:.4f} → {self.config.activation_threshold:.4f}")
               
           elif recent_active > self.config.max_active_ratio:
               # Too many active neurons - raise threshold
               old_threshold = self.config.activation_threshold
               self.config.activation_threshold *= 1.1
               logger.info(f"Raising activation threshold: {old_threshold:.4f} → {self.config.activation_threshold:.4f}")
       
       # Adjust gradient threshold based on gradient magnitudes
       if 'avg_gradient' in network_stats:
           if network_stats['avg_gradient'] < self.config.gradient_threshold * 0.1:
               self.config.gradient_threshold *= 0.5
           elif network_stats['avg_gradient'] > self.config.gradient_threshold * 10:
               self.config.gradient_threshold *= 2
   
   def compute_network_stats(self, network, dataloader):
       """Compute statistics for threshold adjustment."""
       stats = {
           'active_ratio': [],
           'avg_gradient': [],
           'max_activation': [],
           'dead_layers': 0
       }
       
       network.eval()
       
       # Forward pass to collect activation stats
       with torch.no_grad():
           for data, _ in dataloader:
               x = data.view(data.size(0), -1)
               
               for i, layer in enumerate(network.layers):
                   x = layer(x)
                   
                   # Active ratio
                   active = (x.abs() > self.config.activation_threshold).float().mean()
                   stats['active_ratio'].append(active.item())
                   
                   # Max activation
                   stats['max_activation'].append(x.abs().max().item())
                   
                   # Check for dead layer
                   if active < 0.001:
                       stats['dead_layers'] += 1
                   
                   x = F.relu(x)
               
               break  # Just one batch for stats
       
       # Compute gradients
       network.train()
       data, target = next(iter(dataloader))
       output = network(data.view(data.size(0), -1))
       loss = F.cross_entropy(output, target)
       loss.backward()
       
       grad_norms = []
       for p in network.parameters():
           if p.grad is not None:
               grad_norms.append(p.grad.abs().mean().item())
       
       stats['avg_gradient'] = np.mean(grad_norms) if grad_norms else 0
       stats['active_ratio'] = np.mean(stats['active_ratio'])
       
       return stats

# ============================================================================
# PART 8: MAIN INTEGRATED SYSTEM
# ============================================================================

class IntegratedGrowthSystem:
   """Complete system integrating all components."""
   
   def __init__(self, network, config: ThresholdConfig = None,
                metrics_config: MetricsConfig = None):
       self.network = network
       self.threshold_config = config or ThresholdConfig()
       self.metrics_config = metrics_config or MetricsConfig()
       
       # Initialize components
       self.threshold_manager = AdaptiveThresholdManager(self.threshold_config)
       self.tournament = ParallelGrowthTournament(
           network, self.threshold_config, self.metrics_config
       )
       
       # Growth history
       self.growth_history = []
       self.performance_history = []
       
   def grow_network(self, train_loader, val_loader,
                   growth_iterations: int = 3,
                   epochs_per_iteration: int = 20,
                   tournament_epochs: int = 5):
       """Main growth loop with all systems integrated."""
       
       logger.info("\n" + "="*80)
       logger.info("🌱 INTEGRATED GROWTH SYSTEM")
       logger.info("="*80)
       
       # Initial evaluation
       initial_acc = self.tournament._evaluate_network(self.network, val_loader)
       logger.info(f"\nInitial accuracy: {initial_acc:.2%}")
       self.performance_history.append(initial_acc)
       
       for iteration in range(growth_iterations):
           logger.info(f"\n{'='*80}")
           logger.info(f"🌿 GROWTH ITERATION {iteration + 1}/{growth_iterations}")
           logger.info(f"{'='*80}")
           
           # Update thresholds if adaptive
           if self.threshold_config.adaptive:
               logger.info("\n📊 Updating thresholds...")
               stats = self.threshold_manager.compute_network_stats(
                   self.network, train_loader
               )
               self.threshold_manager.update_thresholds(stats)
               logger.info(f"  Active ratio: {stats['active_ratio']:.3%}")
               logger.info(f"  Dead layers: {stats['dead_layers']}")
           
           # Run tournament
           logger.info("\n🏆 Running growth tournament...")
           tournament_results = self.tournament.run_tournament(
               train_loader, val_loader,
               growth_iterations=1,
               epochs_per_iteration=tournament_epochs
           )
           
           # Apply winning strategy
           winner = tournament_results['winner']
           self.network = winner['network']
           self.tournament.base_network = self.network
           
           # Full training
           logger.info(f"\n📚 Training for {epochs_per_iteration} epochs...")
           self._train_network(train_loader, val_loader, epochs_per_iteration)
           
           # Evaluate
           current_acc = self.tournament._evaluate_network(self.network, val_loader)
           self.performance_history.append(current_acc)
           
           # Record growth
           self.growth_history.append({
               'iteration': iteration,
               'winner_strategy': winner['strategy'],
               'actions': winner['actions'],
               'improvement': winner['improvement'],
               'accuracy': current_acc,
               'threshold': self.threshold_config.activation_threshold
           })
           
           logger.info(f"\n✅ Iteration complete. Accuracy: {current_acc:.2%}")
       
       self._print_final_summary()
       
       return self.network
   
   def _train_network(self, train_loader, val_loader, epochs):
       """Full training with monitoring."""
       optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
       
       best_val_acc = 0
       
       for epoch in range(epochs):
           # Train
           self.network.train()
           train_loss = 0
           train_correct = 0
           train_total = 0
           
           for data, target in train_loader:
               optimizer.zero_grad()
               output = self.network(data.view(data.size(0), -1))
               loss = F.cross_entropy(output, target)
               loss.backward()
               optimizer.step()
               
               train_loss += loss.item()
               pred = output.argmax(dim=1)
               train_correct += (pred == target).sum().item()
               train_total += len(target)
           
           # Validate
           val_acc = self.tournament._evaluate_network(self.network, val_loader)
           
           if val_acc > best_val_acc:
               best_val_acc = val_acc
           
           if epoch % 5 == 0:
               train_acc = train_correct / train_total
               logger.info(f"  Epoch {epoch}: Train Loss={train_loss/len(train_loader):.3f}, "
                         f"Train Acc={train_acc:.2%}, Val Acc={val_acc:.2%}")
           
           scheduler.step()
       
       logger.info(f"  Best validation accuracy: {best_val_acc:.2%}")
   
   def _print_final_summary(self):
       """Print final summary of growth process."""
       logger.info("\n" + "="*80)
       logger.info("📊 GROWTH SUMMARY")
       logger.info("="*80)
       
       # Performance trajectory
       logger.info("\nPerformance trajectory:")
       for i, acc in enumerate(self.performance_history):
           if i == 0:
               logger.info(f"  Initial: {acc:.2%}")
           else:
               improvement = acc - self.performance_history[i-1]
               logger.info(f"  After iteration {i}: {acc:.2%} ({improvement:+.2%})")
       
       # Total improvement
       total_improvement = self.performance_history[-1] - self.performance_history[0]
       logger.info(f"\nTotal improvement: {total_improvement:.2%}")
       
       # Growth actions summary
       logger.info("\nGrowth actions taken:")
       for record in self.growth_history:
           logger.info(f"  Iteration {record['iteration'] + 1}: {record['winner_strategy']}")
           for action in record['actions'][:3]:  # First 3 actions
               logger.info(f"    - {action['action']}: {action.get('reason', 'N/A')}")
       
       # Final network stats
       logger.info("\nFinal network statistics:")
       # This is a placeholder for a proper stats call
       # stats = self.threshold_manager.compute_network_stats(self.network, 
       #                                                    next(iter(train_loader))[0].unsqueeze(0))
       # logger.info(f"  Active neuron ratio: {stats['active_ratio']:.2%}")
       # logger.info(f"  Dead layers: {stats['dead_layers']}")
       # logger.info(f"  Average gradient: {stats['avg_gradient']:.6f}")
