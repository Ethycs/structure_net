#!/usr/bin/env python3
"""
Composable Network Evolution System

This module provides a complete composable evolution system that coordinates
analyzers, strategies, and trainers to evolve networks over time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import time
import copy

from ..interfaces import (
    NetworkEvolutionSystem, NetworkComponent, NetworkAnalyzer, 
    GrowthStrategy, LearningRateStrategy, NetworkTrainer,
    NetworkContext, AnalysisResult, GrowthAction, ActionType,
    FullyConfigurableComponent
)
from .analyzers import StandardExtremaAnalyzer, NetworkStatsAnalyzer, SimpleInformationFlowAnalyzer
from .strategies import ExtremaGrowthStrategy, InformationFlowGrowthStrategy, ResidualBlockGrowthStrategy, HybridGrowthStrategy


class StandardNetworkTrainer(NetworkTrainer, FullyConfigurableComponent):
    """
    Standard implementation of network trainer.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 optimizer_type: str = 'adam',
                 criterion_type: str = 'cross_entropy'):
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.criterion_type = criterion_type
        self._metrics = {}
    
    def can_apply(self, context: NetworkContext) -> bool:
        """Check if trainer can be applied."""
        return self.validate_context(context)
    
    def train_epoch(self, context: NetworkContext) -> Dict[str, float]:
        """Train the network for one epoch."""
        network = context.network
        data_loader = context.data_loader
        device = context.device
        
        # Get or create optimizer
        optimizer = context.metadata.get('optimizer')
        if optimizer is None:
            if self.optimizer_type == 'adam':
                optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == 'sgd':
                optimizer = optim.SGD(network.parameters(), lr=self.learning_rate)
            else:
                optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
            context.metadata['optimizer'] = optimizer
        
        # Get criterion
        if self.criterion_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        network.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten for dense networks
            
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            # Limit batches for efficiency
            if batch_idx >= 10:
                break
        
        metrics = {
            'train_loss': total_loss / (batch_idx + 1),
            'train_accuracy': correct / total if total > 0 else 0.0,
            'batches_processed': batch_idx + 1
        }
        
        # Update monitoring
        self._metrics['total_epochs'] = self._metrics.get('total_epochs', 0) + 1
        self._metrics['last_train_loss'] = metrics['train_loss']
        self._metrics['last_train_accuracy'] = metrics['train_accuracy']
        
        return metrics
    
    def evaluate(self, context: NetworkContext) -> Dict[str, float]:
        """Evaluate the network."""
        network = context.network
        data_loader = context.data_loader
        device = context.device
        
        network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                
                output = network(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                # Limit batches for efficiency
                if batch_idx >= 5:
                    break
        
        return {
            'val_loss': total_loss / (batch_idx + 1),
            'val_accuracy': correct / total if total > 0 else 0.0,
            'batches_evaluated': batch_idx + 1
        }
    
    # Configurable interface
    def configure(self, config: Dict[str, Any]):
        """Configure the trainer."""
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        self.optimizer_type = config.get('optimizer_type', self.optimizer_type)
        self.criterion_type = config.get('criterion_type', self.criterion_type)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'learning_rate': self.learning_rate,
            'optimizer_type': self.optimizer_type,
            'criterion_type': self.criterion_type
        }
    
    # Serializable interface
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'StandardNetworkTrainer',
            'config': self.get_configuration(),
            'metrics': self._metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardNetworkTrainer':
        """Deserialize from dictionary."""
        trainer = cls(**data.get('config', {}))
        trainer._metrics = data.get('metrics', {})
        return trainer
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset monitoring metrics."""
        self._metrics.clear()


class ComposableEvolutionSystem(NetworkEvolutionSystem):
    """
    Complete composable evolution system that coordinates multiple components.
    
    This is an orchestrator class that manages and coordinates other components.
    It does not implement NetworkComponent interfaces since it's not a component
    itself, but rather a system that manages components.
    """
    
    def __init__(self):
        self.analyzers: List[NetworkAnalyzer] = []
        self.strategies: List[GrowthStrategy] = []
        self.lr_strategies: List[LearningRateStrategy] = []
        self.trainers: List[NetworkTrainer] = []
        self._metrics = {}
        self._evolution_history = []
    
    def add_component(self, component: NetworkComponent):
        """Add a component to the evolution system."""
        if isinstance(component, NetworkAnalyzer):
            self.analyzers.append(component)
        elif isinstance(component, GrowthStrategy):
            self.strategies.append(component)
        elif isinstance(component, LearningRateStrategy):
            self.lr_strategies.append(component)
        elif isinstance(component, NetworkTrainer):
            self.trainers.append(component)
        else:
            raise ValueError(f"Unknown component type: {type(component)}")
    
    def remove_component(self, component: NetworkComponent):
        """Remove a component from the evolution system."""
        if component in self.analyzers:
            self.analyzers.remove(component)
        elif component in self.strategies:
            self.strategies.remove(component)
        elif component in self.lr_strategies:
            self.lr_strategies.remove(component)
        elif component in self.trainers:
            self.trainers.remove(component)
    
    def get_components(self) -> List[NetworkComponent]:
        """Get all components in the evolution system."""
        return self.analyzers + self.strategies + self.lr_strategies + self.trainers
    
    def evolve_network(self, 
                      context: NetworkContext, 
                      num_iterations: int = 1) -> NetworkContext:
        """
        Evolve the network for the specified number of iterations.
        """
        print(f"\n🧬 COMPOSABLE EVOLUTION SYSTEM")
        print("=" * 60)
        print(f"Components: {len(self.analyzers)} analyzers, {len(self.strategies)} strategies, "
              f"{len(self.lr_strategies)} LR strategies, {len(self.trainers)} trainers")
        
        for iteration in range(num_iterations):
            print(f"\n🔄 Evolution Iteration {iteration + 1}/{num_iterations}")
            print("-" * 40)
            
            context.iteration = iteration
            iteration_start_time = time.time()
            
            # Step 1: Run all analyzers
            print("📊 Running analysis...")
            analysis_results = self._run_analyzers(context)
            
            # Step 2: Apply learning rate strategies
            print("📈 Updating learning rates...")
            self._apply_learning_rate_strategies(context)
            
            # Step 3: Train for a few epochs
            print("🎓 Training network...")
            training_metrics = self._train_network(context, epochs=3)
            
            # Step 4: Try growth strategies
            print("🌱 Attempting growth...")
            growth_applied = self._apply_growth_strategies(context, analysis_results)
            
            # Step 5: Record iteration
            iteration_metrics = {
                'iteration': iteration,
                'analysis_results': {analyzer.get_name(): result.metrics 
                                   for analyzer, result in analysis_results.items()},
                'training_metrics': training_metrics,
                'growth_applied': growth_applied,
                'iteration_time': time.time() - iteration_start_time
            }
            
            self._evolution_history.append(iteration_metrics)
            
            print(f"✅ Iteration {iteration + 1} complete "
                  f"(growth: {'Yes' if growth_applied else 'No'}, "
                  f"time: {iteration_metrics['iteration_time']:.1f}s)")
        
        # Update final metrics
        self._metrics.update({
            'total_iterations': len(self._evolution_history),
            'total_growth_events': sum(1 for h in self._evolution_history if h['growth_applied']),
            'average_iteration_time': sum(h['iteration_time'] for h in self._evolution_history) / len(self._evolution_history)
        })
        
        print(f"\n🎯 Evolution complete! {num_iterations} iterations, "
              f"{self._metrics['total_growth_events']} growth events")
        
        return context
    
    def _run_analyzers(self, context: NetworkContext) -> Dict[NetworkAnalyzer, AnalysisResult]:
        """Run all analyzers and collect results."""
        results = {}
        
        for analyzer in self.analyzers:
            if analyzer.can_apply(context):
                try:
                    result = analyzer.analyze(context)
                    results[analyzer] = result
                    
                    # Store result in context for strategies to use
                    analyzer.apply(context)
                    
                    print(f"  ✅ {analyzer.get_name()}: "
                          f"confidence={result.confidence:.2f}, "
                          f"recommendations={len(result.recommendations)}")
                    
                except Exception as e:
                    print(f"  ❌ {analyzer.get_name()} failed: {e}")
            else:
                print(f"  ⏭️  {analyzer.get_name()}: not applicable")
        
        return results
    
    def _apply_learning_rate_strategies(self, context: NetworkContext):
        """Apply learning rate strategies."""
        for lr_strategy in self.lr_strategies:
            if lr_strategy.can_apply(context):
                try:
                    lr_strategy.apply(context)
                    print(f"  ✅ {lr_strategy.get_name()}: applied")
                except Exception as e:
                    print(f"  ❌ {lr_strategy.get_name()} failed: {e}")
    
    def _train_network(self, context: NetworkContext, epochs: int = 3) -> Dict[str, Any]:
        """Train the network using available trainers."""
        if not self.trainers:
            # Use default trainer if none provided
            default_trainer = StandardNetworkTrainer()
            self.trainers.append(default_trainer)
        
        trainer = self.trainers[0]  # Use first available trainer
        
        training_metrics = []
        
        for epoch in range(epochs):
            if trainer.can_apply(context):
                try:
                    epoch_metrics = trainer.train_epoch(context)
                    training_metrics.append(epoch_metrics)
                    
                    if epoch == 0:  # Print first epoch details
                        print(f"    Epoch {epoch + 1}: "
                              f"loss={epoch_metrics.get('train_loss', 0):.3f}, "
                              f"acc={epoch_metrics.get('train_accuracy', 0):.2%}")
                    
                except Exception as e:
                    print(f"    ❌ Training epoch {epoch + 1} failed: {e}")
                    break
        
        # Evaluate final performance
        try:
            eval_metrics = trainer.evaluate(context)
            final_accuracy = eval_metrics.get('val_accuracy', 0.0)
            context.performance_history.append(final_accuracy)
            
            print(f"    Final: val_acc={final_accuracy:.2%}")
            
            return {
                'training_epochs': training_metrics,
                'evaluation': eval_metrics,
                'final_accuracy': final_accuracy
            }
            
        except Exception as e:
            print(f"    ❌ Evaluation failed: {e}")
            return {'training_epochs': training_metrics}
    
    def _apply_growth_strategies(self, 
                               context: NetworkContext, 
                               analysis_results: Dict[NetworkAnalyzer, AnalysisResult]) -> bool:
        """Apply growth strategies based on analysis results."""
        
        # Try each strategy in order of priority
        for strategy in self.strategies:
            if strategy.can_apply(context):
                try:
                    # Strategy will use analysis results from context metadata
                    if strategy.apply(context):
                        print(f"  ✅ {strategy.get_name()}: growth applied")
                        return True
                    else:
                        print(f"  ⏭️  {strategy.get_name()}: no growth needed")
                        
                except Exception as e:
                    print(f"  ❌ {strategy.get_name()} failed: {e}")
            else:
                print(f"  ⏭️  {strategy.get_name()}: not applicable")
        
        return False
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process."""
        return {
            'total_iterations': len(self._evolution_history),
            'components': {
                'analyzers': len(self.analyzers),
                'strategies': len(self.strategies),
                'lr_strategies': len(self.lr_strategies),
                'trainers': len(self.trainers)
            },
            'metrics': self._metrics,
            'history': self._evolution_history
        }
    
    # Configurable interface
    def configure(self, config: Dict[str, Any]):
        """Configure all components."""
        # Configure analyzers
        analyzer_configs = config.get('analyzers', {})
        for i, analyzer in enumerate(self.analyzers):
            if hasattr(analyzer, 'configure'):
                analyzer_config = analyzer_configs.get(f'analyzer_{i}', {})
                analyzer.configure(analyzer_config)
        
        # Configure strategies
        strategy_configs = config.get('strategies', {})
        for i, strategy in enumerate(self.strategies):
            if hasattr(strategy, 'configure'):
                strategy_config = strategy_configs.get(f'strategy_{i}', {})
                strategy.configure(strategy_config)
        
        # Configure trainers
        trainer_configs = config.get('trainers', {})
        for i, trainer in enumerate(self.trainers):
            if hasattr(trainer, 'configure'):
                trainer_config = trainer_configs.get(f'trainer_{i}', {})
                trainer.configure(trainer_config)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get configuration of all components."""
        config = {
            'analyzers': {},
            'strategies': {},
            'lr_strategies': {},
            'trainers': {}
        }
        
        for i, analyzer in enumerate(self.analyzers):
            if hasattr(analyzer, 'get_configuration'):
                config['analyzers'][f'analyzer_{i}'] = analyzer.get_configuration()
        
        for i, strategy in enumerate(self.strategies):
            if hasattr(strategy, 'get_configuration'):
                config['strategies'][f'strategy_{i}'] = strategy.get_configuration()
        
        for i, trainer in enumerate(self.trainers):
            if hasattr(trainer, 'get_configuration'):
                config['trainers'][f'trainer_{i}'] = trainer.get_configuration()
        
        return config
    
    # Serializable interface
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'ComposableEvolutionSystem',
            'components': {
                'analyzers': [comp.to_dict() if hasattr(comp, 'to_dict') else str(comp) 
                             for comp in self.analyzers],
                'strategies': [comp.to_dict() if hasattr(comp, 'to_dict') else str(comp) 
                              for comp in self.strategies],
                'lr_strategies': [comp.to_dict() if hasattr(comp, 'to_dict') else str(comp) 
                                 for comp in self.lr_strategies],
                'trainers': [comp.to_dict() if hasattr(comp, 'to_dict') else str(comp) 
                            for comp in self.trainers]
            },
            'metrics': self._metrics,
            'history': self._evolution_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComposableEvolutionSystem':
        """Deserialize from dictionary."""
        system = cls()
        system._metrics = data.get('metrics', {})
        system._evolution_history = data.get('history', [])
        # Component reconstruction would need a factory
        return system
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        combined_metrics = self._metrics.copy()
        
        # Add component metrics
        for i, analyzer in enumerate(self.analyzers):
            if hasattr(analyzer, 'get_metrics'):
                analyzer_metrics = analyzer.get_metrics()
                for key, value in analyzer_metrics.items():
                    combined_metrics[f'analyzer_{i}_{key}'] = value
        
        for i, strategy in enumerate(self.strategies):
            if hasattr(strategy, 'get_metrics'):
                strategy_metrics = strategy.get_metrics()
                for key, value in strategy_metrics.items():
                    combined_metrics[f'strategy_{i}_{key}'] = value
        
        return combined_metrics
    
    def reset_metrics(self):
        """Reset monitoring metrics."""
        self._metrics.clear()
        self._evolution_history.clear()
        
        for component in self.get_components():
            if hasattr(component, 'reset_metrics'):
                component.reset_metrics()


def create_standard_evolution_system() -> ComposableEvolutionSystem:
    """Create a standard evolution system with common components."""
    system = ComposableEvolutionSystem()
    
    # Add standard analyzers
    system.add_component(StandardExtremaAnalyzer())
    system.add_component(NetworkStatsAnalyzer())
    system.add_component(SimpleInformationFlowAnalyzer())
    
    # Add standard strategies
    system.add_component(ExtremaGrowthStrategy())
    system.add_component(InformationFlowGrowthStrategy())
    system.add_component(ResidualBlockGrowthStrategy())
    
    # Add standard trainer
    system.add_component(StandardNetworkTrainer())
    
    return system


def create_extrema_focused_system() -> ComposableEvolutionSystem:
    """Create an evolution system focused on extrema-driven growth."""
    system = ComposableEvolutionSystem()
    
    # Extrema-focused components
    system.add_component(StandardExtremaAnalyzer(max_batches=10))
    system.add_component(NetworkStatsAnalyzer())
    system.add_component(ExtremaGrowthStrategy(extrema_threshold=0.2))
    system.add_component(StandardNetworkTrainer())
    
    return system


def create_hybrid_system() -> ComposableEvolutionSystem:
    """Create a hybrid evolution system with multiple strategies."""
    system = ComposableEvolutionSystem()
    
    # All analyzers
    system.add_component(StandardExtremaAnalyzer())
    system.add_component(NetworkStatsAnalyzer())
    system.add_component(SimpleInformationFlowAnalyzer())
    
    # Hybrid strategy combining multiple approaches
    hybrid_strategy = HybridGrowthStrategy([
        ExtremaGrowthStrategy(),
        InformationFlowGrowthStrategy(),
        ResidualBlockGrowthStrategy()
    ])
    system.add_component(hybrid_strategy)
    
    # Standard trainer
    system.add_component(StandardNetworkTrainer())
    
    return system


# Export all components
__all__ = [
    'StandardNetworkTrainer',
    'ComposableEvolutionSystem',
    'create_standard_evolution_system',
    'create_extrema_focused_system',
    'create_hybrid_system'
]
