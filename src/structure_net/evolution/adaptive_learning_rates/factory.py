"""
Factory functions for creating adaptive learning rate systems.

This module provides convenient factory functions for creating adaptive learning
rate managers and training loops with sensible defaults.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from .base import LearningRateStrategy
from .unified_manager import AdaptiveLearningRateManager
from .phase_schedulers import *
from .layer_schedulers import *
from .connection_schedulers import *


def create_adaptive_manager(network: nn.Module,
                          base_lr: float = 0.001,
                          strategy: Union[str, LearningRateStrategy] = "basic",
                          custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                          **kwargs) -> AdaptiveLearningRateManager:
    """
    Create an adaptive learning rate manager with predefined strategies.
    
    Args:
        network: The neural network
        base_lr: Base learning rate
        strategy: Strategy type ('basic', 'advanced', 'comprehensive', 'ultimate')
        custom_configs: Custom configurations for individual schedulers
        **kwargs: Additional arguments passed to AdaptiveLearningRateManager
    
    Returns:
        Configured AdaptiveLearningRateManager
    """
    
    # Convert string strategy to enum
    if isinstance(strategy, str):
        strategy_map = {
            'basic': LearningRateStrategy.BASIC,
            'advanced': LearningRateStrategy.ADVANCED,
            'comprehensive': LearningRateStrategy.COMPREHENSIVE,
            'ultimate': LearningRateStrategy.ULTIMATE
        }
        strategy = strategy_map.get(strategy.lower(), LearningRateStrategy.BASIC)
    
    # Create manager with strategy
    manager = AdaptiveLearningRateManager(
        network=network,
        base_lr=base_lr,
        strategy=strategy,
        scheduler_configs=custom_configs,
        **kwargs
    )
    
    return manager


def create_basic_manager(network: nn.Module, base_lr: float = 0.001, **kwargs) -> AdaptiveLearningRateManager:
    """Create a basic adaptive learning rate manager with essential strategies."""
    return create_adaptive_manager(
        network=network,
        base_lr=base_lr,
        strategy="basic",
        **kwargs
    )


def create_advanced_manager(network: nn.Module, base_lr: float = 0.001, **kwargs) -> AdaptiveLearningRateManager:
    """Create an advanced adaptive learning rate manager with extrema detection."""
    return create_adaptive_manager(
        network=network,
        base_lr=base_lr,
        strategy="advanced",
        **kwargs
    )


def create_comprehensive_manager(network: nn.Module, base_lr: float = 0.001, **kwargs) -> AdaptiveLearningRateManager:
    """Create a comprehensive adaptive learning rate manager with most features."""
    return create_adaptive_manager(
        network=network,
        base_lr=base_lr,
        strategy="comprehensive",
        **kwargs
    )


def create_ultimate_manager(network: nn.Module, base_lr: float = 0.001, **kwargs) -> AdaptiveLearningRateManager:
    """Create the ultimate adaptive learning rate manager with all features."""
    return create_adaptive_manager(
        network=network,
        base_lr=base_lr,
        strategy="ultimate",
        **kwargs
    )


def create_custom_manager(network: nn.Module,
                         base_lr: float = 0.001,
                         schedulers: List[str] = None,
                         scheduler_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                         **kwargs) -> AdaptiveLearningRateManager:
    """
    Create a custom adaptive learning rate manager with specific schedulers.
    
    Args:
        network: The neural network
        base_lr: Base learning rate
        schedulers: List of scheduler names to enable
        scheduler_configs: Custom configurations for schedulers
        **kwargs: Additional arguments
    
    Returns:
        Configured AdaptiveLearningRateManager
    """
    
    if schedulers is None:
        schedulers = ['exponential_backoff', 'layerwise_rates']
    
    # Map scheduler names to enable flags
    enable_flags = {
        'enable_exponential_backoff': 'exponential_backoff' in schedulers,
        'enable_layerwise_rates': 'layerwise_rates' in schedulers,
        'enable_soft_clamping': 'soft_clamping' in schedulers,
        'enable_scale_dependent': 'scale_dependent' in schedulers,
        'enable_phase_based': 'phase_based' in schedulers,
        'enable_extrema_phase': 'extrema_phase' in schedulers,
        'enable_layer_age_aware': 'layer_age_aware' in schedulers,
        'enable_multi_scale': 'multi_scale' in schedulers,
        'enable_unified_system': 'unified_system' in schedulers,
    }
    
    return AdaptiveLearningRateManager(
        network=network,
        base_lr=base_lr,
        strategy=LearningRateStrategy.BASIC,  # Use basic as base, override with flags
        scheduler_configs=scheduler_configs,
        **enable_flags,
        **kwargs
    )


def create_adaptive_training_loop(network: nn.Module, 
                                train_loader,
                                val_loader,
                                epochs: int = 50,
                                base_lr: float = 0.001,
                                strategy: str = "basic",
                                optimizer_class=optim.Adam,
                                criterion=None,
                                device: str = 'auto',
                                scheduler_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                                progress_callback: Optional[Callable] = None,
                                **kwargs) -> Tuple[nn.Module, List[Dict]]:
    """
    Create a complete training loop with adaptive learning rate strategies.
    
    Args:
        network: The neural network to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        base_lr: Base learning rate
        strategy: Adaptive learning rate strategy
        optimizer_class: Optimizer class to use
        criterion: Loss criterion (defaults to CrossEntropyLoss)
        device: Device to use ('auto', 'cuda', 'cpu')
        scheduler_configs: Custom scheduler configurations
        progress_callback: Optional callback for progress updates
        **kwargs: Additional arguments for the manager
    
    Returns:
        Tuple of (trained network, training history)
    """
    
    # Set up device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    network = network.to(device)
    
    # Set up criterion
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Initialize adaptive learning rate manager
    lr_manager = create_adaptive_manager(
        network=network,
        base_lr=base_lr,
        strategy=strategy,
        custom_configs=scheduler_configs,
        **kwargs
    )
    
    # Create adaptive optimizer
    optimizer = lr_manager.create_adaptive_optimizer(optimizer_class)
    
    # Training history
    history = []
    
    print(f"\n🚀 Starting Adaptive Training Loop")
    print(f"   Strategy: {strategy}")
    print(f"   Epochs: {epochs}")
    print(f"   Base LR: {base_lr}")
    print(f"   Device: {device}")
    print(f"   Optimizer: {optimizer_class.__name__}")
    
    for epoch in range(epochs):
        # Update learning rates for current epoch
        lr_manager.update_learning_rates(optimizer, epoch, 
                                       network=network, 
                                       data_loader=train_loader,
                                       device=device)
        
        # Training phase
        network.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Flatten data if needed
            if len(data.shape) > 2:
                data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Apply gradient modifications
            lr_manager.apply_gradient_modifications(optimizer)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += len(target)
        
        # Validation phase
        network.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # Flatten data if needed
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                output = network(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += len(target)
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Record history
        epoch_data = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'learning_rates': lr_manager.get_current_rates_summary()
        }
        history.append(epoch_data)
        
        # Progress callback
        if progress_callback:
            progress_callback(epoch_data)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2%}, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2%}")
            
            if epoch % 20 == 0:
                lr_manager.print_rates_summary()
    
    print(f"\n✅ Adaptive Training Complete")
    print(f"   Final Val Accuracy: {history[-1]['val_acc']:.2%}")
    
    return network, history


def create_scheduler_presets() -> Dict[str, Dict[str, Any]]:
    """
    Create predefined scheduler configuration presets.
    
    Returns:
        Dictionary of preset configurations
    """
    
    presets = {
        'conservative': {
            'exponential_backoff': {
                'initial_lr': 0.1,
                'decay_rate': 0.98,
                'min_lr': 1e-5
            },
            'layerwise_rates': {
                'early_rate': 0.001,
                'middle_rate': 0.005,
                'late_rate': 0.01
            },
            'soft_clamping': {
                'max_age': 200,
                'min_clamp_factor': 0.05
            }
        },
        
        'aggressive': {
            'exponential_backoff': {
                'initial_lr': 1.0,
                'decay_rate': 0.9,
                'min_lr': 1e-4
            },
            'layerwise_rates': {
                'early_rate': 0.05,
                'middle_rate': 0.02,
                'late_rate': 0.01
            },
            'soft_clamping': {
                'max_age': 50,
                'min_clamp_factor': 0.2
            }
        },
        
        'balanced': {
            'exponential_backoff': {
                'initial_lr': 0.5,
                'decay_rate': 0.95,
                'min_lr': 1e-5
            },
            'layerwise_rates': {
                'early_rate': 0.01,
                'middle_rate': 0.01,
                'late_rate': 0.005
            },
            'soft_clamping': {
                'max_age': 100,
                'min_clamp_factor': 0.1
            }
        },
        
        'extrema_focused': {
            'extrema_phase': {
                'explosive_threshold': 0.15,
                'steady_threshold': 0.05,
                'explosive_multiplier': 2.0,
                'steady_multiplier': 0.5,
                'refinement_multiplier': 0.1
            },
            'layer_age_aware': {
                'decay_constant': 30.0,
                'early_layer_rate': 0.05,
                'late_layer_boost': 1.5
            }
        },
        
        'fine_tuning': {
            'progressive_freezing': {
                'warmup_lr': 0.001,
                'refinement_early_lr': 0.00001,
                'refinement_late_lr': 0.0001,
                'final_lr': 0.00005,
                'warmup_epochs': 5,
                'refinement_epochs': 20
            },
            'component_specific': {
                'scaffold_lr': 0.00001,
                'patch_lr': 0.0001,
                'neck_lr': 0.0005,
                'new_layer_lr': 0.001
            }
        }
    }
    
    return presets


def create_preset_manager(network: nn.Module,
                         preset_name: str,
                         base_lr: float = 0.001,
                         **kwargs) -> AdaptiveLearningRateManager:
    """
    Create an adaptive learning rate manager using a predefined preset.
    
    Args:
        network: The neural network
        preset_name: Name of the preset ('conservative', 'aggressive', 'balanced', etc.)
        base_lr: Base learning rate
        **kwargs: Additional arguments
    
    Returns:
        Configured AdaptiveLearningRateManager
    """
    
    presets = create_scheduler_presets()
    
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    preset_config = presets[preset_name]
    
    # Determine strategy based on preset
    if preset_name in ['extrema_focused']:
        strategy = "advanced"
    elif preset_name in ['fine_tuning']:
        strategy = "comprehensive"
    else:
        strategy = "basic"
    
    return create_adaptive_manager(
        network=network,
        base_lr=base_lr,
        strategy=strategy,
        custom_configs=preset_config,
        **kwargs
    )


# Convenience functions for common use cases
def create_structure_net_manager(network: nn.Module, 
                               base_lr: float = 0.001,
                               enable_extrema: bool = True,
                               **kwargs) -> AdaptiveLearningRateManager:
    """Create a manager optimized for Structure Net training."""
    strategy = "comprehensive" if enable_extrema else "advanced"
    return create_adaptive_manager(network, base_lr, strategy, **kwargs)


def create_transfer_learning_manager(network: nn.Module,
                                   pretrained_layers: List[str],
                                   base_lr: float = 0.001,
                                   **kwargs) -> AdaptiveLearningRateManager:
    """Create a manager optimized for transfer learning scenarios."""
    
    # Configure for transfer learning
    custom_configs = {
        'pretrained_new_layer': {
            'pretrained_lr': base_lr * 0.01,  # Very low for pretrained
            'new_lr': base_lr,                # Normal for new layers
            'adapter_lr': base_lr * 0.1       # Medium for adapters
        }
    }
    
    return create_custom_manager(
        network=network,
        base_lr=base_lr,
        schedulers=['pretrained_new_layer', 'soft_clamping'],
        scheduler_configs=custom_configs,
        **kwargs
    )


def create_continual_learning_manager(network: nn.Module,
                                    base_lr: float = 0.001,
                                    **kwargs) -> AdaptiveLearningRateManager:
    """Create a manager optimized for continual learning scenarios."""
    
    custom_configs = {
        'sedimentary_learning': {
            'geological_lr': base_lr * 0.001,
            'sediment_lr': base_lr * 0.01,
            'active_lr': base_lr,
            'patch_lr': base_lr * 0.1
        }
    }
    
    return create_custom_manager(
        network=network,
        base_lr=base_lr,
        schedulers=['sedimentary_learning', 'age_based', 'soft_clamping'],
        scheduler_configs=custom_configs,
        **kwargs
    )
