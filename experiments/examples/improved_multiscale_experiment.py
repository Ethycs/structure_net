"""
Improved Multi-Scale Experiment with Better Coarse Scale and Transfer Learning Analysis

This implements the improved multi-scale concept with:
- Better coarse scale connectivity (5% instead of 0.94%)
- Transfer learning comparison (random vs hierarchical initialization)
- Detailed analysis of learning curves and representations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import logging
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List
import time

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.structure_net.core.minimal_network import MinimalNetwork


def setup_logging(log_file: str = "improved_multiscale_experiment.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def downsample_mnist(factor: int, data_dir: str = "./data"):
    """Load MNIST and downsample by given factor."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    def downsample_batch(batch):
        images, labels = batch
        downsampled = F.avg_pool2d(images, kernel_size=factor, stride=factor)
        flattened = downsampled.view(downsampled.size(0), -1)
        return flattened, labels
    
    # Process data
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    train_loader_temp = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    for batch in train_loader_temp:
        downsampled_batch, labels = downsample_batch(batch)
        train_images.append(downsampled_batch)
        train_labels.append(labels)
    
    test_loader_temp = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    for batch in test_loader_temp:
        downsampled_batch, labels = downsample_batch(batch)
        test_images.append(downsampled_batch)
        test_labels.append(labels)
    
    train_images = torch.cat(train_images, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_images = torch.cat(test_images, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    train_dataset_down = TensorDataset(train_images, train_labels)
    test_dataset_down = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset_down, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset_down, batch_size=128, shuffle=False)
    
    return train_loader, test_loader, train_images.size(1)


def create_improved_scale_network(input_size: int, sparsity: float, device: torch.device):
    """Create improved networks with better architectures for each scale."""
    if input_size <= 49:  # 7x7 coarse - IMPROVED
        hidden_sizes = [64, 32]  # Wider layers for better learning
    elif input_size <= 196:  # 14x14 medium
        hidden_sizes = [128, 64]  # Increased capacity
    else:  # 28x28 fine
        hidden_sizes = [256, 128]  # Even more capacity
    
    network = MinimalNetwork(
        layer_sizes=[input_size] + hidden_sizes + [10],
        sparsity=sparsity,
        activation='tanh',
        device=device
    )
    
    return network


def transfer_weights_improved(source_network: MinimalNetwork, target_network: MinimalNetwork):
    """Improved weight transfer with better compatibility checking."""
    source_state = source_network.state_dict_sparse()
    target_state = target_network.state_dict_sparse()
    
    transferred_layers = []
    
    # Transfer compatible layers (skip input layer due to size mismatch)
    for key in target_state:
        if 'layers.0' not in key:  # Skip first layer
            if key in source_state:
                if (isinstance(source_state[key], torch.Tensor) and 
                    isinstance(target_state[key], torch.Tensor) and
                    source_state[key].shape == target_state[key].shape):
                    
                    target_state[key] = source_state[key].clone()
                    transferred_layers.append(key)
    
    target_network.load_state_dict_sparse(target_state)
    
    # Transfer compatible connection masks
    transferred_masks = []
    for i, (source_mask, target_mask) in enumerate(zip(source_network.connection_masks[1:], target_network.connection_masks[1:])):
        if source_mask.shape == target_mask.shape:
            target_network.connection_masks[i+1] = source_mask.clone()
            transferred_masks.append(f"layer_{i+1}")
    
    return transferred_layers, transferred_masks


def train_with_detailed_logging(
    network: MinimalNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    scale_name: str,
    epochs: int = 15,
    learning_rate: float = 0.001
) -> Dict:
    """Train with detailed epoch-by-epoch logging."""
    logger = logging.getLogger(__name__)
    
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'connectivity_ratio': [],
        'epoch_times': []
    }
    
    logger.info(f"Training {scale_name} scale for {epochs} epochs")
    connectivity_stats = network.get_connectivity_stats()
    logger.info(f"Network: {network.layer_sizes}")
    logger.info(f"Connectivity: {connectivity_stats['connectivity_ratio']:.6f} ({connectivity_stats['total_active_connections']:,} connections)")
    
    for epoch in tqdm(range(epochs), desc=f"Training {scale_name}"):
        epoch_start = time.time()
        
        # Training phase
        network.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(network.device), target.to(network.device)
            
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        
        # Evaluation phase
        network.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(network.device), target.to(network.device)
                output = network(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        
        # Record metrics
        epoch_time = time.time() - epoch_start
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        connectivity = network.get_connectivity_stats()['connectivity_ratio']
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_accuracy'].append(train_acc)
        history['test_loss'].append(test_loss / len(test_loader))
        history['test_accuracy'].append(test_acc)
        history['connectivity_ratio'].append(connectivity)
        history['epoch_times'].append(epoch_time)
        
        if epoch % 3 == 0 or epoch == epochs - 1:
            logger.info(f"{scale_name} Epoch {epoch:2d}: Train={train_acc:.4f}, Test={test_acc:.4f}, Time={epoch_time:.1f}s")
    
    final_stats = network.get_connectivity_stats()
    logger.info(f"{scale_name} training completed:")
    logger.info(f"  Final test accuracy: {history['test_accuracy'][-1]:.4f}")
    logger.info(f"  Total parameters: {final_stats['total_active_connections']:,}")
    logger.info(f"  Training time: {sum(history['epoch_times']):.1f}s")
    
    return history


def compare_initialization_methods(
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_size: int,
    sparsity: float,
    device: torch.device,
    source_network: MinimalNetwork = None,
    epochs: int = 15
) -> Dict:
    """Compare random vs transfer initialization."""
    results = {}
    
    # Method 1: Random initialization
    logger = logging.getLogger(__name__)
    logger.info("Testing random initialization...")
    
    random_network = create_improved_scale_network(input_size, sparsity, device)
    random_history = train_with_detailed_logging(
        random_network, train_loader, test_loader, 
        "random_init", epochs=epochs
    )
    results['random'] = {
        'network': random_network,
        'history': random_history
    }
    
    # Method 2: Transfer initialization (if source provided)
    if source_network is not None:
        logger.info("Testing transfer initialization...")
        
        transfer_network = create_improved_scale_network(input_size, sparsity, device)
        transferred_layers, transferred_masks = transfer_weights_improved(source_network, transfer_network)
        
        logger.info(f"Transferred layers: {transferred_layers}")
        logger.info(f"Transferred masks: {transferred_masks}")
        
        transfer_history = train_with_detailed_logging(
            transfer_network, train_loader, test_loader,
            "transfer_init", epochs=epochs
        )
        results['transfer'] = {
            'network': transfer_network,
            'history': transfer_history,
            'transferred_layers': transferred_layers,
            'transferred_masks': transferred_masks
        }
    
    return results


def create_detailed_comparison_plots(results: Dict, save_dir: Path, scale_name: str):
    """Create detailed comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Test accuracy comparison
    for method, color in [('random', 'red'), ('transfer', 'blue')]:
        if method in results:
            epochs = range(len(results[method]['history']['test_accuracy']))
            axes[0, 0].plot(epochs, results[method]['history']['test_accuracy'], 
                          label=f'{method.capitalize()} Init', color=color, alpha=0.8)
    
    axes[0, 0].set_title(f'{scale_name.capitalize()} Scale - Test Accuracy Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training loss comparison
    for method, color in [('random', 'red'), ('transfer', 'blue')]:
        if method in results:
            epochs = range(len(results[method]['history']['train_loss']))
            axes[0, 1].plot(epochs, results[method]['history']['train_loss'],
                          label=f'{method.capitalize()} Init', color=color, alpha=0.8)
    
    axes[0, 1].set_title(f'{scale_name.capitalize()} Scale - Training Loss Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Training Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning speed comparison (accuracy improvement per epoch)
    for method, color in [('random', 'red'), ('transfer', 'blue')]:
        if method in results:
            accuracies = results[method]['history']['test_accuracy']
            improvements = [accuracies[i] - accuracies[0] for i in range(len(accuracies))]
            epochs = range(len(improvements))
            axes[0, 2].plot(epochs, improvements,
                          label=f'{method.capitalize()} Init', color=color, alpha=0.8)
    
    axes[0, 2].set_title(f'{scale_name.capitalize()} Scale - Learning Progress')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy Improvement from Start')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Final performance comparison
    methods = []
    final_accuracies = []
    final_times = []
    
    for method in ['random', 'transfer']:
        if method in results:
            methods.append(method.capitalize())
            final_accuracies.append(results[method]['history']['test_accuracy'][-1])
            final_times.append(sum(results[method]['history']['epoch_times']))
    
    x = np.arange(len(methods))
    bars = axes[1, 0].bar(x, final_accuracies, color=['red', 'blue'][:len(methods)], alpha=0.7)
    axes[1, 0].set_title(f'{scale_name.capitalize()} Scale - Final Accuracy')
    axes[1, 0].set_xlabel('Initialization Method')
    axes[1, 0].set_ylabel('Final Test Accuracy')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    
    # Add values on bars
    for i, v in enumerate(final_accuracies):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    axes[1, 1].bar(x, final_times, color=['red', 'blue'][:len(methods)], alpha=0.7)
    axes[1, 1].set_title(f'{scale_name.capitalize()} Scale - Training Time')
    axes[1, 1].set_xlabel('Initialization Method')
    axes[1, 1].set_ylabel('Total Training Time (s)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods)
    
    # Add values on bars
    for i, v in enumerate(final_times):
        axes[1, 1].text(i, v + max(final_times) * 0.02, f'{v:.1f}s', ha='center', va='bottom')
    
    # Summary statistics
    summary_text = f"""Transfer Learning Analysis - {scale_name.capitalize()} Scale

"""
    
    if 'random' in results and 'transfer' in results:
        random_acc = results['random']['history']['test_accuracy'][-1]
        transfer_acc = results['transfer']['history']['test_accuracy'][-1]
        improvement = transfer_acc - random_acc
        
        random_time = sum(results['random']['history']['epoch_times'])
        transfer_time = sum(results['transfer']['history']['epoch_times'])
        
        summary_text += f"""Random Initialization:
  Final Accuracy: {random_acc:.4f}
  Training Time: {random_time:.1f}s

Transfer Initialization:
  Final Accuracy: {transfer_acc:.4f}
  Training Time: {transfer_time:.1f}s

Transfer Benefit:
  Accuracy Gain: {improvement:+.4f}
  Time Difference: {transfer_time - random_time:+.1f}s

Transferred Components:
  Layers: {len(results['transfer'].get('transferred_layers', []))}
  Masks: {len(results['transfer'].get('transferred_masks', []))}
"""
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title(f'{scale_name.capitalize()} Scale - Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{scale_name}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main improved experiment function."""
    parser = argparse.ArgumentParser(description='Improved Multi-Scale MNIST Experiment')
    parser.add_argument('--epochs-per-scale', type=int, default=15, help='Epochs per scale')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results-dir', type=str, default='./improved_multiscale_results', help='Results directory')
    
    args = parser.parse_args()
    
    setup_logging("improved_multiscale_experiment.log")
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    save_dir = Path(args.results_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Improved sparsity configuration
    improved_config = {
        'coarse': {'sparsity': 0.05, 'factor': 4},    # 5% instead of 0.94%
        'medium': {'sparsity': 0.08, 'factor': 2},    # 8% instead of 5%
        'fine': {'sparsity': 0.25, 'factor': 1}       # 25% instead of 20%
    }
    
    results = {}
    
    # Phase 1: Improved Coarse Scale (7x7)
    logger.info("="*70)
    logger.info("PHASE 1: IMPROVED COARSE SCALE (7x7) - 5% Connectivity")
    logger.info("="*70)
    
    train_loader_coarse, test_loader_coarse, input_size_coarse = downsample_mnist(
        factor=improved_config['coarse']['factor'], data_dir=args.data_dir
    )
    
    coarse_results = compare_initialization_methods(
        train_loader_coarse, test_loader_coarse, input_size_coarse,
        improved_config['coarse']['sparsity'], device, 
        source_network=None, epochs=args.epochs_per_scale
    )
    
    create_detailed_comparison_plots(coarse_results, save_dir, 'coarse')
    results['coarse'] = coarse_results['random']  # Use random for next phase
    
    # Phase 2: Medium Scale (14x14) with Transfer Learning Test
    logger.info("="*70)
    logger.info("PHASE 2: MEDIUM SCALE (14x14) - Transfer Learning Test")
    logger.info("="*70)
    
    train_loader_medium, test_loader_medium, input_size_medium = downsample_mnist(
        factor=improved_config['medium']['factor'], data_dir=args.data_dir
    )
    
    medium_results = compare_initialization_methods(
        train_loader_medium, test_loader_medium, input_size_medium,
        improved_config['medium']['sparsity'], device,
        source_network=results['coarse']['network'], epochs=args.epochs_per_scale
    )
    
    create_detailed_comparison_plots(medium_results, save_dir, 'medium')
    results['medium'] = medium_results
    
    # Phase 3: Fine Scale (28x28) with Transfer Learning Test
    logger.info("="*70)
    logger.info("PHASE 3: FINE SCALE (28x28) - Transfer Learning Test")
    logger.info("="*70)
    
    # Load full resolution MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train_dataset_fine = datasets.MNIST(args.data_dir, train=True, transform=transform)
    test_dataset_fine = datasets.MNIST(args.data_dir, train=False, transform=transform)
    
    train_loader_fine = DataLoader(train_dataset_fine, batch_size=128, shuffle=True)
    test_loader_fine = DataLoader(test_dataset_fine, batch_size=128, shuffle=False)
    
    # Test both random and transfer initialization
    fine_results = compare_initialization_methods(
        train_loader_fine, test_loader_fine, 784,
        improved_config['fine']['sparsity'], device,
        source_network=medium_results['transfer']['network'], epochs=args.epochs_per_scale
    )
    
    create_detailed_comparison_plots(fine_results, save_dir, 'fine')
    results['fine'] = fine_results
    
    # Final comprehensive summary
    logger.info("="*70)
    logger.info("COMPREHENSIVE EXPERIMENT SUMMARY")
    logger.info("="*70)
    
    summary = {
        'experiment_type': 'improved_multiscale_with_transfer_analysis',
        'improved_config': improved_config,
        'results': {}
    }
    
    for scale in ['coarse', 'medium', 'fine']:
        if scale in results:
            scale_summary = {}
            
            if scale == 'coarse':
                # Only random for coarse
                network = results[scale]['network']
                history = results[scale]['history']
                scale_summary['random'] = {
                    'final_accuracy': history['test_accuracy'][-1],
                    'architecture': network.layer_sizes,
                    'connectivity': network.get_connectivity_stats()['connectivity_ratio'],
                    'parameters': network.get_connectivity_stats()['total_active_connections']
                }
                
                logger.info(f"{scale.upper()} SCALE (Random Only):")
                logger.info(f"  Architecture: {network.layer_sizes}")
                logger.info(f"  Final accuracy: {history['test_accuracy'][-1]:.4f}")
                logger.info(f"  Connectivity: {network.get_connectivity_stats()['connectivity_ratio']:.6f}")
                
            else:
                # Both random and transfer for medium/fine
                for method in ['random', 'transfer']:
                    if method in results[scale]:
                        network = results[scale][method]['network']
                        history = results[scale][method]['history']
                        scale_summary[method] = {
                            'final_accuracy': history['test_accuracy'][-1],
                            'architecture': network.layer_sizes,
                            'connectivity': network.get_connectivity_stats()['connectivity_ratio'],
                            'parameters': network.get_connectivity_stats()['total_active_connections']
                        }
                
                logger.info(f"{scale.upper()} SCALE:")
                logger.info(f"  Architecture: {results[scale]['random']['network'].layer_sizes}")
                
                if 'random' in results[scale] and 'transfer' in results[scale]:
                    random_acc = results[scale]['random']['history']['test_accuracy'][-1]
                    transfer_acc = results[scale]['transfer']['history']['test_accuracy'][-1]
                    improvement = transfer_acc - random_acc
                    
                    logger.info(f"  Random init accuracy: {random_acc:.4f}")
                    logger.info(f"  Transfer init accuracy: {transfer_acc:.4f}")
                    logger.info(f"  Transfer benefit: {improvement:+.4f}")
            
            summary['results'][scale] = scale_summary
            logger.info("")
    
    # Save complete results
    with open(save_dir / "comprehensive_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Improved experiment completed. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
