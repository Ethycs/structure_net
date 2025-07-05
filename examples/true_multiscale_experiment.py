"""
True Multi-Scale Experiment Implementation

This implements the ACTUAL multi-scale concept from Experiment 1:
- Phase 1: Train coarse network on downsampled data (7x7)
- Phase 2: Train medium network on medium resolution (14x14) 
- Phase 3: Train fine network on full resolution (28x28)
- Each phase builds on the previous with hierarchical initialization
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
from typing import Dict

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.structure_net.core.minimal_network import MinimalNetwork
from src.structure_net.snapshots.snapshot_manager import SnapshotManager


def setup_logging(log_file: str = "true_multiscale_experiment.log"):
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
    """
    Load MNIST and downsample by given factor.
    
    Args:
        factor: Downsampling factor (2 = 14x14, 4 = 7x7)
        data_dir: Directory containing MNIST data
        
    Returns:
        train_loader, test_loader with downsampled data
    """
    # Load original MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    # Downsample the data
    def downsample_batch(batch):
        images, labels = batch
        # Downsample using average pooling
        downsampled = F.avg_pool2d(images, kernel_size=factor, stride=factor)
        # Flatten for MLP
        flattened = downsampled.view(downsampled.size(0), -1)
        return flattened, labels
    
    # Create downsampled datasets
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    # Process training data
    train_loader_temp = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    for batch in train_loader_temp:
        downsampled_batch, labels = downsample_batch(batch)
        train_images.append(downsampled_batch)
        train_labels.append(labels)
    
    # Process test data
    test_loader_temp = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    for batch in test_loader_temp:
        downsampled_batch, labels = downsample_batch(batch)
        test_images.append(downsampled_batch)
        test_labels.append(labels)
    
    # Concatenate all batches
    train_images = torch.cat(train_images, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_images = torch.cat(test_images, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # Create new datasets
    train_dataset_down = TensorDataset(train_images, train_labels)
    test_dataset_down = TensorDataset(test_images, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset_down, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset_down, batch_size=128, shuffle=False)
    
    input_size = train_images.size(1)
    
    return train_loader, test_loader, input_size


def create_scale_network(input_size: int, sparsity: float, device: torch.device):
    """
    Create a network for a specific scale.
    
    Args:
        input_size: Size of input (49 for 7x7, 196 for 14x14, 784 for 28x28)
        sparsity: Sparsity level for this scale
        device: Device to run on
        
    Returns:
        MinimalNetwork instance
    """
    # Scale hidden sizes based on input size
    if input_size <= 49:  # 7x7 coarse
        hidden_sizes = [32, 16]
    elif input_size <= 196:  # 14x14 medium
        hidden_sizes = [64, 32]
    else:  # 28x28 fine
        hidden_sizes = [128, 64]
    
    network = MinimalNetwork(
        layer_sizes=[input_size] + hidden_sizes + [10],
        sparsity=sparsity,
        activation='tanh',
        device=device
    )
    
    return network


def grow_from_previous_scale(previous_network: MinimalNetwork, target_input_size: int, target_sparsity: float):
    """
    Initialize a new network from a previous scale.
    
    Args:
        previous_network: Network from previous scale
        target_input_size: Input size for new scale
        target_sparsity: Target sparsity for new scale
        
    Returns:
        New network initialized from previous scale
    """
    # Create new network
    new_network = create_scale_network(target_input_size, target_sparsity, previous_network.device)
    
    # Copy compatible weights from previous network
    prev_state = previous_network.state_dict_sparse()
    new_state = new_network.state_dict_sparse()
    
    # Copy hidden and output layer weights (skip input layer due to size mismatch)
    for key in new_state:
        if 'layers.0' not in key:  # Skip first layer (input layer)
            if key in prev_state:
                # Check if both are tensors and sizes match
                if (isinstance(prev_state[key], torch.Tensor) and 
                    isinstance(new_state[key], torch.Tensor) and
                    prev_state[key].shape == new_state[key].shape):
                    new_state[key] = prev_state[key].clone()
                    print(f"Copied {key} from previous scale")
    
    # Load the modified state
    new_network.load_state_dict_sparse(new_state)
    
    # Copy connection masks for compatible layers
    for i, (prev_mask, new_mask) in enumerate(zip(previous_network.connection_masks[1:], new_network.connection_masks[1:])):
        if prev_mask.shape == new_mask.shape:
            new_network.connection_masks[i+1] = prev_mask.clone()
            print(f"Copied connection mask for layer {i+1}")
    
    return new_network


def train_scale(
    network: MinimalNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    scale_name: str,
    epochs: int = 20,
    learning_rate: float = 0.001
) -> Dict:
    """
    Train a network at a specific scale.
    
    Args:
        network: Network to train
        train_loader: Training data loader
        test_loader: Test data loader
        scale_name: Name of the scale ('coarse', 'medium', 'fine')
        epochs: Number of epochs to train
        learning_rate: Learning rate
        
    Returns:
        Training history
    """
    logger = logging.getLogger(__name__)
    
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'connectivity_ratio': []
    }
    
    logger.info(f"Training {scale_name} scale for {epochs} epochs")
    logger.info(f"Network connectivity: {network.get_connectivity_stats()}")
    
    for epoch in tqdm(range(epochs), desc=f"Training {scale_name}"):
        # Training phase
        network.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
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
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        connectivity = network.get_connectivity_stats()['connectivity_ratio']
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_accuracy'].append(train_acc)
        history['test_loss'].append(test_loss / len(test_loader))
        history['test_accuracy'].append(test_acc)
        history['connectivity_ratio'].append(connectivity)
        
        if epoch % 5 == 0:
            logger.info(f"{scale_name} Epoch {epoch}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    final_train_acc = history['train_accuracy'][-1]
    final_test_acc = history['test_accuracy'][-1]
    
    logger.info(f"{scale_name} training completed:")
    logger.info(f"  Final train accuracy: {final_train_acc:.4f}")
    logger.info(f"  Final test accuracy: {final_test_acc:.4f}")
    logger.info(f"  Final connectivity: {connectivity:.6f}")
    
    return history


def save_scale_results(network: MinimalNetwork, history: Dict, scale_name: str, save_dir: Path):
    """Save results for a specific scale."""
    scale_dir = save_dir / scale_name
    scale_dir.mkdir(exist_ok=True)
    
    # Save training history
    with open(scale_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save network state
    torch.save(network.state_dict_sparse(), scale_dir / "network_state.pt")
    
    # Save connectivity stats
    connectivity_stats = network.get_connectivity_stats()
    with open(scale_dir / "connectivity_stats.json", 'w') as f:
        json.dump(connectivity_stats, f, indent=2)
    
    # Create training plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = range(len(history['train_loss']))
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['test_loss'], label='Test')
    axes[0, 0].set_title(f'{scale_name.capitalize()} Scale - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_accuracy'], label='Train')
    axes[0, 1].plot(epochs, history['test_accuracy'], label='Test')
    axes[0, 1].set_title(f'{scale_name.capitalize()} Scale - Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Connectivity plot
    axes[1, 0].plot(epochs, history['connectivity_ratio'])
    axes[1, 0].set_title(f'{scale_name.capitalize()} Scale - Connectivity')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Connectivity Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final stats
    final_stats = f"""Final {scale_name.capitalize()} Scale Results:

Train Accuracy: {history['train_accuracy'][-1]:.4f}
Test Accuracy: {history['test_accuracy'][-1]:.4f}
Connectivity: {history['connectivity_ratio'][-1]:.6f}

Network Architecture:
{network.layer_sizes}

Total Parameters: {connectivity_stats['total_active_connections']:,}
Sparsity: {connectivity_stats['sparsity']:.4f}
"""
    
    axes[1, 1].text(0.1, 0.5, final_stats, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title(f'{scale_name.capitalize()} Scale - Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(scale_dir / f"{scale_name}_training.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_plots(results: Dict, save_dir: Path):
    """Create comparison plots across all scales."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    scales = ['coarse', 'medium', 'fine']
    colors = ['blue', 'orange', 'green']
    
    # Test accuracy comparison
    for scale, color in zip(scales, colors):
        if scale in results:
            epochs = range(len(results[scale]['history']['test_accuracy']))
            axes[0, 0].plot(epochs, results[scale]['history']['test_accuracy'], 
                          label=f'{scale.capitalize()}', color=color, alpha=0.7)
    
    axes[0, 0].set_title('Test Accuracy Across Scales')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Connectivity comparison
    for scale, color in zip(scales, colors):
        if scale in results:
            epochs = range(len(results[scale]['history']['connectivity_ratio']))
            axes[0, 1].plot(epochs, results[scale]['history']['connectivity_ratio'],
                          label=f'{scale.capitalize()}', color=color, alpha=0.7)
    
    axes[0, 1].set_title('Connectivity Across Scales')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Connectivity Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final performance comparison
    final_accuracies = []
    final_connectivities = []
    scale_names = []
    
    for scale in scales:
        if scale in results:
            final_accuracies.append(results[scale]['history']['test_accuracy'][-1])
            final_connectivities.append(results[scale]['history']['connectivity_ratio'][-1])
            scale_names.append(scale.capitalize())
    
    x = np.arange(len(scale_names))
    axes[1, 0].bar(x, final_accuracies, color=colors[:len(scale_names)], alpha=0.7)
    axes[1, 0].set_title('Final Test Accuracy by Scale')
    axes[1, 0].set_xlabel('Scale')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(scale_names)
    
    # Add accuracy values on bars
    for i, v in enumerate(final_accuracies):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Connectivity comparison
    axes[1, 1].bar(x, final_connectivities, color=colors[:len(scale_names)], alpha=0.7)
    axes[1, 1].set_title('Final Connectivity by Scale')
    axes[1, 1].set_xlabel('Scale')
    axes[1, 1].set_ylabel('Connectivity Ratio')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(scale_names)
    
    # Add connectivity values on bars
    for i, v in enumerate(final_connectivities):
        axes[1, 1].text(i, v + max(final_connectivities) * 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / "multiscale_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='True Multi-Scale MNIST Experiment')
    parser.add_argument('--epochs-per-scale', type=int, default=20, help='Epochs per scale')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results-dir', type=str, default='./multiscale_results', help='Results directory')
    parser.add_argument('--log-file', type=str, default='true_multiscale_experiment.log', help='Log file')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    save_dir = Path(args.results_dir)
    save_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # Phase 1: Coarse Scale (7x7)
    logger.info("="*60)
    logger.info("PHASE 1: COARSE SCALE (7x7)")
    logger.info("="*60)
    
    train_loader_coarse, test_loader_coarse, input_size_coarse = downsample_mnist(factor=4, data_dir=args.data_dir)
    logger.info(f"Coarse data: {input_size_coarse} input features (7x7)")
    
    coarse_network = create_scale_network(input_size_coarse, sparsity=0.01, device=device)
    coarse_history = train_scale(
        coarse_network, train_loader_coarse, test_loader_coarse, 
        'coarse', epochs=args.epochs_per_scale, learning_rate=args.lr
    )
    
    save_scale_results(coarse_network, coarse_history, 'coarse', save_dir)
    results['coarse'] = {'network': coarse_network, 'history': coarse_history}
    
    # Phase 2: Medium Scale (14x14)
    logger.info("="*60)
    logger.info("PHASE 2: MEDIUM SCALE (14x14)")
    logger.info("="*60)
    
    train_loader_medium, test_loader_medium, input_size_medium = downsample_mnist(factor=2, data_dir=args.data_dir)
    logger.info(f"Medium data: {input_size_medium} input features (14x14)")
    
    medium_network = grow_from_previous_scale(coarse_network, input_size_medium, target_sparsity=0.05)
    medium_history = train_scale(
        medium_network, train_loader_medium, test_loader_medium,
        'medium', epochs=args.epochs_per_scale, learning_rate=args.lr
    )
    
    save_scale_results(medium_network, medium_history, 'medium', save_dir)
    results['medium'] = {'network': medium_network, 'history': medium_history}
    
    # Phase 3: Fine Scale (28x28)
    logger.info("="*60)
    logger.info("PHASE 3: FINE SCALE (28x28)")
    logger.info("="*60)
    
    # Load full resolution MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])
    
    train_dataset_fine = datasets.MNIST(args.data_dir, train=True, transform=transform)
    test_dataset_fine = datasets.MNIST(args.data_dir, train=False, transform=transform)
    
    train_loader_fine = DataLoader(train_dataset_fine, batch_size=128, shuffle=True)
    test_loader_fine = DataLoader(test_dataset_fine, batch_size=128, shuffle=False)
    
    input_size_fine = 784
    logger.info(f"Fine data: {input_size_fine} input features (28x28)")
    
    fine_network = grow_from_previous_scale(medium_network, input_size_fine, target_sparsity=0.20)
    fine_history = train_scale(
        fine_network, train_loader_fine, test_loader_fine,
        'fine', epochs=args.epochs_per_scale, learning_rate=args.lr
    )
    
    save_scale_results(fine_network, fine_history, 'fine', save_dir)
    results['fine'] = {'network': fine_network, 'history': fine_history}
    
    # Create comparison plots
    create_comparison_plots(results, save_dir)
    
    # Final summary
    logger.info("="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    for scale in ['coarse', 'medium', 'fine']:
        if scale in results:
            final_acc = results[scale]['history']['test_accuracy'][-1]
            final_conn = results[scale]['history']['connectivity_ratio'][-1]
            network = results[scale]['network']
            
            logger.info(f"{scale.upper()} SCALE:")
            logger.info(f"  Architecture: {network.layer_sizes}")
            logger.info(f"  Final test accuracy: {final_acc:.4f}")
            logger.info(f"  Final connectivity: {final_conn:.6f}")
            logger.info(f"  Active connections: {network.get_connectivity_stats()['total_active_connections']:,}")
            logger.info("")
    
    # Save complete results
    summary = {
        'experiment_type': 'true_multiscale',
        'scales': {
            scale: {
                'final_test_accuracy': results[scale]['history']['test_accuracy'][-1],
                'final_connectivity': results[scale]['history']['connectivity_ratio'][-1],
                'architecture': results[scale]['network'].layer_sizes,
                'total_parameters': results[scale]['network'].get_connectivity_stats()['total_active_connections']
            }
            for scale in results
        }
    }
    
    with open(save_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Experiment completed. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
