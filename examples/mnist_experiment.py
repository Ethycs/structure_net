"""
MNIST Experiment for Multi-Scale Snapshots Network

This script demonstrates the complete multi-scale network on MNIST classification,
showing how the network grows from minimal connectivity to full performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.structure_net.models.multi_scale_network import create_multi_scale_network


def setup_logging(log_file: str = "mnist_experiment.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_mnist_data(batch_size: int = 64, data_dir: str = "./data"):
    """
    Load MNIST dataset.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
        
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader


def create_experiment_network(
    sparsity: float = 0.0001,
    activation: str = 'tanh',
    hidden_sizes: list = None,
    device: torch.device = None
):
    """
    Create network for MNIST experiment.
    
    Args:
        sparsity: Initial connectivity ratio
        activation: Activation function
        hidden_sizes: Hidden layer sizes
        device: Device to run on
        
    Returns:
        MultiScaleNetwork instance
    """
    if hidden_sizes is None:
        hidden_sizes = [256, 128]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    network = create_multi_scale_network(
        input_size=784,  # 28x28 flattened
        hidden_sizes=hidden_sizes,
        output_size=10,  # 10 classes
        sparsity=sparsity,
        activation=activation,
        device=device,
        snapshot_dir="mnist_snapshots"
    )
    
    return network


def train_experiment(
    network,
    train_loader,
    test_loader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device = None
):
    """
    Train the multi-scale network on MNIST.
    
    Args:
        network: MultiScaleNetwork instance
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to run on
        
    Returns:
        Training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    network = network.to(device)
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'growth_events': [],
        'connectivity_ratio': [],
        'phase_transitions': []
    }
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Initial connectivity: {network.network.get_connectivity_stats()}")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Train epoch
        epoch_stats = network.train_epoch(train_loader, optimizer, criterion, epoch)
        
        # Evaluate on test set
        test_stats = network.evaluate(test_loader, criterion)
        
        # Record history
        history['train_loss'].append(epoch_stats['loss'])
        history['train_accuracy'].append(epoch_stats['performance'])
        history['test_loss'].append(test_stats['loss'])
        history['test_accuracy'].append(test_stats['performance'])
        history['connectivity_ratio'].append(epoch_stats['connectivity_ratio'])
        
        # Record growth events
        if epoch_stats['growth_events'] > 0:
            history['growth_events'].append({
                'epoch': epoch,
                'connections_added': epoch_stats['connections_added'],
                'phase': epoch_stats['phase']
            })
        
        # Record phase transitions
        if epoch > 0:
            prev_phase = history.get('last_phase', 'coarse')
            current_phase = epoch_stats['phase']
            if prev_phase != current_phase:
                history['phase_transitions'].append({
                    'epoch': epoch,
                    'from_phase': prev_phase,
                    'to_phase': current_phase
                })
        history['last_phase'] = epoch_stats['phase']
        
        # Log progress
        if epoch % 10 == 0 or epoch_stats['growth_events'] > 0:
            logger.info(
                f"Epoch {epoch}: Train Acc={epoch_stats['performance']:.4f}, "
                f"Test Acc={test_stats['performance']:.4f}, "
                f"Growth={epoch_stats['growth_events']}, "
                f"Connections={epoch_stats['total_connections']}, "
                f"Phase={epoch_stats['phase']}"
            )
    
    logger.info("Training completed")
    
    return history


def analyze_results(network, history, save_dir: str = "results"):
    """
    Analyze and visualize experiment results.
    
    Args:
        network: Trained network
        history: Training history
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    
    # Get final statistics
    growth_stats = network.get_growth_stats()
    snapshots = network.get_snapshots()
    
    # Save statistics
    with open(save_dir / "growth_stats.json", 'w') as f:
        json.dump(growth_stats, f, indent=2, default=str)
    
    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    # Create visualizations
    create_training_plots(history, save_dir)
    create_growth_plots(network, history, save_dir)
    create_connectivity_plots(network, history, save_dir)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*50)
    
    final_connectivity = network.network.get_connectivity_stats()
    logger.info(f"Initial sparsity: {network.network.sparsity:.6f}")
    logger.info(f"Final connectivity ratio: {final_connectivity['connectivity_ratio']:.6f}")
    logger.info(f"Final test accuracy: {history['test_accuracy'][-1]:.4f}")
    logger.info(f"Total growth events: {len(history['growth_events'])}")
    logger.info(f"Total snapshots saved: {len(snapshots)}")
    
    # Phase analysis
    phase_snapshots = {
        'coarse': network.get_phase_snapshots('coarse'),
        'medium': network.get_phase_snapshots('medium'),
        'fine': network.get_phase_snapshots('fine')
    }
    
    for phase, snaps in phase_snapshots.items():
        logger.info(f"{phase.capitalize()} phase snapshots: {len(snaps)}")
    
    logger.info("="*50)


def create_training_plots(history, save_dir):
    """Create training progress plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(len(history['train_loss']))
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', alpha=0.7)
    axes[0, 0].plot(epochs, history['test_loss'], label='Test Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_accuracy'], label='Train Accuracy', alpha=0.7)
    axes[0, 1].plot(epochs, history['test_accuracy'], label='Test Accuracy', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Test Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Connectivity ratio
    axes[1, 0].plot(epochs, history['connectivity_ratio'], color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Connectivity Ratio')
    axes[1, 0].set_title('Network Connectivity Growth')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Growth events
    if history['growth_events']:
        growth_epochs = [event['epoch'] for event in history['growth_events']]
        growth_connections = [event['connections_added'] for event in history['growth_events']]
        
        axes[1, 1].scatter(growth_epochs, growth_connections, alpha=0.7, s=50)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Connections Added')
        axes[1, 1].set_title('Growth Events')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Mark phase transitions
    for ax in axes.flat:
        for transition in history.get('phase_transitions', []):
            ax.axvline(x=transition['epoch'], color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_progress.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_growth_plots(network, history, save_dir):
    """Create growth-specific plots."""
    growth_stats = network.get_growth_stats()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Growth events by phase
    phase_counts = {'coarse': 0, 'medium': 0, 'fine': 0}
    for event in history['growth_events']:
        phase_counts[event['phase']] += 1
    
    phases = list(phase_counts.keys())
    counts = list(phase_counts.values())
    
    axes[0, 0].bar(phases, counts, alpha=0.7)
    axes[0, 0].set_title('Growth Events by Phase')
    axes[0, 0].set_ylabel('Number of Events')
    
    # Connections added over time
    if history['growth_events']:
        growth_epochs = [event['epoch'] for event in history['growth_events']]
        cumulative_connections = np.cumsum([event['connections_added'] for event in history['growth_events']])
        
        axes[0, 1].plot(growth_epochs, cumulative_connections, marker='o', alpha=0.7)
        axes[0, 1].set_title('Cumulative Connections Added')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Total Connections Added')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Structural limits
    limits_stats = growth_stats['structural_limits']
    if 'current_counts' in limits_stats:
        phases = list(limits_stats['current_counts'].keys())
        current = [limits_stats['current_counts'][phase] for phase in phases]
        limits = [limits_stats['limits'][phase] for phase in phases]
        
        x = np.arange(len(phases))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, current, width, label='Current', alpha=0.7)
        axes[1, 0].bar(x + width/2, limits, width, label='Limit', alpha=0.7)
        axes[1, 0].set_title('Structural Limits Usage')
        axes[1, 0].set_xlabel('Phase')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(phases)
        axes[1, 0].legend()
    
    # Snapshot timeline
    snapshots = network.get_snapshots()
    if snapshots:
        snapshot_epochs = [s['epoch'] for s in snapshots]
        snapshot_phases = [s['phase'] for s in snapshots]
        
        phase_colors = {'coarse': 'blue', 'medium': 'orange', 'fine': 'green'}
        colors = [phase_colors[phase] for phase in snapshot_phases]
        
        axes[1, 1].scatter(snapshot_epochs, range(len(snapshot_epochs)), c=colors, alpha=0.7)
        axes[1, 1].set_title('Snapshot Timeline')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Snapshot Index')
        
        # Add legend
        for phase, color in phase_colors.items():
            axes[1, 1].scatter([], [], c=color, label=phase.capitalize())
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "growth_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_connectivity_plots(network, history, save_dir):
    """Create connectivity analysis plots."""
    connectivity_stats = network.network.get_connectivity_stats()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Connectivity evolution
    epochs = range(len(history['connectivity_ratio']))
    axes[0, 0].plot(epochs, history['connectivity_ratio'], alpha=0.7)
    axes[0, 0].set_title('Connectivity Ratio Evolution')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Connectivity Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Layer-wise connectivity (if available)
    if hasattr(network.network, 'connection_masks'):
        layer_connectivity = []
        for i, mask in enumerate(network.network.connection_masks):
            total_possible = mask.numel()
            total_active = mask.sum().item()
            ratio = total_active / total_possible
            layer_connectivity.append(ratio)
        
        axes[0, 1].bar(range(len(layer_connectivity)), layer_connectivity, alpha=0.7)
        axes[0, 1].set_title('Layer-wise Connectivity')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Connectivity Ratio')
    
    # Growth rate analysis
    if len(history['connectivity_ratio']) > 1:
        growth_rates = np.diff(history['connectivity_ratio'])
        axes[1, 0].plot(range(1, len(history['connectivity_ratio'])), growth_rates, alpha=0.7)
        axes[1, 0].set_title('Connectivity Growth Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Growth Rate')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Final connectivity statistics
    stats_text = f"""Final Connectivity Statistics:
    
Total Possible: {connectivity_stats['total_possible_connections']:,}
Total Active: {connectivity_stats['total_active_connections']:,}
Connectivity Ratio: {connectivity_stats['connectivity_ratio']:.6f}
Sparsity: {connectivity_stats['sparsity']:.6f}

Growth Factor: {connectivity_stats['connectivity_ratio'] / network.network.sparsity:.1f}x
"""
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title('Final Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / "connectivity_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='MNIST Multi-Scale Network Experiment')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sparsity', type=float, default=0.0001, help='Initial sparsity')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'sigmoid', 'relu'])
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 128], help='Hidden layer sizes')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results-dir', type=str, default='./results', help='Results directory')
    parser.add_argument('--log-file', type=str, default='mnist_experiment.log', help='Log file')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(args.batch_size, args.data_dir)
    logger.info(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Create network
    logger.info("Creating multi-scale network...")
    network = create_experiment_network(
        sparsity=args.sparsity,
        activation=args.activation,
        hidden_sizes=args.hidden_sizes,
        device=device
    )
    
    logger.info(f"Network architecture: {network.network.layer_sizes}")
    logger.info(f"Initial connectivity: {network.network.get_connectivity_stats()}")
    
    # Train
    logger.info("Starting training...")
    history = train_experiment(
        network, train_loader, test_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )
    
    # Analyze results
    logger.info("Analyzing results...")
    analyze_results(network, history, args.results_dir)
    
    logger.info(f"Experiment completed. Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()
