"""
Multi-Scale Network Implementation

This module integrates all components to create the complete multi-scale
snapshots network that grows dynamically during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

from ..core.minimal_network import MinimalNetwork
from ..core.growth_scheduler import GrowthScheduler, StructuralLimits
from ..core.connection_router import ConnectionRouter
from ..snapshots.snapshot_manager import SnapshotManager


class MultiScaleNetwork(nn.Module):
    """
    Complete multi-scale network with dynamic growth capabilities.
    
    Integrates:
    - Minimal network with sparse connectivity
    - Growth scheduler with credit system
    - Connection router for extrema-based growth
    - Snapshot manager for multi-scale preservation
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        sparsity: float = 0.0001,
        activation: str = 'tanh',
        device: Optional[torch.device] = None,
        snapshot_dir: str = "snapshots",
        growth_config: Optional[Dict] = None,
        routing_config: Optional[Dict] = None
    ):
        """
        Initialize multi-scale network.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            sparsity: Initial connectivity ratio (0.01% = 0.0001)
            activation: Activation function ('tanh', 'sigmoid', 'relu')
            device: Device to run on
            snapshot_dir: Directory for saving snapshots
            growth_config: Configuration for growth scheduler
            routing_config: Configuration for connection router
        """
        super().__init__()
        
        # Initialize base network
        self.network = MinimalNetwork(
            layer_sizes=layer_sizes,
            sparsity=sparsity,
            activation=activation,
            device=device
        )
        
        # Initialize growth components
        growth_config = growth_config or {}
        self.growth_scheduler = GrowthScheduler(**growth_config)
        self.structural_limits = StructuralLimits()
        
        # Initialize routing
        routing_config = routing_config or {}
        self.connection_router = ConnectionRouter(**routing_config)
        
        # Initialize snapshot manager
        self.snapshot_manager = SnapshotManager(save_dir=snapshot_dir)
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        self.growth_log = []
        
        # Performance tracking
        self.best_performance = None
        self.performance_history = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Move to device
        self.to(self.network.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Perform one training step with potential growth.
        
        Args:
            batch: (inputs, targets) tuple
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Dictionary with training metrics
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.network.device), targets.to(self.network.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Get gradient norm before optimizer step
        gradient_norm = self.network.get_gradient_norm()
        
        # Optimizer step
        optimizer.step()
        
        # Update growth scheduler
        self.growth_scheduler.update_epoch(self.current_epoch)
        should_grow = self.growth_scheduler.add_gradient_norm(gradient_norm)
        
        # Check for growth
        growth_occurred = False
        connections_added = 0
        
        if should_grow and self.structural_limits.can_grow(self.current_epoch):
            growth_occurred, connections_added = self._perform_growth()
        
        # Calculate performance metrics
        with torch.no_grad():
            if targets.dim() == 1:  # Classification
                predictions = outputs.argmax(dim=1)
                accuracy = (predictions == targets).float().mean().item()
                performance = accuracy
            else:  # Regression
                performance = -loss.item()  # Use negative loss as performance
        
        # Update performance history
        self.performance_history.append(performance)
        if self.best_performance is None or performance > self.best_performance:
            self.best_performance = performance
        
        # Check for snapshot saving
        if self.snapshot_manager.should_save_snapshot(
            self.current_epoch, performance, growth_occurred, connections_added
        ):
            self._save_snapshot(performance, growth_occurred, connections_added)
        
        # Record training step
        step_info = {
            'epoch': self.current_epoch,
            'loss': loss.item(),
            'performance': performance,
            'gradient_norm': gradient_norm,
            'growth_occurred': growth_occurred,
            'connections_added': connections_added,
            'total_connections': sum(mask.sum().item() for mask in self.network.connection_masks),
            'phase': self.growth_scheduler.get_current_phase()
        }
        
        self.training_history.append(step_info)
        
        return step_info
    
    def _perform_growth(self) -> Tuple[bool, int]:
        """
        Perform network growth based on extrema detection.
        
        Returns:
            (growth_occurred, connections_added)
        """
        # Detect extrema in current network state with epoch-aware detection
        extrema = self.network.detect_extrema(epoch=self.current_epoch)
        
        if not extrema:
            self.logger.info("No extrema detected, skipping growth")
            return False, 0
        
        # Route new connections
        new_connections = self.connection_router.route_connections(
            extrema, self.network.layer_sizes
        )
        
        if not new_connections:
            self.logger.info("No new connections routed, skipping growth")
            return False, 0
        
        # Apply load balancing
        balanced_connections = self.connection_router.apply_load_balancing(new_connections)
        
        # Add reciprocal connections
        final_connections = self.connection_router.add_reciprocal_connections(balanced_connections)
        
        # Check structural limits
        num_connections = len(final_connections)
        if not self.structural_limits.can_grow(self.current_epoch, num_connections):
            # Reduce connections to fit within limits
            remaining_capacity = self.structural_limits.get_remaining_capacity(self.current_epoch)
            final_connections = final_connections[:remaining_capacity]
            num_connections = len(final_connections)
        
        if num_connections == 0:
            self.logger.info("No connections fit within structural limits")
            return False, 0
        
        # Apply connections to network
        self.network.add_connections(final_connections)
        
        # Record growth
        self.structural_limits.record_growth(self.current_epoch, num_connections)
        
        growth_info = {
            'epoch': self.current_epoch,
            'phase': self.growth_scheduler.get_current_phase(),
            'extrema_detected': extrema,
            'connections_added': num_connections,
            'total_connections': sum(mask.sum().item() for mask in self.network.connection_masks),
            'routing_stats': self.connection_router.get_routing_stats()
        }
        
        self.growth_log.append(growth_info)
        
        self.logger.info(f"Growth performed: {num_connections} connections added at epoch {self.current_epoch}")
        
        return True, num_connections
    
    def _save_snapshot(self, performance: float, growth_occurred: bool, connections_added: int):
        """Save a network snapshot."""
        phase = self.growth_scheduler.get_current_phase()
        
        growth_info = {
            'growth_occurred': growth_occurred,
            'connections_added': connections_added,
            'phase': phase,
            'epoch': self.current_epoch
        }
        
        metadata = {
            'training_step': len(self.training_history),
            'best_performance': self.best_performance,
            'growth_stats': self.get_growth_stats(),
            'connectivity_stats': self.network.get_connectivity_stats()
        }
        
        snapshot_id = self.snapshot_manager.save_snapshot(
            self.network, self.current_epoch, performance, growth_info, phase, metadata
        )
        
        self.logger.info(f"Snapshot saved: {snapshot_id}")
    
    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Epoch statistics
        """
        self.current_epoch = epoch
        self.train()
        
        epoch_stats = {
            'loss': 0.0,
            'performance': 0.0,
            'gradient_norm': 0.0,
            'growth_events': 0,
            'connections_added': 0
        }
        
        num_batches = 0
        
        for batch in dataloader:
            step_info = self.training_step(batch, optimizer, criterion)
            
            # Accumulate statistics
            epoch_stats['loss'] += step_info['loss']
            epoch_stats['performance'] += step_info['performance']
            epoch_stats['gradient_norm'] += step_info['gradient_norm']
            
            if step_info['growth_occurred']:
                epoch_stats['growth_events'] += 1
                epoch_stats['connections_added'] += step_info['connections_added']
            
            num_batches += 1
        
        # Average statistics
        for key in ['loss', 'performance', 'gradient_norm']:
            epoch_stats[key] /= num_batches
        
        # Add epoch-level information
        epoch_stats.update({
            'epoch': epoch,
            'phase': self.growth_scheduler.get_current_phase(),
            'total_connections': sum(mask.sum().item() for mask in self.network.connection_masks),
            'connectivity_ratio': self.network.get_connectivity_stats()['connectivity_ratio']
        })
        
        self.logger.info(
            f"Epoch {epoch}: Loss={epoch_stats['loss']:.4f}, "
            f"Performance={epoch_stats['performance']:.4f}, "
            f"Growth={epoch_stats['growth_events']}, "
            f"Connections={epoch_stats['total_connections']}"
        )
        
        return epoch_stats
    
    def evaluate(self, dataloader, criterion: nn.Module) -> Dict[str, float]:
        """
        Evaluate the network.
        
        Args:
            dataloader: Evaluation data loader
            criterion: Loss function
            
        Returns:
            Evaluation metrics
        """
        self.eval()
        
        total_loss = 0.0
        total_performance = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.network.device), targets.to(self.network.device)
                
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Calculate performance
                if targets.dim() == 1:  # Classification
                    predictions = outputs.argmax(dim=1)
                    accuracy = (predictions == targets).float().mean().item()
                    total_performance += accuracy
                else:  # Regression
                    total_performance += -loss.item()
                
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'performance': total_performance / num_batches
        }
    
    def get_growth_stats(self) -> Dict:
        """Get comprehensive growth statistics."""
        return {
            'scheduler_stats': self.growth_scheduler.get_stats(),
            'structural_limits': self.structural_limits.get_stats(),
            'routing_stats': self.connection_router.get_routing_stats(),
            'snapshot_stats': self.snapshot_manager.get_stats(),
            'network_stats': self.network.get_connectivity_stats(),
            'total_growth_events': len(self.growth_log),
            'current_phase': self.growth_scheduler.get_current_phase()
        }
    
    def get_snapshots(self) -> List[Dict]:
        """Get list of saved snapshots."""
        return self.snapshot_manager.get_snapshot_list()
    
    def load_snapshot(self, snapshot_id: str):
        """Load a specific snapshot."""
        network, metadata = self.snapshot_manager.load_snapshot(snapshot_id, self.network.device)
        self.network = network
        return metadata
    
    def get_phase_snapshots(self, phase: str) -> List[Dict]:
        """Get snapshots for a specific phase."""
        return self.snapshot_manager.get_phase_snapshots(phase)
    
    def reset_growth_state(self):
        """Reset growth-related state (useful for experiments)."""
        self.growth_scheduler.reset()
        self.structural_limits.reset()
        self.connection_router.reset_stats()
        self.growth_log.clear()
        self.training_history.clear()
        self.performance_history.clear()
        self.best_performance = None
        self.current_epoch = 0
    
    def state_dict(self) -> Dict:
        """Get complete state dictionary."""
        return {
            'network': self.network.state_dict_sparse(),
            'growth_scheduler': {
                'gradient_history': list(self.growth_scheduler.gradient_history),
                'credits': self.growth_scheduler.credits,
                'growth_events': self.growth_scheduler.growth_events,
                'last_growth_epoch': self.growth_scheduler.last_growth_epoch,
                'current_epoch': self.growth_scheduler.current_epoch
            },
            'structural_limits': self.structural_limits.get_stats(),
            'training_history': self.training_history,
            'growth_log': self.growth_log,
            'performance_history': self.performance_history,
            'best_performance': self.best_performance,
            'current_epoch': self.current_epoch
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load complete state dictionary."""
        # Load network
        self.network.load_state_dict_sparse(state_dict['network'])
        
        # Load growth scheduler state
        scheduler_state = state_dict['growth_scheduler']
        self.growth_scheduler.gradient_history.extend(scheduler_state['gradient_history'])
        self.growth_scheduler.credits = scheduler_state['credits']
        self.growth_scheduler.growth_events = scheduler_state['growth_events']
        self.growth_scheduler.last_growth_epoch = scheduler_state['last_growth_epoch']
        self.growth_scheduler.current_epoch = scheduler_state['current_epoch']
        
        # Load other state
        self.training_history = state_dict.get('training_history', [])
        self.growth_log = state_dict.get('growth_log', [])
        self.performance_history = state_dict.get('performance_history', [])
        self.best_performance = state_dict.get('best_performance')
        self.current_epoch = state_dict.get('current_epoch', 0)


def create_multi_scale_network(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    sparsity: float = 0.0001,  # Back to specification: 0.01% initial connectivity
    activation: str = 'tanh',
    device: Optional[torch.device] = None,
    snapshot_dir: str = "snapshots"
) -> MultiScaleNetwork:
    """
    Factory function to create a multi-scale network.
    
    Args:
        input_size: Size of input layer
        hidden_sizes: List of hidden layer sizes
        output_size: Size of output layer
        sparsity: Initial connectivity ratio
        activation: Activation function
        device: Device to run on
        snapshot_dir: Directory for snapshots
        
    Returns:
        MultiScaleNetwork instance
    """
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    return MultiScaleNetwork(
        layer_sizes=layer_sizes,
        sparsity=sparsity,
        activation=activation,
        device=device,
        snapshot_dir=snapshot_dir
    )


# Example usage and testing
if __name__ == "__main__":
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create synthetic dataset
    torch.manual_seed(42)
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create multi-scale network
    network = create_multi_scale_network(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        sparsity=0.0001,
        activation='tanh'
    )
    
    print("Created multi-scale network:")
    print(f"Initial connectivity: {network.network.get_connectivity_stats()}")
    
    # Setup training
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few epochs
    for epoch in range(5):
        epoch_stats = network.train_epoch(dataloader, optimizer, criterion, epoch)
        print(f"Epoch {epoch}: {epoch_stats}")
    
    # Print final statistics
    print("\nFinal Growth Statistics:")
    growth_stats = network.get_growth_stats()
    for component, stats in growth_stats.items():
        print(f"{component}: {stats}")
    
    # Print snapshots
    snapshots = network.get_snapshots()
    print(f"\nSnapshots saved: {len(snapshots)}")
    for snapshot in snapshots:
        print(f"  {snapshot['snapshot_id']}: epoch {snapshot['epoch']}, phase {snapshot['phase']}")
