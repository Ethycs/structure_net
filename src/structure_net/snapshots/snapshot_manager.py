"""
Snapshot Manager for Multi-Scale Snapshots Experiment

This module manages the saving and loading of network snapshots at different
growth phases, implementing delta-based storage for efficiency.
"""

import torch
import numpy as np
import os
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime


class SnapshotManager:
    """
    Manages multi-scale network snapshots with delta-based storage.
    
    Implements the snapshot saving rules:
    - Save at each growth event (not arbitrary epochs)
    - Store: weights, structure, growth history
    - Only save if performance improved >2%
    - Or if major structural change occurred
    - Delta-based storage for memory efficiency
    """
    
    def __init__(
        self,
        save_dir: str = "snapshots",
        performance_threshold: float = 0.02,  # 2% improvement
        save_deltas: bool = True,
        compression: bool = True
    ):
        """
        Initialize snapshot manager.
        
        Args:
            save_dir: Directory to save snapshots
            performance_threshold: Minimum performance improvement to save
            save_deltas: Whether to use delta-based storage
            compression: Whether to compress snapshots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_threshold = performance_threshold
        self.save_deltas = save_deltas
        self.compression = compression
        
        # Snapshot tracking
        self.snapshots = []
        self.base_snapshot = None
        self.last_performance = None
        
        # Statistics
        self.save_stats = {
            'total_snapshots': 0,
            'performance_saves': 0,
            'structural_saves': 0,
            'total_size_mb': 0.0
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Create metadata file
        self.metadata_file = self.save_dir / "metadata.json"
        self._load_metadata()
    
    def should_save_snapshot(
        self,
        epoch: int,
        performance: float,
        growth_occurred: bool,
        structural_change_size: int = 0
    ) -> bool:
        """
        Determine if a snapshot should be saved.
        
        Args:
            epoch: Current epoch
            performance: Current performance metric (accuracy, loss, etc.)
            growth_occurred: Whether growth occurred this epoch
            structural_change_size: Number of connections/neurons added
            
        Returns:
            True if snapshot should be saved
        """
        # Always save if growth occurred
        if growth_occurred:
            self.logger.info(f"Saving snapshot at epoch {epoch}: growth occurred")
            return True
        
        # Save if performance improved significantly
        if self.last_performance is not None:
            performance_improvement = performance - self.last_performance
            if performance_improvement > self.performance_threshold:
                self.logger.info(f"Saving snapshot at epoch {epoch}: performance improved by {performance_improvement:.4f}")
                return True
        
        # Save if major structural change
        if structural_change_size > 10:  # Arbitrary threshold for "major"
            self.logger.info(f"Saving snapshot at epoch {epoch}: major structural change ({structural_change_size} connections)")
            return True
        
        # Save at specific milestone epochs
        milestone_epochs = [20, 50, 100, 150, 200]
        if epoch in milestone_epochs:
            self.logger.info(f"Saving snapshot at epoch {epoch}: milestone epoch")
            return True
        
        return False
    
    def save_snapshot(
        self,
        network,
        epoch: int,
        performance: float,
        growth_info: Dict,
        phase: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a network snapshot.
        
        Args:
            network: Network to save
            epoch: Current epoch
            performance: Current performance
            growth_info: Growth information
            phase: Growth phase ('coarse', 'medium', 'fine')
            metadata: Additional metadata
            
        Returns:
            Snapshot ID
        """
        snapshot_id = f"snapshot_{epoch:04d}_{phase}_{datetime.now().strftime('%H%M%S')}"
        snapshot_dir = self.save_dir / snapshot_id
        snapshot_dir.mkdir(exist_ok=True)
        
        # Prepare snapshot data
        snapshot_data = {
            'snapshot_id': snapshot_id,
            'epoch': epoch,
            'performance': performance,
            'phase': phase,
            'growth_info': growth_info,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Save network state
        if self.save_deltas and self.base_snapshot is not None:
            # Save as delta from base snapshot
            network_data = self._create_delta_snapshot(network, snapshot_data)
            save_type = 'delta'
        else:
            # Save full snapshot
            network_data = self._create_full_snapshot(network, snapshot_data)
            save_type = 'full'
            
            # Update base snapshot if this is the first one
            if self.base_snapshot is None:
                self.base_snapshot = snapshot_id
        
        snapshot_data['save_type'] = save_type
        snapshot_data['base_snapshot'] = self.base_snapshot
        
        # Save to disk
        file_size = self._save_snapshot_to_disk(snapshot_dir, network_data, snapshot_data)
        
        # Update tracking
        self.snapshots.append(snapshot_data)
        self.last_performance = performance
        
        # Update statistics
        self.save_stats['total_snapshots'] += 1
        self.save_stats['total_size_mb'] += file_size
        
        if growth_info.get('growth_occurred', False):
            self.save_stats['structural_saves'] += 1
        else:
            self.save_stats['performance_saves'] += 1
        
        # Save metadata
        self._save_metadata()
        
        self.logger.info(f"Saved snapshot {snapshot_id} ({save_type}, {file_size:.2f} MB)")
        
        return snapshot_id
    
    def _create_full_snapshot(self, network, snapshot_data: Dict) -> Dict:
        """Create a full network snapshot."""
        return {
            'state_dict': network.state_dict_sparse(),
            'architecture': {
                'layer_sizes': network.layer_sizes,
                'sparsity': network.sparsity,
                'activation_name': network.activation_name
            },
            'connectivity': {
                'connection_masks': network.connection_masks,
                'connectivity_stats': network.get_connectivity_stats()
            },
            'growth_history': network.growth_history,
            'extrema_history': network.extrema_history
        }
    
    def _create_delta_snapshot(self, network, snapshot_data: Dict) -> Dict:
        """Create a delta snapshot relative to base snapshot."""
        if self.base_snapshot is None:
            return self._create_full_snapshot(network, snapshot_data)
        
        # Load base snapshot for comparison
        base_data = self._load_snapshot_data(self.base_snapshot)
        base_state = base_data['state_dict']
        current_state = network.state_dict_sparse()
        
        # Compute deltas
        weight_deltas = {}
        for key in current_state:
            if key in base_state and isinstance(current_state[key], torch.Tensor):
                delta = current_state[key] - base_state[key]
                # Only store non-zero deltas
                if delta.abs().sum() > 1e-8:
                    weight_deltas[key] = delta
        
        # Compute connectivity deltas
        connectivity_deltas = self._compute_connectivity_deltas(
            base_data['connectivity']['connection_masks'],
            network.connection_masks
        )
        
        return {
            'weight_deltas': weight_deltas,
            'connectivity_deltas': connectivity_deltas,
            'new_growth_events': network.growth_history[len(base_data.get('growth_history', [])):],
            'architecture': {
                'layer_sizes': network.layer_sizes,
                'sparsity': network.sparsity,
                'activation_name': network.activation_name
            }
        }
    
    def _compute_connectivity_deltas(self, base_masks: List, current_masks: List) -> Dict:
        """Compute connectivity changes between snapshots."""
        deltas = {}
        
        for i, (base_mask, current_mask) in enumerate(zip(base_masks, current_masks)):
            # Find new connections
            new_connections = current_mask & (~base_mask)
            removed_connections = base_mask & (~current_mask)
            
            if new_connections.any() or removed_connections.any():
                deltas[f'layer_{i}'] = {
                    'new_connections': new_connections.nonzero(as_tuple=True),
                    'removed_connections': removed_connections.nonzero(as_tuple=True)
                }
        
        return deltas
    
    def _save_snapshot_to_disk(self, snapshot_dir: Path, network_data: Dict, snapshot_data: Dict) -> float:
        """Save snapshot data to disk and return file size in MB."""
        # Save network data
        network_file = snapshot_dir / "network.pkl"
        with open(network_file, 'wb') as f:
            if self.compression:
                import gzip
                with gzip.open(network_file.with_suffix('.pkl.gz'), 'wb') as gz_f:
                    pickle.dump(network_data, gz_f)
                network_file.unlink()  # Remove uncompressed version
                network_file = network_file.with_suffix('.pkl.gz')
            else:
                pickle.dump(network_data, f)
        
        # Save metadata
        metadata_file = snapshot_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            # Convert non-serializable objects
            serializable_data = self._make_serializable(snapshot_data)
            json.dump(serializable_data, f, indent=2)
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in snapshot_dir.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, '__dict__'):
            return str(data)  # Convert objects to string representation
        else:
            return data
    
    def load_snapshot(self, snapshot_id: str, device: Optional[torch.device] = None):
        """
        Load a network snapshot.
        
        Args:
            snapshot_id: ID of snapshot to load
            device: Device to load tensors to
            
        Returns:
            Loaded network
        """
        snapshot_dir = self.save_dir / snapshot_id
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        
        # Load metadata
        metadata_file = snapshot_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load network data
        network_file = snapshot_dir / "network.pkl"
        if not network_file.exists():
            network_file = snapshot_dir / "network.pkl.gz"
            if network_file.exists():
                import gzip
                with gzip.open(network_file, 'rb') as f:
                    network_data = pickle.load(f)
            else:
                raise FileNotFoundError(f"Network data not found for snapshot {snapshot_id}")
        else:
            with open(network_file, 'rb') as f:
                network_data = pickle.load(f)
        
        # Reconstruct network
        if metadata['save_type'] == 'full':
            network = self._reconstruct_from_full(network_data, device)
        else:  # delta
            network = self._reconstruct_from_delta(network_data, metadata, device)
        
        return network, metadata
    
    def _reconstruct_from_full(self, network_data: Dict, device: Optional[torch.device]):
        """Reconstruct network from full snapshot."""
        from ..core.minimal_network import MinimalNetwork
        
        arch = network_data['architecture']
        network = MinimalNetwork(
            layer_sizes=arch['layer_sizes'],
            sparsity=arch['sparsity'],
            activation=arch['activation_name'],
            device=device
        )
        
        # Load state
        network.load_state_dict_sparse(network_data['state_dict'])
        network.connection_masks = network_data['connectivity']['connection_masks']
        
        return network
    
    def _reconstruct_from_delta(self, network_data: Dict, metadata: Dict, device: Optional[torch.device]):
        """Reconstruct network from delta snapshot."""
        # Load base snapshot
        base_network, _ = self.load_snapshot(metadata['base_snapshot'], device)
        
        # Apply weight deltas
        with torch.no_grad():
            for key, delta in network_data['weight_deltas'].items():
                if hasattr(base_network, key.split('.')[0]):
                    param = base_network.state_dict()[key]
                    param += delta.to(param.device)
        
        # Apply connectivity deltas
        for layer_key, deltas in network_data['connectivity_deltas'].items():
            layer_idx = int(layer_key.split('_')[1])
            mask = base_network.connection_masks[layer_idx]
            
            # Add new connections
            if 'new_connections' in deltas:
                new_conn = deltas['new_connections']
                mask[new_conn] = True
            
            # Remove connections
            if 'removed_connections' in deltas:
                removed_conn = deltas['removed_connections']
                mask[removed_conn] = False
        
        # Update growth history
        base_network.growth_history.extend(network_data.get('new_growth_events', []))
        
        return base_network
    
    def _load_snapshot_data(self, snapshot_id: str) -> Dict:
        """Load raw snapshot data."""
        snapshot_dir = self.save_dir / snapshot_id
        network_file = snapshot_dir / "network.pkl"
        
        if not network_file.exists():
            network_file = snapshot_dir / "network.pkl.gz"
            import gzip
            with gzip.open(network_file, 'rb') as f:
                return pickle.load(f)
        else:
            with open(network_file, 'rb') as f:
                return pickle.load(f)
    
    def get_snapshot_list(self) -> List[Dict]:
        """Get list of all snapshots with metadata."""
        return [
            {
                'snapshot_id': s['snapshot_id'],
                'epoch': s['epoch'],
                'phase': s['phase'],
                'performance': s['performance'],
                'timestamp': s['timestamp']
            }
            for s in self.snapshots
        ]
    
    def get_phase_snapshots(self, phase: str) -> List[Dict]:
        """Get snapshots for a specific phase."""
        return [s for s in self.snapshots if s['phase'] == phase]
    
    def get_stats(self) -> Dict:
        """Get snapshot manager statistics."""
        return {
            **self.save_stats,
            'snapshots_by_phase': {
                phase: len(self.get_phase_snapshots(phase))
                for phase in ['coarse', 'medium', 'fine']
            },
            'average_size_mb': self.save_stats['total_size_mb'] / max(1, self.save_stats['total_snapshots']),
            'base_snapshot': self.base_snapshot
        }
    
    def _save_metadata(self):
        """Save manager metadata to disk."""
        metadata = {
            'snapshots': self.snapshots,
            'base_snapshot': self.base_snapshot,
            'save_stats': self.save_stats,
            'last_performance': self.last_performance
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self._make_serializable(metadata), f, indent=2)
    
    def _load_metadata(self):
        """Load manager metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                
            self.snapshots = metadata.get('snapshots', [])
            self.base_snapshot = metadata.get('base_snapshot')
            self.save_stats = metadata.get('save_stats', self.save_stats)
            self.last_performance = metadata.get('last_performance')
    
    def cleanup_old_snapshots(self, keep_latest: int = 10):
        """Remove old snapshots to save disk space."""
        if len(self.snapshots) <= keep_latest:
            return
        
        # Sort by epoch
        sorted_snapshots = sorted(self.snapshots, key=lambda x: x['epoch'])
        
        # Keep latest snapshots and base snapshot
        to_remove = sorted_snapshots[:-keep_latest]
        
        for snapshot in to_remove:
            if snapshot['snapshot_id'] != self.base_snapshot:
                snapshot_dir = self.save_dir / snapshot['snapshot_id']
                if snapshot_dir.exists():
                    import shutil
                    shutil.rmtree(snapshot_dir)
                    self.logger.info(f"Removed old snapshot: {snapshot['snapshot_id']}")
        
        # Update snapshot list
        self.snapshots = [s for s in self.snapshots if s not in to_remove or s['snapshot_id'] == self.base_snapshot]
        self._save_metadata()


# Example usage and testing
if __name__ == "__main__":
    from ..core.minimal_network import create_minimal_network
    
    # Create test network
    network = create_minimal_network(784, [256, 128], 10)
    
    # Create snapshot manager
    manager = SnapshotManager("test_snapshots")
    
    # Simulate training with snapshots
    for epoch in range(0, 101, 25):
        # Simulate performance improvement
        performance = 0.5 + 0.4 * (epoch / 100) + 0.1 * np.random.random()
        
        # Simulate growth
        growth_occurred = epoch % 50 == 0 and epoch > 0
        growth_info = {
            'growth_occurred': growth_occurred,
            'connections_added': 5 if growth_occurred else 0
        }
        
        # Determine phase
        if epoch < 50:
            phase = 'coarse'
        elif epoch < 100:
            phase = 'medium'
        else:
            phase = 'fine'
        
        # Check if should save
        if manager.should_save_snapshot(epoch, performance, growth_occurred):
            snapshot_id = manager.save_snapshot(
                network, epoch, performance, growth_info, phase,
                metadata={'test_run': True}
            )
            print(f"Saved snapshot: {snapshot_id}")
    
    # Print statistics
    print("\nSnapshot Manager Statistics:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test loading
    snapshots = manager.get_snapshot_list()
    if snapshots:
        latest_snapshot = snapshots[-1]
        print(f"\nLoading snapshot: {latest_snapshot['snapshot_id']}")
        
        loaded_network, metadata = manager.load_snapshot(latest_snapshot['snapshot_id'])
        print(f"Loaded network with {len(loaded_network.layers)} layers")
        print(f"Connectivity: {loaded_network.get_connectivity_stats()}")
