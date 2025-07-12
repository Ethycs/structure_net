"""
Snapshot Orchestrator Component.

Orchestrates the saving and loading of network snapshots during evolution,
implementing intelligent snapshot policies and delta-based storage.
"""

import torch
import torch.nn as nn
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

from ...core import (
    BaseOrchestrator, ComponentContract, ComponentVersion,
    Maturity, ResourceRequirements, ResourceLevel,
    IModel, EvolutionContext, AnalysisReport
)


class SnapshotOrchestrator(BaseOrchestrator):
    """
    Orchestrates snapshot management for evolving networks.
    
    Implements intelligent snapshot policies:
    - Save at growth events
    - Save on significant performance improvements
    - Delta-based storage for efficiency
    - Automatic snapshot pruning
    - Metadata tracking and recovery
    """
    
    def __init__(self,
                 save_dir: str = "snapshots",
                 performance_threshold: float = 0.02,  # 2% improvement
                 max_snapshots: int = 50,
                 use_deltas: bool = True,
                 compression: bool = True,
                 name: str = None):
        """
        Initialize snapshot orchestrator.
        
        Args:
            save_dir: Directory for snapshots
            performance_threshold: Minimum performance improvement to save
            max_snapshots: Maximum snapshots to keep
            use_deltas: Use delta-based storage
            compression: Compress snapshots
            name: Optional custom name
        """
        super().__init__(name or "SnapshotOrchestrator")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_threshold = performance_threshold
        self.max_snapshots = max_snapshots
        self.use_deltas = use_deltas
        self.compression = compression
        
        # Snapshot tracking
        self.snapshots: List[Dict[str, Any]] = []
        self.base_snapshot_id: Optional[str] = None
        self.last_performance: Optional[float] = None
        self.last_architecture: Optional[List[int]] = None
        
        # Statistics
        self.stats = {
            'total_saved': 0,
            'growth_saves': 0,
            'performance_saves': 0,
            'milestone_saves': 0,
            'total_size_mb': 0.0,
            'deltas_saved': 0
        }
        
        # Load existing metadata
        self.metadata_file = self.save_dir / "snapshot_metadata.json"
        self._load_metadata()
        
        # Define contract
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={'model', 'context'},
            optional_inputs={'analysis_report', 'force_save'},
            provided_outputs={
                'snapshot_saved',
                'snapshot_id',
                'snapshot_stats',
                'recovery_info'
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=False  # File I/O
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Return component contract."""
        return self._contract
    
    def run_cycle(self, context: EvolutionContext) -> Dict[str, Any]:
        """
        Run snapshot management cycle.
        
        Args:
            context: Evolution context with model and metrics
            
        Returns:
            Dict with snapshot results
        """
        results = {
            'snapshot_saved': False,
            'snapshot_id': None,
            'reason': None,
            'stats': self.stats.copy()
        }
        
        # Get model and analysis
        model = context.get('model')
        if model is None:
            self.log(logging.WARNING, "No model in context, skipping snapshot")
            return results
        
        # Check if we should save
        should_save, reason = self._should_save_snapshot(context)
        
        if should_save or context.get('force_save', False):
            snapshot_id = self._save_snapshot(model, context, reason)
            results['snapshot_saved'] = True
            results['snapshot_id'] = snapshot_id
            results['reason'] = reason
            
            # Prune old snapshots if needed
            self._prune_snapshots()
        
        results['stats'] = self.stats.copy()
        return results
    
    def _should_save_snapshot(self, context: EvolutionContext) -> Tuple[bool, str]:
        """Determine if snapshot should be saved."""
        epoch = context.epoch
        
        # Check for growth events
        growth_info = context.get('growth_info', {})
        if growth_info.get('growth_occurred', False):
            return True, 'growth_event'
        
        # Check for structural changes
        if 'model' in context:
            model = context['model']
            current_arch = self._get_architecture(model)
            if self.last_architecture and current_arch != self.last_architecture:
                return True, 'structural_change'
        
        # Check performance improvement
        performance = context.get('performance', {})
        current_perf = performance.get('accuracy', performance.get('loss', None))
        
        if current_perf is not None and self.last_performance is not None:
            improvement = current_perf - self.last_performance
            if abs(improvement) > self.performance_threshold:
                return True, f'performance_improvement_{improvement:.4f}'
        
        # Check milestone epochs
        milestone_epochs = [10, 25, 50, 100, 200, 500, 1000]
        if epoch in milestone_epochs:
            # Check if we have a recent snapshot
            recent = any(abs(s['epoch'] - epoch) < 5 for s in self.snapshots[-3:])
            if not recent:
                return True, f'milestone_epoch_{epoch}'
        
        return False, None
    
    def _save_snapshot(self, model: IModel, context: EvolutionContext, 
                      reason: str) -> str:
        """Save a model snapshot."""
        epoch = context.epoch
        timestamp = datetime.now()
        
        # Generate snapshot ID
        snapshot_id = f"snapshot_{epoch:06d}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        snapshot_dir = self.save_dir / snapshot_id
        snapshot_dir.mkdir(exist_ok=True)
        
        # Prepare metadata
        metadata = {
            'snapshot_id': snapshot_id,
            'epoch': epoch,
            'timestamp': timestamp.isoformat(),
            'reason': reason,
            'architecture': self._get_architecture(model),
            'performance': context.get('performance', {}),
            'growth_info': context.get('growth_info', {}),
            'save_type': 'delta' if self.use_deltas and self.base_snapshot_id else 'full'
        }
        
        # Save model state
        if self.use_deltas and self.base_snapshot_id:
            file_size = self._save_delta(model, snapshot_dir, metadata)
            self.stats['deltas_saved'] += 1
        else:
            file_size = self._save_full(model, snapshot_dir, metadata)
            if not self.base_snapshot_id:
                self.base_snapshot_id = snapshot_id
        
        metadata['file_size_mb'] = file_size
        
        # Save metadata
        with open(snapshot_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update tracking
        self.snapshots.append(metadata)
        self.stats['total_saved'] += 1
        self.stats['total_size_mb'] += file_size
        
        # Update reason-specific stats
        if 'growth' in reason:
            self.stats['growth_saves'] += 1
        elif 'performance' in reason:
            self.stats['performance_saves'] += 1
        elif 'milestone' in reason:
            self.stats['milestone_saves'] += 1
        
        # Update last values
        perf = context.get('performance', {})
        self.last_performance = perf.get('accuracy', perf.get('loss'))
        self.last_architecture = self._get_architecture(model)
        
        # Save global metadata
        self._save_metadata()
        
        self.log(logging.INFO, 
                f"Saved snapshot {snapshot_id} (reason: {reason}, size: {file_size:.2f}MB)")
        
        return snapshot_id
    
    def _save_full(self, model: IModel, snapshot_dir: Path, 
                   metadata: Dict[str, Any]) -> float:
        """Save full model snapshot."""
        # Save model state dict
        state_file = snapshot_dir / 'model_state.pt'
        torch.save(model.state_dict(), state_file, _use_new_zipfile_serialization=self.compression)
        
        # Save architecture info
        arch_info = {
            'architecture': self._get_architecture(model),
            'component_info': model.get_architecture_summary() if hasattr(model, 'get_architecture_summary') else {}
        }
        
        with open(snapshot_dir / 'architecture.json', 'w') as f:
            json.dump(arch_info, f, indent=2)
        
        # Calculate size
        file_size = sum(f.stat().st_size for f in snapshot_dir.iterdir()) / (1024 * 1024)
        
        return file_size
    
    def _save_delta(self, model: IModel, snapshot_dir: Path,
                    metadata: Dict[str, Any]) -> float:
        """Save delta from base snapshot."""
        # Load base snapshot
        base_dir = self.save_dir / self.base_snapshot_id
        base_state = torch.load(base_dir / 'model_state.pt')
        
        # Compute deltas
        current_state = model.state_dict()
        deltas = {}
        
        for key in current_state:
            if key in base_state:
                diff = current_state[key] - base_state[key]
                # Only save if changed
                if diff.abs().max() > 1e-8:
                    deltas[key] = diff
            else:
                # New parameter
                deltas[key] = current_state[key]
        
        # Save deltas
        delta_file = snapshot_dir / 'delta_state.pt'
        torch.save({
            'base_snapshot': self.base_snapshot_id,
            'deltas': deltas
        }, delta_file, _use_new_zipfile_serialization=self.compression)
        
        # Save metadata about which keys changed
        with open(snapshot_dir / 'delta_info.json', 'w') as f:
            json.dump({
                'base_snapshot': self.base_snapshot_id,
                'changed_keys': list(deltas.keys()),
                'num_changes': len(deltas)
            }, f, indent=2)
        
        # Calculate size
        file_size = sum(f.stat().st_size for f in snapshot_dir.iterdir()) / (1024 * 1024)
        
        return file_size
    
    def load_snapshot(self, snapshot_id: str, model: IModel) -> Dict[str, Any]:
        """Load a snapshot into a model."""
        snapshot_dir = self.save_dir / snapshot_id
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot {snapshot_id} not found")
        
        # Load metadata
        with open(snapshot_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load model state
        if metadata['save_type'] == 'delta':
            self._load_delta(model, snapshot_dir)
        else:
            self._load_full(model, snapshot_dir)
        
        self.log(logging.INFO, f"Loaded snapshot {snapshot_id}")
        
        return metadata
    
    def _load_full(self, model: IModel, snapshot_dir: Path):
        """Load full snapshot."""
        state_dict = torch.load(snapshot_dir / 'model_state.pt')
        model.load_state_dict(state_dict)
    
    def _load_delta(self, model: IModel, snapshot_dir: Path):
        """Load delta snapshot."""
        # Load delta info
        delta_data = torch.load(snapshot_dir / 'delta_state.pt')
        base_id = delta_data['base_snapshot']
        deltas = delta_data['deltas']
        
        # First load base snapshot
        self._load_full(model, self.save_dir / base_id)
        
        # Apply deltas
        state_dict = model.state_dict()
        for key, delta in deltas.items():
            if key in state_dict:
                state_dict[key] = state_dict[key] + delta
            else:
                state_dict[key] = delta
        
        model.load_state_dict(state_dict)
    
    def _prune_snapshots(self):
        """Remove old snapshots if over limit."""
        if len(self.snapshots) <= self.max_snapshots:
            return
        
        # Sort by importance (keep growth events, milestones, recent)
        def snapshot_priority(snap):
            score = 0
            if 'growth' in snap.get('reason', ''):
                score += 100
            if 'milestone' in snap.get('reason', ''):
                score += 50
            if 'performance' in snap.get('reason', ''):
                score += 25
            # Recency bonus
            score += snap['epoch'] / 10000
            return score
        
        # Sort and keep top snapshots
        sorted_snaps = sorted(self.snapshots, key=snapshot_priority, reverse=True)
        to_remove = sorted_snaps[self.max_snapshots:]
        
        # Remove snapshots
        for snap in to_remove:
            snap_dir = self.save_dir / snap['snapshot_id']
            if snap_dir.exists():
                import shutil
                shutil.rmtree(snap_dir)
                self.log(logging.INFO, f"Pruned snapshot {snap['snapshot_id']}")
        
        # Update list
        self.snapshots = sorted_snaps[:self.max_snapshots]
    
    def _get_architecture(self, model: IModel) -> List[int]:
        """Extract architecture from model."""
        if hasattr(model, 'get_architecture_summary'):
            summary = model.get_architecture_summary()
            return summary.get('architecture', [])
        
        # Fallback: count parameters in each layer
        arch = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                arch.append(module.out_features)
        
        return arch
    
    def _save_metadata(self):
        """Save global metadata."""
        metadata = {
            'snapshots': self.snapshots,
            'base_snapshot_id': self.base_snapshot_id,
            'stats': self.stats,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self):
        """Load existing metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.snapshots = metadata.get('snapshots', [])
                self.base_snapshot_id = metadata.get('base_snapshot_id')
                self.stats.update(metadata.get('stats', {}))
                
                self.log(logging.INFO, f"Loaded {len(self.snapshots)} existing snapshots")
            except Exception as e:
                self.log(logging.WARNING, f"Could not load metadata: {e}")
    
    def get_composition_health(self) -> Dict[str, Any]:
        """Get health status of snapshot system."""
        return {
            'total_snapshots': len(self.snapshots),
            'disk_usage_mb': self.stats['total_size_mb'],
            'compression_ratio': self.stats['deltas_saved'] / max(1, self.stats['total_saved']),
            'snapshot_types': {
                'growth': self.stats['growth_saves'],
                'performance': self.stats['performance_saves'],
                'milestone': self.stats['milestone_saves']
            },
            'base_snapshot': self.base_snapshot_id,
            'health_status': 'healthy' if len(self.snapshots) > 0 else 'no_snapshots'
        }
    
    def get_recovery_info(self) -> List[Dict[str, Any]]:
        """Get information for recovery options."""
        recovery_options = []
        
        for snap in sorted(self.snapshots, key=lambda x: x['epoch'], reverse=True)[:10]:
            recovery_options.append({
                'snapshot_id': snap['snapshot_id'],
                'epoch': snap['epoch'],
                'reason': snap['reason'],
                'performance': snap.get('performance', {}),
                'size_mb': snap.get('file_size_mb', 0),
                'timestamp': snap['timestamp']
            })
        
        return recovery_options