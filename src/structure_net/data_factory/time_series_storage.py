#!/usr/bin/env python3
"""
Time Series Storage for Structure Net

Provides efficient storage for large time series data like training histories,
keeping them out of memory while maintaining quick access.
"""

import json
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import h5py


@dataclass
class TimeSeriesConfig:
    """Configuration for time series storage."""
    storage_dir: str = "/data/timeseries"
    compression: str = "gzip"  # none, gzip, lzf
    chunk_size: int = 1000  # Chunk size for HDF5
    max_in_memory_cache: int = 100  # Max time series to keep in memory
    use_hdf5: bool = True  # Use HDF5 for numeric data
    use_json: bool = False  # Use JSON for small data (fallback)


class TimeSeriesStorage:
    """
    Efficient storage for time series data from experiments.
    
    Uses HDF5 for large numeric arrays and compressed JSON for metadata.
    Keeps minimal data in memory while providing fast access.
    """
    
    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()
        self.storage_path = Path(self.config.storage_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Memory cache for recently accessed data
        self.cache = {}
        self.cache_order = []  # LRU tracking
        
        # HDF5 file handles (kept open for performance)
        self.hdf5_files = {}
        
    def store_training_history(
        self,
        experiment_id: str,
        epoch_data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store training history efficiently.
        
        Args:
            experiment_id: Unique experiment identifier
            epoch_data: List of per-epoch metrics
            metadata: Additional metadata about the training
            
        Returns:
            Storage key for retrieval
        """
        storage_key = f"training_{experiment_id}"
        
        if len(epoch_data) > 50 and self.config.use_hdf5:
            # Large dataset - use HDF5
            return self._store_hdf5(storage_key, epoch_data, metadata)
        else:
            # Small dataset - use compressed JSON
            return self._store_json(storage_key, epoch_data, metadata)
    
    def _store_hdf5(
        self,
        storage_key: str,
        epoch_data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store large time series in HDF5 format."""
        file_path = self.storage_path / f"{storage_key}.h5"
        
        with h5py.File(file_path, 'w') as f:
            # Extract numeric columns
            if epoch_data:
                # Get all numeric keys
                numeric_keys = []
                for key, value in epoch_data[0].items():
                    if isinstance(value, (int, float)):
                        numeric_keys.append(key)
                
                # Create datasets for each numeric column
                for key in numeric_keys:
                    data = [epoch.get(key, np.nan) for epoch in epoch_data]
                    f.create_dataset(
                        key,
                        data=data,
                        chunks=(min(self.config.chunk_size, len(data)),),
                        compression=self.config.compression if self.config.compression != 'none' else None
                    )
                
                # Store non-numeric data separately
                non_numeric_data = []
                for epoch in epoch_data:
                    non_numeric = {k: v for k, v in epoch.items() if k not in numeric_keys}
                    if non_numeric:
                        non_numeric_data.append(non_numeric)
                
                if non_numeric_data:
                    f.attrs['non_numeric_data'] = json.dumps(non_numeric_data)
            
            # Store metadata
            if metadata:
                f.attrs['metadata'] = json.dumps(metadata)
            
            f.attrs['storage_time'] = datetime.now().isoformat()
            f.attrs['num_epochs'] = len(epoch_data)
        
        return storage_key
    
    def _store_json(
        self,
        storage_key: str,
        epoch_data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store small time series as compressed JSON."""
        file_path = self.storage_path / f"{storage_key}.json.gz"
        
        data = {
            'epochs': epoch_data,
            'metadata': metadata or {},
            'storage_time': datetime.now().isoformat(),
            'num_epochs': len(epoch_data)
        }
        
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return storage_key
    
    def retrieve_training_history(
        self,
        storage_key: str,
        epochs: Optional[Union[int, slice, List[int]]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve training history.
        
        Args:
            storage_key: Storage key returned by store_training_history
            epochs: Optional epoch selection (int, slice, or list of indices)
            
        Returns:
            Tuple of (epoch_data, metadata)
        """
        # Check cache first
        if storage_key in self.cache:
            self._update_cache_lru(storage_key)
            data, metadata = self.cache[storage_key]
            return self._filter_epochs(data, epochs), metadata
        
        # Try HDF5 first
        hdf5_path = self.storage_path / f"{storage_key}.h5"
        if hdf5_path.exists():
            data, metadata = self._retrieve_hdf5(storage_key, epochs)
        else:
            # Try JSON
            json_path = self.storage_path / f"{storage_key}.json.gz"
            if json_path.exists():
                data, metadata = self._retrieve_json(storage_key, epochs)
            else:
                raise ValueError(f"No data found for key: {storage_key}")
        
        # Update cache
        self._add_to_cache(storage_key, (data, metadata))
        
        return data, metadata
    
    def _retrieve_hdf5(
        self,
        storage_key: str,
        epochs: Optional[Union[int, slice, List[int]]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Retrieve from HDF5 storage."""
        file_path = self.storage_path / f"{storage_key}.h5"
        
        with h5py.File(file_path, 'r') as f:
            # Get metadata
            metadata = json.loads(f.attrs.get('metadata', '{}'))
            
            # Get numeric data
            epoch_data = []
            
            # Find all datasets (numeric columns)
            numeric_keys = [key for key in f.keys()]
            
            if numeric_keys:
                # Determine which epochs to load
                num_epochs = len(f[numeric_keys[0]])
                if epochs is None:
                    epoch_indices = list(range(num_epochs))
                elif isinstance(epochs, int):
                    epoch_indices = [epochs]
                elif isinstance(epochs, slice):
                    epoch_indices = list(range(num_epochs))[epochs]
                else:
                    epoch_indices = epochs
                
                # Load data for selected epochs
                for i in epoch_indices:
                    epoch_dict = {}
                    for key in numeric_keys:
                        epoch_dict[key] = float(f[key][i])
                    epoch_data.append(epoch_dict)
                
                # Add non-numeric data if present
                non_numeric_json = f.attrs.get('non_numeric_data')
                if non_numeric_json:
                    non_numeric_data = json.loads(non_numeric_json)
                    for i, idx in enumerate(epoch_indices):
                        if idx < len(non_numeric_data):
                            epoch_data[i].update(non_numeric_data[idx])
        
        return epoch_data, metadata
    
    def _retrieve_json(
        self,
        storage_key: str,
        epochs: Optional[Union[int, slice, List[int]]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Retrieve from JSON storage."""
        file_path = self.storage_path / f"{storage_key}.json.gz"
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        epoch_data = data['epochs']
        metadata = data.get('metadata', {})
        
        return self._filter_epochs(epoch_data, epochs), metadata
    
    def _filter_epochs(
        self,
        epoch_data: List[Dict[str, Any]],
        epochs: Optional[Union[int, slice, List[int]]] = None
    ) -> List[Dict[str, Any]]:
        """Filter epochs based on selection."""
        if epochs is None:
            return epoch_data
        elif isinstance(epochs, int):
            return [epoch_data[epochs]]
        elif isinstance(epochs, slice):
            return epoch_data[epochs]
        else:  # List of indices
            return [epoch_data[i] for i in epochs]
    
    def _add_to_cache(self, key: str, data: Any):
        """Add to LRU cache."""
        if key in self.cache:
            self.cache_order.remove(key)
        
        self.cache[key] = data
        self.cache_order.append(key)
        
        # Evict if over limit
        while len(self.cache) > self.config.max_in_memory_cache:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
    
    def _update_cache_lru(self, key: str):
        """Update LRU order for cache hit."""
        self.cache_order.remove(key)
        self.cache_order.append(key)
    
    def get_summary_statistics(self, storage_key: str) -> Dict[str, Any]:
        """
        Get summary statistics without loading full data.
        
        Useful for ChromaDB metadata.
        """
        hdf5_path = self.storage_path / f"{storage_key}.h5"
        
        if hdf5_path.exists():
            with h5py.File(hdf5_path, 'r') as f:
                stats = {
                    'num_epochs': f.attrs.get('num_epochs', 0),
                    'storage_time': f.attrs.get('storage_time', ''),
                }
                
                # Calculate statistics for numeric columns
                for key in f.keys():
                    data = f[key][:]
                    stats[f'{key}_final'] = float(data[-1])
                    stats[f'{key}_max'] = float(np.max(data))
                    stats[f'{key}_min'] = float(np.min(data))
                    stats[f'{key}_mean'] = float(np.mean(data))
                
                return stats
        else:
            # Fallback to loading JSON
            _, metadata = self.retrieve_training_history(storage_key)
            return metadata
    
    def delete_training_history(self, storage_key: str):
        """Delete stored training history."""
        # Remove from cache
        if storage_key in self.cache:
            del self.cache[storage_key]
            self.cache_order.remove(storage_key)
        
        # Delete files
        for suffix in ['.h5', '.json.gz']:
            file_path = self.storage_path / f"{storage_key}{suffix}"
            if file_path.exists():
                file_path.unlink()
    
    def close(self):
        """Close any open file handles."""
        for f in self.hdf5_files.values():
            f.close()
        self.hdf5_files.clear()


class HybridExperimentStorage:
    """
    Hybrid storage combining ChromaDB for search and time series storage for large data.
    """
    
    def __init__(
        self,
        chroma_config: Optional[Any] = None,
        timeseries_config: Optional[TimeSeriesConfig] = None
    ):
        from data_factory.search import ExperimentSearcher
        
        self.searcher = ExperimentSearcher(chroma_config)
        self.timeseries = TimeSeriesStorage(timeseries_config)
    
    def store_experiment(
        self,
        experiment_id: str,
        experiment_data: Dict[str, Any],
        training_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Store experiment with hybrid approach.
        
        Large time series data goes to specialized storage,
        searchable metadata goes to ChromaDB.
        """
        # Extract training history if present
        if 'training_history' in experiment_data and len(experiment_data['training_history']) > 50:
            training_history = experiment_data['training_history']
            experiment_data = experiment_data.copy()
            del experiment_data['training_history']
        
        # Store time series data if provided
        timeseries_key = None
        if training_history and len(training_history) > 10:
            timeseries_key = self.timeseries.store_training_history(
                experiment_id,
                training_history,
                metadata={
                    'experiment_id': experiment_id,
                    'num_epochs': len(training_history)
                }
            )
            
            # Add summary statistics to searchable data
            stats = self.timeseries.get_summary_statistics(timeseries_key)
            experiment_data['training_summary'] = stats
            experiment_data['timeseries_key'] = timeseries_key
        
        # Store in ChromaDB for search
        self.searcher.index_experiment(
            experiment_id=experiment_id,
            experiment_data=experiment_data
        )
        
        return experiment_id
    
    def retrieve_experiment(
        self,
        experiment_id: str,
        include_training_history: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve experiment data.
        
        Args:
            experiment_id: Experiment ID
            include_training_history: Whether to load full training history
            
        Returns:
            Complete experiment data
        """
        # Get from ChromaDB
        client = self.searcher.client
        result = client.get_experiment(experiment_id)
        
        if not result:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment_data = result['metadata']
        
        # Load training history if requested
        if include_training_history and 'timeseries_key' in experiment_data:
            history, _ = self.timeseries.retrieve_training_history(
                experiment_data['timeseries_key']
            )
            experiment_data['training_history'] = history
        
        return experiment_data