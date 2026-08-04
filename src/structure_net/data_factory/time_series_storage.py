#!/usr/bin/env python3
"""
Time Series Storage for Structure Net

Provides efficient storage for large time series data like training histories,
keeping them out of memory while maintaining quick access.
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import h5py


@dataclass
class TimeSeriesConfig:
    """Configuration for time series storage."""
    storage_dir: str = field(
        default_factory=lambda: str(Path(__file__).resolve().parents[1] / "data" / "timeseries_db")
    )
    compression: Optional[str] = "gzip"  # none, gzip, lzf
    chunk_size: int = 1000  # Chunk size for HDF5
    max_in_memory_cache: int = 100  # Max time series to keep in memory
    use_hdf5: bool = True  # Use HDF5 for numeric data
    use_json: bool = False  # Use JSON for small data (fallback)
    hdf5_threshold: int = 50


class TimeSeriesKey(str):
    """String storage key with path conveniences used by the legacy API."""

    def __new__(cls, key: str, path: Path):
        instance = super().__new__(cls, key)
        instance.path = path
        return instance

    def exists(self) -> bool:
        return self.path.exists()

    @property
    def suffix(self) -> str:
        if self.path.name.endswith('.json.gz'):
            return '.json.gz'
        return self.path.suffix

    def __fspath__(self) -> str:
        return str(self.path)


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
        self.epochs_threshold = self.config.hdf5_threshold
        
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
        
        if len(epoch_data) >= self.epochs_threshold and self.config.use_hdf5:
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
        
        compression = self.config.compression
        if compression in (None, 'none'):
            compression = None

        with h5py.File(file_path, 'w') as f:
            # Store the canonical history representation so mixed numeric and
            # structured values round-trip without losing types or alignment.
            history_json = [json.dumps(epoch, default=_json_default) for epoch in epoch_data]
            string_type = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(
                'history',
                data=np.asarray(history_json, dtype=object),
                dtype=string_type,
                chunks=(min(self.config.chunk_size, max(len(history_json), 1)),),
                compression=compression,
            )

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
                        f'metric_{key}',
                        data=data,
                        chunks=(min(self.config.chunk_size, len(data)),),
                        compression=compression,
                    )
            
            # Store metadata
            if metadata:
                f.attrs['metadata'] = json.dumps(metadata)
            
            f.attrs['storage_time'] = datetime.now().isoformat()
            f.attrs['num_epochs'] = len(epoch_data)
        
        return TimeSeriesKey(storage_key, file_path)
    
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
            json.dump(data, f, indent=2, default=_json_default)
        
        return TimeSeriesKey(storage_key, file_path)
    
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
        storage_key = self._normalize_storage_key(storage_key)

        # Check cache first
        if storage_key in self.cache:
            self._update_cache_lru(storage_key)
            data, metadata = self.cache[storage_key]
            return self._filter_epochs(data, epochs), metadata
        
        # Try HDF5 first
        hdf5_path = self.storage_path / f"{storage_key}.h5"
        if hdf5_path.exists():
            data, metadata = self._retrieve_hdf5(storage_key)
        else:
            # Try JSON
            json_path = self.storage_path / f"{storage_key}.json.gz"
            if json_path.exists():
                data, metadata = self._retrieve_json(storage_key)
            else:
                raise ValueError(f"No data found for key: {storage_key}")
        
        # Update cache
        self._add_to_cache(storage_key, (data, metadata))
        
        return self._filter_epochs(data, epochs), metadata
    
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
            
            if 'history' in f:
                raw_history = f['history'].asstr()[:]
                epoch_data = [json.loads(value) for value in raw_history]
                epoch_data = self._filter_epochs(epoch_data, epochs)
            else:
                # Read files produced by the original column-oriented format.
                numeric_keys = [key for key in f.keys()]
                epoch_data = []
                if numeric_keys:
                    num_epochs = len(f[numeric_keys[0]])
                    epoch_indices = self._epoch_indices(num_epochs, epochs)
                    for index in epoch_indices:
                        epoch_data.append({key: float(f[key][index]) for key in numeric_keys})
        
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

    def _epoch_indices(
        self,
        count: int,
        epochs: Optional[Union[int, slice, List[int]]],
    ) -> List[int]:
        if epochs is None:
            return list(range(count))
        if isinstance(epochs, int):
            return [epochs]
        if isinstance(epochs, slice):
            return list(range(count))[epochs]
        return epochs

    def _normalize_storage_key(self, storage_key: str) -> str:
        key = str(storage_key)
        if key.endswith('.json.gz'):
            key = key[:-8]
        elif key.endswith('.h5'):
            key = key[:-3]
        if not key.startswith('training_'):
            key = f'training_{key}'
        return key

    @property
    def storage_dir(self) -> Path:
        return self.storage_path

    @property
    def use_hdf5(self) -> bool:
        return self.config.use_hdf5

    def load_training_history(
        self,
        experiment_id_or_key: str,
        epochs: Optional[Union[int, slice, List[int]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Compatibility alias accepting either an experiment ID or storage key."""
        return self.retrieve_training_history(experiment_id_or_key, epochs)
    
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
        storage_key = self._normalize_storage_key(storage_key)
        hdf5_path = self.storage_path / f"{storage_key}.h5"
        
        if hdf5_path.exists():
            with h5py.File(hdf5_path, 'r') as f:
                stats = {
                    'num_epochs': f.attrs.get('num_epochs', 0),
                    'storage_time': f.attrs.get('storage_time', ''),
                }
                
                # Calculate statistics for numeric columns
                for key in f.keys():
                    if key == 'history' or not key.startswith('metric_'):
                        continue
                    data = f[key][:]
                    metric_name = key.removeprefix('metric_')
                    stats[f'{metric_name}_final'] = float(data[-1])
                    stats[f'{metric_name}_max'] = float(np.max(data))
                    stats[f'{metric_name}_min'] = float(np.min(data))
                    stats[f'{metric_name}_mean'] = float(np.mean(data))
                
                return stats
        else:
            # Fallback to loading JSON
            _, metadata = self.retrieve_training_history(storage_key)
            return metadata
    
    def delete_training_history(self, storage_key: str):
        """Delete stored training history."""
        storage_key = self._normalize_storage_key(storage_key)
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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if hasattr(value, 'value'):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class HybridExperimentStorage:
    """
    Hybrid storage combining ChromaDB for search and time series storage for large data.
    """
    
    def __init__(
        self,
        chroma_config: Optional[Any] = None,
        timeseries_config: Optional[TimeSeriesConfig] = None,
        timeseries_threshold: int = 0,
    ):
        from .search import ExperimentSearcher
        
        self.searcher = ExperimentSearcher(chroma_config)
        self.timeseries = TimeSeriesStorage(timeseries_config)
        self.timeseries_threshold = timeseries_threshold

    @property
    def timeseries_storage(self) -> TimeSeriesStorage:
        """Compatibility name for the time-series backend."""
        return self.timeseries
    
    def store_experiment(
        self,
        experiment_id: Union[str, Dict[str, Any]],
        experiment_data: Optional[Dict[str, Any]] = None,
        training_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Store experiment with hybrid approach.
        
        Large time series data goes to specialized storage,
        searchable metadata goes to ChromaDB.
        """
        # The original API accepted a single experiment dictionary. Keep that
        # form while making the explicit ID + envelope signature canonical.
        if isinstance(experiment_id, dict):
            if experiment_data is not None:
                raise TypeError("experiment_data must be omitted when the first argument is a dictionary")
            experiment_data = experiment_id
            experiment_id = experiment_data.get('experiment_id')

        if not experiment_id:
            raise KeyError("experiment_id is required")
        if experiment_data is None:
            raise TypeError("experiment_data is required")

        experiment_data = experiment_data.copy()
        experiment_data['experiment_id'] = str(experiment_id)
        experiment_data.setdefault('schema_version', 1)

        embedded_history = experiment_data.pop('training_history', None)
        if training_history is None:
            training_history = embedded_history
        
        # Store time series data if provided
        timeseries_key = None
        if training_history and len(training_history) >= self.timeseries_threshold:
            timeseries_key = self.timeseries.store_training_history(
                str(experiment_id),
                training_history,
                metadata={
                    'experiment_id': experiment_id,
                    'num_epochs': len(training_history)
                }
            )
            
            # Add summary statistics to searchable data
            stats = self.timeseries.get_summary_statistics(timeseries_key)
            experiment_data['training_summary'] = stats
            experiment_data['timeseries_key'] = str(timeseries_key)
        elif training_history:
            experiment_data['training_history'] = training_history
        
        # Store in ChromaDB for search
        self.searcher.index_experiment(
            experiment_id=str(experiment_id),
            experiment_data=experiment_data
        )
        
        return str(experiment_id)
    
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
        
        metadata = result['metadata']
        envelope = metadata.get('experiment_json')
        if envelope:
            try:
                experiment_data = json.loads(envelope)
            except (TypeError, json.JSONDecodeError):
                experiment_data = metadata.copy()
        else:
            experiment_data = metadata.copy()
        
        # Load training history if requested
        if include_training_history and 'timeseries_key' in experiment_data:
            history, _ = self.timeseries.retrieve_training_history(
                experiment_data['timeseries_key']
            )
            experiment_data['training_history'] = history
        
        return experiment_data

    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load a complete experiment, returning ``None`` when it is absent."""
        try:
            return self.retrieve_experiment(experiment_id, include_training_history=True)
        except ValueError:
            return None

    def search_similar(
        self,
        query: Dict[str, Any],
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        return self.searcher.search_similar_experiments(query, n_results=n_results)

    def search_by_performance(
        self,
        min_accuracy: Optional[float] = None,
        max_parameters: Optional[int] = None,
        dataset: Optional[str] = None,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.searcher.search_by_performance(
            min_accuracy=min_accuracy,
            max_parameters=max_parameters,
            dataset=dataset,
            n_results=n_results,
        )

    def search_by_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return canonical experiment envelopes matching flat metadata."""
        if not filters:
            return []
        where: Dict[str, Any]
        if len(filters) == 1:
            where = filters
        else:
            where = {'$and': [{key: value} for key, value in filters.items()]}
        result = self.searcher.client.collection.get(where=where, include=['metadatas'])
        experiments = []
        for experiment_id, metadata in zip(result['ids'], result['metadatas']):
            envelope = metadata.get('experiment_json')
            experiment = json.loads(envelope) if envelope else metadata.copy()
            experiment['experiment_id'] = experiment_id
            experiments.append(experiment)
        return experiments

    def clear_generation(self, generation: int) -> None:
        """Delete one generated cohort and its external training histories."""
        for experiment in self.search_by_metadata({'generation': generation}):
            timeseries_key = experiment.get('timeseries_key')
            if timeseries_key:
                self.timeseries.delete_training_history(timeseries_key)
            self.searcher.client.delete_experiment(experiment['experiment_id'])
