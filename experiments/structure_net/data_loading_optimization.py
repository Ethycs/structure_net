#!/usr/bin/env python3
"""
Data Loading Optimization for Ultimate Stress Test

This module provides optimized data loading strategies to prevent memory issues:
1. Lazy loading - datasets are loaded only when needed
2. Shared memory - use torch.multiprocessing shared memory for dataset
3. GPU prefetching - load data to GPU in parallel with model training
"""

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import weakref
from pathlib import Path
import json


class DatasetReference:
    """
    A lightweight reference to a dataset that can be pickled efficiently.
    The actual dataset is loaded only when needed.
    """
    
    def __init__(self, dataset_name: str, batch_size: int, config: Optional[Dict] = None):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.config = config or {}
        self._cache = {}
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and test loaders, creating them if needed."""
        cache_key = f"{self.dataset_name}_{self.batch_size}"
        
        if cache_key not in self._cache:
            # Import here to avoid circular imports
            from data_factory import create_dataset
            
            dataset_dict = create_dataset(
                dataset_name=self.dataset_name,
                batch_size=self.batch_size,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,  # Keep workers alive
                **self.config
            )
            
            self._cache[cache_key] = (
                dataset_dict['train_loader'],
                dataset_dict['test_loader']
            )
        
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Clear the dataset cache to free memory."""
        self._cache.clear()


class SharedDatasetManager:
    """
    Manager for shared datasets across processes using torch.multiprocessing.
    """
    
    _instances = {}
    
    @classmethod
    def get_instance(cls, dataset_name: str) -> 'SharedDatasetManager':
        """Get or create a shared dataset manager instance."""
        if dataset_name not in cls._instances:
            cls._instances[dataset_name] = cls(dataset_name)
        return cls._instances[dataset_name]
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.shared_tensors = None
        self.metadata = None
        
    def load_to_shared_memory(self):
        """Load dataset into shared memory for efficient multi-process access."""
        from data_factory import create_dataset
        
        # Load dataset
        dataset_dict = create_dataset(
            dataset_name=self.dataset_name,
            batch_size=1,  # Load all at once
            shuffle=False
        )
        
        # Convert to tensors and move to shared memory
        train_data = []
        train_labels = []
        for data, labels in dataset_dict['train_loader']:
            train_data.append(data)
            train_labels.append(labels)
        
        train_data = torch.cat(train_data, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        
        # Move to shared memory
        train_data.share_memory_()
        train_labels.share_memory_()
        
        self.shared_tensors = {
            'train_data': train_data,
            'train_labels': train_labels
        }
        
        self.metadata = {
            'num_samples': len(train_data),
            'data_shape': list(train_data.shape[1:]),
            'num_classes': dataset_dict['config'].num_classes
        }
        
        return self
    
    def get_shared_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from shared memory tensors."""
        if self.shared_tensors is None:
            raise ValueError("Dataset not loaded to shared memory yet")
        
        from torch.utils.data import TensorDataset
        
        dataset = TensorDataset(
            self.shared_tensors['train_data'],
            self.shared_tensors['train_labels']
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Already in memory
            pin_memory=False  # Already in shared memory
        )


def create_gpu_prefetch_loader(loader: DataLoader, device: str) -> 'GPUPrefetchLoader':
    """
    Create a DataLoader that prefetches data to GPU while the model is training.
    This overlaps data transfer with computation for better efficiency.
    """
    return GPUPrefetchLoader(loader, device)


class GPUPrefetchLoader:
    """
    DataLoader wrapper that prefetches data to GPU in background.
    """
    
    def __init__(self, loader: DataLoader, device: str):
        self.loader = loader
        self.device = device
        
    def __iter__(self):
        stream = torch.cuda.Stream() if self.device.startswith('cuda') else None
        first = True
        
        for next_data, next_target in self.loader:
            if stream is not None:
                with torch.cuda.stream(stream):
                    next_data = next_data.to(self.device, non_blocking=True)
                    next_target = next_target.to(self.device, non_blocking=True)
            else:
                next_data = next_data.to(self.device)
                next_target = next_target.to(self.device)
            
            if not first:
                yield data, target
            else:
                first = False
            
            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)
            
            data = next_data
            target = next_target
        
        yield data, target
    
    def __len__(self):
        return len(self.loader)


def optimize_dataloader_memory(loader: DataLoader) -> DataLoader:
    """
    Optimize a DataLoader for memory efficiency.
    """
    # Set memory-efficient options
    loader.num_workers = min(loader.num_workers, 2)  # Limit workers
    loader.pin_memory = torch.cuda.is_available()  # Pin memory for GPU
    loader.persistent_workers = True  # Keep workers alive
    loader.prefetch_factor = 2  # Limit prefetching
    
    return loader


# Example usage in evaluate_competitor_task
def evaluate_with_optimized_loading(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Example of how to use optimized data loading in evaluation.
    """
    device = config.get('device', 'cpu')
    
    # Use DatasetReference for lazy loading
    dataset_ref = DatasetReference(
        dataset_name=config['dataset'],
        batch_size=config['batch_size']
    )
    
    # Get loaders only when needed
    train_loader, test_loader = dataset_ref.get_loaders()
    
    # Use GPU prefetching if available
    if device.startswith('cuda'):
        train_loader = create_gpu_prefetch_loader(train_loader, device)
        test_loader = create_gpu_prefetch_loader(test_loader, device)
    
    # ... rest of training code ...
    
    # Clear cache when done
    dataset_ref.clear_cache()
    
    return {'accuracy': 0.95}  # Example metrics