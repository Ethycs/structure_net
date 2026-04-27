"""
Dataset Factory with Caching

Provides high-level factory functions for dataset creation with:
- Automatic caching of datasets
- Metadata tracking
- Version management
- Easy dataset swapping
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, Dict, Any
import pickle
from pathlib import Path
import hashlib
import logging
from datetime import datetime

from .config import get_dataset_config, DatasetConfig
from .datasets import load_dataset, create_data_loaders
from .metadata import DatasetMetadata, track_dataset_usage, save_dataset_metadata

logger = logging.getLogger(__name__)

# Global cache for loaded datasets
_DATASET_CACHE: Dict[str, Dataset] = {}


def _get_cache_key(
    dataset_name: str,
    train: bool,
    subset_fraction: Optional[float] = None,
    seed: Optional[int] = None
) -> str:
    """Generate a cache key for a dataset configuration."""
    key_parts = [
        dataset_name,
        "train" if train else "test",
        f"subset_{subset_fraction}" if subset_fraction else "full",
        f"seed_{seed}" if seed is not None else "no_seed"
    ]
    return "_".join(key_parts)


def _get_dataset_hash(dataset_name: str, config: DatasetConfig) -> str:
    """Generate a hash for dataset version tracking."""
    content = f"{dataset_name}_{config.version}_{config.input_shape}_{config.num_classes}"
    return hashlib.md5(content.encode()).hexdigest()[:8]


def get_dataset(
    dataset_name: str,
    train: bool = True,
    use_cache: bool = True,
    subset_fraction: Optional[float] = None,
    seed: Optional[int] = None,
    transform: Optional[Any] = None,
    download: bool = True,
    experiment_id: Optional[str] = None
) -> Dataset:
    """
    Get a dataset with caching support.
    
    Args:
        dataset_name: Name of the dataset
        train: Whether to get training set
        use_cache: Whether to use cached dataset if available
        subset_fraction: Optional fraction of data to use
        seed: Random seed for subset selection
        transform: Optional transform (uses default if None)
        download: Whether to download if not found
        experiment_id: Optional experiment ID for tracking
        
    Returns:
        PyTorch Dataset object
    """
    cache_key = _get_cache_key(dataset_name, train, subset_fraction, seed)
    
    # Check cache
    if use_cache and cache_key in _DATASET_CACHE:
        logger.info(f"Using cached dataset: {cache_key}")
        dataset = _DATASET_CACHE[cache_key]
    else:
        # Load dataset
        logger.info(f"Loading dataset: {dataset_name} (train={train})")
        dataset = load_dataset(
            dataset_name,
            train=train,
            transform=transform,
            download=download,
            subset_fraction=subset_fraction,
            seed=seed
        )
        
        # Cache it
        if use_cache:
            _DATASET_CACHE[cache_key] = dataset
            logger.info(f"Cached dataset: {cache_key}")
    
    # Track usage if experiment_id provided
    if experiment_id:
        config = get_dataset_config(dataset_name)
        track_dataset_usage(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            dataset_version=config.version,
            subset_used="train" if train else "test",
            samples_used=len(dataset),
            transformations_applied=[]  # TODO: Extract from transform
        )
    
    return dataset


def create_dataset(
    dataset_name: str,
    batch_size: Optional[int] = None,
    train_transform: Optional[Any] = None,
    test_transform: Optional[Any] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    subset_fraction: Optional[float] = None,
    seed: Optional[int] = None,
    use_cache: bool = True,
    experiment_id: Optional[str] = None,
    download: bool = True
) -> Dict[str, Any]:
    """
    Create a complete dataset configuration with loaders and metadata.
    
    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size (uses default if None)
        train_transform: Transform for training data
        test_transform: Transform for test data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        subset_fraction: Optional fraction to use
        seed: Random seed
        use_cache: Whether to use caching
        experiment_id: Optional experiment ID for tracking
        download: Whether to download if needed
        
    Returns:
        Dictionary containing:
        - config: DatasetConfig
        - train_loader: Training DataLoader
        - test_loader: Test DataLoader
        - metadata: DatasetMetadata
    """
    config = get_dataset_config(dataset_name)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
        subset_fraction=subset_fraction,
        seed=seed,
        download=download
    )
    
    # Create metadata
    dataset_hash = _get_dataset_hash(dataset_name, config)
    metadata = DatasetMetadata(
        dataset_name=dataset_name,
        dataset_version=config.version,
        download_timestamp=datetime.now().isoformat(),
        size_bytes=0,  # TODO: Calculate actual size
        num_samples=config.num_train_samples + config.num_test_samples,
        shape=list(config.input_shape),
        checksum=dataset_hash,
        config=config.__dict__
    )
    
    # Save metadata
    save_dataset_metadata(metadata)
    
    # Track usage if experiment_id provided
    if experiment_id:
        # Track training set usage
        track_dataset_usage(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            dataset_version=config.version,
            subset_used="train",
            samples_used=len(train_loader.dataset),
            transformations_applied=[]
        )
        
        # Track test set usage
        track_dataset_usage(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            dataset_version=config.version,
            subset_used="test",
            samples_used=len(test_loader.dataset),
            transformations_applied=[]
        )
    
    return {
        "config": config,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "metadata": metadata,
        "input_size": config.input_size,
        "num_classes": config.num_classes,
        "input_shape": config.input_shape,
    }


def clear_dataset_cache(dataset_name: Optional[str] = None) -> int:
    """
    Clear cached datasets.
    
    Args:
        dataset_name: Optional specific dataset to clear (clears all if None)
        
    Returns:
        Number of cache entries cleared
    """
    global _DATASET_CACHE
    
    if dataset_name:
        # Clear specific dataset
        keys_to_remove = [k for k in _DATASET_CACHE.keys() if k.startswith(dataset_name)]
        for key in keys_to_remove:
            del _DATASET_CACHE[key]
        cleared = len(keys_to_remove)
    else:
        # Clear all
        cleared = len(_DATASET_CACHE)
        _DATASET_CACHE.clear()
    
    logger.info(f"Cleared {cleared} dataset cache entries")
    return cleared


def get_dataset_for_hypothesis(
    hypothesis_config: Dict[str, Any],
    experiment_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get dataset based on hypothesis configuration.
    
    This is a convenience function that extracts dataset parameters
    from a hypothesis configuration and creates the dataset.
    
    Args:
        hypothesis_config: Hypothesis or experiment configuration
        experiment_id: Optional experiment ID for tracking
        
    Returns:
        Dataset configuration dictionary
    """
    # Extract dataset parameters
    dataset_name = hypothesis_config.get('dataset', 'mnist')
    batch_size = hypothesis_config.get('batch_size', None)
    subset_fraction = hypothesis_config.get('subset_fraction', None)
    seed = hypothesis_config.get('seed', None)
    
    # Create dataset
    return create_dataset(
        dataset_name=dataset_name,
        batch_size=batch_size,
        subset_fraction=subset_fraction,
        seed=seed,
        experiment_id=experiment_id
    )


def preload_datasets(dataset_names: list[str]) -> None:
    """
    Preload multiple datasets into cache.
    
    Useful for experiments that will use multiple datasets.
    
    Args:
        dataset_names: List of dataset names to preload
    """
    for name in dataset_names:
        logger.info(f"Preloading dataset: {name}")
        
        # Load train and test sets
        get_dataset(name, train=True, use_cache=True)
        get_dataset(name, train=False, use_cache=True)
    
    logger.info(f"Preloaded {len(dataset_names)} datasets")