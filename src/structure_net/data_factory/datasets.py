"""
Dataset Loading Implementations

Provides actual dataset loading functionality for each registered dataset.
Handles downloading, caching, and transformations.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from pathlib import Path
import logging

from .config import get_dataset_config, DatasetConfig

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.cache_path = Path(config.cache_dir) / config.name
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    def load(
        self,
        train: bool = True,
        transform: Optional[Any] = None,
        download: bool = True,
        subset_fraction: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dataset:
        """Load the dataset."""
        raise NotImplementedError
    
    def get_default_transform(self, train: bool = True) -> Any:
        """Get default transformations for the dataset."""
        raise NotImplementedError


class MNISTLoader(DatasetLoader):
    """MNIST dataset loader."""
    
    def get_default_transform(self, train: bool = True) -> Any:
        """Get default MNIST transformations."""
        transform_list = []
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if specified in config
        if "normalize" in (self.config.default_train_transforms if train else self.config.default_test_transforms):
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        return transforms.Compose(transform_list)
    
    def load(
        self,
        train: bool = True,
        transform: Optional[Any] = None,
        download: bool = True,
        subset_fraction: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dataset:
        """Load MNIST dataset."""
        if transform is None:
            transform = self.get_default_transform(train)
        
        dataset = torchvision.datasets.MNIST(
            root=str(self.cache_path),
            train=train,
            transform=transform,
            download=download
        )
        
        # Apply subset if requested
        if subset_fraction is not None and 0 < subset_fraction < 1:
            n_samples = len(dataset)
            n_subset = int(n_samples * subset_fraction)
            
            if seed is not None:
                np.random.seed(seed)
            
            indices = np.random.permutation(n_samples)[:n_subset]
            dataset = Subset(dataset, indices)
            
            logger.info(f"Using {n_subset}/{n_samples} samples from MNIST {'train' if train else 'test'} set")
        
        return dataset


class CIFAR10Loader(DatasetLoader):
    """CIFAR-10 dataset loader."""
    
    def get_default_transform(self, train: bool = True) -> Any:
        """Get default CIFAR-10 transformations."""
        transform_list = []
        
        # Training augmentations
        if train:
            if "random_horizontal_flip" in self.config.default_train_transforms:
                transform_list.append(transforms.RandomHorizontalFlip())
            if "random_crop" in self.config.default_train_transforms:
                transform_list.append(transforms.RandomCrop(32, padding=4))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if specified
        if "normalize" in (self.config.default_train_transforms if train else self.config.default_test_transforms):
            transform_list.append(transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            ))
        
        return transforms.Compose(transform_list)
    
    def load(
        self,
        train: bool = True,
        transform: Optional[Any] = None,
        download: bool = True,
        subset_fraction: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dataset:
        """Load CIFAR-10 dataset."""
        if transform is None:
            transform = self.get_default_transform(train)
        
        dataset = torchvision.datasets.CIFAR10(
            root=str(self.cache_path),
            train=train,
            transform=transform,
            download=download
        )
        
        # Apply subset if requested
        if subset_fraction is not None and 0 < subset_fraction < 1:
            n_samples = len(dataset)
            n_subset = int(n_samples * subset_fraction)
            
            if seed is not None:
                np.random.seed(seed)
            
            indices = np.random.permutation(n_samples)[:n_subset]
            dataset = Subset(dataset, indices)
            
            logger.info(f"Using {n_subset}/{n_samples} samples from CIFAR-10 {'train' if train else 'test'} set")
        
        return dataset


class CIFAR100Loader(DatasetLoader):
    """CIFAR-100 dataset loader."""
    
    def get_default_transform(self, train: bool = True) -> Any:
        """Get default CIFAR-100 transformations."""
        transform_list = []
        
        # Training augmentations
        if train:
            if "random_horizontal_flip" in self.config.default_train_transforms:
                transform_list.append(transforms.RandomHorizontalFlip())
            if "random_crop" in self.config.default_train_transforms:
                transform_list.append(transforms.RandomCrop(32, padding=4))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if specified
        if "normalize" in (self.config.default_train_transforms if train else self.config.default_test_transforms):
            transform_list.append(transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761)
            ))
        
        return transforms.Compose(transform_list)
    
    def load(
        self,
        train: bool = True,
        transform: Optional[Any] = None,
        download: bool = True,
        subset_fraction: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dataset:
        """Load CIFAR-100 dataset."""
        if transform is None:
            transform = self.get_default_transform(train)
        
        dataset = torchvision.datasets.CIFAR100(
            root=str(self.cache_path),
            train=train,
            transform=transform,
            download=download
        )
        
        # Apply subset if requested
        if subset_fraction is not None and 0 < subset_fraction < 1:
            n_samples = len(dataset)
            n_subset = int(n_samples * subset_fraction)
            
            if seed is not None:
                np.random.seed(seed)
            
            indices = np.random.permutation(n_samples)[:n_subset]
            dataset = Subset(dataset, indices)
            
            logger.info(f"Using {n_subset}/{n_samples} samples from CIFAR-100 {'train' if train else 'test'} set")
        
        return dataset


class CustomDatasetLoader(DatasetLoader):
    """Loader for custom datasets."""
    
    def __init__(self, config: DatasetConfig, dataset_class: type, **kwargs):
        super().__init__(config)
        self.dataset_class = dataset_class
        self.kwargs = kwargs
    
    def get_default_transform(self, train: bool = True) -> Any:
        """Get default transformations for custom dataset."""
        # Basic transform - just convert to tensor
        return transforms.ToTensor()
    
    def load(
        self,
        train: bool = True,
        transform: Optional[Any] = None,
        download: bool = True,
        subset_fraction: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dataset:
        """Load custom dataset."""
        if transform is None:
            transform = self.get_default_transform(train)
        
        # Pass all kwargs to the custom dataset class
        dataset = self.dataset_class(
            train=train,
            transform=transform,
            **self.kwargs
        )
        
        # Apply subset if requested
        if subset_fraction is not None and 0 < subset_fraction < 1:
            n_samples = len(dataset)
            n_subset = int(n_samples * subset_fraction)
            
            if seed is not None:
                np.random.seed(seed)
            
            indices = np.random.permutation(n_samples)[:n_subset]
            dataset = Subset(dataset, indices)
            
            logger.info(f"Using {n_subset}/{n_samples} samples from custom dataset")
        
        return dataset


# Loader registry
_LOADER_REGISTRY: Dict[str, type] = {
    "mnist": MNISTLoader,
    "cifar10": CIFAR10Loader,
    "cifar100": CIFAR100Loader,
}


def register_loader(dataset_name: str, loader_class: type) -> None:
    """Register a custom dataset loader."""
    _LOADER_REGISTRY[dataset_name] = loader_class
    logger.info(f"Registered loader for dataset: {dataset_name}")


def get_loader(dataset_name: str) -> DatasetLoader:
    """Get a dataset loader instance."""
    config = get_dataset_config(dataset_name)
    
    if dataset_name not in _LOADER_REGISTRY:
        raise ValueError(f"No loader registered for dataset: {dataset_name}")
    
    loader_class = _LOADER_REGISTRY[dataset_name]
    return loader_class(config)


def load_dataset(
    dataset_name: str,
    train: bool = True,
    transform: Optional[Any] = None,
    download: bool = True,
    subset_fraction: Optional[float] = None,
    seed: Optional[int] = None
) -> Dataset:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
        train: Whether to load training set (True) or test set (False)
        transform: Optional transform to apply (uses default if None)
        download: Whether to download the dataset if not found
        subset_fraction: Optional fraction of data to use (for quick experiments)
        seed: Random seed for subset selection
        
    Returns:
        PyTorch Dataset object
    """
    loader = get_loader(dataset_name)
    return loader.load(
        train=train,
        transform=transform,
        download=download,
        subset_fraction=subset_fraction,
        seed=seed
    )


def create_data_loaders(
    dataset_name: str,
    batch_size: Optional[int] = None,
    train_transform: Optional[Any] = None,
    test_transform: Optional[Any] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    subset_fraction: Optional[float] = None,
    seed: Optional[int] = None,
    shuffle_train: bool = True,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size (uses dataset default if None)
        train_transform: Transform for training data
        test_transform: Transform for test data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        subset_fraction: Optional fraction of data to use
        seed: Random seed for subset selection
        shuffle_train: Whether to shuffle training data
        download: Whether to download the dataset if needed
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    config = get_dataset_config(dataset_name)
    
    if batch_size is None:
        batch_size = config.default_batch_size
    
    # Load datasets
    train_dataset = load_dataset(
        dataset_name,
        train=True,
        transform=train_transform,
        download=download,
        subset_fraction=subset_fraction,
        seed=seed
    )
    
    test_dataset = load_dataset(
        dataset_name,
        train=False,
        transform=test_transform,
        download=download,
        subset_fraction=subset_fraction,
        seed=seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader