"""
Dataset Configuration System

Provides a registry of dataset configurations with metadata about:
- Input dimensions
- Number of classes
- Default transformations
- Download locations
- Version information
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    
    # Basic information
    name: str
    full_name: str
    description: str
    
    # Data characteristics
    input_shape: Tuple[int, ...]  # e.g., (28, 28) for MNIST, (3, 32, 32) for CIFAR
    num_classes: int
    num_train_samples: int
    num_test_samples: int
    
    # Flattened input size for fully connected networks
    @property
    def input_size(self) -> int:
        """Get flattened input size."""
        size = 1
        for dim in self.input_shape:
            size *= dim
        return size
    
    # Task information
    task_type: str = "classification"  # classification, regression, etc.
    
    # Storage and versioning
    version: str = "1.0"
    download_url: Optional[str] = None
    cache_dir: str = "data/datasets"
    
    # Default parameters
    default_batch_size: int = 128
    default_train_transforms: List[str] = field(default_factory=list)
    default_test_transforms: List[str] = field(default_factory=list)
    
    # Loader function (to be set by dataset implementation)
    loader_function: Optional[Callable] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if any(dim <= 0 for dim in self.input_shape):
            raise ValueError(f"All input_shape dimensions must be positive, got {self.input_shape}")


# Global dataset registry
_DATASET_REGISTRY: Dict[str, DatasetConfig] = {}


def register_dataset(config: DatasetConfig) -> None:
    """Register a dataset configuration."""
    if config.name in _DATASET_REGISTRY:
        logger.warning(f"Overwriting existing dataset config for {config.name}")
    _DATASET_REGISTRY[config.name] = config
    logger.info(f"Registered dataset: {config.name} ({config.full_name})")


def get_dataset_config(name: str) -> DatasetConfig:
    """Get a dataset configuration by name."""
    if name not in _DATASET_REGISTRY:
        available = list(_DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return _DATASET_REGISTRY[name]


def list_available_datasets() -> List[str]:
    """List all available dataset names."""
    return list(_DATASET_REGISTRY.keys())


# Standard dataset configurations
MNIST_CONFIG = DatasetConfig(
    name="mnist",
    full_name="MNIST Handwritten Digits",
    description="28x28 grayscale images of handwritten digits (0-9)",
    input_shape=(28, 28),
    num_classes=10,
    num_train_samples=60000,
    num_test_samples=10000,
    default_train_transforms=["normalize"],
    default_test_transforms=["normalize"],
    metadata={
        "citation": "LeCun et al., 1998",
        "website": "http://yann.lecun.com/exdb/mnist/",
    }
)

CIFAR10_CONFIG = DatasetConfig(
    name="cifar10",
    full_name="CIFAR-10",
    description="32x32 color images in 10 classes",
    input_shape=(3, 32, 32),
    num_classes=10,
    num_train_samples=50000,
    num_test_samples=10000,
    default_train_transforms=["normalize", "random_horizontal_flip"],
    default_test_transforms=["normalize"],
    metadata={
        "citation": "Krizhevsky, 2009",
        "website": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "classes": ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"],
    }
)

CIFAR100_CONFIG = DatasetConfig(
    name="cifar100",
    full_name="CIFAR-100",
    description="32x32 color images in 100 classes",
    input_shape=(3, 32, 32),
    num_classes=100,
    num_train_samples=50000,
    num_test_samples=10000,
    default_train_transforms=["normalize", "random_horizontal_flip", "random_crop"],
    default_test_transforms=["normalize"],
    metadata={
        "citation": "Krizhevsky, 2009",
        "website": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "superclasses": 20,
    }
)

IMAGENET_CONFIG = DatasetConfig(
    name="imagenet",
    full_name="ImageNet",
    description="Large-scale image classification dataset",
    input_shape=(3, 224, 224),
    num_classes=1000,
    num_train_samples=1281167,
    num_test_samples=50000,
    default_batch_size=256,
    default_train_transforms=["resize", "random_crop", "random_horizontal_flip", "normalize"],
    default_test_transforms=["resize", "center_crop", "normalize"],
    metadata={
        "citation": "Deng et al., 2009",
        "website": "http://www.image-net.org/",
        "requires_download": True,
    }
)

# Custom dataset template
CUSTOM_DATASET_TEMPLATE = DatasetConfig(
    name="custom",
    full_name="Custom Dataset",
    description="Template for custom datasets",
    input_shape=(1, 28, 28),  # Should be overridden
    num_classes=10,  # Should be overridden
    num_train_samples=0,  # Should be overridden
    num_test_samples=0,  # Should be overridden
    metadata={
        "note": "This is a template. Create your own DatasetConfig for custom datasets.",
    }
)


# Register standard datasets on module import
def _register_standard_datasets():
    """Register all standard dataset configurations."""
    for config in [MNIST_CONFIG, CIFAR10_CONFIG, CIFAR100_CONFIG, IMAGENET_CONFIG]:
        register_dataset(config)


# Auto-register standard datasets
_register_standard_datasets()


# Utility functions for dataset configuration
def create_custom_dataset_config(
    name: str,
    input_shape: Tuple[int, ...],
    num_classes: int,
    num_train_samples: int,
    num_test_samples: int,
    **kwargs
) -> DatasetConfig:
    """
    Create a custom dataset configuration.
    
    Args:
        name: Short name for the dataset
        input_shape: Shape of input data
        num_classes: Number of output classes
        num_train_samples: Number of training samples
        num_test_samples: Number of test samples
        **kwargs: Additional configuration parameters
        
    Returns:
        DatasetConfig instance
    """
    config = DatasetConfig(
        name=name,
        full_name=kwargs.get('full_name', name.title()),
        description=kwargs.get('description', f"Custom dataset: {name}"),
        input_shape=input_shape,
        num_classes=num_classes,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        **{k: v for k, v in kwargs.items() 
           if k not in ['full_name', 'description']}
    )
    
    # Auto-register if requested
    if kwargs.get('auto_register', True):
        register_dataset(config)
    
    return config


def get_dataset_input_size(dataset_name: str) -> int:
    """Get the flattened input size for a dataset."""
    config = get_dataset_config(dataset_name)
    return config.input_size


def get_dataset_num_classes(dataset_name: str) -> int:
    """Get the number of classes for a dataset."""
    config = get_dataset_config(dataset_name)
    return config.num_classes


def get_dataset_shape(dataset_name: str) -> Tuple[int, ...]:
    """Get the input shape for a dataset."""
    config = get_dataset_config(dataset_name)
    return config.input_shape