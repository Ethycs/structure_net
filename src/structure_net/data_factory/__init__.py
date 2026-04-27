"""
Data Factory for Structure Net

Provides a unified interface for dataset management with:
- Dynamic dataset loading
- Metadata tracking
- Version management
- Caching support
- ChromaDB integration for search
"""

# Import config adapter to enable unified config support
from . import config_adapter

from .config import (
    DatasetConfig,
    MNIST_CONFIG,
    CIFAR10_CONFIG,
    CIFAR100_CONFIG,
    IMAGENET_CONFIG,
    register_dataset,
    get_dataset_config,
    list_available_datasets,
)

from .factory import (
    create_dataset,
    get_dataset,
    clear_dataset_cache,
)

from .metadata import (
    DatasetMetadata,
    DatasetUsage,
    track_dataset_usage,
    get_dataset_metadata,
)

# Import search components
from .search import (
    ExperimentSearcher,
    search_similar_experiments,
    search_by_architecture,
    search_by_performance,
    search_by_hypothesis,
    get_chroma_client,
    ChromaConfig,
)

__all__ = [
    # Config
    'DatasetConfig',
    'MNIST_CONFIG',
    'CIFAR10_CONFIG',
    'CIFAR100_CONFIG',
    'IMAGENET_CONFIG',
    'register_dataset',
    'get_dataset_config',
    'list_available_datasets',
    
    # Factory
    'create_dataset',
    'get_dataset',
    'clear_dataset_cache',
    
    # Metadata
    'DatasetMetadata',
    'DatasetUsage',
    'track_dataset_usage',
    'get_dataset_metadata',
    
    # Search
    'ExperimentSearcher',
    'search_similar_experiments',
    'search_by_architecture',
    'search_by_performance',
    'search_by_hypothesis',
    'get_chroma_client',
    'ChromaConfig',
]