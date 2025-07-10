# Structure Net Data System Integration Guide

## Overview

The Structure Net Data System provides a flexible, extensible framework for dataset management with integrated semantic search capabilities through ChromaDB. This guide covers installation, configuration, usage, and extension of the data system.

## Table of Contents

1. [Installation](#installation)
2. [Architecture Overview](#architecture-overview)
3. [Basic Usage](#basic-usage)
4. [Dataset Configuration](#dataset-configuration)
5. [ChromaDB Integration](#chromadb-integration)
6. [Metadata Tracking](#metadata-tracking)
7. [Adding New Datasets](#adding-new-datasets)
8. [Advanced Features](#advanced-features)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

```bash
# Install pixi environment
pixi install

# The data factory and ChromaDB are included in the dependencies
```

### Verify Installation

```python
from data_factory import create_dataset, list_available_datasets
from data_factory.search import get_chroma_client

# List available datasets
print("Available datasets:", list_available_datasets())

# Verify ChromaDB
client = get_chroma_client()
print(f"ChromaDB collections: {client.count()} experiments indexed")
```

## Architecture Overview

The data system consists of several key components:

```
src/data_factory/
├── __init__.py          # Main entry point
├── config.py            # Dataset configurations
├── datasets.py          # Dataset loading implementations
├── factory.py           # High-level factory functions
├── metadata.py          # Metadata tracking
├── nal_integration.py   # NAL-ChromaDB integration
├── time_series_storage.py # Efficient time series storage
└── search/              # ChromaDB search layer
    ├── embedder.py      # Embedding generation
    ├── chroma_client.py # ChromaDB wrapper
    └── search_api.py    # Search API
```

### Component Responsibilities

- **Config**: Defines dataset properties (shape, classes, samples)
- **Datasets**: Handles actual data loading and transformations
- **Factory**: Provides caching and high-level interface
- **Metadata**: Tracks dataset usage and integrates with logging
- **Search**: Enables semantic search across experiments
- **NAL Integration**: Memory-efficient experiment tracking for NAL
- **Time Series Storage**: Efficient storage for large training histories

## Basic Usage

### Loading a Dataset

```python
from data_factory import create_dataset

# Load MNIST (default)
dataset = create_dataset()
train_loader = dataset['train_loader']
test_loader = dataset['test_loader']
config = dataset['config']

print(f"Dataset: {config.name}")
print(f"Input shape: {config.input_shape}")
print(f"Classes: {config.num_classes}")

# Load a specific dataset
cifar10 = create_dataset('cifar10', batch_size=128)

# Load with subset for quick testing
small_dataset = create_dataset('cifar100', subset_size=1000)
```

### Using with the Neural Architecture Lab (NAL)

The most common way to use the `data_factory` is indirectly, through the NAL. The NAL is designed to handle all data loading for you automatically.

You simply specify the `dataset_name` in the `control_parameters` of your `Hypothesis`:

```python
from neural_architecture_lab import Hypothesis, HypothesisCategory

my_hypothesis = Hypothesis(
    # ... other hypothesis parameters
    control_parameters={
        'dataset': 'cifar10',  # NAL will use the data_factory to load this
        'epochs': 50,
        'batch_size': 128
    }
)

# The NAL's runner will automatically call create_dataset('cifar10')
# and pass the data loaders to your test function.
```


## Dataset Configuration

### Available Datasets

The system comes pre-configured with:

- **MNIST**: 28x28 grayscale digits (10 classes)
- **CIFAR-10**: 32x32 color images (10 classes)
- **CIFAR-100**: 32x32 color images (100 classes)
- **ImageNet**: 224x224 color images (1000 classes)

### Configuration Structure

```python
@dataclass
class DatasetConfig:
    name: str                    # Short identifier
    full_name: str              # Human-readable name
    description: str            # Dataset description
    input_shape: Tuple[int, ...] # (C, H, W) or (features,)
    num_classes: int            # Number of output classes
    num_train_samples: int      # Training set size
    num_test_samples: int       # Test set size
```

### Accessing Configuration

```python
from data_factory import get_dataset_config

config = get_dataset_config('cifar10')
print(f"Input size for network: {config.input_size}")  # Flattened size
print(f"Number of classes: {config.num_classes}")
```

## ChromaDB Integration

### Overview

ChromaDB provides semantic search capabilities for experiments, architectures, and results. It's designed as a complementary layer to the primary data storage.

### Basic Search Operations

```python
from data_factory.search import ExperimentSearcher

searcher = ExperimentSearcher()

# Index a new experiment
searcher.index_experiment(
    experiment_id="exp_001",
    experiment_data={
        'architecture': [784, 512, 256, 10],
        'final_performance': {'accuracy': 0.95},
        'config': {'dataset': 'mnist', 'epochs': 20}
    }
)

# Search for similar experiments
results = searcher.search_similar_experiments(
    query_experiment={'architecture': [784, 500, 250, 10]},
    n_results=5
)

# Search by performance criteria
high_performers = searcher.search_by_performance(
    min_accuracy=0.90,
    max_parameters=1_000_000,
    dataset='mnist'
)

# Search by architecture pattern
similar_archs = searcher.search_by_architecture(
    architecture=[784, 512, 256, 128, 10],
    n_results=10
)
```

### Automatic Integration

The system automatically indexes experiments when using NAL:

```python
from neural_architecture_lab.lab import NeuralArchitectureLab

lab = NeuralArchitectureLab()
# Experiments are automatically indexed in ChromaDB
result = await lab.test_hypothesis("optimal_seeds")
```

### Advanced Search Queries

```python
# Find experiments for a specific hypothesis
hypothesis_results = searcher.search_by_hypothesis(
    hypothesis_id="canonical_operators",
    n_results=20
)

# Complex metadata filtering
filtered_results = searcher.search_similar_experiments(
    query_experiment=my_experiment,
    filters={
        'dataset': 'cifar10',
        'accuracy': {'$gte': 0.85},
        'parameters': {'$lte': 5_000_000}
    }
)
```

## Metadata Tracking

### Automatic Tracking

The data factory automatically tracks dataset usage:

```python
# Every dataset creation is logged
dataset = create_dataset('cifar10')
# Automatically logs: dataset access, configuration, timestamp

# Integrated with StandardizedLogger
from structure_net.logging.standardized_logging import StandardizedLogger
logger = StandardizedLogger()
# Dataset metadata is included in experiment logs
```

### Manual Metadata

```python
from data_factory.metadata import DatasetMetadata, track_dataset_usage

# Create custom metadata
metadata = DatasetMetadata(
    dataset_name="custom_dataset",
    version="1.0.0",
    source="local_files",
    preprocessing_steps=["normalize", "augment"],
    statistics={
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
)

# Track usage
track_dataset_usage(
    experiment_id="exp_123",
    dataset_metadata=metadata,
    subset_info={"size": 5000, "split": "validation"}
)
```

## Adding New Datasets

### Step 1: Define Configuration

```python
# In src/data_factory/config.py
from data_factory.config import DatasetConfig, register_dataset

# Define your dataset
MY_DATASET = DatasetConfig(
    name="mydataset",
    full_name="My Custom Dataset",
    description="Description of my dataset",
    input_shape=(3, 64, 64),  # C, H, W
    num_classes=50,
    num_train_samples=10000,
    num_test_samples=2000
)

# Register it
register_dataset(MY_DATASET)
```

### Step 2: Implement Loader

```python
# In src/data_factory/datasets.py
from data_factory.datasets import DatasetLoader, register_loader

class MyDatasetLoader(DatasetLoader):
    def load_data(self) -> Tuple[Any, Any]:
        """Load raw data from source."""
        # Implement data loading logic
        # Return (train_data, test_data)
        pass
    
    def get_transform(self, train: bool = True):
        """Define data transformations."""
        # Return torchvision transforms
        pass
    
    def create_datasets(self, train_data, test_data):
        """Create PyTorch datasets."""
        # Return (train_dataset, test_dataset)
        pass

# Register the loader
register_loader('mydataset', MyDatasetLoader)
```

### Step 3: Use Your Dataset

```python
# Your dataset is now available everywhere
dataset = create_dataset('mydataset')
hypothesis = find_optimal_seeds(dataset_name='mydataset')
```

## Advanced Features

### Caching

The data factory implements intelligent caching:

```python
# First load - downloads and caches
dataset1 = create_dataset('cifar10')  # Takes time

# Subsequent loads - uses cache
dataset2 = create_dataset('cifar10')  # Instant

# Force refresh
dataset3 = create_dataset('cifar10', force_reload=True)
```

### Dynamic Architecture Adaptation

```python
from seed_search.architecture_generator import ArchitectureGenerator

# Architectures automatically adapt to dataset
for dataset_name in list_available_datasets():
    arch_gen = ArchitectureGenerator.from_dataset(dataset_name)
    architecture = arch_gen.generate_seed_architecture()
    print(f"{dataset_name}: {architecture}")
```

### Subset Loading for Development

```python
# Load small subset for quick iteration
dev_dataset = create_dataset(
    'imagenet',
    subset_size=1000,  # Only 1000 samples
    batch_size=32
)

# Useful for:
# - Quick prototyping
# - Debugging
# - CI/CD pipelines
```

### Custom Transformations

```python
import torchvision.transforms as transforms

# Define custom transform
custom_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create loader with custom transform
class CustomLoader(DatasetLoader):
    def get_transform(self, train=True):
        return custom_transform if train else transforms.ToTensor()
```

## Best Practices

### 1. Dataset Selection

```python
# Always specify dataset explicitly in experiments
result = run_experiment(dataset_name='cifar10')  # Good
result = run_experiment()  # Relies on default - less clear
```

### 2. Memory Management

```python
# Use subset_size for memory-constrained environments
if torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
    dataset = create_dataset('imagenet', subset_size=5000)
```

### 3. Consistent Preprocessing

```python
# Store preprocessing in metadata for reproducibility
metadata = DatasetMetadata(
    dataset_name="custom",
    preprocessing_steps=["resize_224", "normalize_imagenet", "random_crop"],
    statistics={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
)
```

### 4. Search Strategy

```python
# Index experiments immediately after completion
result = run_experiment(...)
searcher.index_experiment(
    experiment_id=result.experiment_id,
    experiment_data=result.to_dict()
)

# Batch index for efficiency
experiments = [...]  # List of completed experiments
searcher.index_experiments_batch([
    (exp.id, exp.data) for exp in experiments
])
```

### 5. Error Handling

```python
from data_factory import create_dataset, DatasetNotFoundError

try:
    dataset = create_dataset('unknown_dataset')
except DatasetNotFoundError as e:
    print(f"Dataset not found: {e}")
    print(f"Available datasets: {list_available_datasets()}")
```

## Troubleshooting

### Common Issues

#### 1. Dataset Download Failures

```python
# Specify alternative data directory
import os
os.environ['DATA_DIR'] = '/path/to/data'
dataset = create_dataset('cifar10')
```

#### 2. ChromaDB Connection Issues

```python
from data_factory.search import ChromaConfig, reset_chroma_client

# Reset client
reset_chroma_client()

# Use custom configuration
config = ChromaConfig(
    persist_directory="/custom/chroma/path",
    collection_name="my_experiments"
)
searcher = ExperimentSearcher(config)
```

#### 3. Memory Issues with Large Datasets

```python
# Use smaller batch sizes
dataset = create_dataset('imagenet', batch_size=16)

# Or use subset
dataset = create_dataset('imagenet', subset_size=10000)
```

#### 4. Import Errors

```python
# Ensure you're importing from data_factory, not src.data_factory
from data_factory import create_dataset  # Correct
# from src.data_factory import create_dataset  # Incorrect
```

### Debugging

```python
# Enable verbose logging
import logging
logging.getLogger('data_factory').setLevel(logging.DEBUG)

# Check cache status
from data_factory.factory import DATASET_CACHE
print(f"Cached datasets: {list(DATASET_CACHE.keys())}")

# Verify ChromaDB status
client = get_chroma_client()
print(f"Total experiments indexed: {client.count()}")
```

## NAL Integration

### Memory-Efficient NAL with ChromaDB

The data system now includes integration with NAL to offload experiment data and prevent memory accumulation:

```python
from data_factory.nal_integration import create_memory_efficient_nal
from neural_architecture_lab.core import LabConfig
from data_factory.search import ChromaConfig
from data_factory.time_series_storage import TimeSeriesConfig

# Configure NAL with minimal memory usage
nal_config = LabConfig(
    max_parallel_experiments=8,
    save_best_models=False,
    results_dir="/tmp/nal_results"
)

# Configure ChromaDB
chroma_config = ChromaConfig(
    persist_directory="/data/chroma_nal",
    collection_name="nal_experiments"
)

# Configure time series storage for large histories
timeseries_config = TimeSeriesConfig(
    storage_dir="/data/timeseries",
    use_hdf5=True,
    compression="gzip"
)

# Create memory-efficient NAL
nal, chroma_integration = create_memory_efficient_nal(
    nal_config, 
    chroma_config,
    timeseries_config
)
```

### Time Series Storage

For experiments with large training histories, the system automatically offloads to efficient storage:

```python
from data_factory.time_series_storage import TimeSeriesStorage

storage = TimeSeriesStorage()

# Store large training history
storage_key = storage.store_training_history(
    experiment_id="exp_001",
    epoch_data=[
        {'epoch': 0, 'loss': 2.3, 'accuracy': 0.1},
        {'epoch': 1, 'loss': 1.8, 'accuracy': 0.3},
        # ... many more epochs
    ],
    metadata={'model': 'resnet50', 'dataset': 'imagenet'}
)

# Retrieve specific epochs efficiently
recent_epochs, metadata = storage.retrieve_training_history(
    storage_key,
    epochs=slice(-10, None)  # Last 10 epochs
)

# Get summary without loading all data
stats = storage.get_summary_statistics(storage_key)
```

### Hybrid Storage Pattern

The system uses a hybrid approach:
- **ChromaDB**: Searchable metadata and small experiments
- **HDF5/Time Series**: Large training histories and numeric data

```python
from data_factory.time_series_storage import HybridExperimentStorage

hybrid = HybridExperimentStorage()

# Store experiment with automatic routing
hybrid.store_experiment(
    experiment_id="exp_001",
    experiment_data={
        'architecture': [784, 512, 256, 10],
        'accuracy': 0.95,
        'parameters': 1.2e6
    },
    training_history=[...]  # Automatically stored in time series if large
)

# Retrieve with optional history loading
exp_data = hybrid.retrieve_experiment(
    "exp_001", 
    include_training_history=True
)
```

## Conclusion

The Structure Net Data System provides a powerful, flexible foundation for dataset management and experiment search. By decoupling datasets from hypotheses and integrating semantic search, it enables:

- Easy dataset switching without code changes
- Comprehensive experiment tracking and search
- Efficient caching and memory management
- Extensibility for custom datasets
- Memory-efficient NAL integration
- Scalable time series storage for large experiments

For questions or contributions, please refer to the main Structure Net documentation or open an issue on GitHub.