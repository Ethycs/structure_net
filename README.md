# Structure Net: A Composable Framework for Dynamic Network Growth

A PyTorch implementation of a modular, metrics-driven framework for dynamically growing neural networks during training. This project has been refactored to a composable architecture that allows for flexible and extensible research into network evolution.

## Overview

This project provides a framework for creating and training neural networks that evolve their own structure. Instead of a fixed architecture, networks start with minimal connectivity and grow based on a deep analysis of their internal state. The core of the project is the **Composable Evolution System**, an interface-based framework that allows researchers to easily mix and match different components to create novel growth strategies.

## Key Features

### 🧬 Composable Evolution System
- **Modular by Design:** The evolution system is built from a set of clean, interchangeable components:
    - **Analyzers:** Inspect the network's state (e.g., `StandardExtremaAnalyzer`, `GraphAnalyzer`).
    - **Growth Strategies:** Propose and execute changes to the network architecture (e.g., `ExtremaGrowthStrategy`).
    - **Trainers:** Handle the training loop for each evolution cycle.
- **Flexible & Extensible:** Easily create new components and combine them to build custom evolution systems without modifying the core framework.

### 📊 Standardized Logging & Artifacts
- **Schema-Driven:** All experiment results are validated against Pydantic schemas, ensuring data consistency.
- **WandB Integration:** Automatically saves each experiment as a versioned, immutable artifact in Weights & Biases.
- **Offline-First:** A local queuing system ensures that no data is lost, even without a network connection.

### 🚀 GPU Acceleration
- **Optimized Metrics:** Key metrics computations, including graph analysis, are accelerated on the GPU using `cuGraph` and `cuPy`.
- **CPU Fallback:** The system gracefully falls back to CPU-based computation if a compatible GPU is not available.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd structure_net

# Install with pixi (recommended)
pixi install
```

## Quick Start

The following example demonstrates how to use the new composable system to create a standard network and evolve it for a few iterations.

```python
import torch
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.evolution.components import (
    create_standard_evolution_system,
    NetworkContext
)

# 1. Create a standard sparse network
network = create_standard_network(
    architecture=[784, 256, 128, 10],
    sparsity=0.01
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network.to(device)

# 2. Create a synthetic dataset
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, y),
    batch_size=32
)

# 3. Create a standard evolution system
evolution_system = create_standard_evolution_system()

# 4. Create a network context
context = NetworkContext(network, dataloader, device)

# 5. Evolve the network
evolved_context = evolution_system.evolve_network(context, num_iterations=3)

print("Evolution complete!")
print(f"Final network has {evolved_context.network.get_network_stats()['total_connections']} connections.")
```

## Usage

### Running Experiments

The project includes several example experiments that demonstrate how to use the framework.

```bash
# Run the main example for the composable evolution system
pixi run python examples/composable_evolution_example.py

# Run an experiment focused on modular metrics
pixi run python examples/modular_metrics_example.py
```

### Running Tests

The project uses `pytest` for testing.

```bash
# Run the full test suite
pixi run pytest tests/
```

## Architecture

The project has been refactored to a modular, composable architecture. The core components are:

1.  **`ComposableEvolutionSystem`** (`src/structure_net/evolution/components/evolution_system.py`)
    - The central orchestrator that manages the evolution process.
    - Coordinates the execution of analyzers, strategies, and trainers.

2.  **Analyzers** (`src/structure_net/evolution/components/analyzers.py`)
    - Components that inspect the network and produce metrics.
    - Examples: `StandardExtremaAnalyzer`, `NetworkStatsAnalyzer`.

3.  **Growth Strategies** (`src/structure_net/evolution/components/strategies.py`)
    - Components that use the analysis results to propose and execute changes to the network.
    - Examples: `ExtremaGrowthStrategy`, `InformationFlowGrowthStrategy`.

4.  **`StandardNetworkTrainer`** (`src/structure_net/evolution/components/evolution_system.py`)
    - A dedicated component that handles the training loop within an evolution cycle.

5.  **Core Network Components** (`src/structure_net/core/`)
    - **`StandardSparseLayer`**: The fundamental building block for all sparse networks.
    - **`create_standard_network`**: The canonical factory function for creating new networks.

6.  **Standardized Logging** (`src/structure_net/logging/`)
    - A robust system for logging experiment data, using Pydantic schemas and a local queue to ensure data integrity and offline support.

## File Structure

```
structure_net/
├── examples/                   # Example implementations
│   ├── composable_evolution_example.py
│   └── modular_metrics_example.py
├── src/structure_net/          # Core package
│   ├── core/                   # Core network components (layers, factory)
│   ├── evolution/              # The evolution system
│   │   ├── components/         # Analyzers, strategies, trainers
│   │   ├── metrics/            # GPU-accelerated metrics
│   │   └── interfaces.py       # Core interfaces for the composable system
│   ├── logging/                # Standardized logging system
│   └── profiling/              # Performance profiling system
├── tests/                      # Pytest test suite
│   ├── conftest.py
│   ├── test_core_functionality.py
│   └── test_evolution.py
├── pyproject.toml             # Project configuration
└── README.md                  # This documentation
```

## Research Context

This project is a research platform for exploring:

- **Dynamic Network Growth**: How networks can grow from minimal to full connectivity in a metrics-driven way.
- **Composable Evolution**: Building complex growth behaviors from simple, reusable components.
- **Metrics-Driven Architecture Search**: Using deep analysis of a network's internal state to guide its evolution.

## Contributing

1.  Fork the repository.
2.  Create a feature branch.
3.  Make changes and add corresponding tests in the `tests/` directory.
4.  Submit a pull request.

## License

[Add your license here]