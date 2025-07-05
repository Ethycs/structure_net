# Structure Net: Multi-Scale Snapshots Neural Network

A PyTorch implementation of neural networks that grow dynamically during training based on extrema detection and multi-scale snapshot preservation, implementing the "Experiment 1: Multi-Scale Snapshots" from the research specification.

## Overview

This implementation creates neural networks that:
- Start with minimal connectivity (0.01% of possible connections)
- Grow dynamically based on gradient variance spikes and extrema detection
- Route connections from high extrema to low extrema
- Save multi-scale snapshots at different growth phases
- Support ensemble inference across different scales

## Key Features

### 🌱 Dynamic Growth
- **Minimal Start**: Networks begin with only 0.01% connectivity
- **Gradient-Based Triggers**: Growth occurs when gradient variance spikes are detected
- **Credit System**: 10 credits per gradient spike, 100 credits trigger growth
- **Phase-Based Limits**: Coarse (10), Medium (50), Fine (200) structures per phase

### 🎯 Extrema-Based Routing
- **High→Low Routing**: Connections route from saturated to unsaturated neurons
- **Fan-out Control**: Maximum 3-4 connections per high extrema
- **Load Balancing**: Prevents overconnection to single neurons
- **Reciprocal Connections**: 20% chance of bidirectional connections

### 📸 Multi-Scale Snapshots
- **Growth-Triggered Saves**: Snapshots saved at each growth event
- **Performance-Based**: Save when performance improves >2%
- **Delta Compression**: Efficient storage using weight deltas
- **Phase Preservation**: Separate snapshots for coarse, medium, fine phases

### 🔄 Renormalization Flow
- **Scale Separation**: Different phases handle different feature scales
- **Information Flow**: Extrema indicate when to change scales
- **Multi-Scale Inference**: Can use individual snapshots or ensembles

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd structure_net

# Install with pixi (recommended)
pixi install

# Or install with pip
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.structure_net import create_multi_scale_network
import torch

# Create a multi-scale network
network = create_multi_scale_network(
    input_size=784,      # MNIST: 28x28 flattened
    hidden_sizes=[256, 128],
    output_size=10,      # 10 classes
    sparsity=0.0001,     # 0.01% initial connectivity
    activation='tanh'
)

# Check initial connectivity
stats = network.network.get_connectivity_stats()
print(f"Initial connectivity: {stats['connectivity_ratio']:.6f}")
print(f"Total connections: {stats['total_active_connections']}")

# Forward pass
x = torch.randn(32, 784)
output = network(x)
print(f"Output shape: {output.shape}")
```

### Training with Growth

```python
import torch.nn as nn
import torch.optim as optim

# Setup training
optimizer = optim.Adam(network.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop with automatic growth
for epoch in range(50):
    epoch_stats = network.train_epoch(
        train_loader, optimizer, criterion, epoch
    )
    
    if epoch_stats['growth_events'] > 0:
        print(f"Growth at epoch {epoch}: "
              f"{epoch_stats['connections_added']} connections added")
```

### Main Experiment

Run the main Experiment 1 implementation:

```bash
# Run Experiment 1 with default settings
pixi run python experiment_1.py

# Custom settings
pixi run python experiment_1.py \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.002 \
    --device auto
```

### MNIST Experiment

Run the complete MNIST experiment:

```bash
# Run with default settings
pixi run python examples/mnist_experiment.py

# Custom settings
pixi run python examples/mnist_experiment.py \
    --epochs 100 \
    --sparsity 0.0001 \
    --activation tanh \
    --hidden-sizes 256 128 \
    --lr 0.001
```

## Architecture

### Core Components

1. **MinimalNetwork** (`src/structure_net/core/minimal_network.py`)
   - Sparse neural network with dynamic connectivity
   - Extrema detection for growth triggers
   - Connection mask management

2. **GrowthScheduler** (`src/structure_net/core/growth_scheduler.py`)
   - Gradient variance spike detection
   - Credit system management
   - Growth phase determination

3. **ConnectionRouter** (`src/structure_net/core/connection_router.py`)
   - High→Low extrema routing
   - Fan-out and load balancing
   - Reciprocal connection generation

4. **SnapshotManager** (`src/structure_net/snapshots/snapshot_manager.py`)
   - Multi-scale snapshot saving
   - Delta-based compression
   - Performance-triggered saves

5. **MultiScaleNetwork** (`src/structure_net/models/multi_scale_network.py`)
   - Main integration class
   - Training orchestration
   - Growth coordination

### Growth Rules Implementation

The implementation follows all 13 rules from the experiment specification:

1. **Initialization**: 0.01% connectivity, Xavier/He weights
2. **Growth Detection**: Gradient variance spikes (50% change)
3. **Extrema Detection**: >0.95 (high), <0.05 (low) for tanh/sigmoid
4. **Connection Routing**: High→Low with 3-layer search radius
5. **Vertical Cloning**: Adjacent layer clones with 0.7x weights
6. **Growth Economy**: 10 credits/spike, 100 threshold
7. **Structural Limits**: 10/50/200 for coarse/medium/fine
8. **Snapshot Saving**: Growth events + performance improvements
9. **Load Balancing**: Max 5 incoming connections per neuron
10. **Training Continuation**: 10-epoch stabilization periods
11. **Multi-Scale Preservation**: Phase-based snapshot organization
12. **Termination**: Epoch limits or extrema absence
13. **Ensemble Inference**: Individual or combined snapshot usage

## Experiment Results

### Expected Behavior

- **Phase 1 (Epochs 0-50)**: Coarse features, sparse growth
- **Phase 2 (Epochs 50-100)**: Medium features, moderate growth  
- **Phase 3 (Epochs 100+)**: Fine features, dense growth

### Performance Metrics

- **Connectivity Growth**: From 0.01% to ~1-5% over training
- **Accuracy**: Competitive with standard networks despite sparsity
- **Growth Events**: 5-15 growth events across all phases
- **Snapshots**: 3-8 snapshots capturing different scales

## Hardware Requirements

### Recommended Setup
- **GPU**: NVIDIA GeForce RTX 2060s or better
- **Memory**: 8GB+ RAM, 6GB+ VRAM
- **Storage**: 1GB+ for snapshots and data

### CPU Fallback
The implementation automatically falls back to CPU if CUDA is unavailable, though training will be slower.

## File Structure

```
structure_net/
├── experiment_1.py             # 🎯 Main Experiment 1 implementation
├── src/structure_net/          # Core package
│   ├── core/                   # Core components
│   │   ├── minimal_network.py  # Sparse network implementation
│   │   ├── growth_scheduler.py # Growth detection and scheduling
│   │   └── connection_router.py # Extrema-based routing
│   ├── models/                 # High-level models
│   │   └── multi_scale_network.py # Main network class
│   ├── snapshots/              # Snapshot management
│   │   └── snapshot_manager.py # Multi-scale snapshot saving
│   └── __init__.py            # Package exports
├── examples/                   # Example implementations
│   ├── mnist_experiment.py    # MNIST-specific experiment
│   ├── true_multiscale_experiment.py # Multi-scale concept demo
│   └── improved_multiscale_experiment.py # Transfer learning demo
├── comprehensive_test_suite.py # Complete test suite
├── test_basic.py              # Basic functionality tests
├── check_cuda.py              # CUDA availability check
├── experiment 1.md            # Experiment specification
├── pyproject.toml             # Project configuration
└── README.md                  # This documentation
```

## Testing

### Basic Test
```bash
pixi run python test_basic.py
```

### Full MNIST Test
```bash
pixi run python examples/mnist_experiment.py --epochs 10
```

## Research Context

This implementation is based on the "Multi-Scale Snapshots" experiment, which explores:

- **Dynamic Network Growth**: How networks can grow from minimal to full connectivity
- **Extrema-Based Routing**: Using activation extrema as growth signals
- **Multi-Scale Learning**: Capturing different feature scales in separate snapshots
- **Renormalization Theory**: Applying physics concepts to neural network growth

### Key Insights

1. **Sparse Start**: Networks can achieve good performance starting from minimal connectivity
2. **Gradient Signals**: Gradient variance spikes indicate when growth is needed
3. **Scale Separation**: Different growth phases naturally capture different feature scales
4. **Snapshot Ensembles**: Multiple snapshots can be combined for better performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

[Add your license here]

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{structure_net_2025,
  title={Structure Net: Multi-Scale Snapshots Neural Network},
  author={Ethycs},
  year={2025},
  url={https://github.com/Ethycs/structure_net}
}
```

## Acknowledgments

This implementation is based on research into dynamic neural network growth and multi-scale learning, drawing inspiration from renormalization group theory in physics.
