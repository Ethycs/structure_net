# Structure Net: Dynamic Neural Architecture Evolution Framework

A PyTorch-based research framework for evolving neural network architectures dynamically during training. Structure Net implements a composable, metrics-driven approach to network growth, enabling networks to adapt their topology based on internal state analysis.

## Overview

Structure Net enables neural networks to evolve their architecture during training, starting from minimal connectivity and growing based on comprehensive internal state analysis. The framework provides:

- **Dynamic Architecture Evolution**: Networks grow from sparse to complex topologies guided by performance metrics
- **Composable Evolution System**: Mix and match analyzers, growth strategies, and trainers to create custom evolution pipelines
- **Multi-Scale Growth**: Support for coarse-to-fine network development through phased evolution
- **Advanced Metrics Suite**: GPU-accelerated analysis including extrema detection, information flow, topological features, and homological analysis
- **Neural Architecture Lab (NAL)**: Scientific framework for systematic hypothesis testing on network architectures

## Key Features

### 🧬 Composable Evolution System
- **Modular Architecture**: Clean, interface-based design with swappable components:
  - **Analyzers**: Examine network state (extrema, information flow, topology, homology)
  - **Growth Strategies**: Execute architectural changes (layer addition, connection growth, residual blocks)
  - **Trainers**: Manage training loops with configurable parameters
- **Pre-built Components**: Rich library of analyzers and strategies ready to use
- **Custom Components**: Simple interface for creating your own evolution logic

### 🔬 Neural Architecture Lab (NAL)
- **Hypothesis-Driven Research**: Formalize architectural insights as testable hypotheses
- **Automated Experimentation**: Run systematic experiments with statistical analysis
- **Insight Extraction**: Automatically identify patterns and generate new hypotheses
- **Result Tracking**: Comprehensive logging and visualization of experiment outcomes

### 📊 Advanced Metrics & Analysis
- **Extrema Detection**: Identify dead and saturated neurons for targeted growth
- **Information Flow**: Analyze bottlenecks and efficiency in network communication
- **Topological Analysis**: Compute persistence diagrams and Betti numbers
- **Homological Features**: Track network complexity through homology groups
- **Graph Metrics**: Analyze connectivity patterns, clustering, and centrality
- **GPU Acceleration**: Fast computation using cuGraph and cuPy when available

### 🎯 Growth Strategies
- **Extrema-Based Growth**: Add connections/layers based on activation patterns
- **Information-Driven Growth**: Optimize architecture for information flow
- **Residual Blocks**: Add skip connections for deeper networks
- **Hybrid Strategies**: Combine multiple growth approaches
- **Multi-Scale Evolution**: Phased growth from coarse to fine structures

### 📝 Logging & Reproducibility
- **Standardized Schemas**: Pydantic-validated data structures
- **WandB Integration**: Automatic artifact versioning and tracking
- **Offline Queue**: Reliable data persistence without network dependency
- **Comprehensive Profiling**: Performance analysis and bottleneck identification

### 💾 Data System & Storage
- **Flexible Dataset Management**: Easy dataset switching without code changes
- **ChromaDB Integration**: Semantic search for experiments and architectures
- **Time Series Storage**: Efficient HDF5-based storage for large training histories
- **Memory-Efficient NAL**: Automatic offloading of experiment data to prevent memory accumulation
- **Optimized Stress Test**: Dataset sharing, disk caching, and aggressive cleanup for large-scale experiments

## Installation

### Using Pixi (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd structure_net

# Install with pixi
pixi install

# Install PyTorch with CUDA support
pixi run install-torch

# Verify CUDA installation
pixi run test-cuda
```

### Using pip
```bash
# Clone and install in development mode
git clone <repository-url>
cd structure_net
pip install -e .

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Quick Start

### Basic Network Evolution

```python
import torch
from structure_net import create_standard_network
from structure_net.evolution.components import (
    create_standard_evolution_system,
    NetworkContext
)

# Create a sparse network
network = create_standard_network(
    architecture=[784, 256, 128, 10],
    sparsity=0.01
)

# Setup device and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network.to(device)

# Create dataset (using your actual data)
train_loader = create_your_dataloader()

# Create evolution system and evolve
evolution_system = create_standard_evolution_system()
context = NetworkContext(network, train_loader, device)
evolved_context = evolution_system.evolve_network(context, num_iterations=5)

print(f"Network grew from {context.network.get_network_stats()['total_connections']} to "
      f"{evolved_context.network.get_network_stats()['total_connections']} connections")
```

### Custom Evolution Pipeline

```python
from structure_net.evolution.components import (
    ComposableEvolutionSystem,
    StandardExtremaAnalyzer,
    ExtremaGrowthStrategy,
    StandardNetworkTrainer
)

# Build custom evolution system
evolution_system = ComposableEvolutionSystem()

# Add analyzer to detect dead/saturated neurons
evolution_system.add_component(
    StandardExtremaAnalyzer(
        max_batches=10,
        dead_threshold=0.01,
        saturated_multiplier=2.5
    )
)

# Add growth strategy
evolution_system.add_component(
    ExtremaGrowthStrategy(
        extrema_threshold=0.3,
        patch_size=20,
        add_layer_on_extrema=True
    )
)

# Add trainer
evolution_system.add_component(
    StandardNetworkTrainer(epochs=5, learning_rate=0.01)
)

# Evolve network
evolved_context = evolution_system.evolve_network(context, num_iterations=3)
```

### Using the Neural Architecture Lab

```python
from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis,
    HypothesisCategory
)

# Configure lab
config = LabConfig(
    max_parallel_experiments=4,
    results_dir="./nal_results",
    device="cuda"
)

# Create lab
lab = NeuralArchitectureLab(config)

# Define hypothesis
hypothesis = Hypothesis(
    name="extrema_growth_efficiency",
    category=HypothesisCategory.GROWTH_DYNAMICS,
    question="Does extrema-based growth lead to more efficient networks?",
    prediction="Networks grown with extrema detection will achieve higher accuracy with fewer parameters"
)

# Register and test hypothesis
lab.register_hypothesis(hypothesis)
results = lab.run_hypothesis_test(hypothesis.id)

# View insights
insights = lab.extract_insights()
print(lab.generate_report())
```

## Core Components

### Network Layers
- **StandardSparseLayer**: Base sparse layer with configurable connectivity
- **ExtremaAwareSparseLayer**: Tracks activation extrema for growth decisions
- **TemporaryPatchLayer**: Temporary connections for testing growth impact

### Analyzers
- **StandardExtremaAnalyzer**: Detects dead and saturated neurons
- **NetworkStatsAnalyzer**: Computes basic network statistics
- **SimpleInformationFlowAnalyzer**: Identifies information bottlenecks
- **GraphAnalyzer**: Analyzes network topology (clustering, centrality)
- **TopologicalAnalyzer**: Computes persistence diagrams
- **HomologicalAnalyzer**: Tracks homology groups

### Growth Strategies
- **ExtremaGrowthStrategy**: Grows based on neuron activation patterns
- **InformationFlowGrowthStrategy**: Optimizes for information transmission
- **ResidualBlockGrowthStrategy**: Adds skip connections
- **HybridGrowthStrategy**: Combines multiple strategies
- **PruningGrowthStrategy**: Removes ineffective connections

### Adaptive Learning Rates
- **ExponentialBackoffScheduler**: Reduces LR on performance plateaus
- **LayerwiseAdaptiveRates**: Per-layer learning rate adjustment
- **GrowthPhaseScheduler**: Different rates for growth vs. training
- **UnifiedAdaptiveLearning**: Combines multiple LR strategies

## Usage Examples

### Running Experiments

```bash
# Basic evolution example
pixi run python examples/composable_evolution_example.py

# Multi-scale network evolution
pixi run python examples/modern_multi_scale_evolution.py

# Modular metrics demonstration
pixi run python examples/modular_metrics_example.py

# Neural Architecture Lab experiments
pixi run python examples/nal_example.py
```

### Running Tests

```bash
# Run all tests
pixi run pytest

# Run specific test modules
pixi run pytest tests/test_core_functionality.py
pixi run pytest tests/test_evolution.py
pixi run pytest tests/test_nal.py

# Run with coverage
pixi run pytest --cov=src/structure_net
```

## Architecture

### Design Principles

1. **Composable Components**: All evolution logic is built from interchangeable components
2. **Interface-Driven**: Clean interfaces enable easy extension without modifying core code
3. **Metrics-First**: All decisions are based on quantitative network analysis
4. **GPU-Optimized**: Critical paths use GPU acceleration when available

### Core Architecture

```
structure_net/
├── core/                          # Foundation components
│   ├── layers.py                  # Sparse layer implementations
│   ├── network_factory.py         # Network construction utilities
│   ├── validation.py              # Model quality validation
│   └── lsuv.py                    # Layer-sequential unit variance init
│
├── evolution/                     # Evolution system
│   ├── components/                # Composable building blocks
│   │   ├── evolution_system.py    # Main orchestrator
│   │   ├── analyzers.py           # Network analysis components
│   │   └── strategies.py          # Growth strategy implementations
│   │
│   ├── metrics/                   # Advanced analysis metrics
│   │   ├── extrema_analyzer.py    # Dead/saturated neuron detection
│   │   ├── topological_analysis.py # Persistence diagrams
│   │   ├── homological_analysis.py # Homology computation
│   │   └── graph_analysis.py      # Network topology metrics
│   │
│   └── adaptive_learning_rates/   # Learning rate strategies
│
├── models/                        # Network architectures
│   ├── minimal_network.py         # Baseline sparse network
│   └── modern_multi_scale_network.py # Multi-scale growth support
│
├── neural_architecture_lab/       # Hypothesis testing framework
│   ├── lab.py                     # Main lab implementation
│   ├── runners.py                 # Experiment execution
│   └── analyzers.py               # Result analysis
│
└── logging/                       # Experiment tracking
    ├── standardized_logging.py    # Schema-based logging
    ├── artifact_manager.py        # WandB integration
    └── schemas.py                 # Data validation schemas
```

## Advanced Usage

### Creating Custom Analyzers

```python
from structure_net.evolution.interfaces import Analyzer
from structure_net.evolution.components import NetworkContext

class MyCustomAnalyzer(Analyzer):
    def analyze(self, context: NetworkContext) -> dict:
        # Perform your analysis
        metrics = {
            "my_metric": compute_something(context.network),
            "another_metric": analyze_something_else(context)
        }
        return metrics
```

### Creating Custom Growth Strategies

```python
from structure_net.evolution.interfaces import GrowthStrategy
from structure_net.evolution.components import NetworkContext

class MyGrowthStrategy(GrowthStrategy):
    def should_grow(self, context: NetworkContext, analysis: dict) -> bool:
        # Decide whether to grow based on analysis
        return analysis.get("my_metric", 0) > self.threshold
    
    def grow(self, context: NetworkContext, analysis: dict) -> NetworkContext:
        # Implement growth logic
        # Modify network architecture
        return updated_context
```

### Multi-Scale Evolution Example

```python
# Phase 1: Coarse structure (few connections, basic topology)
coarse_system = ComposableEvolutionSystem()
coarse_system.add_component(StandardExtremaAnalyzer())
coarse_system.add_component(ExtremaGrowthStrategy(patch_size=50))
coarse_system.add_component(StandardNetworkTrainer(epochs=10))

# Phase 2: Medium detail (add layers, increase connectivity)
medium_system = ComposableEvolutionSystem()
medium_system.add_component(StandardExtremaAnalyzer())
medium_system.add_component(ExtremaGrowthStrategy(add_layer_on_extrema=True))
medium_system.add_component(StandardNetworkTrainer(epochs=5))

# Phase 3: Fine detail (optimize connections, add residuals)
fine_system = ComposableEvolutionSystem()
fine_system.add_component(SimpleInformationFlowAnalyzer())
fine_system.add_component(ResidualBlockGrowthStrategy())
fine_system.add_component(StandardNetworkTrainer(epochs=3))

# Execute phased evolution
context = NetworkContext(network, train_loader, device)
context = coarse_system.evolve_network(context, num_iterations=3)
context = medium_system.evolve_network(context, num_iterations=2)
context = fine_system.evolve_network(context, num_iterations=1)
```

## Research Applications

Structure Net provides a platform for exploring several research directions:

### Dynamic Architecture Search
- **Metrics-Driven Growth**: Networks evolve based on quantitative analysis rather than random search
- **Efficient Exploration**: Start sparse and grow only where needed
- **Multi-Scale Development**: Mimic biological development from coarse to fine structures

### Network Science
- **Topology Evolution**: Study how network structure emerges from local growth rules
- **Information Theory**: Analyze information flow and bottlenecks in evolving networks
- **Homological Analysis**: Track topological features through growth phases

### Sparse Neural Networks
- **Adaptive Sparsity**: Networks maintain efficiency while growing capacity
- **Connection Importance**: Identify critical pathways through extrema analysis
- **Pruning Strategies**: Remove connections that don't contribute to performance

### Scientific ML Research
- **Hypothesis Testing**: Formalize architectural insights as testable hypotheses
- **Systematic Experimentation**: Run controlled experiments with statistical validation
- **Insight Generation**: Automatically identify patterns across experiments

## Performance Considerations

### GPU Acceleration
- Metrics computation uses cuGraph and cuPy when available
- Automatic fallback to CPU for unsupported operations
- Profiling system tracks performance bottlenecks

### Memory Efficiency
- Sparse representations reduce memory footprint
- Lazy metric computation avoids unnecessary calculations
- Batch processing for large networks

### Scalability
- Distributed experiment execution in Neural Architecture Lab
- Parallel hypothesis testing
- Efficient serialization for large models

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork and Clone**: Fork the repository and clone locally
2. **Create Branch**: Use descriptive branch names (e.g., `feature/new-analyzer`)
3. **Write Tests**: Add tests for new functionality in `tests/`
4. **Documentation**: Update docstrings and README as needed
5. **Code Style**: Follow existing patterns and conventions
6. **Pull Request**: Submit PR with clear description of changes

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/structure_net.git
cd structure_net

# Install in development mode
pixi install
pixi run install-torch

# Run tests to verify setup
pixi run pytest

# Create new branch
git checkout -b feature/my-new-feature
```

## Citation

If you use Structure Net in your research, please cite:

```bibtex
@software{structure_net,
  title = {Structure Net: Dynamic Neural Architecture Evolution Framework},
  author = {Ethycs},
  year = {2024},
  url = {https://github.com/ethycs/structure_net}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- RAPIDS team for GPU-accelerated graph analytics
- The broader neural architecture search community for inspiration

---

For questions, issues, or discussions, please open an issue on GitHub.