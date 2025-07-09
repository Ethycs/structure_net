# Ultimate Structure Net Stress Test

This is the most comprehensive test of the entire structure_net ecosystem, designed to push every component to its absolute limits while demonstrating the full capabilities of the system.

## 🚀 What This Test Does

The Ultimate Stress Test is a **tournament-style architecture evolution experiment** that combines:

### Core Features
- **Multi-GPU Parallel Processing**: Uses `torch.multiprocessing` to saturate all available GPUs
- **Tournament Evolution**: Competing architectures evolve over generations through mutation and crossover
- **CIFAR-10 Dataset**: Full-scale image classification with 32x32x3 RGB images
- **Memory Optimization**: Dynamic batch size adjustment and memory cleanup

### Advanced Systems Integration
- **All Adaptive Learning Rate Strategies**: BASIC, ADVANCED, COMPREHENSIVE, and ULTIMATE strategies running simultaneously
- **Complete Metrics System**: Mutual information, activity analysis, sensitivity analysis, and graph analysis
- **Comprehensive Profiling**: Memory, compute, and I/O profiling with WandB integration
- **Standardized Logging**: Pydantic-validated experiment tracking with artifact management
- **Residual Block Insertion**: Dynamic insertion of sparse residual blocks during growth
- **Extrema-Driven Growth**: Network evolution based on activation extrema analysis

## 🏗️ Architecture

### Tournament System
```
Population (64 competitors) → Parallel GPU Evaluation → Selection → Mutation → Next Generation
```

### Multi-GPU Processing
```
GPU 0: Process 0, Process 1
GPU 1: Process 0, Process 1
...
GPU N: Process 0, Process 1
```

### Feature Integration
```
Network Creation → Residual Blocks → Adaptive LR → Metrics → Growth → Profiling → Logging
```

## 🔧 Configuration Options

### Tournament Parameters
- `tournament_size`: Number of competing architectures (default: 64)
- `generations`: Number of evolution generations (default: 10)
- `survivors_per_generation`: Top performers kept each generation (default: 16)
- `mutation_rate`: Probability of architecture mutation (default: 0.3)

### Training Parameters
- `epochs_per_generation`: Training epochs per competitor (default: 20)
- `batch_size_base`: Base batch size (auto-optimized per GPU)
- `learning_rate_strategies`: All 4 strategies used by default

### Growth and Evolution
- `enable_growth`: Extrema-driven network growth (default: True)
- `enable_residual_blocks`: Sparse residual block insertion (default: True)
- `growth_frequency`: Growth attempt frequency in epochs (default: 5)
- `max_layers`: Maximum network depth (default: 20)

### Metrics and Profiling
- `enable_comprehensive_metrics`: Full metrics suite (default: True)
- `enable_profiling`: System profiling (default: True)
- `metrics_frequency`: Metrics computation frequency (default: 2 epochs)

### Memory Optimization
- `gradient_checkpointing`: Memory-efficient gradients (default: True)
- `mixed_precision`: FP16 training (default: True)
- `memory_cleanup_frequency`: GPU memory cleanup (default: 10 epochs)

## 🚀 Running the Stress Test

### Quick Start
```bash
# Run with default balanced configuration
pixi run python run_stress_test.py

# Quick test (smaller scale)
pixi run python run_stress_test.py --quick

# Full stress test (maximum scale)
pixi run python run_stress_test.py --full
```

### Advanced Usage
```bash
# Custom configuration
pixi run python run_stress_test.py \
    --tournament-size 128 \
    --generations 15 \
    --epochs 25 \
    --no-profiling

# Minimal test (fastest)
pixi run python run_stress_test.py \
    --quick \
    --tournament-size 8 \
    --generations 2 \
    --epochs 5 \
    --no-metrics \
    --no-profiling
```

### Direct Execution
```bash
# Run the stress test directly
pixi run python experiments/ultimate_stress_test.py
```

## 📊 What Gets Tested

### 1. GPU Seed Hunter Integration
- CIFAR-10 dataset loading and caching
- Multi-GPU memory management
- Parallel seed evaluation

### 2. Adaptive Learning Rate Systems
- **BASIC**: Exponential backoff, layerwise rates, soft clamping
- **ADVANCED**: Extrema phase detection, layer age awareness
- **COMPREHENSIVE**: Multi-scale learning, sparsity awareness
- **ULTIMATE**: Unified system combining all strategies

### 3. Complete Metrics System
- **Mutual Information**: Information flow between layers
- **Activity Analysis**: Neuron activation patterns and health
- **Sensitivity Analysis**: Gradient-based bottleneck detection
- **Graph Analysis**: Network connectivity and reachability

### 4. Growth and Evolution
- **Extrema Detection**: Identification of saturated/dead neurons
- **Network Growth**: Adding layers and connections
- **Residual Blocks**: Sparse residual connections for gradient flow
- **Architecture Evolution**: Tournament-based optimization

### 5. Profiling and Logging
- **Memory Profiling**: GPU memory usage tracking
- **Compute Profiling**: Operation timing and efficiency
- **Standardized Logging**: Pydantic-validated experiment records
- **WandB Integration**: Artifact management and visualization

## 📈 Expected Results

### Performance Metrics
- **Throughput**: 10-50+ experiments per second (depending on hardware)
- **Accuracy**: 60-80% on CIFAR-10 (varies by architecture)
- **Efficiency**: Accuracy per parameter optimized through evolution

### System Utilization
- **GPU Memory**: 80% utilization target with dynamic optimization
- **Multi-GPU**: All available GPUs saturated with parallel processes
- **CPU**: Efficient data loading and preprocessing

### Evolution Outcomes
- **Architecture Optimization**: Better performing networks through generations
- **Growth Events**: Successful network expansion based on extrema analysis
- **Learning Rate Adaptation**: Optimal rates discovered for each architecture

## 🔍 Monitoring and Analysis

### Real-time Output
```
🏁 GENERATION 1/5: Evaluating 64 competitors
🔥 GPU 0 Process 0: Training arch_001
   Architecture: [3072, 512, 256, 128, 10]
   LR Strategy: ULTIMATE
   Batch size: 256
     Epoch 0: Acc=0.234, Loss=2.1234
     🌱 Attempting growth at epoch 5
     ✅ Growth successful: [512, 256, 128] -> [512, 384, 256, 128]
📊 Progress: 32/64 completed
```

### Generated Files
- `stress_test_results_YYYYMMDD_HHMMSS.json`: Complete tournament results
- `stress_test_profiling_YYYYMMDD_HHMMSS/`: Profiling data
- `experiment_queue/`: Logged experiment data
- `experiment_sent/`: Successfully uploaded experiments

### WandB Integration
- Experiment artifacts automatically uploaded
- Real-time metrics tracking
- Profiling data visualization
- Architecture evolution tracking

## 🛠️ System Requirements

### Minimum Requirements
- **GPU**: 4GB VRAM (single GPU)
- **RAM**: 8GB system memory
- **Storage**: 2GB free space
- **Python**: 3.8+ with PyTorch and CUDA

### Recommended Configuration
- **GPU**: 16GB+ VRAM (multiple GPUs preferred)
- **RAM**: 32GB+ system memory
- **Storage**: 10GB+ free space (for profiling data)
- **Network**: Internet connection for WandB logging

### Optimal Setup
- **GPU**: Multiple high-memory GPUs (A100, V100, RTX 4090)
- **RAM**: 64GB+ system memory
- **Storage**: SSD with 50GB+ free space
- **Network**: High-speed internet for artifact uploads

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Reduce batch size and tournament size
pixi run python run_stress_test.py --tournament-size 16 --quick
```

#### Import Errors
```bash
# Ensure you're in the structure_net directory
cd /path/to/structure_net
pixi run python run_stress_test.py
```

#### Slow Performance
```bash
# Disable expensive features
pixi run python run_stress_test.py --no-metrics --no-profiling --no-growth
```

#### WandB Issues
```bash
# Run without WandB integration
# Edit logging_config in the code to set enable_wandb=False
```

### Performance Tuning

#### For Limited Memory
- Reduce `tournament_size` to 16-32
- Reduce `epochs_per_generation` to 10-15
- Disable comprehensive metrics with `--no-metrics`
- Disable profiling with `--no-profiling`

#### For Maximum Speed
- Use `--quick` mode
- Reduce `generations` to 2-3
- Disable growth with `--no-growth`
- Use smaller architectures

#### For Maximum Coverage
- Use `--full` mode
- Increase `tournament_size` to 128+
- Increase `generations` to 15+
- Enable all features (default)

## 📚 Understanding the Output

### Tournament Results Structure
```json
{
  "config": { /* StressTestConfig parameters */ },
  "generations": [
    {
      "generation": 0,
      "population_size": 64,
      "survivors": 16,
      "best_fitness": 0.456,
      "avg_fitness": 0.234,
      "duration": 123.45,
      "top_performers": [ /* Top 5 architectures */ ]
    }
  ],
  "final_best": { /* Champion architecture details */ },
  "system_stats": {
    "total_experiments": 320,
    "experiments_per_second": 12.34,
    "gpu_utilization": 4,
    "memory_usage": 67.8
  }
}
```

### Fitness Calculation
```python
fitness = accuracy * 0.7 + efficiency * 0.3
efficiency = accuracy / (parameters / 1000)  # Accuracy per K parameters
```

### Growth Events
- Triggered when `extrema_ratio > 0.3`
- Adds layers or connections based on extrema analysis
- Logged with before/after architectures
- Tracked in tournament results

## 🎯 Success Criteria

A successful stress test should demonstrate:

1. **System Integration**: All components work together without conflicts
2. **Scalability**: Efficient utilization of available hardware resources
3. **Evolution**: Improving architectures over generations
4. **Robustness**: Handling failures gracefully and continuing execution
5. **Logging**: Complete experiment tracking and artifact management
6. **Performance**: Reasonable throughput given system constraints

## 🔬 Research Applications

This stress test can be used for:

- **Architecture Search**: Finding optimal sparse network architectures
- **Learning Rate Research**: Comparing adaptive learning rate strategies
- **Growth Algorithm Validation**: Testing extrema-driven growth approaches
- **System Benchmarking**: Evaluating multi-GPU training performance
- **Metrics Validation**: Verifying comprehensive network analysis tools

## 📝 Citation

If you use this stress test in your research, please cite:

```bibtex
@software{structure_net_stress_test,
  title={Ultimate Structure Net Stress Test},
  author={Structure Net Development Team},
  year={2025},
  url={https://github.com/Ethycs/structure_net}
}
```

---

**Happy stress testing! 🚀**

For questions or issues, please check the troubleshooting section above or open an issue in the repository.
