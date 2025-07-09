# Migration Guide: Ultimate Stress Test to Neural Architecture Lab

This guide helps you transition from the tournament-based `ultimate_stress_test.py` to the hypothesis-driven Neural Architecture Lab (NAL) system.

## Overview of Changes

### Old Approach (ultimate_stress_test.py)
- Tournament-based evolution with fixed structure
- Hard-coded evaluation logic
- Limited analysis capabilities
- Monolithic design

### New Approach (Neural Architecture Lab)
- Hypothesis-driven experiments
- Modular and extensible design
- Comprehensive metrics integration
- Statistical analysis built-in
- Reusable experiment framework

## Key Concepts Mapping

| Old Concept | NAL Equivalent | Benefits |
|------------|----------------|----------|
| `ArchitectureTournament` | `Hypothesis` + `Experiment` | More flexible, reusable |
| `evaluate_competitor` | `ExperimentRunner` | Parallel execution, better error handling |
| Tournament generations | Hypothesis parameter space | Systematic exploration |
| Manual result analysis | `InsightExtractor` + `StatisticalAnalyzer` | Automated insights |
| Fixed metrics | `CompleteMetricsSystem` integration | Comprehensive analysis |

## Migration Examples

### 1. Basic Tournament → NAL Hypothesis

**Old way:**
```python
tournament = ArchitectureTournament(config)
results = tournament.run_tournament()
```

**NAL way:**
```python
# Define hypothesis
tournament_hypothesis = Hypothesis(
    id="architecture_tournament",
    name="Architecture Evolution Tournament",
    category=HypothesisCategory.ARCHITECTURE,
    parameter_space={
        'tournament_size': [32],
        'generations': [5],
        'mutation_strategy': ['mixed']
    },
    # ... other configuration
)

# Run with NAL
lab = NeuralArchitectureLab(config)
lab.register_hypothesis(tournament_hypothesis)
results = await lab.test_hypothesis("architecture_tournament")
```

### 2. Seed Model Loading

**Old way:**
```python
tournament = SeedArchitectureTournament(config, seed_models)
# Seeds handled internally
```

**NAL way:**
```python
# Create specific hypothesis for seed models
seed_hypothesis = create_seed_evolution_hypothesis(seed_models)
lab.register_hypothesis(seed_hypothesis)

# Seeds become part of parameter space
parameter_space={
    'use_seed': [True, False],
    'seed_path': seed_models,
    # ... other parameters
}
```

### 3. Parallel Evaluation

**Old way:**
```python
# Complex multiprocessing setup
with mp.Pool(processes=num_processes) as pool:
    results = pool.map(evaluate_competitor, competitors)
```

**NAL way:**
```python
# Handled automatically by NAL
config = LabConfig(
    max_parallel_experiments=8,
    device_ids=[0, 1]
)
# NAL manages parallel execution internally
```

### 4. Results Analysis

**Old way:**
```python
# Manual analysis
best_accuracy = max(results, key=lambda x: x['accuracy'])
print(f"Best: {best_accuracy}")
```

**NAL way:**
```python
# Automatic statistical analysis
result = await lab.test_hypothesis(hypothesis_id)
print(f"Confirmed: {result.confirmed}")
print(f"Effect size: {result.effect_size}")
print(f"Key insights: {result.key_insights}")

# Access comprehensive metrics
metrics_analysis = result.statistical_summary
```

## Step-by-Step Migration

### Step 1: Identify Your Experiments
List what you're testing:
- Architecture variations
- Learning rate strategies
- Growth patterns
- Seed model effectiveness

### Step 2: Convert to Hypotheses
For each experiment:
```python
hypothesis = Hypothesis(
    id="unique_id",
    name="Descriptive Name",
    question="What are you testing?",
    prediction="Expected outcome",
    parameter_space={...},  # Variables to test
    control_parameters={...},  # Fixed settings
    success_metrics={...}  # Success criteria
)
```

### Step 3: Implement Test Functions
NAL uses test functions that return (model, metrics):
```python
def test_my_hypothesis(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    # Your experiment logic
    model = create_model(config)
    metrics = train_and_evaluate(model, config)
    return model, metrics
```

### Step 4: Configure NAL
```python
config = LabConfig(
    max_parallel_experiments=8,
    results_dir="my_results",
    save_best_models=True,
    # ... other settings
)
lab = NeuralArchitectureLab(config)
```

### Step 5: Run Experiments
```python
# Register all hypotheses
for hypothesis in my_hypotheses:
    lab.register_hypothesis(hypothesis)

# Run all
results = await lab.run_all_hypotheses()
```

## Feature Comparison

### Metrics and Analysis

**Old system:**
- Basic accuracy, loss tracking
- Manual metric calculation
- Limited insight extraction

**NAL system:**
- Full integration with CompleteMetricsSystem
- Automatic statistical significance testing
- Extrema analysis
- Topological and homological analysis
- Autocorrelation performance tracking
- Automated insight generation

### Growth and Evolution

**Old system:**
- Fixed growth strategies
- Manual architecture mutation

**NAL system:**
- Parameterized growth strategies
- Systematic testing of growth hypotheses
- Integration with IntegratedGrowthSystem v2

### Learning Rate Management

**Old system:**
- Basic strategy selection
- Limited adaptation

**NAL system:**
- Full adaptive learning rate testing
- Comparison across all strategies
- Statistical analysis of effectiveness

## Example: Complete Migration

Here's a complete example migrating a tournament experiment:

```python
# Old ultimate_stress_test.py approach
config = StressTestConfig(
    tournament_size=32,
    generations=5,
    epochs_per_generation=20
)
tournament = ArchitectureTournament(config)
results = tournament.run_tournament()

# New NAL approach
from src.neural_architecture_lab import NeuralArchitectureLab, LabConfig, Hypothesis

# 1. Define what you're testing
hypothesis = Hypothesis(
    id="tournament_evolution",
    name="Tournament Architecture Evolution",
    description="Test tournament-based architecture evolution",
    category=HypothesisCategory.ARCHITECTURE,
    question="Which architectures emerge from competitive evolution?",
    prediction="Deeper architectures with residual connections will dominate",
    test_function=tournament_evolution_test,
    parameter_space={
        'tournament_size': [16, 32],
        'generations': [5, 10],
        'mutation_rate': [0.1, 0.2, 0.3],
        'selection_pressure': [0.1, 0.2]
    },
    control_parameters={
        'dataset': 'cifar10',
        'epochs_per_generation': 20,
        'batch_size': 128
    },
    success_metrics={
        'accuracy': 0.55,
        'improvement': 1.1  # 10% over baseline
    }
)

# 2. Configure the lab
config = LabConfig(
    max_parallel_experiments=8,
    min_experiments_per_hypothesis=5,
    results_dir="tournament_results"
)

# 3. Run experiments
lab = NeuralArchitectureLab(config)
lab.register_hypothesis(hypothesis)
result = await lab.test_hypothesis("tournament_evolution")

# 4. Analyze results
print(f"Hypothesis confirmed: {result.confirmed}")
print(f"Statistical significance: p={result.statistical_summary['p_value']}")
print(f"Best configuration: {result.best_parameters}")
for insight in result.key_insights:
    print(f"- {insight}")
```

## Benefits of Migration

1. **Scientific Rigor**: Hypothesis-driven approach with statistical validation
2. **Reusability**: Experiments can be easily repeated and modified
3. **Better Analysis**: Automatic insight extraction and metric integration
4. **Modularity**: Easy to add new experiments without changing core code
5. **Comprehensive Metrics**: Full access to structure_net's metrics system
6. **Parallel Execution**: Built-in support for efficient parallel experiments
7. **Result Tracking**: Automatic result storage and report generation

## Tips for Successful Migration

1. Start small - migrate one experiment type at a time
2. Use the existing hypothesis library as templates
3. Leverage NAL's automatic parallel execution
4. Take advantage of statistical analysis features
5. Use adaptive hypotheses for follow-up experiments
6. Review generated insights to guide future experiments

## Getting Help

- See `examples/neural_architecture_lab_demo.py` for basic usage
- Check `nal_stress_test_with_seed.py` for seed model integration
- Review `nal_tournament_evolution.py` for tournament-style experiments
- Examine the hypothesis library for common patterns

The NAL system provides a more powerful and flexible framework for neural architecture experiments while maintaining the ability to run tournament-style evolution when needed.