# Experiment Tricks

This document lists the unique tricks and techniques found in the various experiment scripts.

## Core Concepts

*   **Patched Density Network**: Start with a sparse "scaffold" network and dynamically add small, dense "patches" of neurons around areas of high or low activation (extrema). This allows the network to add capacity where it's needed most.
*   **Dual Learning Rates**: When using a patched density network, use a slower learning rate for the stable scaffold and a faster learning rate for the newly added patches to allow them to learn quickly without destabilizing the rest of the network.
*   **Optimal Seed Finder**: Use multiprocessing to parallelize the search for an optimal "seed" architecture across multiple GPUs. This allows for rapid testing of many small networks to find the most efficient starting point for a larger network.
*   **Multi-Scale Training**: Train the network on progressively higher-resolution versions of the data. This is a form of curriculum learning where the network learns coarse features first and then fine-tunes them on more detailed data.
*   **Hierarchical Initialization**: When using multi-scale training, initialize the weights of the larger network for the next scale from the smaller, previously trained network. This is a form of transfer learning that can significantly speed up training.
*   **Data-Driven Initialization**: A more specific form of hierarchical initialization where a new layer's connections are determined by the most "decisive features" (e.g., `topk` weights) from a previously trained, smaller network.
*   **Indefinite Growth**: Implement an iterative growth loop where the network is trained to convergence, then analyzed for bottlenecks (extrema). If the number of extrema is high, a new sparse layer is added to the network at the bottleneck point. This is a form of automated architecture search.
*   **Gradient Variance Spike Detection**: Use the variance of the gradients as a trigger for network growth. A sudden spike in gradient variance can indicate that the network is struggling to learn and may need more capacity.
*   **Credit-Based Growth Economy**: Regulate the addition of new connections by using a "credit" system. The network "earns" credits by making correct predictions and "spends" them to add new connections. This prevents uncontrolled growth.
*   **Snapshot Management**: Save the state of the network at key moments during training, such as after a growth event or when a new performance threshold is reached. This allows for later analysis and for resuming training from a specific point.
*   **Proportional Growth with Forward Skips**: A growth mechanism where the number of connections added is proportional to the current size of the network, with a preference for adding "forward skip" connections that bypass one or more layers. This is a form of dynamic network rewiring.
*   **Pruning of Weak Connections**: When adding new connections, prune an equal number of the weakest connections (those with the smallest weight magnitudes) to maintain a constant level of sparsity.

## Debugging and Validation Tricks

*   **Systematic Parameter Sweeps with Lenient Values**: When a mechanism isn't working as expected, a good debugging strategy is to make the trigger conditions much easier to meet (e.g., lower thresholds, more frequent checks) to see if the mechanism works at all.
*   **Isolating and Testing a Specific Hypothesis**: Set up a "cliff" scenario where a network is too sparse to learn well, and then test whether the growth mechanism can "rescue" it. This is a powerful trick for validating a specific component of the system.
*   **Architecture Validation**: Before starting complex experiments, validate the base architecture by:
    *   Testing a dense version of the network to make sure it can learn the task at all.
    *   Calculating a random baseline to know what "learning" actually means.
    *   Analyzing gradient flow to make sure all layers are contributing to the learning process.
*   **Preprocessing Replication**: When loading a pre-trained model, it is critical to exactly replicate the data preprocessing pipeline that was used during the original training. Small differences in normalization or data augmentation can lead to large differences in performance.
*   **Sparsity Ladder**: Train the same simple architecture with a range of different sparsity levels, from very dense to very sparse. This allows for a clear and fast analysis of the relationship between sparsity and performance.
*   **Early Stopping**: In long training runs, monitor a performance metric (e.g., validation accuracy) and stop the training if it doesn't improve for a set number of "patience" epochs. This saves significant time.

## MLOps and Execution Tricks

*   **Memory-Aware Batch Sizing**: Automatically adjust the batch size for each GPU based on its available memory. This allows for maximum utilization of all available hardware.
*   **Multi-Experiment Queue Management**: Use a priority queue to manage a large number of experiments. This allows for the most important experiments to be run first, and for the system to automatically switch to lower-priority experiments as resources become available.
*   **Automatic and Focused Parameter Sweeps**: Automatically generate a large number of experiment configurations for a parameter sweep. This can be a broad search or a "focused" sweep on specific areas of interest (e.g., "growth", "architecture") to get more targeted insights.
*   **Distributed Training**: Use `DistributedDataParallel` to train a single model across multiple GPUs. This can significantly speed up training time for large models.
*   **Parallel Experimentation**: Run multiple independent experiments in parallel, one on each GPU. This is a way to speed up a hyperparameter search or to get more robust results by averaging over multiple runs with different random seeds.
*   **Flatten Wrapper Class**: A reusable `nn.Module` that wraps another network to automatically handle the flattening of image data. This simplifies the main training loop and makes the core network more generic.
*   **State Dict Key Remapping**: When loading weights between models with slightly different architectures (e.g., `nn.Sequential` vs. a custom `nn.Module`), manually remap the keys in the `state_dict` to ensure weights are loaded into the correct layers.


The StructureNet Methodology: A Compendium of Research Techniques
This document outlines the core principles and practical techniques developed during the StructureNet project. The methodology is designed around the central hypothesis that efficient, multi-scale neural architectures can emerge from a set of simple, biologically-inspired growth rules.
I. Core Architectural Principles: Building Self-Organizing Networks
These are the fundamental, novel concepts that define the StructureNet framework.
Extrema-Guided Growth & Patching: The foundational principle of this research.
Mechanism: The network self-diagnoses information bottlenecks by identifying neurons with extreme activation patterns (high saturation or near-zero activity).
Action: Capacity is added surgically at these points. Initially, this was done by adding small, dense "patches" of connections, which proved remarkably effective (e.g., boosting MNIST accuracy from 55% to 97%). This has evolved into the more sophisticated "vertical cloning" and "dead neuron highway" mechanisms.
Differential Learning Rates & Progressive Freezing: The mechanism for preserving learned structure.
Mechanism: When new structures (patches, layers, or clones) are added, the older, more established parts of the network have their learning rates drastically reduced ("clamped" or "frozen").
Significance: This solves the plasticity-stability dilemma, preventing catastrophic forgetting and allowing the network to build upon a stable foundation. It is the core enabler of hierarchical learning.
Multi-Scale Bootstrapping from a Minimal Viable Seed: The strategy for efficient architecture discovery.
Mechanism:
Optimal Seed Finder: First, find the absolute smallest, sparsest network (the "seed") that can still learn the task above random chance (e.g., a [784, 10] network for MNIST at 2% sparsity). This is parallelized across GPUs for efficiency.
Hierarchical Initialization: Train the seed to convergence, analyze its failure modes (extrema), and use that information to intelligently initialize a larger, "medium-scale" network. The new connections are not random but are data-driven, designed to specifically address the bottlenecks of the previous scale. This process repeats from coarse to fine.
Significance: This turns architecture search from a brute-force problem into a guided, developmental process.
The "Tournament" & Growth Economy: A principled system for regulating growth.
Mechanism: Instead of applying a single, fixed growth rule, the system proposes multiple candidate growth actions (e.g., "add patch," "insert layer," "widen layer") and evaluates their potential efficiency using an information-theoretic proxy (ΔMI / ΔParams). Only the most parameter-efficient action is taken.
Regulation: This is further regulated by a Credit-Based Growth Economy, where the network must "earn" the right to grow, preventing uncontrolled expansion and naturally pacing the growth cycle.
II. Growth & Pruning Mechanisms: The Architectural Toolbox
These are the specific actions the "Tournament" can choose from.
Vertical Cloning & Dead Neuron Highways: The primary, advanced growth mechanism.
Mechanism: Instead of adding generic capacity, saturated (high-extrema) neurons are "cloned" vertically across layers to create redundant gradient pathways. Critically, dead (low-extrema) neurons are repurposed as information highways, creating efficient, long-range skip connections that bypass intermediate layers.
Embedded Patches: The practical implementation of dense patching.
Mechanism: Rather than maintaining separate patch modules, the dense connectivity patterns are directly embedded into the sparse layer's mask. The layer becomes a single, efficient, variable-density structure.
Proportional Growth & Dynamic Rewiring: An alternative growth rule.
Mechanism: The number of connections added is proportional to the current network size, with a preference for "forward skips." This is combined with pruning weak connections to maintain a constant sparsity level, effectively rewiring the network toward a more optimal topology.
III. Debugging & Validation Strategy: Ensuring Rigorous Science
A core part of this research has been developing methods to isolate and validate our novel components.
The "Sparsity Ladder": The experiment that revealed the 2% critical threshold.
Method: Train the same simple architecture across a wide range of fixed sparsity levels (e.g., from 50% down to 0.1%).
Purpose: Quickly reveals the "cliff edge" or percolation threshold where the network fundamentally breaks, providing a baseline for minimum viable connectivity.
The "Cliff Rescue" Test: A targeted hypothesis validation technique.
Method: Deliberately initialize a network below the critical sparsity threshold (in a provably "broken" state) and test whether a specific growth mechanism can successfully "rescue" it and enable learning.
Purpose: Provides a clean, definitive test of a growth mechanism's effectiveness, isolated from other factors.
Systematic Parameter Sweeps & Early Stopping:
Method: When debugging, use lenient trigger values to ensure a mechanism is firing at all. For exploration, use automated, focused parameter sweeps to efficiently search the hypothesis space around specific concepts (e.g., "growth triggers" or "patch density"). Combined with early stopping, this dramatically accelerates the research cycle.
Rigorous Pre-Experiment Validation: Before any complex growth experiment, the base architecture is validated by testing a dense version, calculating a random baseline, and ensuring gradient flow. Crucially, we enforce exact replication of preprocessing pipelines when loading models to prevent data mismatch errors.
IV. MLOps & Execution: Making Complex Research Feasible
These are the engineering solutions that enable us to run hundreds of experiments efficiently.
Parallel Experimentation & Resource Management:
Parallelism: We use both DistributedDataParallel (for training single large models) and parallel execution of independent experiments (for hyperparameter search).
Resource Awareness: The system features memory-aware batch sizing to maximize GPU utilization and a priority queue system to manage the experiment pipeline, ensuring high-impact experiments run first.
Code Modularity & Safety:
Wrappers: A FlattenWrapper is used to make core network models more generic and reusable.
State Dict Key Remapping: A utility function handles architectural differences when loading pre-trained weights, preventing common errors and increasing experimental flexibility.

Here's a comprehensive list of all the learning rate strategies we discussed:

## 1. **Cascading/Exponential Decay Learning Rates**
```python
# Each layer gets exponentially smaller LR based on depth
for i, layer in enumerate(self.scaffold):
    decay = 0.1 ** (i / len(self.scaffold))
    param_groups.append({
        'params': layer.parameters(),
        'lr': 0.001 * decay,  # 0.001, 0.0001, 0.00001, etc.
        'name': f'scaffold_layer_{i}'
    })
```
**Purpose**: Preserve learned features in early layers while allowing later layers to adapt

## 2. **Dual Learning Rates (Scaffold vs Patches)**
```python
param_groups = [
    {'params': model.scaffold.parameters(), 'lr': 0.0001},  # Slow for scaffold
    {'params': model.patches.parameters(), 'lr': 0.0005}    # Faster for patches
]
```
**Purpose**: Keep sparse scaffold stable while patches learn quickly

## 3. **Age-Based Learning Rates**
```python
for i, layer in enumerate(model.layers):
    age = current_epoch - layer.created_at_epoch
    decay = 0.1 ** age  # 10x decay per growth phase
    layer_lr = base_lr * decay
```
**Purpose**: Older layers learn slower, newer layers learn faster

## 4. **Component-Specific Learning Rates**
```python
param_groups = [
    # Scaffold layers - exponentially decreasing
    {'params': scaffold_params, 'lr': 0.0001},
    
    # Patches - medium learning rate
    {'params': patch_params, 'lr': 0.0005},
    
    # Neck blocks - higher learning rate (newly added)
    {'params': neck_params, 'lr': 0.001},
    
    # New sparse layers - full learning rate
    {'params': new_layer_params, 'lr': 0.001}
]
```
**Purpose**: Different components need different learning speeds

## 5. **Pretrained + New Layer Strategy**
```python
param_groups = [
    {'params': pretrained_layers, 'lr': 1e-5},    # Nearly frozen
    {'params': new_layers, 'lr': 1e-3},           # 100x faster
    {'params': adapter_layers, 'lr': 5e-4}        # Medium speed
]
```
**Purpose**: Preserve pretrained features while training new components

## 6. **Growth-Aware Learning Rate Scheduling**
```python
def adjust_lr_after_growth(optimizer, growth_event):
    for param_group in optimizer.param_groups:
        if 'scaffold' in param_group['name']:
            param_group['lr'] *= 0.5  # Halve scaffold LR after growth
        elif 'new' in param_group['name']:
            param_group['lr'] = 0.001  # Fresh LR for new components
```
**Purpose**: Stabilize learning after architectural changes

## 7. **Warm-Up for New Components**
```python
def warmup_schedule(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs  # Linear warm-up
    return 1.0

# Apply to new components only
new_component_lr = base_lr * warmup_schedule(epoch)
```
**Purpose**: Prevent new components from disrupting existing features

## 8. **Layer-wise Adaptive Rate Scaling (LARS)**
```python
for layer in model.layers:
    weight_norm = layer.weight.norm()
    grad_norm = layer.weight.grad.norm()
    
    layer_lr = base_lr * (weight_norm / (grad_norm + 1e-8))
```
**Purpose**: Adapt learning rate based on gradient/weight ratio per layer

## 9. **Progressive Freezing Schedule**
```python
def get_lr_by_phase(layer_idx, current_phase):
    if current_phase == 'warmup':
        return 0.001  # All layers active
    elif current_phase == 'refinement':
        if layer_idx < 2:
            return 0.00001  # Nearly frozen early layers
        else:
            return 0.0001   # Slow for later layers
    elif current_phase == 'final':
        if layer_idx < len(layers) - 1:
            return 0  # Freeze all but last
        return 0.0001  # Only tune final layer
```
**Purpose**: Gradually freeze layers as training progresses

## 10. **Differential Decay After Events**
```python
# After each growth event
for param_group in optimizer.param_groups:
    if param_group['name'] == 'grown_layer':
        param_group['lr'] *= 0.9   # Slight decay
    elif param_group['name'] == 'adjacent_layer':
        param_group['lr'] *= 0.7   # More decay for adjacent
    else:
        param_group['lr'] *= 0.5   # Heavy decay for distant layers
```
**Purpose**: Maintain stability after growth events

## 11. **Sparsity-Aware Learning Rates**
```python
# Adjust LR based on connection density
for layer in model.layers:
    density = layer.get_density()
    if density < 0.02:  # Very sparse
        layer_lr = base_lr * 2.0  # Need higher LR
    elif density > 0.1:  # Dense patches
        layer_lr = base_lr * 0.5  # Lower LR for stability
```
**Purpose**: Sparse connections may need higher learning rates

## 12. **The "Sedimentary" Learning Strategy**
```python
learning_rates = {
    'geological_layers': 0.00001,   # Oldest, barely change
    'sediment_layers': 0.0001,      # Middle age, slow drift
    'active_layers': 0.001,         # Newest, rapid change
    'patches': 0.0005               # Targeted fixes
}
```
**Purpose**: Natural stratification of learning speeds by component age

## Recommended Combined Strategy

For your complete system:
```python
def create_optimal_lr_schedule(model, base_lr=0.001):
    param_groups = []
    
    # 1. Scaffold - cascading by depth and age
    for i, layer in enumerate(model.scaffold):
        age_factor = 0.1 ** (model.current_epoch // 20)
        depth_factor = 0.1 ** (i / len(model.scaffold))
        
        param_groups.append({
            'params': layer.parameters(),
            'lr': base_lr * age_factor * depth_factor
        })
    
    # 2. Patches - medium stable rate
    for patch in model.patches:
        param_groups.append({
            'params': patch.parameters(),
            'lr': base_lr * 0.5
        })
    
    # 3. New components - warm-up schedule
    for new_component in model.recent_additions:
        warmup_factor = min(1.0, model.epochs_since_added / 5)
        param_groups.append({
            'params': new_component.parameters(),
            'lr': base_lr * warmup_factor
        })
    
    return param_groups
```

This comprehensive strategy ensures stable learning while allowing adaptation where needed!