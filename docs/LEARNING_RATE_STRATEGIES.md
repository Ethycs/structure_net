# Learning Rate Strategies in StructureNet

This document provides a comprehensive overview of the various learning rate adaptation strategies implemented and discussed in the StructureNet project. These strategies are crucial for maintaining stability and promoting efficient learning in dynamically growing networks.

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
**Purpose**: Preserve learned features in early layers while allowing later layers to adapt.

## 2. **Dual Learning Rates (Scaffold vs Patches)**
```python
param_groups = [
    {'params': model.scaffold.parameters(), 'lr': 0.0001},  # Slow for scaffold
    {'params': model.patches.parameters(), 'lr': 0.0005}    # Faster for patches
]
```
**Purpose**: Keep the sparse scaffold stable while newly added patches learn quickly.

## 3. **Age-Based Learning Rates**
```python
for i, layer in enumerate(model.layers):
    age = current_epoch - layer.created_at_epoch
    decay = 0.1 ** age  # 10x decay per growth phase
    layer_lr = base_lr * decay
```
**Purpose**: Older, more established layers learn slower, while newer layers learn faster. This is a core principle of the "sedimentary" learning strategy.

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
**Purpose**: Acknowledges that different architectural components have different learning needs and stability.

## 5. **Pretrained + New Layer Strategy**
```python
param_groups = [
    {'params': pretrained_layers, 'lr': 1e-5},    # Nearly frozen
    {'params': new_layers, 'lr': 1e-3},           # 100x faster
    {'params': adapter_layers, 'lr': 5e-4}        # Medium speed for layers connecting old and new
]
```
**Purpose**: Preserve pretrained features while efficiently training new components.

## 6. **Growth-Aware Learning Rate Scheduling**
```python
def adjust_lr_after_growth(optimizer, growth_event):
    for param_group in optimizer.param_groups:
        if 'scaffold' in param_group['name']:
            param_group['lr'] *= 0.5  # Halve scaffold LR after growth
        elif 'new' in param_group['name']:
            param_group['lr'] = 0.001  # Fresh LR for new components
```
**Purpose**: Explicitly stabilize the network after an architectural change.

## 7. **Warm-Up for New Components**
```python
def warmup_schedule(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs  # Linear warm-up
    return 1.0

# Apply to new components only
new_component_lr = base_lr * warmup_schedule(epoch)
```
**Purpose**: Prevent new, randomly initialized components from disrupting existing, stable features with large initial gradients.

## 8. **Layer-wise Adaptive Rate Scaling (LARS)**
```python
for layer in model.layers:
    weight_norm = layer.weight.norm()
    grad_norm = layer.weight.grad.norm()
    
    layer_lr = base_lr * (weight_norm / (grad_norm + 1e-8))
```
**Purpose**: Adapt the learning rate for each layer based on the ratio of its weight norm to its gradient norm, a form of trust-ratio optimization.

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
**Purpose**: Gradually freeze layers as training progresses, shifting focus from feature extraction to fine-tuning.

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
**Purpose**: Maintain stability after growth events by decaying learning rates based on proximity to the change.

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
**Purpose**: Sparse connections may need higher learning rates to have a meaningful impact on the network's output.

## 12. **The "Sedimentary" Learning Strategy**
```python
learning_rates = {
    'geological_layers': 0.00001,   # Oldest, barely change
    'sediment_layers': 0.0001,      # Middle age, slow drift
    'active_layers': 0.001,         # Newest, rapid change
    'patches': 0.0005               # Targeted fixes
}
```
**Purpose**: A conceptual model that encapsulates age-based learning, leading to a natural stratification of learning speeds by component age.

## Recommended Combined Strategy

For a complete, sophisticated system, these strategies can be combined. The `adaptive_learning_rates` module is designed to facilitate such compositions.

```python
def create_optimal_lr_schedule(model, base_lr=0.001):
    param_groups = []
    
    # 1. Scaffold - cascading by depth and age
    for i, layer in enumerate(model.scaffold):
        age_factor = 0.1 ** (model.current_epoch // 20) # Age-based decay
        depth_factor = 0.1 ** (i / len(model.scaffold)) # Depth-based decay
        
        param_groups.append({
            'params': layer.parameters(),
            'lr': base_lr * age_factor * depth_factor,
            'name': f'scaffold_layer_{i}'
        })
    
    # 2. Patches - medium stable rate
    for patch in model.patches:
        param_groups.append({
            'params': patch.parameters(),
            'lr': base_lr * 0.5,
            'name': 'patch'
        })
    
    # 3. New components - warm-up schedule
    for new_component in model.recent_additions:
        warmup_factor = min(1.0, model.epochs_since_added / 5) # Warm-up
        param_groups.append({
            'params': new_component.parameters(),
            'lr': base_lr * warmup_factor,
            'name': 'new_component'
        })
    
    return param_groups
```

This comprehensive strategy ensures stable learning by preserving foundational knowledge in older layers while allowing new components to adapt and integrate quickly.