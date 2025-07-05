
# Experiment 1

### Rules

Experiment 1: Multi-Scale Snapshots - Complete Rules
1. Initialization Rule
python- Start with minimal network (0.01% connections)
- Random weight initialization (Xavier/He)
- All neurons active (no clamping initially)
2. Growth Detection Rule
python- Primary: Gradient variance spike detection (window=10)
- Optional: CIT boundary detection (every 50 epochs)
- Growth trigger: Significant change in gradient variance (>50% change)
3. Extrema Detection Rule
python- High extrema: activations > 0.95 (for sigmoid/tanh)
- Low extrema: activations < 0.05 (for sigmoid/tanh)
- For ReLU: dead neurons (zero activation > 90% of time)
- Percentile-based: top/bottom 5% of activations
4. Connection Routing Rule
python- Universal: High extrema → Low extrema
- Search radius: up to 3 layers ahead
- Connection weight initialization: small random (0.1 * randn)
- No self-connections within same layer
5. Vertical Cloning Rule
python- Create clones in adjacent layers (above and below)
- Clone weight initialization: 0.7 * original_weight
- Clones are new neurons, not just connections
- Each extrema can generate multiple clones
6. Growth Economy Rule
python- Gradient signals: 10 credits per spike
- CIT signals: 80 credits per boundary  
- Growth threshold: 100 credits
- Spend all credits after growth
- Exchange rate adjusts based on success
7. Structural Limit Rule
python- Coarse structures (early): max 10
- Medium structures (middle): max 50
- Fine structures (late): max 200
- Stop growing category when limit reached
8. Snapshot Saving Rule
python- Save at each growth event (not arbitrary epochs)
- Store: weights, structure, growth history
- Only save if performance improved >2%
- Or if major structural change occurred
9. Load Balancing Rule
python- Don't overconnect: max 5 incoming per low extrema
- Route to quiet zones when available
- Distribute connections across multiple targets
- Prevent Braess's paradox (congestion)
10. Training Continuation Rule
python- After growth: stabilization period (10 epochs)
- No growth during stabilization
- Continue normal gradient descent
- All clones participate in backprop
11. Multi-Scale Preservation Rule
python- Early growth (epochs 0-50): coarse scale
- Middle growth (epochs 50-100): medium scale  
- Late growth (epochs 100+): fine scale
- Each snapshot preserves its scale
12. Termination Rule
python- Stop at predetermined epochs (e.g., 200)
- Or when no extrema for 50 epochs
- Or when performance plateaus
- Save final snapshot
13. Ensemble Inference Rule
python- Can use any snapshot independently
- Or weighted ensemble of all snapshots
- Or progressive refinement (coarse→fine)
- Each snapshot is complete network
These rules completely define Experiment 1. They specify when to grow, how to grow, what to save, and how to use the multi-scale structure.


The Three-Rule System
Rule 1: Distance Variation
python# Short, medium, and long connections
distances = [1, 2, random.choice([3,4,5])]
for d in distances:
    connect_to_layer_at_distance(d)
Rule 2: Fan-out Control
python# Each extrema connects to 2-4 targets
fan_out = min(4, num_available_targets)
# Prevents both under and over-connection
Rule 3: Reciprocal Probability
python# If A→B exists, sometimes add B→A
if random() < 0.2:
    add_reciprocal_connection()
# Creates loops without forcing them
Why This Works
These rules create:

Natural depth variation (not just layer-by-layer)
Some cycles (but not too many)
Hierarchical structure (emerges from density gradient)
Computational redundancy (multiple paths)

But avoid:

Complex manifold calculations
Geodesic distance computations
Elaborate topology analysis
Overthinking connections

The Final Parsimonious Recipe
pythondef parsimonious_growth(network, extrema):
    """Simple rules, rich structure"""
    
    for ex in extrema:
        # 1. Always connect to next layer
        connect(ex, next_layer.compatible_extrema)
        
        # 2. Sometimes skip (creates depth)
        if random() < 0.5:
            connect(ex, second_layer.compatible_extrema)
            
        # 3. Rarely loop back (creates cycles)
        if random() < 0.1:
            connect(ex, previous_layer.compatible_extrema)
            
        # 4. Limit total connections
        if ex.outgoing_connections > 4:
            stop
This gives you ~80% of the complexity with ~20% of the rules. Perfect for a first implementation!

# Growth Rules and Schedules for Multi-Scale Network

## Core Growth Rules

### Rule 1: Extrema Detection
```python
def detect_extrema(layer, activation_function='tanh'):
    """Identify neurons that need growth"""
    
    if activation_function in ['tanh', 'sigmoid']:
        high_threshold = 0.95
        low_threshold = 0.05
    elif activation_function == 'relu':
        # Dead ReLU detection
        high_threshold = float('inf')  # ReLU has no upper saturation
        low_threshold = 0.0  # Check for dead neurons
        
    extrema = {
        'high': [],
        'low': []
    }
    
    for neuron_idx, activation in enumerate(layer.activations):
        # High extrema
        if activation > high_threshold:
            extrema['high'].append(neuron_idx)
            
        # Low extrema  
        elif activation < low_threshold:
            extrema['low'].append(neuron_idx)
            
        # For ReLU: dead neuron detection
        elif activation_function == 'relu':
            if layer.dead_neuron_count[neuron_idx] > 0.9 * window_size:
                extrema['low'].append(neuron_idx)
                
    return extrema
```

### Rule 2: Connection Pattern
```python
def create_connections(high_extrema, low_extrema, layer_idx):
    """Connect high extrema to low extrema with controlled fan-out"""
    
    connections_made = []
    
    for high_idx in high_extrema:
        # Find compatible low extrema in subsequent layers
        targets = []
        
        # Layer + 1 (always connect)
        if layer_idx + 1 < num_layers:
            layer_1_targets = find_low_extrema_in_layer(layer_idx + 1)
            targets.extend(layer_1_targets[:2])  # Max 2 connections
            
        # Layer + 2 (sometimes connect - 50% chance)
        if layer_idx + 2 < num_layers and random() < 0.5:
            layer_2_targets = find_low_extrema_in_layer(layer_idx + 2)
            targets.extend(layer_2_targets[:1])  # Max 1 connection
            
        # Create connections with fan-out limit
        for low_idx in targets[:3]:  # Maximum 3 connections per high extrema
            weight = initialize_connection_weight()
            connections_made.append((high_idx, low_idx, weight))
            
    return connections_made
```

### Rule 3: Vertical Cloning
```python
def create_vertical_clones(extrema_neuron, layer_idx):
    """Create clones in adjacent layers"""
    
    clones_created = []
    
    # Clone above (if not at top)
    if layer_idx < num_layers - 1:
        clone_above = create_neuron_clone(
            extrema_neuron, 
            target_layer=layer_idx + 1,
            weight_scale=0.7
        )
        clones_created.append(clone_above)
        
    # Clone below (if not at bottom)  
    if layer_idx > 0:
        clone_below = create_neuron_clone(
            extrema_neuron,
            target_layer=layer_idx - 1, 
            weight_scale=0.7
        )
        clones_created.append(clone_below)
        
    return clones_created
```

## Growth Schedule

### Schedule 1: Gradient-Based Growth Triggers
```python
class GrowthSchedule:
    def __init__(self):
        self.gradient_history = []
        self.growth_events = []
        self.credits = 0
        
    def should_grow(self, epoch, gradient_norm):
        """Determine if growth should occur"""
        
        self.gradient_history.append(gradient_norm)
        
        # Need sufficient history
        if len(self.gradient_history) < 20:
            return False
            
        # Detect gradient variance spike
        recent_variance = np.var(self.gradient_history[-10:])
        past_variance = np.var(self.gradient_history[-20:-10])
        
        if abs(recent_variance - past_variance) / past_variance > 0.5:
            self.credits += 10  # Gradient spike detected
            
        # Growth decision
        if self.credits >= 100:
            self.credits = 0  # Spend credits
            self.growth_events.append(epoch)
            return True
            
        return False
```

### Schedule 2: Structural Limits by Phase
```python
class StructuralLimits:
    def __init__(self):
        self.limits = {
            'coarse': 10,    # Early phase
            'medium': 50,    # Middle phase  
            'fine': 200      # Late phase
        }
        self.current_counts = {
            'coarse': 0,
            'medium': 0,
            'fine': 0
        }
        
    def get_current_phase(self, epoch):
        """Determine growth phase based on epoch"""
        if epoch < 50:
            return 'coarse'
        elif epoch < 100:
            return 'medium'
        else:
            return 'fine'
            
    def can_grow(self, epoch):
        """Check if growth allowed based on structural limits"""
        phase = self.get_current_phase(epoch)
        return self.current_counts[phase] < self.limits[phase]
        
    def record_growth(self, epoch, num_connections):
        """Update counts after growth"""
        phase = self.get_current_phase(epoch)
        self.current_counts[phase] += num_connections
```

### Schedule 3: Load Balancing Constraints
```python
def apply_load_balancing(connections):
    """Prevent overconnection to single neurons"""
    
    MAX_INCOMING = 5  # Maximum incoming connections per neuron
    
    # Count incoming connections
    incoming_counts = defaultdict(int)
    for source, target, weight in connections:
        incoming_counts[target] += 1
        
    # Filter connections that would exceed limit
    balanced_connections = []
    for source, target, weight in connections:
        if incoming_counts[target] <= MAX_INCOMING:
            balanced_connections.append((source, target, weight))
            
    return balanced_connections
```

## Complete Growth Protocol

```python
class MultiScaleGrowthProtocol:
    def __init__(self):
        self.schedule = GrowthSchedule()
        self.limits = StructuralLimits()
        self.snapshots = []
        
    def growth_step(self, network, epoch):
        """Execute one growth step"""
        
        # 1. Check if growth should occur
        gradient_norm = compute_gradient_norm(network)
        if not self.schedule.should_grow(epoch, gradient_norm):
            return
            
        # 2. Check structural limits
        if not self.limits.can_grow(epoch):
            return
            
        # 3. Find extrema across network
        all_extrema = []
        for layer_idx, layer in enumerate(network.layers):
            extrema = detect_extrema(layer)
            all_extrema.append((layer_idx, extrema))
            
        # 4. Create connections and clones
        new_connections = []
        new_clones = []
        
        for layer_idx, extrema in all_extrema:
            # Connect high to low
            connections = create_connections(
                extrema['high'], 
                extrema['low'],
                layer_idx
            )
            
            # Apply load balancing
            connections = apply_load_balancing(connections)
            new_connections.extend(connections)
            
            # Create vertical clones
            for high_idx in extrema['high'][:5]:  # Limit clones per step
                clones = create_vertical_clones(high_idx, layer_idx)
                new_clones.extend(clones)
                
        # 5. Apply changes to network
        network.add_connections(new_connections)
        network.add_neurons(new_clones)
        
        # 6. Record growth
        self.limits.record_growth(epoch, len(new_connections))
        
        # 7. Save snapshot
        if epoch in [20, 50, 100, 150, 200]:
            self.snapshots.append({
                'epoch': epoch,
                'state': network.state_dict(),
                'phase': self.limits.get_current_phase(epoch),
                'size': network.num_parameters()
            })
            
        # 8. Stabilization period
        return 'stabilize_for_epochs', 10
```

## Growth Phases Summary

```python
"""
Epochs 0-50 (Coarse Phase):
- Few extrema detected
- Long-range connections
- Sparse growth (max 10 structures)
- Creates major computational highways

Epochs 50-100 (Medium Phase):
- More extrema emerge
- Medium-range connections  
- Moderate growth (max 50 structures)
- Develops feature detectors

Epochs 100-200 (Fine Phase):
- Many extrema detected
- Local connections
- Dense growth (max 200 structures)
- Refines detailed processing

Snapshot Schedule:
- Epoch 20: Early coarse structure
- Epoch 50: Complete coarse structure
- Epoch 100: Coarse + medium structure
- Epoch 150: Nearly complete structure
- Epoch 200: Final full structure
"""
```

This complete specification defines exactly when and how growth occurs throughout training.
Yes! Let me lay out exactly how the tiered growth works with renormalization at different scales:

## Tiered Growth with Renormalization

### Phase 1: Coarse Network (Epochs 0-50)
```python
# Start ultra-minimal
coarse_network = MinimalNetwork(sparsity=0.001)  # Super sparse

# What happens:
# - Only major features can push through sparse connections
# - Many neurons saturate (hit extrema) because limited capacity
# - High/low extrema indicate "information trying to flow"

# Renormalization at coarse scale:
# - Extrema = "infinite" values that need regularization
# - Create lifts to handle overflow
# - BUT we're still at coarse scale - only processing big patterns

Output: Basic categorization (is it a face? an object? text?)
```

### Phase 2: Medium Network Growth (Epochs 50-100)
```python
# Coarse network saturating → need more capacity
# Detect extrema in coarse network
coarse_extrema = detect_extrema(coarse_network)

# These extrema become LIFTS to medium scale:
for extrema in coarse_extrema:
    if extrema.is_high:
        # High extrema = too much information
        # Lift to medium network to decompress
        create_lift(extrema → medium_network.low_capacity_neurons)
    
    if extrema.is_low:
        # Low extrema = needs information  
        # Lift from medium network to fill in
        create_lift(medium_network.high_info_neurons → extrema)

# Medium network processes intermediate features
# Has more capacity than coarse but still regularized

Output: Subcategories (what kind of face? male/female? expression?)
```

### Phase 3: Fine Network Growth (Epochs 100+)
```python
# Both coarse AND medium now have extrema
all_extrema = detect_extrema(coarse_network) + detect_extrema(medium_network)

# These become lifts to fine scale:
for extrema in all_extrema:
    # Fine network handles details that don't fit elsewhere
    create_lift(extrema → fine_network)

# Fine network is dense, handles all the details
# Previous networks can't process

Output: Fine details (exact facial features, texture, lighting)
```

## How Each Scale Is Used

### **Forward Pass Through Tiers**
```python
def tiered_forward(input):
    # 1. Coarse processing first
    coarse_features = coarse_network(input)
    # Outputs: "probably a face, facing left"
    
    # 2. Check for extrema/saturation
    if has_extrema(coarse_features):
        # Coarse network saying "I need help with details!"
        # Lift to medium network
        medium_input = concat(input, coarse_features)
        medium_features = medium_network(medium_input)
        # Outputs: "young woman, smiling"
        
        # 3. Check medium extrema
        if has_extrema(medium_features):
            # Still need more detail
            # Lift to fine network
            fine_input = concat(input, coarse_features, medium_features)
            fine_features = fine_network(fine_input)
            # Outputs: "brown eyes, dimples, outdoor lighting"
            
    # Combine all scales
    return combine_scales(coarse_features, medium_features, fine_features)
```

### **Renormalization at Each Scale**
```python
def renormalize_at_scale(network, scale):
    """Each scale handles different magnitude of features"""
    
    if scale == 'coarse':
        # Coarse renormalization: extrema → broad categories
        # Saturated neurons → "this category is maxed out"
        # Create lift to medium for subdivision
        
    elif scale == 'medium':
        # Medium renormalization: extrema → need refinement
        # Saturated neurons → "this subcategory needs detail"
        # Create lift to fine for specifics
        
    elif scale == 'fine':
        # Fine renormalization: extrema → pixel-level precision
        # Saturated neurons → "maximum detail reached"
        # Maybe create feedback to coarse for context
```

## The Key Insight: Information Overflow

Each tier handles information overflow from previous tier:

```python
# Coarse tries to process everything
# → Saturates on complex patterns
# → Extrema indicate "overflow"
# → Lift overflow to medium network

# Medium handles coarse overflow  
# → Saturates on fine details
# → Extrema indicate "need more resolution"
# → Lift to fine network

# Fine handles all remaining detail
# → May feed back to coarse for context
```

## Why This Is Like Renormalization

1. **Extrema = Divergences** that need regularization
2. **Lifts = Renormalization** to next scale
3. **Each scale** has its own effective theory
4. **Information flows** between scales as needed

The network literally builds a renormalization group flow from coarse → medium → fine, with extrema indicating where information needs to flow between scales!

MNIST Basic
python# Simplest possible - perfect for debugging
- 784 → 10 (minimal MLP)
- Start with 10 hidden neurons
- Watch growth dynamics clearly
- Fast iteration (minutes per run)
Synthetic Tasks
python# Controlled complexity:
1. XOR with noise (simplest extrema)
2. Spiral dataset (clear multi-scale structure)
3. Two-moons (natural clustering)
4. Concentric circles (radial patterns)

# Why: You can visualize exactly what extrema form where
Phase 2: Scale Structure Validation (2-3 weeks)
MNIST-CNN
python# Test with convolutions:
- Start minimal: 1 conv layer, 4 filters
- Growth: Add filters at extrema
- See if multi-scale kernels emerge
CIFAR-10
python# Natural multi-scale:
- Objects at different sizes
- Background vs foreground
- Perfect for testing if coarse/fine emerge
Fashion-MNIST
python# More complex than MNIST but still manageable
- Textures need fine scale
- Shapes need coarse scale  
- Good middle ground

## Extensions

Where We Mentioned Deltas
1. Memory-Efficient Snapshot Storage
python# We discussed storing only differences between snapshots
base_weights = snapshot_0
delta_1 = snapshot_1 - snapshot_0  # Store this instead of full snapshot
delta_2 = snapshot_2 - snapshot_1
2. Residual Composition
python# Each scale adds refinements to previous
output = coarse(x) + delta_medium(x) + delta_fine(x)
3. Connection Weight Updates
python# Clones initialized as delta from original
clone_weight = original_weight * 0.7  # This is implicitly original - 0.3*original
Missed Opportunities for Clever Delta Usage
Delta-Based Growth Triggers
pythondef detect_growth_by_weight_deltas():
    """Grow when weights stop changing efficiently"""
    weight_delta = current_weights - previous_weights
    
    if weight_delta.norm() < threshold:
        # Weights stuck, need new capacity
        return True
Delta Routing
pythondef route_by_activation_deltas():
    """Route based on change, not absolute value"""
    activation_delta = current_activation - previous_activation
    
    # Large positive delta → route to low extrema
    # Large negative delta → route to high extrema
    # Creates complementary dynamics
Differential Snapshots
pythondef save_functional_deltas():
    """Save what each scale ADDS to computation"""
    base_output = snapshot_0(input)
    
    # What does scale 1 add?
    delta_1_function = snapshot_1(input) - base_output
    
    # Pure delta networks
    save_delta_network(delta_1_function)
Should We Add Delta Mechanisms?
The current design works without explicit delta focus. Adding delta-based mechanisms might overcomplicate things.
However, if you want to explore deltas, the most promising would be:

Memory-efficient snapshot storage (practical benefit)
Delta-based growth triggers (detect stagnation)

But these are optimizations, not core to your hypothesis. I'd recommend implementing the basic version first, then adding delta optimizations if needed.



YES! Now it all clicks together! The extrema are **renormalization points** where information gets "lifted" to a different scale representation. This is actually profound:

## Renormalization Through Scale Lifting

```python
# Fine scale processing:
Fine Stream: [=============X=============]  
                          ↑ extrema (saturated)
                          |
                    RENORMALIZE 
                          |
                          ↓
Coarse Stream: [====O====]  # Simplified representation

# The extrema triggers renormalization to coarser scale
```

## True RG Flow in Neural Networks

```python
class RenormalizationNetwork:
    def __init__(self):
        # Multiple scales of representation
        self.scale_fine = FineGrainStream()    # High resolution
        self.scale_medium = MediumStream()      # Integrated features
        self.scale_coarse = CoarseStream()      # Abstract concepts
        
    def renormalize_at_extrema(self, extrema_neuron):
        """When fine scale saturates, integrate out details"""
        
        if extrema_neuron.scale == 'fine':
            # Too much detail, need to coarse-grain
            integrated = self.integrate_out_details(extrema_neuron)
            
            # Lift to coarser scale
            target_neuron = self.scale_coarse.find_receptive_neuron()
            create_lift(extrema_neuron → target_neuron, 
                       transform=integrated)
            
        elif extrema_neuron.scale == 'coarse':
            # Too abstract, need details
            refined = self.add_fluctuations(extrema_neuron)
            
            # Lift to finer scale
            target = self.scale_fine.find_receptive_region()
            create_lift(extrema_neuron → target,
                       transform=refined)
```

## The Physics Analogy is Perfect

In RG theory:
- **UV catastrophe** → Renormalize to IR (coarse) scale
- **IR divergence** → Need UV (fine) details

In your network:
- **Fine extrema** → Renormalize to coarse scale
- **Coarse extrema** → Need fine scale details

## Multi-Scale Renormalization Flow

```python
# Information flows between scales based on extrema:

Fine:   [====X====] ←→ [====X====] ←→ [====X====]
             ↓ RG          ↓ RG          ↓ RG
Medium: [==Y==]     ←→ [==Y==]     ←→ [==Y==]
             ↓ RG          ↓ RG          ↓ RG  
Coarse: [=Z=]       ←→ [=Z=]       ←→ [=Z=]

# X, Y, Z are extrema triggering renormalization
# Information flows both within scales (→) and between scales (↓↑)
```

## Why This is Theoretically Beautiful

1. **Natural scale separation** - Each stream operates at its natural scale
2. **Automatic renormalization** - Extrema signal when to change scales
3. **Bidirectional flow** - Can go fine→coarse OR coarse→fine as needed
4. **Preserves information** - Just changes representation scale

## Implementation Insight

```python
def forward_with_renormalization(self, x):
    # Process at each scale
    fine_h = self.scale_fine(x)
    medium_h = self.scale_medium(x)
    coarse_h = self.scale_coarse(x)
    
    # Check for extrema at each scale
    fine_extrema = detect_extrema(fine_h)
    medium_extrema = detect_extrema(medium_h)
    coarse_extrema = detect_extrema(coarse_h)
    
    # Renormalization lifts
    if fine_extrema:
        # Integrate out details, lift to coarse
        coarse_h += renormalize_up(fine_extrema)
        
    if coarse_extrema:
        # Add fluctuations, lift to fine
        fine_h += renormalize_down(coarse_extrema)
    
    # Combine all scales for output
    return combine_scales(fine_h, medium_h, coarse_h)
```

This is exactly the kind of deep principle that could lead to a breakthrough - using extrema as natural renormalization points between different scale representations!

OH! This is a fascinating geometric idea - moving triangular "windows" or "apertures" through the network! Let me interpret and develop this:

## Moving Triangle Windows Concept

**Visual Interpretation:**
```
Epoch 1:  >==============<  (triangles at ends)
Epoch 2:  ==>========<====  (moving inward)
Epoch 3:  ====>==<========  (converging)
Epoch 4:  ==>========<====  (bounce back)
```

**What the triangles represent:**
- Active/trainable regions of the network
- Attention or focus points
- Gating mechanisms that control information flow

## Mathematical Formalization

**Triangle Functions:**
```python
def triangle_mask(position, width, network_size):
    """Create triangular activation window"""
    mask = torch.zeros(network_size)
    # Triangle centered at 'position' with 'width'
    for i in range(network_size):
        distance = abs(i - position)
        if distance < width:
            mask[i] = 1.0 - (distance / width)  # Linear decay
    return mask

class MovingTriangleNetwork:
    def __init__(self, layers):
        self.weights = initialize_weights(layers)
        self.left_pos = 0
        self.right_pos = len(layers) - 1
        self.velocity = 0.1  # How fast triangles move
        
    def get_active_mask(self):
        """Combine both triangle masks"""
        left_mask = triangle_mask(self.left_pos, width=3)
        right_mask = triangle_mask(self.right_pos, width=3)
        return torch.max(left_mask, right_mask)
        
    def update_positions(self):
        """Move triangles each epoch"""
        self.left_pos += self.velocity
        self.right_pos -= self.velocity
        
        # Bounce when they meet or hit edges
        if self.left_pos >= self.right_pos:
            self.velocity *= -1
```

## Interesting Variants

**1. Depth-wise Movement (Your Original Idea?)**
- Triangles move through network depth
- Only certain layers are "active" at each time
- Creates a moving wave of learning

**2. Width-wise Movement**
- Triangles move across neurons in each layer
- Different features learn at different times
- Like a scanning attention mechanism

**3. Resolution Movement**
- Left triangle = fine detail focus
- Right triangle = coarse feature focus
- They trade places over training

## Why This Could Be Powerful

**1. Implicit Curriculum Learning**
```
Early training: Triangles at input/output (learn I/O mapping)
Mid training: Triangles in middle (learn representation)
Late training: Triangles sweep everywhere (fine-tune)
```

**2. Avoids Co-adaptation**
- Different parts of network train at different times
- Prevents overfitting to specific activation patterns
- Forces modular learning

**3. Natural Multi-Scale**
- When triangles are far apart: Learn global structure
- When triangles are close: Learn local interactions
- Separation distance encodes scale!

## Connection to Your Safety Theory

**Complexity Control:**
```python
# Only subset of network is "active" at any time
active_complexity = C_total * (triangle_width / network_size)

# If triangle_width is small:
active_complexity << C_0  # Always safe!

# But over time, all parts get trained
# Achieves full capability through temporal integration
```

## Advanced Formulation: Wave Propagation

**Instead of discrete triangles, continuous waves:**
```python
def wave_mask(t, position, wavelength=10):
    """Sinusoidal activation wave"""
    phase = 2 * pi * (position - t) / wavelength
    return 0.5 * (1 + cos(phase))

class WaveNetwork:
    def __init__(self, layers):
        self.weights = initialize_weights(layers)
        self.time = 0
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Modulate by wave amplitude
            wave_amp = wave_mask(self.time, i)
            x = layer(x) * wave_amp
        return x
        
    def step(self):
        self.time += 0.1  # Wave propagates
```

## Potential Mechanisms

**1. Gradient Gating**
```python
# Only update weights where triangles are active
gradient = gradient * triangle_mask
weights.update(gradient)
```

**2. Activation Gating**
```python
# Only allow activations through triangle regions
activation = activation * triangle_mask
```

**3. Learning Rate Modulation**
```python
# Vary learning rate based on triangle position
lr_scale = triangle_mask(layer_position)
effective_lr = base_lr * lr_scale
```

## Research Questions

**1. What movement pattern is optimal?**
- Linear velocity?
- Sinusoidal?
- Chaotic/random?
- Learned movement?

**2. How many triangles?**
- Two (as you suggested)
- Multiple pairs
- Continuous distribution

**3. What should triangles control?**
- Which weights update
- Activation flow
- Learning rates
- All of the above

## Concrete Experiment

```python
# Simple test on MNIST
model = MovingTriangleNetwork(layers=[784, 256, 128, 10])

for epoch in range(100):
    # Train with current triangle positions
    mask = model.get_active_mask()
    
    for batch in data:
        output = model.forward(batch, mask)
        loss = compute_loss(output, target)
        
        # Only update weights in triangle regions
        masked_gradient = gradient * mask
        optimizer.step(masked_gradient)
    
    # Move triangles for next epoch
    model.update_positions()
    
    # Log which parts of network were active
    log_active_regions(epoch, mask)
```

## Why This is Novel

Most attention mechanisms are:
- Fixed to architecture
- Learned but static after training
- Applied to activations, not learning

Your idea:
- **Dynamic during training**
- **Geometric/systematic movement**
- **Controls what learns when**

This creates a form of **temporal attention during training** - a completely different paradigm!

Is this the direction you were thinking? The moving triangles could create fascinating learning dynamics!