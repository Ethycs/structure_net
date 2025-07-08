# Structure Net Logging System - Beginner's Guide

A comprehensive guide to using the standardized logging system with WandB artifacts and Pydantic validation.

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Core Concepts](#core-concepts)
3. [Step-by-Step Tutorials](#step-by-step-tutorials)
4. [Common Scenarios](#common-scenarios)
5. [Troubleshooting](#troubleshooting)
6. [CLI Tools](#cli-tools)
7. [Best Practices](#best-practices)
8. [Advanced Usage](#advanced-usage)
9. [FAQ](#faq)

## Quick Start Guide

### 5-Minute Setup

1. **Install dependencies** (already done if you have the project):
   ```bash
   # Dependencies are in pyproject.toml: pydantic, wandb, jsonschema
   ```

2. **Basic usage** - Replace your old logger:
   ```python
   # OLD WAY
   from src.structure_net.logging import StructureNetWandBLogger
   logger = StructureNetWandBLogger(project_name="my_project")
   
   # NEW WAY (with validation)
   from src.structure_net.logging import create_growth_logger
   logger = create_growth_logger(project_name="my_project")
   ```

3. **Log your first experiment**:
   ```python
   # Start experiment
   logger.log_experiment_start(
       network=my_network,
       target_accuracy=0.95
   )
   
   # Log growth iteration
   logger.log_growth_iteration(
       iteration=1,
       network=my_network,
       accuracy=0.85,
       growth_occurred=True
   )
   
   # Finish and save
   logger.finish_experiment(final_accuracy=0.95)
   ```

4. **Check your data**:
   ```bash
   # Check queue status
   python -m structure_net.logging.cli status
   
   # Process uploads
   python -m structure_net.logging.cli process
   ```

That's it! Your experiment is now logged with validation and saved as a WandB artifact.

## Core Concepts

### What Are WandB Artifacts?

WandB artifacts are **versioned, immutable data objects** that store your experiment results. Think of them as:
- **Git commits** for your experiment data
- **Guaranteed persistence** - never lose data
- **Automatic deduplication** - same data = same artifact
- **Version tracking** - see how experiments evolve

### The Queue → Validate → Upload Pipeline

```
Your Experiment → Queue → Validation → WandB Artifact
     ↓              ↓         ↓            ↓
  Log data    Local JSON   Schema     Permanent
              (safe)      Check      Storage
```

**Benefits:**
- ✅ **Never lose data** - Local queue survives network outages
- ✅ **Data quality** - Only valid data reaches WandB
- ✅ **Offline-safe** - Experiments never block on network
- ✅ **Automatic retry** - Failed uploads retry automatically

### Schema Validation

**Pydantic schemas** ensure your data is consistent:

```python
# This will be validated automatically
experiment_data = {
    "experiment_type": "growth_experiment",
    "accuracy": 0.95,  # Must be 0.0-1.0
    "architecture": [784, 128, 10],  # Must be list of positive ints
    "timestamp": "2025-01-08T21:00:00"  # Auto-generated
}
```

**What gets validated:**
- ✅ Required fields are present
- ✅ Data types are correct (float, int, string, etc.)
- ✅ Value ranges are valid (accuracy 0-1, positive layer sizes)
- ✅ Relationships are consistent (depth matches layer count)

## Step-by-Step Tutorials

### Tutorial 1: Logging Your First Experiment

**Goal:** Log a simple growth experiment with validation.

```python
from src.structure_net.logging import create_growth_logger
from src.structure_net import create_standard_network

# 1. Create logger
logger = create_growth_logger(
    project_name="my_first_experiment",
    experiment_name="tutorial_1",
    config={
        'dataset': 'mnist',
        'batch_size': 64,
        'learning_rate': 0.001,
        'device': 'cpu'
    },
    tags=['tutorial', 'beginner']
)

# 2. Create a simple network
network = create_standard_network(
    architecture=[784, 128, 10],
    sparsity=0.02,
    device='cpu'
)

# 3. Start experiment
logger.log_experiment_start(
    network=network,
    target_accuracy=0.90
)

# 4. Simulate training iterations
for iteration in range(3):
    # Simulate some training...
    accuracy = 0.7 + (iteration * 0.1)  # Fake progress
    
    logger.log_growth_iteration(
        iteration=iteration,
        network=network,
        accuracy=accuracy,
        growth_occurred=(iteration > 0),  # Growth after first iteration
        extrema_analysis={
            'total_extrema': 10 + iteration,
            'extrema_ratio': 0.1 + (iteration * 0.05)
        }
    )

# 5. Finish experiment
artifact_hash = logger.finish_experiment(final_accuracy=0.90)
print(f"✅ Experiment saved as artifact: {artifact_hash}")
```

**What happened:**
1. Logger created with validation enabled
2. Real-time metrics sent to WandB
3. Validated data queued for artifact upload
4. Experiment finished with artifact creation

### Tutorial 2: Understanding Validation Errors

**Goal:** Learn to debug validation errors.

```python
from src.structure_net.logging import create_growth_logger

logger = create_growth_logger()

# This will cause a validation error
try:
    logger.log_growth_iteration(
        iteration=-1,  # ❌ Negative iteration
        network=network,
        accuracy=1.5,  # ❌ Accuracy > 1.0
        growth_occurred="yes"  # ❌ Should be boolean
    )
except Exception as e:
    print(f"Validation error: {e}")
    # Shows exactly what's wrong and how to fix it
```

**Common validation errors:**
- `accuracy must be between 0.0 and 1.0` → Check your accuracy calculation
- `iteration must be non-negative` → Don't use negative iteration numbers
- `layers cannot be empty` → Provide valid architecture
- `experiment_id cannot be empty` → Provide experiment name

**How to debug:**
1. **Read the error message** - It tells you exactly what's wrong
2. **Check your data types** - String vs int vs float vs boolean
3. **Validate ranges** - Accuracy 0-1, positive layer sizes, etc.
4. **Use the CLI validator**:
   ```bash
   python -m structure_net.logging.cli validate my_experiment.json
   ```

### Tutorial 3: Working with Different Experiment Types

**Goal:** Use the right logger for your experiment type.

#### Growth Experiments
```python
from src.structure_net.logging import create_growth_logger

logger = create_growth_logger()
# Use for: Network growth, architecture evolution, extrema-driven experiments
```

#### Training Experiments
```python
from src.structure_net.logging import create_training_logger

logger = create_training_logger()
# Use for: Standard training, hyperparameter sweeps, baseline comparisons

# Log training epochs
logger.log_training_epoch(
    epoch=1,
    train_loss=0.5,
    train_acc=0.85,
    val_loss=0.6,
    val_acc=0.80,
    learning_rate=0.001
)
```

#### Tournament Experiments
```python
from src.structure_net.logging import create_tournament_logger

logger = create_tournament_logger()
# Use for: Strategy competitions, A/B testing, multi-approach comparisons

# Log tournament results
logger.log_tournament_results(
    tournament_results={
        'winner': {
            'strategy': 'extrema_growth',
            'improvement': 0.15,
            'final_accuracy': 0.95
        },
        'all_results': [
            {'strategy': 'extrema_growth', 'improvement': 0.15, 'final_accuracy': 0.95},
            {'strategy': 'random_growth', 'improvement': 0.05, 'final_accuracy': 0.85}
        ]
    },
    iteration=1
)
```

### Tutorial 4: Debugging Failed Uploads

**Goal:** Handle network issues and upload failures.

```bash
# 1. Check what's in the queue
python -m structure_net.logging.cli status

# Output:
# 📊 QUEUE STATUS
# Queue size: 3 files
# Sent count: 15 files  
# Rejected count: 1 files

# 2. Check rejected files
python -m structure_net.logging.cli rejected

# Output:
# ❌ REJECTED FILES (1)
# File: abc123def456.json
# Error: accuracy must be between 0.0 and 1.0

# 3. Fix and requeue
# Edit the file or fix your code, then:
python -m structure_net.logging.cli requeue abc123def456.json

# 4. Process the queue
python -m structure_net.logging.cli process
```

**Common upload issues:**
- **Network down** → Files stay in queue, auto-retry when network returns
- **WandB authentication** → Run `wandb login` 
- **Validation errors** → Check rejected files and fix data
- **Disk space** → Clean old sent files: `python -m structure_net.logging.cli clean`

### Tutorial 5: Viewing Results in WandB

**Goal:** Navigate your artifacts in WandB UI.

1. **Go to your WandB project** → `https://wandb.ai/your-username/structure_net`

2. **Find your artifacts**:
   - Click "Artifacts" tab
   - Look for type "experiment_result"
   - Each artifact is named with a hash (e.g., `abc123def456`)

3. **View artifact contents**:
   - Click on artifact name
   - See metadata (experiment_type, timestamp, etc.)
   - Download JSON file to inspect raw data

4. **Compare experiments**:
   - Use WandB's comparison tools
   - Filter by experiment_type
   - Create custom charts from artifact data

## Common Scenarios

### Scenario 1: Training a Simple Network

```python
from src.structure_net.logging import create_training_logger
import torch.nn as nn
import torch.optim as optim

# Setup
logger = create_training_logger(
    experiment_name="simple_mnist_training",
    config={
        'dataset': 'mnist',
        'batch_size': 64,
        'learning_rate': 0.001,
        'max_epochs': 10
    }
)

network = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = optim.Adam(network.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    # ... training code ...
    
    # Log each epoch
    logger.log_training_epoch(
        epoch=epoch,
        train_loss=train_loss,
        train_acc=train_acc,
        val_loss=val_loss,
        val_acc=val_acc,
        learning_rate=optimizer.param_groups[0]['lr']
    )

# Finish
logger.finish_experiment(final_accuracy=final_val_acc)
```

### Scenario 2: Running Growth Experiments

```python
from src.structure_net.logging import create_growth_logger
from src.structure_net import create_standard_network

# Setup growth experiment
logger = create_growth_logger(
    experiment_name="extrema_driven_growth",
    config={
        'dataset': 'cifar10',
        'target_accuracy': 0.95,
        'growth_strategy': 'extrema_detection'
    }
)

# Initial network
network = create_standard_network([3072, 128, 10], sparsity=0.02)

logger.log_experiment_start(
    network=network,
    target_accuracy=0.95,
    seed_architecture=[3072, 128, 10]
)

# Growth loop
for iteration in range(5):
    # Train to convergence
    accuracy = train_network(network)
    
    # Analyze extrema
    extrema_analysis = detect_extrema(network)
    
    # Decide on growth
    growth_actions = []
    growth_occurred = False
    
    if extrema_analysis['extrema_ratio'] > 0.3:
        # Add layer
        network = add_layer(network, position=1, size=64)
        growth_actions.append({
            'action': 'add_layer',
            'position': 1,
            'size': 64,
            'reason': f"High extrema ratio: {extrema_analysis['extrema_ratio']:.2f}"
        })
        growth_occurred = True
    
    # Log iteration
    logger.log_growth_iteration(
        iteration=iteration,
        network=network,
        accuracy=accuracy,
        extrema_analysis=extrema_analysis,
        growth_actions=growth_actions,
        growth_occurred=growth_occurred
    )
    
    if accuracy >= 0.95:
        break

logger.finish_experiment(final_accuracy=accuracy)
```

### Scenario 3: Comparing Multiple Experiments

```python
# Run multiple experiments with different configurations
configs = [
    {'learning_rate': 0.001, 'batch_size': 32},
    {'learning_rate': 0.01, 'batch_size': 64},
    {'learning_rate': 0.0001, 'batch_size': 128}
]

results = []

for i, config in enumerate(configs):
    logger = create_training_logger(
        experiment_name=f"lr_comparison_{i}",
        config=config,
        tags=['comparison', 'hyperparameter_sweep']
    )
    
    # Run experiment...
    final_accuracy = run_experiment(config)
    
    artifact_hash = logger.finish_experiment(final_accuracy=final_accuracy)
    results.append({
        'config': config,
        'accuracy': final_accuracy,
        'artifact': artifact_hash
    })

# Results are automatically comparable in WandB
print("Experiment comparison complete!")
for result in results:
    print(f"Config {result['config']} → Accuracy: {result['accuracy']:.2%}")
```

### Scenario 4: Importing Existing JSON Data

```python
# You have old experiment files to import
old_files = [
    'data/old_experiment_1.json',
    'data/old_experiment_2.json',
    'data/old_experiment_3.json'
]

for file_path in old_files:
    # Queue for validation and upload
    python -m structure_net.logging.cli queue {file_path} --process-immediately
    
    # Or use the API
    from src.structure_net.logging import queue_experiment
    import json
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    artifact_hash = queue_experiment(data)
    print(f"Queued {file_path} as {artifact_hash}")

# Process all queued files
python -m structure_net.logging.cli process
```

## Troubleshooting

### Problem: "My experiment data isn't showing up in WandB"

**Possible causes:**
1. **Data is queued but not uploaded**
2. **Validation failed**
3. **WandB authentication issues**
4. **Network connectivity problems**

**Solutions:**
```bash
# 1. Check queue status
python -m structure_net.logging.cli status

# 2. Check for rejected files
python -m structure_net.logging.cli rejected

# 3. Try manual upload
python -m structure_net.logging.cli process

# 4. Check WandB login
wandb login

# 5. Test connectivity
wandb status
```

### Problem: "Validation errors I don't understand"

**Example error:**
```
ValidationError: 1 validation error for GrowthExperiment
accuracy
  ensure this value is less than or equal to 1.0 (type=value_error.number.not_le; limit_value=1.0)
```

**How to read this:**
- `GrowthExperiment` → Schema type
- `accuracy` → Field with the problem  
- `ensure this value is less than or equal to 1.0` → What's wrong
- Your accuracy value is > 1.0 (probably a percentage like 95.0 instead of 0.95)

**Common fixes:**
```python
# Wrong: accuracy as percentage
accuracy = 95.0  # ❌

# Right: accuracy as decimal
accuracy = 0.95  # ✅

# Wrong: negative values
iteration = -1  # ❌

# Right: non-negative
iteration = 0   # ✅

# Wrong: empty lists
architecture = []  # ❌

# Right: valid architecture
architecture = [784, 128, 10]  # ✅
```

### Problem: "Queue is backing up"

**Symptoms:**
- Queue size keeps growing
- Files not uploading to WandB
- Disk space filling up

**Solutions:**
```bash
# 1. Check what's failing
python -m structure_net.logging.cli rejected

# 2. Fix validation errors and requeue
python -m structure_net.logging.cli requeue filename.json

# 3. Start background uploader
python -m structure_net.logging.cli uploader --interval 30

# 4. Clean old sent files
python -m structure_net.logging.cli clean --days 7
```

### Problem: "WandB is down, what happens to my data?"

**Answer:** Your data is safe! 

The logging system is **offline-first**:
1. ✅ **Experiments continue running** - Never blocked by network
2. ✅ **Data queued locally** - Stored in `experiments/queue/`
3. ✅ **Automatic retry** - Uploads when WandB comes back online
4. ✅ **No data loss** - Everything persisted locally first

```bash
# Check your local data
ls experiments/queue/  # Your queued experiments
ls experiments/sent/   # Successfully uploaded experiments

# When WandB comes back online
python -m structure_net.logging.cli process  # Upload everything
```

### Problem: "I need to change my experiment schema"

**Scenario:** You want to add new fields or change data structure.

**Solution:** Schema migration is built-in!

1. **Update schema** in `src/structure_net/logging/schemas.py`
2. **Add migration function** in the same file
3. **Old data automatically migrated** during upload

**Example:**
```python
# In schemas.py, add migration
def _migrate_v10_to_v11(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from version 1.0 to 1.1."""
    # Add new field with default value
    if 'new_field' not in data:
        data['new_field'] = 'default_value'
    
    return data

# Old experiments automatically upgraded when processed
```

## CLI Tools

The CLI provides powerful tools for managing your logging system:

### Basic Commands

```bash
# Check system status
python -m structure_net.logging.cli status

# Process upload queue  
python -m structure_net.logging.cli process

# Validate a file
python -m structure_net.logging.cli validate experiment.json

# Show rejected files
python -m structure_net.logging.cli rejected

# Start background uploader
python -m structure_net.logging.cli uploader
```

### Advanced Commands

```bash
# Migrate legacy file
python -m structure_net.logging.cli migrate old.json new.json

# Requeue rejected file
python -m structure_net.logging.cli requeue abc123.json

# Clean old files
python -m structure_net.logging.cli clean --days 30

# Test the system
python -m structure_net.logging.cli test

# Queue experiment from file
python -m structure_net.logging.cli queue experiment.json --process-immediately
```

### Command Options

```bash
# Custom directories
python -m structure_net.logging.cli status \
  --queue-dir custom/queue \
  --sent-dir custom/sent \
  --rejected-dir custom/rejected

# Custom project
python -m structure_net.logging.cli process --project my_project

# Limit processing
python -m structure_net.logging.cli process --max-files 5

# Background uploader with custom interval
python -m structure_net.logging.cli uploader --interval 120  # 2 minutes
```

## Best Practices

### When to Log What Data

**Growth Experiments:**
- ✅ Log every growth iteration
- ✅ Include extrema analysis
- ✅ Record growth actions and reasons
- ✅ Track architecture evolution

**Training Experiments:**
- ✅ Log every epoch (or every N epochs for long training)
- ✅ Include both training and validation metrics
- ✅ Record learning rate changes
- ✅ Track training duration

**Tournament Experiments:**
- ✅ Log all strategy results, not just winner
- ✅ Include execution times
- ✅ Record strategy parameters
- ✅ Track improvement metrics

### Performance Considerations

**Logging frequency:**
```python
# Good: Log every iteration for short experiments
for iteration in range(10):
    logger.log_growth_iteration(...)

# Good: Log every N epochs for long training
for epoch in range(1000):
    if epoch % 10 == 0:  # Every 10 epochs
        logger.log_training_epoch(...)

# Avoid: Logging every batch (too frequent)
for batch in dataloader:  # ❌ Too much data
    logger.log_training_epoch(...)
```

**Data size management:**
```python
# Good: Reasonable extrema analysis
extrema_analysis = {
    'total_extrema': 15,
    'extrema_ratio': 0.12,
    'layer_health': {0: 0.8, 1: 0.9, 2: 0.7}  # Summary stats
}

# Avoid: Storing raw activations
extrema_analysis = {
    'raw_activations': huge_tensor.tolist()  # ❌ Too much data
}
```

### Data Organization Strategies

**Experiment naming:**
```python
# Good: Descriptive, hierarchical names
experiment_name = "mnist_growth_extrema_v2_20250108"
experiment_name = "cifar10_baseline_resnet18_lr001"
experiment_name = "tournament_strategies_comparison_batch64"

# Avoid: Generic names
experiment_name = "experiment1"  # ❌ Not descriptive
experiment_name = "test"         # ❌ Not specific
```

**Tagging strategy:**
```python
# Good: Consistent, searchable tags
tags = ['mnist', 'growth', 'extrema', 'v2']
tags = ['cifar10', 'baseline', 'resnet', 'comparison']
tags = ['tournament', 'strategy_comparison', 'batch_size_study']

# Use tags for:
# - Dataset (mnist, cifar10, imagenet)
# - Method (growth, baseline, tournament)
# - Version (v1, v2, pilot)
# - Study type (comparison, ablation, hyperparameter)
```

**Configuration management:**
```python
# Good: Complete, reproducible config
config = {
    'dataset': 'mnist',
    'batch_size': 64,
    'learning_rate': 0.001,
    'max_epochs': 100,
    'device': 'cuda:0',
    'random_seed': 42,
    'optimizer': 'adam',
    'scheduler': 'cosine',
    'target_accuracy': 0.95,
    'growth_threshold': 0.3,
    'extrema_detection': 'adaptive'
}

# Include everything needed to reproduce the experiment
```

## Advanced Usage

### Custom Schemas for New Experiment Types

If you need to log a new type of experiment:

1. **Define schema** in `schemas.py`:
```python
class MyCustomExperiment(BaseExperimentSchema):
    """Schema for my custom experiment type."""
    
    experiment_type: Literal["my_custom_experiment"] = "my_custom_experiment"
    custom_field: float = Field(..., description="My custom metric")
    custom_config: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('custom_field')
    def validate_custom_field(cls, v):
        if v < 0:
            raise ValueError("custom_field must be non-negative")
        return v
```

2. **Update validation function**:
```python
def validate_experiment_data(data: Dict[str, Any]) -> ExperimentSchema:
    schema_map = {
        'growth_experiment': GrowthExperiment,
        'training_experiment': TrainingExperiment,
        'tournament_experiment': TournamentExperiment,
        'my_custom_experiment': MyCustomExperiment,  # Add your schema
        # ...
    }
    # ...
```

3. **Create custom logger**:
```python
def create_custom_logger(**kwargs) -> StandardizedLogger:
    return StandardizedLogger(
        experiment_type="my_custom_experiment",
        **kwargs
    )
```

### Batch Processing Multiple Experiments

```python
from src.structure_net.logging import ArtifactManager
import json
from pathlib import Path

# Process many experiment files
experiment_files = Path("old_experiments").glob("*.json")
manager = ArtifactManager()

for file_path in experiment_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Queue for processing
    artifact_hash = manager.queue_experiment(data)
    print(f"Queued {file_path.name} → {artifact_hash}")

# Process all at once
stats = manager.process_queue()
print(f"Processed {stats['uploaded']} experiments")
```

### Integration with CI/CD Pipelines

```yaml
# .github/workflows/experiments.yml
name: Run Experiments

on:
  push:
    branches: [main]

jobs:
  experiment:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -e .
        wandb login ${{ secrets.WANDB_API_KEY }}
    
    - name: Run experiment
      run: |
        python experiments/automated_experiment.py
    
    - name: Upload artifacts
      run: |
        python -m structure_net.logging.cli process
    
    - name: Validate results
      run: |
        python -m structure_net.logging.cli status
```

### Team Collaboration Workflows

**Shared project setup:**
```python
# Team configuration
TEAM_CONFIG = {
    'project_name': 'team_structure_net',
    'entity': 'our_organization',  # WandB team/organization
    'queue_dir': 'shared/experiments/queue',
    'sent_dir': 'shared/experiments/sent'
}

# Each team member uses same config
logger = create_growth_logger(**TEAM_CONFIG)
```

**Experiment coordination:**
```bash
# Team lead processes queue regularly
python -m structure_net.logging.cli uploader --interval 300  # 5 minutes

# Team members check status
python -m structure_net.logging.cli status

# Share rejected files for debugging
python -m structure_net.logging.cli rejected > rejected_report.txt
```

## FAQ

### "Why do I need schemas?"

**Without schemas:**
- ❌ Inconsistent data formats
- ❌ Silent data corruption
- ❌ Hard to compare experiments
- ❌ Debugging nightmares

**With schemas:**
- ✅ Guaranteed data consistency
- ✅ Automatic validation
- ✅ Easy experiment comparison
- ✅ Clear error messages

### "What happens if WandB is offline?"

Your experiments **never stop**! The system is offline-first:

1. **Experiments continue** - No network dependency
2. **Data queued locally** - Safe in `experiments/queue/`
3. **Automatic upload** - When network returns
4. **No data loss** - Everything preserved

### "How do I know if my data uploaded?"

```bash
# Check queue status
python -m structure_net.logging.cli status

# If queue_size = 0 and sent_count > 0, you're good!
# If queue_size > 0, run:
python -m structure_net.logging.cli process
```

### "Can I modify data after logging?"

**No** - Artifacts are immutable (like Git commits). This is a **feature**:
- ✅ Guarantees reproducibility
- ✅ Prevents accidental data corruption
- ✅ Creates audit trail

**If you need changes:**
1. Fix your logging code
2. Re-run the experiment
3. New artifact created automatically

### "How do I share experiments with teammates?"

**Easy!** Just share the WandB project:
1. **Same project name** - Everyone uses same `project_name`
2. **WandB permissions** - Add teammates to WandB project
3. **Artifact access** - All artifacts visible to team

```python
# Everyone on team uses this
logger = create_growth_logger(
    project_name="team_structure_net",  # Shared project
    entity="our_organization"           # Team organization
)
```

### "What if I have a lot of old experiment files?"

**Batch migration** is built-in:

```bash
# Migrate and upload all old files
for file in old_experiments/*.json; do
    python -m structure_net.logging.cli queue "$file"
done

# Process everything
python -m structure_net.logging.cli process
```

**Or use the API:**
```python
from src.structure_net.logging import queue_experiment
import json
from pathlib import Path

for file_path in Path("old_experiments").glob("*.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
    queue_experiment(data)
```

### "How much disk space does this use?"

**Very little!** The system is designed to be efficient:

- **Queue files** - Only temporary, deleted after upload
- **Sent files** - Optional cache, can be cleaned regularly
- **Automatic cleanup** - `python -m structure_net.logging.cli clean`

**Typical usage:**
- Small experiment: ~1-10 KB per file
- Large experiment: ~100 KB - 1 MB per file
- Queue processes quickly, so minimal accumulation

### "Can I use this without WandB?"

**Partially** - You get validation and local persistence, but lose:
- ❌ Real-time monitoring
- ❌ Artifact versioning
- ❌ Team collaboration features
- ❌ Web dashboard

**To disable WandB uploads:**
```python
# Just validate and queue (no upload)
logger = StandardizedLogger(...)
logger.finish_experiment(save_artifact=False)

# Or validate files manually
python -m structure_net.logging.cli validate experiment.json
```

---

## Getting Help

**If you're stuck:**

1. **Check this guide** - Most common issues covered
2. **Use CLI tools** - `python -m structure_net.logging.cli status`
3. **Validate your data** - `python -m structure_net.logging.cli validate file.json`
4. **Check WandB status** - `wandb status`
5. **Ask for help** - Include error messages and queue status

**Remember:** The system is designed to be **safe** and **recoverable**. Your data is always preserved locally, so you can always debug and retry!

---

*Happy logging! 🚀*
