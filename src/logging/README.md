# Structure Net Standardized Logging System

A comprehensive logging system that adopts the WandB artifact standard with Pydantic validation to ensure data consistency and reliability.

## 🚀 Quick Start

```python
from src.logging import create_growth_logger

# Create logger with validation
logger = create_growth_logger(
    project_name="my_project",
    experiment_name="my_experiment",
    config={'dataset': 'mnist', 'batch_size': 64}
)

# Log experiment
logger.log_experiment_start(network=my_network)
logger.log_growth_iteration(
    iteration=1,
    network=my_network,
    accuracy=0.85,
    growth_occurred=True
)
logger.finish_experiment(final_accuracy=0.95)
```

## 🎯 Key Features

### ✅ **Guaranteed Data Quality**
- **Pydantic validation** ensures all data meets strict schemas
- **Type safety** prevents silent data corruption
- **Range validation** catches invalid values (accuracy > 1.0, negative iterations, etc.)
- **Relationship validation** ensures data consistency

### ✅ **Never Lose Data**
- **Local-first approach** - experiments never block on network issues
- **Queue system** preserves data during WandB outages
- **Automatic retry** uploads when connectivity returns
- **Offline-safe operation** for long-running experiments

### ✅ **WandB Artifact Standard**
- **Immutable artifacts** with hash-based naming for deduplication
- **Version tracking** shows experiment evolution
- **Metadata enrichment** with experiment context
- **Lineage tracking** for reproducibility

### ✅ **Schema Evolution**
- **Automatic migration** of legacy data formats
- **Backward compatibility** with existing experiments
- **Version management** for smooth schema updates
- **Migration utilities** for bulk data conversion

## 📁 System Architecture

```
Your Experiment → Queue → Validation → WandB Artifact
     ↓              ↓         ↓            ↓
  Log data    Local JSON   Schema     Permanent
              (safe)      Check      Storage
```

### Directory Structure
```
experiments/
├── queue/          # JSON awaiting validation & upload
├── sent/           # Successfully uploaded artifacts  
├── rejected/       # Failed validation (for debugging)
└── logs/           # System logs
```

## 🔧 Components

### 1. **Schemas** (`schemas.py`)
Pydantic models defining strict data structures:
- `GrowthExperiment` - Network growth experiments
- `TrainingExperiment` - Standard training runs
- `TournamentExperiment` - Strategy competitions
- `NetworkArchitecture` - Network structure validation
- `PerformanceMetrics` - Accuracy, loss, precision, etc.

### 2. **Artifact Manager** (`artifact_manager.py`)
Handles the complete artifact lifecycle:
- Queue management and processing
- Schema validation and migration
- WandB upload with deduplication
- Retry logic and error handling

### 3. **Standardized Logger** (`standardized_logger.py`)
Enhanced logger combining real-time logging with artifact persistence:
- Real-time WandB metrics for monitoring
- Validated artifact creation for data integrity
- Automatic schema validation
- Backward compatibility with existing code

### 4. **CLI Tools** (`cli.py`)
Command-line interface for system management:
- Queue status and processing
- File validation and migration
- Background uploader daemon
- Debugging and maintenance tools

## 📊 Experiment Types

### Growth Experiments
```python
from src.logging import create_growth_logger

logger = create_growth_logger()
logger.log_growth_iteration(
    iteration=1,
    network=network,
    accuracy=0.85,
    extrema_analysis={'total_extrema': 15, 'extrema_ratio': 0.12},
    growth_actions=[{'action_type': 'add_layer', 'reason': 'High extrema ratio'}],
    growth_occurred=True
)
```

### Training Experiments
```python
from src.logging import create_training_logger

logger = create_training_logger()
logger.log_training_epoch(
    epoch=1,
    train_loss=0.5,
    train_acc=0.85,
    val_loss=0.6,
    val_acc=0.80
)
```

### Tournament Experiments
```python
from src.logging import create_tournament_logger

logger = create_tournament_logger()
logger.log_tournament_results({
    'winner': {'strategy': 'extrema_growth', 'improvement': 0.15},
    'all_results': [...]
}, iteration=1)
```

## 🛠️ CLI Usage

### Basic Commands
```bash
# Check queue status
python -m structure_net.logging.cli status

# Process upload queue
python -m structure_net.logging.cli process

# Validate a file
python -m structure_net.logging.cli validate experiment.json

# Start background uploader
python -m structure_net.logging.cli uploader --interval 60
```

### Advanced Commands
```bash
# Migrate legacy file
python -m structure_net.logging.cli migrate old.json new.json

# Show rejected files
python -m structure_net.logging.cli rejected

# Requeue rejected file
python -m structure_net.logging.cli requeue filename.json

# Clean old files
python -m structure_net.logging.cli clean --days 30
```

## 🔍 Validation Examples

### Valid Data
```python
# ✅ This will pass validation
experiment_data = {
    "experiment_type": "growth_experiment",
    "accuracy": 0.95,  # 0.0-1.0 range
    "architecture": [784, 128, 10],  # Positive integers
    "iteration": 5,  # Non-negative
    "growth_occurred": True  # Boolean
}
```

### Invalid Data (Caught by Validation)
```python
# ❌ These will fail validation
{
    "accuracy": 1.5,  # > 1.0
    "iteration": -1,  # Negative
    "architecture": [],  # Empty
    "growth_occurred": "yes"  # String instead of boolean
}
```

## 🚨 Error Handling

### Validation Errors
```python
try:
    logger.log_growth_iteration(accuracy=1.5)  # Invalid
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Fix your data and try again
```

### Network Issues
```bash
# Data is safely queued during outages
# When network returns:
python -m structure_net.logging.cli process
```

### Debugging
```bash
# Check what failed
python -m structure_net.logging.cli rejected

# Validate specific file
python -m structure_net.logging.cli validate experiment.json --migrate
```

## 📈 Migration from Old System

### Automatic Migration
```python
# Old experiments are automatically migrated
from src.logging import queue_experiment
import json

with open('old_experiment.json', 'r') as f:
    data = json.load(f)

# Automatically migrated and validated
artifact_hash = queue_experiment(data)
```

### Batch Migration
```bash
# Migrate all old files
for file in old_experiments/*.json; do
    python -m structure_net.logging.cli queue "$file"
done

python -m structure_net.logging.cli process
```

## 🔧 Configuration

### Artifact Manager Config
```python
from src.logging import ArtifactConfig

config = ArtifactConfig(
    queue_dir="custom/queue",
    project_name="my_project",
    max_retries=5,
    auto_migrate=True
)
```

### Logger Config
```python
logger = create_growth_logger(
    project_name="my_project",
    config={
        'dataset': 'mnist',
        'batch_size': 64,
        'learning_rate': 0.001,
        'device': 'cuda:0'
    },
    tags=['experiment', 'v2']
)
```

## 📚 Documentation

- **[Complete Beginner's Guide](../../../docs/logging_guide.md)** - Comprehensive tutorial with examples
- **[Example Script](../../../examples/standardized_logging_example.py)** - Runnable demonstrations
- **[Schema Reference](schemas.py)** - All available data models
- **[CLI Reference](cli.py)** - Command-line tools

## 🔄 Workflow Integration

### Development Workflow
1. **Code your experiment** with standardized logger
2. **Run experiment** - data validated and queued automatically
3. **Monitor in real-time** via WandB dashboard
4. **Process queue** when convenient
5. **Analyze artifacts** for long-term insights

### Team Workflow
1. **Shared project** - everyone uses same project name
2. **Consistent schemas** - all data validated to same standard
3. **Artifact sharing** - immutable, versioned experiment results
4. **Queue coordination** - team lead processes uploads

### CI/CD Integration
```yaml
# .github/workflows/experiments.yml
- name: Run experiments
  run: python experiments/my_experiment.py

- name: Upload artifacts
  run: python -m structure_net.logging.cli process

- name: Validate results
  run: python -m structure_net.logging.cli status
```

## 🎯 Benefits Summary

| Feature | Old System | New System |
|---------|------------|------------|
| **Data Quality** | ❌ No validation | ✅ Pydantic schemas |
| **Data Loss** | ❌ Network dependent | ✅ Local-first queue |
| **Consistency** | ❌ Ad-hoc formats | ✅ Standardized schemas |
| **Debugging** | ❌ Silent failures | ✅ Clear error messages |
| **Migration** | ❌ Manual process | ✅ Automatic migration |
| **Deduplication** | ❌ Manual checking | ✅ Hash-based artifacts |
| **Offline Support** | ❌ Blocks on network | ✅ Fully offline-safe |
| **Team Collaboration** | ❌ Inconsistent data | ✅ Shared standards |

## 🚀 Getting Started

1. **Try the example**:
   ```bash
   python examples/standardized_logging_example.py
   ```

2. **Read the guide**:
   ```bash
   open docs/logging_guide.md
   ```

3. **Replace your logger**:
   ```python
   # OLD
   from src.logging import StructureNetWandBLogger
   
   # NEW
   from src.logging import create_growth_logger
   ```

4. **Process your queue**:
   ```bash
   python -m structure_net.logging.cli process
   ```

## 🆘 Getting Help

- **Check the [Beginner's Guide](../../../docs/logging_guide.md)** for detailed tutorials
- **Run the [example script](../../../examples/standardized_logging_example.py)** to see it in action
- **Use CLI tools** for debugging: `python -m structure_net.logging.cli status`
- **Validate your data**: `python -m structure_net.logging.cli validate file.json`

---

**The standardized logging system ensures your experiment data is always safe, valid, and accessible. Happy experimenting! 🚀**
