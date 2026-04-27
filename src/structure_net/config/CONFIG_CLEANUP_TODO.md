# Configuration Cleanup TODO

This document tracks all the remaining configuration classes that need to be migrated to the unified configuration system.

## High Priority - Core Configs

### 1. Neural Architecture Lab
- [ ] `src/neural_architecture_lab/core.py` - Contains `LabConfig` class
- [ ] `src/neural_architecture_lab/config_factory.py` - Contains `LabConfigFactory`
- [ ] Update all NAL files to use `config.get_config().get_lab_config()`

### 2. Data Factory
- [ ] `src/data_factory/config.py` - Contains `DatasetConfig` class
- [ ] Migrate to use `config.storage` for paths

### 3. Logging System
- [ ] `src/structure_net/logging/standardized_logging.py` - Contains `LoggingConfig`
- [ ] `src/structure_net/logging/artifact_manager.py` - Contains `ArtifactConfig`
- [ ] Migrate to use unified `config.logging` and `config.wandb`

## Medium Priority - Component Configs

### 4. Evolution System
- [ ] `src/structure_net/evolution/advanced_layers.py` - `ThresholdConfig`, `MetricsConfig`
- [ ] `src/structure_net/evolution/metrics/` - Various metric configs
- [ ] Consider if these should be experiment parameters rather than global config

### 5. Profiling System
- [ ] `src/structure_net/profiling/core/profiler_manager.py` - `ProfilerConfig`
- [ ] Migrate to use `config.experiment.profiling` settings

### 6. Search System
- [ ] `src/data_factory/search/chroma_client.py` - `ChromaConfig`
- [ ] `src/data_factory/time_series_storage.py` - `TimeSeriesConfig`
- [ ] Migrate to use `config.storage.chromadb_path` etc.

## Low Priority - Model-Specific Configs

### 7. Model Configs
- [ ] `src/structure_net/models/fiber_bundle_network.py` - `FiberBundleConfig`
- [ ] These might stay as model-specific parameters

### 8. Component Configs
- [ ] Various analyzer and strategy configs in evolution components
- [ ] These are likely better as component parameters, not global config

## Migration Strategy

### Phase 1: Update Core Systems (This Week)
1. Update NAL to use unified config via compatibility layer
2. Update Data Factory to use unified storage paths
3. Update logging systems to use unified config

### Phase 2: Component Migration (Next Week)
1. Migrate evolution configs
2. Migrate profiling configs
3. Migrate search configs

### Phase 3: Cleanup (Following Week)
1. Remove old config classes
2. Update all imports
3. Update documentation

## Code Patterns to Update

### Old Pattern:
```python
from neural_architecture_lab import LabConfig
config = LabConfig(project_name="test", ...)
```

### New Pattern:
```python
from config import get_config
config = get_config()
config.experiment.project_name = "test"
lab_config = config.get_lab_config()  # For compatibility
```

### Or using migration helper:
```python
from config import create_lab_config
lab_config = create_lab_config(project_name="test")
```

## Files to Update

Use this grep command to find all files that need updating:
```bash
grep -r "class.*Config\|from.*Config\|import.*Config" src/ --include="*.py"
```

## Testing Strategy

1. Start with compatibility shims to ensure nothing breaks
2. Update one system at a time
3. Run tests after each system migration
4. Remove old configs only after all dependencies updated

## Notes

- Some configs (like model-specific ones) might be better as parameters
- Component configs should probably stay with their components
- Focus on configs that affect multiple systems first
- Maintain backward compatibility during transition