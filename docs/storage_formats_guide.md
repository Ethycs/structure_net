# Storage Formats in Structure Net

## Current Storage Status

### 1. NAL (Neural Architecture Lab)
- **Format**: JSON only
- **Location**: `results_dir/` (configurable)
- **Files**: 
  - `{hypothesis_id}_results.json` - Individual hypothesis results
  - `lab_report_{lab_id}.json` - Lab session reports
- **No HDF5 support built-in**

### 2. Metrics System
- **Format**: In-memory only (returns dictionaries)
- **No built-in storage**
- **Must be saved manually if persistence needed**

### 3. Data Factory (ChromaDB + HDF5)
- **Format**: Hybrid storage
  - ChromaDB for searchable metadata
  - HDF5 for large time series data
  - Compressed JSON for small data
- **Components**:
  - `TimeSeriesStorage` - HDF5-based storage for training histories
  - `HybridExperimentStorage` - Combines ChromaDB and HDF5
  - `NALChromaIntegration` - Bridge between NAL and hybrid storage

### 4. Ultimate Stress Test v2
- **Default**: JSON files
- **With `aggressive_memory_cleanup=True`**: ChromaDB + HDF5
- **Storage happens in**:
  - `/data/stress_test_v2_*/generation_*_results/` - JSON files
  - `/data/stress_test_v2_*/chroma_db/` - ChromaDB database
  - `/data/stress_test_v2_*/timeseries/` - HDF5 files

## How to Enable HDF5 Storage

### For Ultimate Stress Test v2
```python
# Already enabled by default when aggressive_memory_cleanup=True
config = StressTestConfig(
    aggressive_memory_cleanup=True  # Default is True
)
```

When enabled, the stress test will:
1. Store experiment metadata in ChromaDB
2. Store large training histories (>10 epochs) in HDF5
3. Fall back to JSON if storage fails

### For Custom NAL Usage
```python
from data_factory.nal_integration import NALChromaIntegration
from data_factory.search import ChromaConfig
from data_factory.time_series_storage import TimeSeriesConfig

# Configure storage
chroma_config = ChromaConfig(
    persist_directory="./chroma_db",
    collection_name="experiments"
)

timeseries_config = TimeSeriesConfig(
    storage_dir="./timeseries",
    use_hdf5=True,
    compression="gzip"
)

# Create integration
integration = NALChromaIntegration(
    chroma_config=chroma_config,
    timeseries_config=timeseries_config,
    timeseries_threshold=10  # Use HDF5 for >10 epochs
)

# Index NAL results
integration.index_experiment_result(experiment_result, hypothesis_id)
```

### For Metrics Storage
Currently, metrics must be manually saved. Example:
```python
# Compute metrics
metrics = complete_metrics_system.compute_all_metrics(network, data_loader)

# Save to HDF5
import h5py
with h5py.File("metrics.h5", "w") as f:
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            f.create_dataset(key, data=value, compression="gzip")
        else:
            f.attrs[key] = value
```

## Benefits of HDF5 Storage

1. **Space Efficiency**: 
   - Compressed storage (gzip by default)
   - Binary format vs text JSON
   - ~10x smaller for large arrays

2. **Performance**:
   - Fast random access to datasets
   - Memory-mapped access available
   - Parallel I/O support

3. **Data Organization**:
   - Hierarchical structure (like filesystem)
   - Metadata support
   - Multiple datasets per file

## Storage Recommendations

1. **Small experiments (<100 epochs)**: JSON is fine
2. **Large experiments (>100 epochs)**: Use HDF5
3. **Need search capability**: Use ChromaDB + HDF5 hybrid
4. **Metrics analysis**: Consider dedicated HDF5 files

## Future Integration Plans

To fully integrate HDF5 across the system:
1. Add HDF5 backend option to NAL's `_save_hypothesis_results()`
2. Add storage methods to metrics system
3. Create unified storage interface
4. Add data migration tools (JSON → HDF5)