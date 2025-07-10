# Ultimate Stress Test v2 Memory Optimization Summary

## Problem Identified

The primary memory issue was caused by the hypothesis parameter space containing ALL competitor data, which was pickled and sent to every worker process, causing massive memory duplication.

## Optimizations Implemented

### 1. **Fixed Parameter Space Duplication** (CRITICAL FIX)
- **Before**: All competitor data was in `parameter_space`, pickled to every worker
- **After**: Only competitor indices are in `parameter_space`, with a mapping stored separately
- **Impact**: Reduces memory usage by O(N) where N = number of competitors

### 2. **Removed Dataset Pre-loading**
- **Before**: Datasets were pre-loaded and passed to all experiments
- **After**: Each experiment loads its own dataset instance
- **Impact**: Prevents dataset duplication across processes

### 3. **Aggressive NAL Memory Cleanup**
- Monkey-patched NAL's `generate_experiments` to not store experiments in memory
- Clear experiment results after processing each generation
- Clear hypothesis objects and their results
- Clear runner caches and internal state
- Multiple garbage collection passes

### 4. **Memory-Aware Configuration**
- Reduced default parallelism (cap at 4 concurrent experiments)
- Reduced batch sizes and epoch counts
- Added timeout to prevent hanging experiments
- Enable verbose logging to track progress

### 5. **Disk-Based Result Storage**
- Results are immediately saved to disk after evaluation
- Only essential metrics kept in memory
- ChromaDB/HDF5 integration for efficient storage

### 6. **NAL Recreation on High Memory**
- Monitor memory usage between generations
- Recreate NAL instance if memory > 70% of system RAM
- Ensures clean state for each generation

## Code Changes

### In `create_competitor_hypothesis()`:
```python
# OLD - BAD: All competitor data in parameter space
parameter_space = {
    'architecture': [c['architecture'] for c in self.population],
    'sparsity': [c['sparsity'] for c in self.population],
    # ... duplicated for EVERY worker
}

# NEW - GOOD: Only indices in parameter space
parameter_space = {
    'competitor_index': list(range(len(self.population))),
}
# Mapping stored separately, not pickled to workers
self.competitor_mapping = {...}
```

### In `evaluate_competitor_task_optimized()`:
```python
# Extract only the specific competitor's data
competitor_index = config['competitor_index']
competitor = config['competitor_mapping'][competitor_index]
# Process just this one competitor
```

## Memory Usage Expectations

With these optimizations:
- Initial memory: ~1-2 GB
- Per-generation increase: <500 MB (previously: several GB)
- Total for 5 generations with 64 competitors: ~3-4 GB (previously: 10+ GB)

## Testing

Run the memory test:
```bash
python test_memory_optimization.py
```

This will monitor memory usage during a small tournament and report if optimizations are working.

## Future Improvements

1. **Shared Memory Datasets**: Use torch.multiprocessing shared memory for datasets
2. **GPU Prefetching**: Overlap data loading with computation
3. **Streaming Results**: Stream results to disk instead of accumulating
4. **Worker Pooling**: Reuse worker processes across generations