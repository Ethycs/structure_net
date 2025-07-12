# Ultimate Stress Test v2 Memory Fixes

## Summary of Memory Optimizations

### 1. Dataset Sharing (Major Fix)
**Problem**: Each competitor was creating its own dataset, multiplying memory usage by tournament size.
**Solution**: Pre-load dataset once in `TournamentExecutor` and share across all experiments.

```python
# In TournamentExecutor.__init__
self._preload_dataset()

# In create_competitor_hypothesis
control_parameters = {
    ...
    'train_loader': self.train_loader,
    'test_loader': self.test_loader,
}

# In evaluate_competitor_task
if 'train_loader' in config and 'test_loader' in config:
    train_loader = config['train_loader']
    test_loader = config['test_loader']
```

### 2. Aggressive Result Caching to Disk
**Problem**: NAL accumulates all experiment results in memory.
**Solution**: Save results to disk immediately after evaluation and keep only essential metrics in memory.

```python
# Save full results to disk
results_dir = self.data_dir / f"generation_{generation}_results"
for res in hypothesis_result.experiment_results:
    result_file = results_dir / f"{comp_id}_result.json"
    with open(result_file, 'w') as f:
        json.dump({...}, f)
    # Keep only essential metrics in memory
    results_map[comp_id] = essential_metrics
```

### 3. NAL Memory Cleanup
**Problem**: NAL keeps hypotheses, experiments, and results in memory.
**Solution**: Aggressive cleanup after each generation.

```python
def _cleanup_nal_memory(self, hypothesis_id: str):
    # Clear results
    if hypothesis_id in self.lab.results:
        del self.lab.results[hypothesis_id]
    # Clear experiments
    # Clear hypotheses
    # Force garbage collection
```

### 4. NAL Recreation on High Memory
**Problem**: Even with cleanup, memory can accumulate.
**Solution**: Recreate NAL instance when memory usage exceeds 70%.

```python
if memory_percent > 70:  # If using more than 70% of RAM
    del self.lab
    gc.collect()
    self._init_nal()
```

### 5. Reduced Default Parameters
- `tournament_size`: 64 → 32
- `batch_size`: 256 → 128
- `max_experiments_in_memory`: 100

### 6. Comprehensive Logging
Added logging throughout to track memory usage:
- Before/after NAL evaluation
- Memory increases per generation
- High memory warnings
- Cleanup operations

## Memory Usage Tracking

The stress test now logs:
1. RAM and GPU memory before/after each generation
2. Memory increases during evaluation
3. Memory after cleanup operations
4. Warnings when recreating NAL

Check logs in: `/data/stress_test_v2_*/queue/` for detailed memory tracking.

## ChromaDB Integration (Optional)

When `aggressive_memory_cleanup` is enabled, the system can use ChromaDB to offload experiment data:

```python
if self.config.aggressive_memory_cleanup:
    self._init_chroma_integration()
```

This stores experiment metadata in ChromaDB and large time series data in HDF5 files.

## Expected Results

With these optimizations:
- Memory usage should stabilize between generations
- Dataset memory is reused (not multiplied by tournament size)
- Only essential data kept in memory
- Full results available on disk for analysis

## Usage

To run with all optimizations:
```bash
python experiments/ultimate_stress_test_v2.py --enable-profiling
```

Monitor the memory logs to ensure the fixes are working effectively.