# Magic Trace Integration for Structure Net

## Overview

Magic Trace is a high-performance tracing tool that can capture detailed execution timelines without modifying your code. It's particularly useful for understanding performance bottlenecks in Structure Net's complex execution paths.

## Installation

```bash
# For Linux (Structure Net's primary platform)
wget https://github.com/janestreet/magic-trace/releases/latest/download/magic-trace-x86_64-unknown-linux-gnu.tar.gz
tar -xzf magic-trace-x86_64-unknown-linux-gnu.tar.gz
sudo mv magic-trace /usr/local/bin/

# Verify installation
magic-trace --version
```

## When to Use Magic Trace vs Built-in Profiling

### Use Magic Trace for:
- **Debugging performance mysteries** - When you don't know where the bottleneck is
- **Understanding execution flow** - Visualizing how NAL coordinates experiments  
- **Finding unexpected overhead** - Discovering hidden costs in library calls
- **One-off investigations** - No code changes needed

### Use Built-in Profiling for:
- **Continuous monitoring** - Tracking metrics across all experiments
- **Business logic metrics** - Accuracy vs training time trade-offs
- **Memory tracking** - GPU/CPU memory usage over time
- **Production insights** - Long-term performance trends

## Magic Trace with Structure Net

### 1. Tracing a Full Stress Test Run

```bash
# Start magic-trace in snapshot mode (records last 1-2 seconds on trigger)
magic-trace attach -pid $(pgrep -f "ultimate_stress_test_v2.py") -trace-include="structure_net.*,neural_architecture_lab.*"

# Or trace from start with automatic stop after 10 seconds
magic-trace run -target python experiments/ultimate_stress_test_v2.py -- --enable-profiling
```

### 2. Targeted Tracing with Triggers

Create a trigger script for specific phases:

```python
# In stress test code
import os
import signal

def trigger_trace():
    """Send SIGUSR1 to trigger magic-trace snapshot."""
    os.kill(os.getpid(), signal.SIGUSR1)

# In evaluate_competitor_task
def evaluate_competitor_task(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    # ... setup code ...
    
    # Trigger trace for training phase
    if config.get('trace_training', False):
        print("🎯 Triggering trace for training phase...")
        trigger_trace()
    
    # Training loop
    for epoch in range(config['epochs']):
        # ... training code ...
```

### 3. Analyzing Structure Net Patterns

Key patterns to look for in traces:

```python
# Pattern 1: Data Loading Bottlenecks
# Look for: repeated calls to dataset.__getitem__, transform operations

# Pattern 2: GPU Synchronization
# Look for: torch.cuda.synchronize, .item() calls

# Pattern 3: NAL Coordination Overhead  
# Look for: asyncio event loop, hypothesis scheduling

# Pattern 4: Memory Allocation Storms
# Look for: malloc/free patterns, tensor creation
```

### 4. Integration Script

```python
#!/usr/bin/env python3
"""
Magic Trace integration for Structure Net performance analysis.
"""

import subprocess
import time
import argparse
import os
from pathlib import Path

class StructureNetTracer:
    """Helper for tracing Structure Net with magic-trace."""
    
    def __init__(self, output_dir: str = "/data/traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def trace_experiment(
        self,
        experiment_cmd: str,
        duration: int = 10,
        include_patterns: List[str] = None
    ) -> Path:
        """Run experiment with tracing."""
        
        if include_patterns is None:
            include_patterns = [
                "structure_net.*",
                "neural_architecture_lab.*",
                "torch.*forward",
                "torch.*backward"
            ]
        
        trace_file = self.output_dir / f"trace_{int(time.time())}.fxt"
        
        cmd = [
            "magic-trace", "run",
            "-target", "python",
            "-duration", str(duration),
            "-output", str(trace_file)
        ]
        
        # Add include patterns
        for pattern in include_patterns:
            cmd.extend(["-trace-include", pattern])
        
        # Add the experiment command
        cmd.extend(["--", *experiment_cmd.split()])
        
        print(f"🔍 Starting trace: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        print(f"✅ Trace saved to: {trace_file}")
        self._generate_report(trace_file)
        
        return trace_file
    
    def _generate_report(self, trace_file: Path):
        """Generate HTML report from trace."""
        report_file = trace_file.with_suffix('.html')
        
        # Use magic-trace to generate Perfetto UI compatible HTML
        subprocess.run([
            "magic-trace", "html",
            str(trace_file),
            "-output", str(report_file)
        ])
        
        print(f"📊 View report at: file://{report_file.absolute()}")

# Usage example
if __name__ == "__main__":
    tracer = StructureNetTracer()
    
    # Trace a stress test run
    tracer.trace_experiment(
        "experiments/ultimate_stress_test_v2.py --generations 1",
        duration=30,
        include_patterns=[
            "evaluate_competitor_task",
            "create_standard_network",
            "torch.*"
        ]
    )
```

## Combining Magic Trace with Built-in Profiling

### 1. Two-Phase Analysis

```python
# Phase 1: Use built-in profiling to identify slow generations
results = run_stress_test_async(config)
slow_generations = [
    g for g in results['generations'] 
    if g['avg_time'] > threshold
]

# Phase 2: Re-run slow cases with magic-trace
for gen in slow_generations:
    print(f"Tracing slow generation {gen['id']} with magic-trace...")
    # Run with magic-trace attached
```

### 2. Correlation Analysis

```python
def correlate_traces_with_metrics(trace_file: Path, metrics: Dict):
    """Correlate magic-trace data with experiment metrics."""
    
    # Parse trace file (convert to JSON first)
    trace_data = parse_trace(trace_file)
    
    # Find hot functions
    hot_functions = find_hot_functions(trace_data, top_n=10)
    
    # Correlate with experiment performance
    correlation = {
        'hot_functions': hot_functions,
        'experiment_accuracy': metrics['accuracy'],
        'experiment_duration': metrics['duration'],
        'overhead_analysis': calculate_overhead(trace_data)
    }
    
    return correlation
```

## Best Practices

### 1. Selective Tracing

```bash
# Don't trace everything - focus on specific patterns
magic-trace run -target python experiment.py \
    -trace-include="evaluate_competitor_task" \
    -trace-include="forward" \
    -trace-exclude="logging.*"
```

### 2. Snapshot Mode for Long Runs

```python
# For multi-hour stress tests, use snapshot mode
# Triggers on SIGUSR1 to capture last 1-2 seconds

# In your code
if approaching_interesting_phase():
    os.kill(os.getpid(), signal.SIGUSR1)  # Trigger snapshot
```

### 3. Automated Analysis Pipeline

```python
class TraceAnalyzer:
    """Automated analysis of Structure Net traces."""
    
    def analyze_stress_test_trace(self, trace_file: Path) -> Dict:
        """Extract Structure Net specific insights."""
        
        # Key metrics to extract
        metrics = {
            'data_loading_overhead': self._measure_dataloader_time(trace_file),
            'gpu_utilization': self._measure_gpu_gaps(trace_file),
            'python_overhead': self._measure_interpreter_overhead(trace_file),
            'memory_allocation_cost': self._measure_allocation_overhead(trace_file)
        }
        
        # Generate recommendations
        if metrics['gpu_utilization'] < 0.8:
            print("⚠️  Low GPU utilization - consider larger batch sizes")
        
        if metrics['data_loading_overhead'] > 0.2:
            print("⚠️  High data loading overhead - consider more workers")
        
        return metrics
```

## Example Findings

Here's what magic-trace might reveal in Structure Net:

1. **Hidden NumPy/CPU bottlenecks** in data preprocessing
2. **Excessive tensor copying** between CPU/GPU
3. **Asyncio overhead** in NAL experiment coordination
4. **Memory allocation storms** during model creation
5. **Unexpected serialization costs** in multiprocessing

## Limitations

- **Linux only** (matches Structure Net's target platform)
- **Intel CPUs only** (uses Intel PT)
- **Large trace files** (GB+ for long runs)
- **Learning curve** for trace analysis

## Conclusion

Magic Trace is an excellent complement to Structure Net's built-in profiling:

- **Built-in profiling**: Answers "how long?" and "how much memory?"
- **Magic Trace**: Answers "why?" and "where exactly?"

Use both tools together for comprehensive performance insights.