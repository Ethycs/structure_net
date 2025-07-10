#!/usr/bin/env python3
"""Simple test to check if the stress test runs without memory issues."""

import subprocess
import sys
import time

print("Testing Ultimate Stress Test v2 with minimal configuration...")
print("=" * 80)

# Run with minimal config to test memory usage
cmd = [
    sys.executable, 
    "experiments/ultimate_stress_test_v2.py",
    "--generations", "1",
    "--tournament-size", "4",
    "--dataset", "mnist"  # Use smaller dataset
]

print(f"Running command: {' '.join(cmd)}")
print("This should complete quickly if memory management is working...")
print("-" * 80)

start_time = time.time()
result = subprocess.run(cmd, capture_output=False, text=True)
elapsed = time.time() - start_time

print("-" * 80)
if result.returncode == 0:
    print(f"✅ Test PASSED in {elapsed:.1f} seconds!")
    print("Memory management appears to be working correctly.")
else:
    print(f"❌ Test FAILED after {elapsed:.1f} seconds!")
    print("There may still be memory issues to resolve.")
    
sys.exit(result.returncode)