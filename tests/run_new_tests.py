#!/usr/bin/env python3
"""
Quick test runner for the new data factory and stress test memory tests.
"""

import subprocess
import sys

def run_tests():
    """Run the new test suites."""
    print("Running Data Factory Integration Tests...")
    print("=" * 80)
    
    # Run data factory tests
    result1 = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/test_data_factory_integration.py", 
        "-v", "--tb=short",
        "-k", "not integration"  # Skip integration tests for quick run
    ])
    
    print("\n\nRunning Stress Test Memory Tests...")
    print("=" * 80)
    
    # Run stress test memory tests
    result2 = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/test_stress_test_memory.py", 
        "-v", "--tb=short",
        "-k", "not integration"  # Skip integration tests for quick run
    ])
    
    print("\n\nTest Summary:")
    print("=" * 80)
    print(f"Data Factory Tests: {'PASSED' if result1.returncode == 0 else 'FAILED'}")
    print(f"Stress Test Memory Tests: {'PASSED' if result2.returncode == 0 else 'FAILED'}")
    
    return result1.returncode == 0 and result2.returncode == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)