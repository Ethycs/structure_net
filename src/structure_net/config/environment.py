"""
Centralized environment configuration for Structure Net.

This module should be imported before any torch imports to ensure
proper GPU configuration.
"""

import os

# GPU Configuration
DEFAULT_CUDA_DEVICES = "1,2"

def setup_cuda_devices(devices: str = None):
    """
    Set CUDA_VISIBLE_DEVICES environment variable.
    
    Args:
        devices: Comma-separated device IDs (e.g., "0,1,2"). 
                If None, uses DEFAULT_CUDA_DEVICES.
    """
    if devices is None:
        devices = DEFAULT_CUDA_DEVICES
    
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        # Only print when explicitly setting, not when already set
        if os.environ.get("STRUCTURE_NET_VERBOSE", "").lower() == "true":
            print(f"Set CUDA_VISIBLE_DEVICES={devices}")
    else:
        # Don't print this message - it's too noisy when spawning processes
        pass

# Automatically configure on import
setup_cuda_devices()

# Other centralized settings can go here
PYTORCH_ENABLE_MPS_FALLBACK = "1"  # For Mac M1/M2 compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = PYTORCH_ENABLE_MPS_FALLBACK

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set default number of threads for CPU operations
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

# Memory settings
PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF