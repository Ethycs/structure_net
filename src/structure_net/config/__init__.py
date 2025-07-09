"""
Configuration module for Structure Net.

Import this before torch to ensure proper environment setup.
"""

# Import environment setup first
from .environment import setup_cuda_devices

__all__ = ['setup_cuda_devices']