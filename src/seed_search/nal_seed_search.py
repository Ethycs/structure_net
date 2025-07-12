#!/usr/bin/env python3
"""
DEPRECATED: NAL-Powered GPU Seed Hunter

This script has been replaced by the more robust and configurable
`experiments/seed_search_experiment.py`.

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings

warnings.warn(
    "This script is deprecated. Please use `experiments/seed_search_experiment.py` instead.",
    DeprecationWarning,
    stacklevel=2
)

from experiments.seed_search_experiment import main

if __name__ == "__main__":
    print("--- DEPRECATED SCRIPT ---")
    print("Running the new seed search experiment from `experiments/seed_search_experiment.py`...")
    print("-------------------------")
    main()
