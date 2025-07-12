"""
Argument Parser for Standardized Logging Configuration

This module provides a reusable argument parser for configuring the
standardized logging system from the command line.
"""

import argparse
import logging

def add_logging_arguments(parser: argparse.ArgumentParser):
    """
    Adds logging-related arguments to the given argument parser.
    
    Args:
        parser: The argparse.ArgumentParser instance to add arguments to.
    """
    group = parser.add_argument_group('Logging Options')
    
    group.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the global logging level.'
    )
    group.add_argument(
        '--module-log-level',
        action='append',
        nargs=1,
        metavar='MODULE:LEVEL',
        help='Set logging level for a specific module (e.g., "nal:DEBUG"). Can be used multiple times.'
    )
    group.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to a file to write logs to.'
    )
    group.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='Weights & Biases project name for logging.'
    )
    group.add_argument(
        '--disable-wandb',
        action='store_true',
        help='Disable Weights & Biases logging.'
    )



