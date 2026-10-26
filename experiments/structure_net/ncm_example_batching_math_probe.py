"""Separate SDPA-backend roundoff from the failed default-backend batch test."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.structure_net import ncm_example_batching_probe as original

FAILED = original.ROOT
ROOT = original.broad.runtime.ROOT/'20261026_example_batching_math_probe_v2'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-example-batching-math-control.md')


@dataclass(frozen=True)
class Config(original.Config):
    parity_attention_backend: str = 'PyTorch_SDPA_MATH_only'
    capacity_attention_backend: str = 'unchanged_default_selection'
    original_default_backend_gate_remains_failed: bool = True


def run():
    import json
    prior = json.loads((FAILED/'result.json').read_text())
    receipt = json.loads((FAILED/'actual_tool_receipt.json').read_text())
    if (prior['classification'] != 'FP32_example_batching_parity_failed'
            or receipt['actual_tool_exit_code'] != 0 or receipt['tool_chunk'] != 'f07d30'
            or not prior['weights_unchanged'] or prior['resource_results']):
        raise ValueError('preserved completed default-backend failure required')
    for path, expected in receipt['artifacts'].items():
        if original.broad.file_hash(path) != expected:
            raise ValueError('original failure evidence changed')
    base = original.parity

    def parity(*args, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            result = base(*args, **kwargs)
        return {**result, 'attention_backend': 'MATH'}

    original.ROOT, original.PLAN, original.CONFIG, original.parity = ROOT, PLAN, Config(), parity
    original.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('run',))
    parser.parse_args()
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if not existed and ROOT.exists():
            original.broad.write_json(ROOT/'failure.json', {'exception': type(error).__name__,
                'message': str(error), 'task_accuracy': None})
        raise
