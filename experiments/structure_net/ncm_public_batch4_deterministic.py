"""Repeat the unchanged batch4 lifecycle with deterministic GPU algorithms."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_public_batch4_lifecycle as original

FAILED = original.ROOT
ROOT = original.data.ROOT/'20261026_public_batch4_lifecycle_deterministic_v2'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-batch4-determinism.md')
TEST = Path('tests/structure_net/test_ncm_public_batch4_deterministic.py')


@dataclass(frozen=True)
class Config(original.Config):
    deterministic_algorithms: bool = True
    deterministic_warn_only: bool = False
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    CUBLAS_WORKSPACE_CONFIG: str = ':4096:8'


CONFIG = Config()


def configure_runtime():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != CONFIG.CUBLAS_WORKSPACE_CONFIG:
        raise ValueError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run():
    if ROOT.exists():
        raise ValueError('fresh deterministic lifecycle root required')
    receipt = original.data.read(FAILED/'actual_tool_receipt.json')
    equality = original.data.read(FAILED/'resumed_state_equality.json')
    if (receipt['actual_tool_exit_code'] != 1 or equality['model'] or equality['optimizer']
            or not all(v for k, v in equality.items() if k not in ('model', 'optimizer'))):
        raise ValueError('preserved failed model/optimizer replay with matching RNG states required')
    configure_runtime()
    before = original.freeze
    old_amendment = original.AMENDMENT
    def freeze(training, environment, identities, extra):
        paths = [FAILED/name for name in ('failure.json', 'actual_tool_receipt.json', 'resumed_state_equality.json',
                                         'update_2.json', 'resumed_update_2.json')]
        return before(training, environment, identities, [*extra, *paths, PLAN, TEST, old_amendment])
    original.ROOT, original.CONFIG, original.AMENDMENT, original.freeze = ROOT, CONFIG, PLAN, freeze
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
            original.data.write(ROOT/'failure.json', {'exception': type(error).__name__, 'message': str(error),
                'task_accuracy': None, 'systems_lifecycle_only_not_quality_evidence': True})
        raise
