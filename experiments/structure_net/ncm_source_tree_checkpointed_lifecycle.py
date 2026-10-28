"""Repeat the preserved failed lifecycle with equivalent activation recomputation."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from experiments.structure_net import ncm_source_tree_training_lifecycle as original
from experiments.structure_net import ncm_source_tree_checkpointing as memory

data = original.data
FAILED = original.ROOT
ROOT = data.ROOT/'20261027_source_tree_training_lifecycle_v2'
QUALIFICATION = data.ROOT/'20261027_source_tree_checkpointing_CPU_v1.json'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-source-tree-checkpointing.md')
TEST = Path('tests/structure_net/test_ncm_source_tree_checkpointing.py')


def run():
    receipt = data.read(FAILED/'actual_tool_receipt.json'); data.verify(receipt['artifacts'])
    failure = data.read(FAILED/'failure.json')
    if receipt['actual_tool_exit_code'] != 1 or receipt['completed_optimizer_updates'] != 0 or failure['type'] != 'OutOfMemoryError':
        raise ValueError('preserved pre-optimizer CUDA OOM required')
    qualified = data.read(QUALIFICATION); data.verify(qualified['artifacts'])
    if qualified['actual_tool_exit_code'] != 0 or qualified['tests_passed'] != 2:
        raise ValueError('checkpointing parity/RNG CPU tests required')
    extra = [QUALIFICATION,TEST,original.PLAN,*[FAILED/n for n in ('producer.json','failure.json','actual_tool_receipt.json')]]
    old_parent = original.task.require_parent
    original.task.require_parent = lambda:[*old_parent(),*extra]
    original.tree = SimpleNamespace(install=memory.install,__file__=memory.__file__)
    original.ROOT,original.PLAN = ROOT,PLAN
    original.STORAGE = Path('/data2/structure_net/ncm_source_tree_20261027_v1/lifecycle_checkpointed')
    original.run()


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args(); existed = ROOT.exists()
    try:run()
    except Exception as error:
        if ROOT.exists() and not existed:data.write(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
