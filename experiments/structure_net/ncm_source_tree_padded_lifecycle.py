"""Qualify per-source hierarchy depths after preserving both failed attempts."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from experiments.structure_net import ncm_source_tree_training_lifecycle as original
from experiments.structure_net import ncm_source_tree_checkpointed_lifecycle as prior
from experiments.structure_net import ncm_source_tree_padding as memory

data = original.data
ROOT = data.ROOT/'20261027_source_tree_training_lifecycle_v3'
QUALIFICATION = data.ROOT/'20261027_source_tree_padding_CPU_v1.json'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-source-tree-padding.md')
TEST = Path('tests/structure_net/test_ncm_source_tree_padding.py')


def run():
    extra = [QUALIFICATION,TEST,original.PLAN,prior.PLAN,prior.TEST,prior.QUALIFICATION]
    for failed,kind in ((prior.FAILED,'OutOfMemoryError'),(prior.ROOT,'ValueError')):
        receipt = data.read(failed/'actual_tool_receipt.json'); data.verify(receipt['artifacts'])
        if receipt['actual_tool_exit_code'] != 1 or receipt['completed_optimizer_updates'] != 0 or data.read(failed/'failure.json')['type'] != kind:
            raise ValueError('both actual pre-optimizer failures required')
        extra += [failed/n for n in ('producer.json','failure.json','actual_tool_receipt.json')]
    extra += [prior.ROOT/'parity_clean.json',prior.ROOT/'memory.jsonl']
    qualified = data.read(QUALIFICATION); data.verify(qualified['artifacts'])
    if qualified['actual_tool_exit_code'] != 0 or qualified['tests_passed'] != 4: raise ValueError('new padding/checkpointing CPU regressions required')
    old_parent = original.task.require_parent
    original.task.require_parent = lambda:[*old_parent(),*extra]
    original.tree = SimpleNamespace(install=memory.install,__file__=memory.__file__)
    original.ROOT,original.PLAN = ROOT,PLAN
    original.STORAGE = Path('/data2/structure_net/ncm_source_tree_20261027_v1/lifecycle_per_source')
    original.run()


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args(); existed = ROOT.exists()
    try:run()
    except Exception as error:
        if ROOT.exists() and not existed:data.write(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
