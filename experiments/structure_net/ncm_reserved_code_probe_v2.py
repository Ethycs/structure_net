"""Correct one nonexistent audit filename while retaining the failed pilot."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_reserved_code_probe as original

AMENDMENT = Path('docs/07 - Status Reports/2026-10-27_ncm-reserved-code-probe-input-recovery.md')
TEST = Path('tests/structure_net/test_ncm_reserved_code_probe_v2.py')
FAILED = original.ROOTS['pilot']
MISSING = original.study.pipeline.bundle.catalog.INSPECTION_AUDIT/'campaign_results.json'
OLD_INPUTS = original.inputs
INSTALLED = False


def corrected_inputs(paths, missing=MISSING):
    if paths.count(missing) != 1 or missing.exists():
        raise ValueError('exact known nonexistent inspection campaign input required')
    result = [p for p in paths if p != missing]
    if any(not p.is_file() for p in result):raise ValueError('every remaining fingerprint input must exist')
    return result


def inputs():
    paths = corrected_inputs(OLD_INPUTS())
    paths += [Path(__file__),AMENDMENT,TEST,FAILED/'failure.json',FAILED/'actual_tool_receipt.json']
    paths += sorted((FAILED/'producer_source').glob('*.py'))
    if any(not p.is_file() for p in paths):raise ValueError('complete recovery input list required')
    return paths


def install():
    global INSTALLED
    if INSTALLED:return
    failure = original.data.read(FAILED/'failure.json')
    receipt = original.data.read(FAILED/'actual_tool_receipt.json')
    if (failure['type'] != 'FileNotFoundError' or str(MISSING) not in failure['message']
            or receipt['actual_tool_exit_code'] != 1 or receipt['tool_session'] != 10054
            or receipt['tool_chunk'] != 'c27058'):
        raise ValueError('exact retained pre-model input failure required')
    original.data.verify(receipt['artifacts'])
    inputs()
    original.inputs = inputs
    original.ROOTS = {s:original.data.ROOT/f'20261027_reserved_code_probe_{s}_v2'
                      for s in ('pilot','run','audit','results')}
    INSTALLED = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('pilot','run','audit','report','register','readback'))
    stage = parser.parse_args().stage
    install()
    if stage in ('report','register','readback'):
        from experiments.neural_architecture_lab import ncm_reserved_code_probe_results as results
        getattr(results,stage)()
    else:
        root = original.ROOTS[stage];existed = root.exists()
        try:original.audit() if stage == 'audit' else original.measure(stage)
        except Exception as error:
            if root.exists() and not existed:original.data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
            raise
