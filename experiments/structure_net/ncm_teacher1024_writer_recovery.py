"""Write continuation metadata once; preserve the pre-request v3 writer failure."""
import argparse
import json
from pathlib import Path
import sys

from experiments.structure_net import ncm_teacher1024_continuation as prior

data,base,original = prior.data,prior.base,prior.original
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-teacher1024-writer-recovery.md')
TEST = Path('tests/structure_net/test_ncm_teacher1024_writer_recovery.py')
SERVICE = Path('experiments/neural_architecture_lab/ncm_teacher1024_writer_recovery_service.py')
FAILURE = prior.ROOTS['prepare']
ROOTS = {s:data.ROOT/f'20261027_teacher1024_{s}_v4' for s in prior.ROOTS}


def write_producer(root,proof,sources):
    # data.write is intentionally exclusive-create. Construct the final proof first.
    proof['fingerprint'] = base.fingerprint(proof)
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sources):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(root/'producer.json',proof)
    return proof


def freeze(root,stage,inputs,environment):
    receipt = data.read(FAILURE/'actual_tool_receipt.json');data.verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code'] != 1 or receipt['new_requests'] != 0:
        raise ValueError('preserved actual pre-request writer failure required')
    if root.exists():raise ValueError('fresh append-only continuation root required')
    files = {Path(__file__).resolve(),PLAN.resolve(),TEST.resolve(),SERVICE.resolve(),
        Path(prior.__file__).resolve(),prior.PLAN.resolve(),prior.TEST.resolve(),prior.SERVICE.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):files.add(p)
    inputs = list(dict.fromkeys([*inputs,FAILURE/'producer.json',FAILURE/'failure.json',FAILURE/'actual_tool_receipt.json',
        Path('pixi.lock'),Path('pyproject.toml')]))
    config = data.read(FAILURE/'producer.json')['config']
    config = {**config,'continuation':'only586 untouched reserved sources after drained timeout failure',
        'original438_firsts_preserved':True,'original8_exhausted_failures_preserved':True,
        'exhausted_firsts_receive_no_new_attempt_allowance':True,'producer_written_once':True}
    proof = {'schema':original.SCHEMA,'hypothesis_id':original.HYPOTHESIS,'stage':stage,'environment':environment,
        'config':config,'source_hashes':{str(p):data.sha(p) for p in sorted(files)},
        'input_hashes':{str(p):data.sha(p) for p in inputs}}
    return write_producer(root,proof,sorted(files))


def install():
    prior.ROOTS = ROOTS;prior.freeze = freeze


def configure():
    install();return prior.configure()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=tuple(ROOTS));args = parser.parse_args()
    install();root = ROOTS[args.stage];existed = root.exists()
    try:
        if args.stage == 'prepare':prior.inventory()
        elif args.stage in ('pilot','generate'):prior.generate(args.stage == 'pilot')
        else:prior.audit(args.stage == 'pilot_audit')
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'error_type':type(error).__name__,'error':str(error),
            'preserve_firsts':True,'retry_same_sources_prohibited':True})
        raise
