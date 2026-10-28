"""Unchanged semantic controls on the isolated persistent SQLite configuration."""
import argparse

from experiments.neural_architecture_lab import ncm_persistent_judge_gateway_v2 as gateway
from experiments.structure_net import ncm_isolated_judge_qualification as original

IDENTITY='ncm-persistent-isolated-SQLite-judge-transport-v1'
ROOTS={(role,stage):original.BASE/f'20261028_persistent_SQLite_judge_{role}_{stage}_v1'
       for role in original.transport.MODELS for stage in ('qualification','audit')}


def install():
    gateway.require();gateway.install()
    original.neutral=gateway.base
    original.ROOTS=ROOTS
    original.PLAN,original.TEST=gateway.PLAN,gateway.TEST
    original.transport.IDENTITY=IDENTITY
    original.transport.credential=gateway.credential


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('qualify','audit'))
    parser.add_argument('--role',choices=tuple(original.transport.MODELS),required=True);args=parser.parse_args()
    install();root=ROOTS[args.role,'qualification' if args.stage=='qualify' else 'audit'];existed=root.exists()
    try:original.qualify(args.role) if args.stage=='qualify' else original.audit(args.role)
    except Exception as error:
        if root.exists() and not existed:
            original.transport.write(root/'failure.json',{'type':type(error).__name__,'message':str(error),
                'first_judgments_preserved':True,'automatic_retry':False})
        raise
