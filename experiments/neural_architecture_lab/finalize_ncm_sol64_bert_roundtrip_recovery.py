"""Replay the unchanged fixed-case rendering study under its dependency recovery."""
import argparse

from experiments.structure_net import ncm_sol64_bert_roundtrip_recovery as recovery
from experiments.neural_architecture_lab import finalize_ncm_sol64_bert_roundtrip as audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('finalize', 'register'))
    args = parser.parse_args()
    recovery.configure()
    recovery.require_unstarted()
    audit.run = recovery.base
    {'finalize': audit.finalize, 'register': audit.register}[args.stage]()
