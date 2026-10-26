"""Run the unchanged paired replay with the recorded evaluation recovery bound."""
import argparse
from pathlib import Path

from experiments.neural_architecture_lab import finalize_ncm_onpolicy_residual as original
from experiments.structure_net import ncm_onpolicy_development_recovery as development

RESULTS = development.study.broad.runtime.ROOT/'20261025_onpolicy_residual_results_v2'
REGISTRAR = Path('experiments/neural_architecture_lab/register_ncm_onpolicy_residual_recovery.py')
_REPLAY = original.replay_arm


def replay_arm(arm):
    result = list(_REPLAY(arm))
    result[3] = [*result[3],Path(__file__),REGISTRAR,*development.INPUTS]
    return tuple(result)


def configure():
    original.development = development
    original.RESULTS = RESULTS
    original.replay_arm = replay_arm
    return original


def main():
    argparse.ArgumentParser(__doc__).parse_args()
    configure().finalize()


if __name__ == '__main__':
    main()
