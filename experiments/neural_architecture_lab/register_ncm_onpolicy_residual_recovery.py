"""Keep the conservative registry contract for the paired v2 execution root."""
import argparse
from pathlib import Path

from experiments.neural_architecture_lab import register_ncm_onpolicy_residual as original

ROOT = Path('data/experiments/ncm_paper_regime/20261025_onpolicy_residual_results_v2')


def configure():
    original.ROOT = ROOT
    original.CAMPAIGN = ROOT/'campaign_results.json'
    return original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('register','readback'))
    configure().main(parser.parse_args().stage)
