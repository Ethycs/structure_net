"""Disable unused prompt checkpoints while preserving native MTP rollback."""
import argparse
from pathlib import Path

from .free_gpu_70tps import run
from .target_70tps import Config

PLAN = Path('docs/07 - Status Reports/2026-10-31_word-games-70tps-mtp-checkpoints.md')
PREVIOUS = Path('data/experiments/word_games/20261031_70tps_free_gpus_v1/campaign_results.json')
SOURCE = Path('.pixi/word-games-sampling-v2/source/tools/server/server-context.cpp')
CONFIGS = {'control': Config('control', True, 1024, True, batch=2048,
                            no_checkpoints=True, free_layout=True)}
for microbatch in (512, 1024):
    for tokens in (1, 2):
        name = f'mtp{tokens}_nocp{microbatch}'
        CONFIGS[name] = Config(name, True, microbatch, True, batch=2 * microbatch,
            no_checkpoints=True, free_layout=True, mtp=True, mtp_gpu=True, mtp_tokens=tokens)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--port', default=18088, type=int)
    run(p.parse_args(), configs=CONFIGS, plan=PLAN, previous=PREVIOUS,
        stage='mtp-nocp', extra_sources=(SOURCE,))
