"""Reuse the existing lossless disk loader with a 256MiB feature-cache budget."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_sequential_full_training_v2 as training

PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-sequential-disk-streaming-plan.md')
CACHE_BYTES = 256 * 1024**2
RUNS = {stage:training.ROOT / ('20261025_sequential_full732_disk_' + stage + '_v1')
    for stage in ('pilot','train')}
CONFIG = {**training.CONFIG, 'host_feature_cache_bytes':CACHE_BYTES,
    'feature_loading_variant':'existing_lossless_disk_LRU_256MiB'}


def run(stage):
    # Reuse the frozen implementation without changing its source or other jobs.
    # freeze() records these effective values plus this launcher in its producer.
    previous = training.CONFIG, training.PLAN, training.RUNS
    training.CONFIG, training.PLAN, training.RUNS = CONFIG, PLAN, RUNS
    try:
        training.run(stage)
    finally:
        training.CONFIG, training.PLAN, training.RUNS = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=tuple(RUNS))
    stage = parser.parse_args().stage
    existed = RUNS[stage].exists()
    try:
        run(stage)
    except Exception as error:
        if not existed and RUNS[stage].exists():
            training.write_json(RUNS[stage] / 'failure.json',
                {'exception':type(error).__name__,'message':str(error),'quality_result':None})
        raise
