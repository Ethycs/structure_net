"""Use the existing source-disjoint semantic gates for the Pixi-trained NCM."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_pixi as runtime


def run(stage):
    training, environment = runtime.configure()
    from experiments.structure_net import ncm_sequential_development_v2 as evaluation
    qualification = training.q.read(runtime.CHECK / 'result.json')
    if qualification['runtime'] != environment:
        raise ValueError('The evaluated checkpoint must use the qualified Pixi runtime')
    for path, expected in qualification['input_hashes'].items():
        if training.file_hash(Path(path)) != expected:
            raise ValueError('Qualified runtime input changed: ' + path)
    original_freeze = evaluation.freeze

    def freeze(root, mode, extra):
        proof = original_freeze(root, mode, extra)
        proof['environment'] = environment
        for path in (Path('pyproject.toml'), Path('pixi.lock'), runtime.CHECK / 'result.json'):
            proof['input_hashes'][str(path)] = training.file_hash(path)
        proof['fingerprint'] = training.q.digest({k: v for k, v in proof.items() if k != 'fingerprint'})
        training.write_json(root / 'producer.json', proof)
        return proof

    previous = training.RUNS, training.PLAN, evaluation.RUNS, evaluation.freeze
    training.RUNS, training.PLAN = runtime.RUNS, runtime.PLAN
    evaluation.RUNS = {s: runtime.ROOT / ('20261025_sequential_full732_pixi_DEV_' + s + '_v1')
                       for s in ('inference_pilot', 'outputs', 'judge_pilot', 'grade')}
    evaluation.freeze = freeze
    try:
        if stage in ('inference_pilot', 'outputs'):
            evaluation.generate(stage == 'inference_pilot')
        else:
            evaluation.grade(stage == 'judge_pilot')
    finally:
        training.RUNS, training.PLAN, evaluation.RUNS, evaluation.freeze = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('inference_pilot', 'outputs', 'judge_pilot', 'grade'))
    run(parser.parse_args().stage)
