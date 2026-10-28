"""Apply qualified process-local input hashing to future source-only DEV stages."""
import argparse
from pathlib import Path
import sys

from experiments.structure_net import ncm_recovered_review_lineage as lineage
from experiments.structure_net import ncm_recovered_training_verified_cache as cache

data, task = lineage.data, lineage.task
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-generation-validation-cache.md')
TEST = Path('tests/structure_net/test_ncm_verified_task_generation.py')
CPU = data.ROOT/'20261028_verified_task_generation_CPU_v1.json'
POLICY = {**cache.POLICY, 'scope': 'future source-only DEV generation input verification'}


def bind_policy(proof):
    value = {**proof, 'config': {**proof['config'], 'input_validation_policy': POLICY},
             'input_hashes': dict(proof['input_hashes'])}
    for path in (Path(__file__).resolve(), PLAN, TEST, CPU, Path(cache.__file__).resolve(),
                 cache.PLAN, cache.TEST, cache.CPU):
        name, digest = str(path), data.sha(path)
        if name in value['input_hashes'] and value['input_hashes'][name] != digest:
            raise ValueError('validation policy conflicts with a producer input')
        value['input_hashes'][name] = digest
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('inference_pilot', 'outputs'))
    parser.add_argument('--arm', choices=('existing', 'expanded'), required=True)
    args = parser.parse_args()
    for path, count in ((CPU, 4), (cache.CPU, 3)):
        checked = data.read(path)
        data.verify(checked['artifacts'])
        if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != count:
            raise ValueError('qualified unchanged cache and generation contracts required')
    proofs = cache.ProcessProofs()
    data.sha = proofs.sha
    lineage.require_preflight()
    lineage.install()
    original_write = task.write_proof
    task.write_proof = lambda root, proof, files: original_write(root, bind_policy(proof), files)
    sys.argv = [task.__file__, args.stage, '--arm', args.arm]
    task.main()


if __name__ == '__main__':
    main()
