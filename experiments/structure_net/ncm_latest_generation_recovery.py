"""Publish only an active generation output file by atomic replacement."""
import argparse
import json
import os
from pathlib import Path
import sys

from experiments.structure_net import ncm_latest_recovered_evaluation as latest

BASE = latest.BASE
FAILED = BASE/'20261029_latest_recovered_DEV_inference_pilot_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-29_ncm-latest-generation-publication-recovery.md')
TEST = Path('tests/structure_net/test_ncm_latest_generation_recovery.py')
CPU = BASE/'20261029_latest_generation_recovery_CPU_v1.json'
ROOTS = {stage: root if stage == 'preflight' else root.with_name(root.name.removesuffix('_v1')+'_v2')
         for stage, root in latest.GENERATION.items()}


def check_failure(receipt, failure, outputs):
    if (receipt['actual_tool_exit_code'] != 1 or receipt.get('new_judge_calls') != 0
            or failure.get('type') != 'FileExistsError'
            or str(FAILED/'outputs.json') not in failure.get('message', '')
            or set(outputs) != {'parent_with_source', 'new_codec_only'}
            or len(outputs['parent_with_source']) != 1 or outputs['new_codec_only']):
        raise ValueError('exact preserved first-case publication failure required')


def require():
    checked = latest.data.read(CPU)
    latest.data.verify(checked['artifacts'])
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 7:
        raise ValueError('four publication and three unchanged output contracts required')
    receipt = latest.data.read(FAILED/'actual_tool_receipt.json')
    latest.data.verify(receipt['artifacts'])
    check_failure(receipt, latest.data.read(FAILED/'failure.json'), latest.data.read(FAILED/'outputs.json'))


def install_roots():
    require()
    latest.GENERATION = dict(ROOTS)


def bind_recovery(proof):
    value = {**proof, 'config': {**proof['config'],
        'generation_publication_recovery': 'atomic_replace_only_active_outputs_v2',
        'source_prediction_algorithm_unchanged': True}, 'input_hashes': dict(proof['input_hashes'])}
    for path in (Path(__file__).resolve(), PLAN, TEST, CPU,
                 FAILED/'actual_tool_receipt.json', FAILED/'failure.json', FAILED/'outputs.json'):
        name, digest = str(path), latest.data.sha(path)
        if name in value['input_hashes'] and value['input_hashes'][name] != digest:
            raise ValueError('publication recovery conflicts with an existing binding')
        value['input_hashes'][name] = digest
    return value


def active_writer(original, allowed):
    allowed = Path(allowed).resolve()
    def write(path, value):
        path = Path(path)
        if path.resolve() != allowed:
            return original(path, value)
        if (path.parent/'result.json').exists():
            raise ValueError('completed generation outputs remain immutable')
        pending = path.with_name(path.name+'.pending')
        with pending.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(pending, path)
    return write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('inference_pilot', 'outputs', 'audit'))
    parser.add_argument('--arm', choices=latest.semantic.ARMS)
    args = parser.parse_args()
    if (args.stage == 'audit') != (args.arm is not None):
        parser.error('only independent audit requires a readout arm')
    install_roots(); latest.install()
    root = latest.SEMANTIC[args.arm]['audit'] if args.stage == 'audit' else ROOTS[args.stage]
    if root.exists():
        raise ValueError('fresh recovery generation root required')
    original_write_proof = latest.previous.write_proof
    latest.previous.write_proof = lambda root, proof, files: original_write_proof(root, bind_recovery(proof), files)
    original_write = latest.data.write
    if args.stage != 'audit':
        latest.data.write = active_writer(original_write, root/'outputs.json')
    argv = sys.argv
    try:
        sys.argv = [latest.__file__, args.stage, *(['--arm', args.arm] if args.arm else [])]
        latest.main()
        if args.stage == 'inference_pilot':
            old = latest.data.read(FAILED/'outputs.json')['parent_with_source'][0]
            current = latest.data.read(root/'outputs.json')['parent_with_source']
            if len(current) != 2 or current[0] != old:
                raise ValueError('recovered pilot must reproduce the exact previously saved source prediction')
            original_write(root/'recovery_checks.json', {
                'previous_saved_parent_case_exactly_reproduced': True,
                'source_case_id': old['id'], 'new_judge_calls': 0, 'task_accuracy': None})
    except Exception as error:
        if root.exists() and not (root/'failure.json').exists():
            original_write(root/'failure.json', {'type': type(error).__name__, 'message': str(error),
                'preserve_failed_v1': True, 'automatic_retry': False})
        raise
    finally:
        latest.data.write, latest.previous.write_proof, sys.argv = original_write, original_write_proof, argv


if __name__ == '__main__':
    main()
