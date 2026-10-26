"""Replay batching diagnostics without promoting systems checks to task quality."""
import json
import math
from pathlib import Path

from experiments.structure_net.ncm_memory_data import file_hash, write_json

BASE = Path('data/experiments/ncm_paper_regime')
ROOTS = (BASE/'20261026_example_batching_probe_v1', BASE/'20261026_example_batching_math_probe_v2',
         BASE/'20261026_example_batching_serial_codec_probe_v3')
DEST = BASE/'20261026_example_batching_results_v1'
HYPOTHESIS = 'ncm-example-batching-v1'


def read(path):
    return json.loads(Path(path).read_text())


def parity_gate(row, config):
    return (row['gradient_presence_equal'] and row['finite']
        and row['relative_gradient_L2_error'] <= config['gradient_relative_tolerance']
        and row['max_abs_gradient_difference'] <= config['gradient_absolute_tolerance']
        and row['logits_allclose'] and row['argmax_disagreements'] == 0)


def replay_parity(row, config):
    measured = [r for r in row['parameters'] if 'squared_error' in r]
    total_error = sum(r['squared_error'] for r in measured)
    total_norm = sum(r['reference_squared_norm'] for r in measured)
    maximum = max(r['max_absolute_error'] for r in measured)
    relative = math.sqrt(total_error/max(total_norm, 1e-30))
    if (not math.isclose(relative, row['relative_gradient_L2_error'], rel_tol=1e-12, abs_tol=1e-15)
            or maximum != row['max_abs_gradient_difference'] or row['passed'] != parity_gate(row, config)
            or len({r['name'] for r in row['parameters']}) != len(row['parameters'])):
        raise ValueError('gradient reduction or fixed parity gate differs')
    return {k: v for k, v in row.items() if k != 'parameters'}


def replay(root):
    receipt, result, producer = [read(root/name) for name in ('actual_tool_receipt.json', 'result.json', 'producer.json')]
    if receipt['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed diagnostic required')
    for key in ('artifacts',):
        for name, expected in receipt[key].items():
            if file_hash(name) != expected:
                raise ValueError('completed artifact changed: '+name)
    for key in ('source_hashes', 'input_hashes'):
        for name, expected in producer[key].items():
            if file_hash(name) != expected:
                raise ValueError('producer dependency changed: '+name)
    for i, (name, expected) in enumerate(sorted(producer['source_hashes'].items())):
        snapshot = root/'producer_source'/f'{i}_{Path(name).name}'
        if file_hash(snapshot) != expected:
            raise ValueError('frozen producing source snapshot differs')
    catalog = BASE/'20261025_broad_teacher_catalog_v1'
    lengths = {r['id']: r['source_positions'] for r in read(catalog/'source_index.json')}
    rows = read(BASE/'20261025_broad_generator_prepare_v1/cases.json')
    if (len(rows) != 1376 or {r['id'] for r in rows} != set(lengths)
            or any(r['split'] != 'TRAIN' for r in rows)):
        raise ValueError('all1376 previously admitted TRAIN examples required')
    ascending = sorted(lengths, key=lambda k: (lengths[k], k))
    descending = sorted(lengths, key=lambda k: (-lengths[k], k))
    expected_selection = {'parity_ids': [ascending[0], descending[0]], 'capacity_ids': descending[:8],
        'source_positions': {k: lengths[k] for k in dict.fromkeys([ascending[0], *descending[:8]])}}
    if producer['selection'] != expected_selection:
        raise ValueError('prospective shortest/longest input selection differs')
    if (result['hypothesis_id'] != HYPOTHESIS or not result['weights_unchanged']
            or result['optimizer_steps'] != 0 or result['new_judge_requests'] != 0
            or result['task_accuracy'] is not None or result['final_test_opened']
            or result['application_speed_tested'] or not result['systems_lifecycle_only_not_quality_evidence']):
        raise ValueError('unchanged-weight systems diagnostic boundaries required')
    config = result['config']
    checks = []
    for noisy in (False, True):
        path = root/('parity_noisy.json' if noisy else 'parity_clean.json')
        if path.exists():
            check = replay_parity(read(path), config)
            if check['noisy_history'] != noisy:
                raise ValueError('parity condition differs')
            checks.append(check)
            if not check['passed']:
                break
    if checks != result['parity']:
        raise ValueError('ordered parity evidence differs')
    passed = len(checks) == 2 and all(r['passed'] for r in checks)
    if not passed and result['resource_results']:
        raise ValueError('capacity ran after a failed numerical gate')
    sizes = []
    if len(result['resource_results']) > len(config['candidate_sizes']):
        raise ValueError('extra capacity trials outside fixed sweep')
    for expected_size, capacity in zip(config['candidate_sizes'], result['resource_results'], strict=False):
        size = capacity['microbatch']
        if size != expected_size or read(root/f'capacity_B{size}.json') != capacity:
            raise ValueError('capacity order or raw record differs')
        if capacity['status'] == 'resource_check_passed':
            times = sorted(r['seconds'] for r in capacity['iterations'] if not r['warmup'])
            if len(times) != config['measurements'] or len(times) != 3:
                raise ValueError('complete fixed measured trials required')
            if (capacity['median_effective_batch_seconds'] != times[1]
                    or capacity['examples_per_second'] != config['effective_examples']/times[1]
                    or capacity['peak_allocated_bytes'] > config['peak_limit_bytes']
                    or not all(r['finite_gradients'] for r in capacity['iterations'])):
                raise ValueError('resource measurement arithmetic differs')
            sizes.append(size)
        elif capacity != result['resource_results'][-1]:
            raise ValueError('sweep continued after a resource failure')
        for iteration in capacity.get('iterations', []):
            if read(root/f"capacity_B{size}_iteration{iteration['iteration']}.json") != iteration:
                raise ValueError('raw iteration differs')
    if sizes != result['resource_qualified_sizes']:
        raise ValueError('resource-qualified sizes differ')
    return {'root': str(root), 'parity_passed': passed, 'parity': checks,
        'resource_qualified_sizes': sizes, 'resource_results': result['resource_results'],
        'selection': producer['selection'], 'weights_unchanged': True,
        'producer_fingerprint': result['producer_fingerprint']}


def main():
    if DEST.exists():
        raise ValueError('fresh aggregate required')
    arms = [replay(root) for root in ROOTS]
    if any(a['parity_passed'] or a['resource_results'] for a in arms[:2]):
        raise ValueError('both original failed backend comparisons must be preserved')
    if any(a['selection'] != arms[0]['selection'] for a in arms):
        raise ValueError('matched TRAIN inputs required')
    artifacts = [p for root in ROOTS for p in sorted(root.glob('*.json'))]
    result = {'schema': 'nal.ncm-example-batching-aggregate.v1', 'hypothesis_id': HYPOTHESIS,
        'classification': 'original_batching_parity_failed_serial_codec_'+('passed' if arms[2]['parity_passed'] else 'failed'),
        'arms': arms, 'default_backend_numerically_qualified': False,
        'math_reference_numerically_qualified': arms[1]['parity_passed'],
        'serial_codec_numerically_qualified': arms[2]['parity_passed'],
        'resource_qualified_sizes': arms[2]['resource_qualified_sizes'],
        'training_optimizer_lifecycle_qualified': False, 'weights_unchanged': True,
        'optimizer_steps': 0, 'new_judge_requests': 0, 'task_accuracy': None,
        'final_test_opened': False, 'application_speed_tested': False,
        'systems_lifecycle_only_not_quality_evidence': True,
        'independent_GPU_forwards_repeated': False,
        'auditor_sha256': file_hash(__file__),
        'artifacts': {str(p): file_hash(p) for p in artifacts}}
    DEST.mkdir(); write_json(DEST/'campaign_results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('arms', 'artifacts')}))


if __name__ == '__main__':
    main()
