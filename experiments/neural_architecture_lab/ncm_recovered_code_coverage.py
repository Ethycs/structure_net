"""Descriptive target-code coverage and marginal entropy; no prediction accuracy."""
from collections import Counter
import json
import os
from pathlib import Path
import resource

import numpy as np

from experiments.structure_net import ncm_recovered_training_inputs as inputs

BASE = inputs.BASE
ROOT = BASE/'20261028_recovered_code_coverage_v1'
DIAGNOSTIC = BASE/'20261028_recovered_decoder_readout_eval_v1'
LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)


def histogram(codes, weights):
    if codes.shape[1] != 641 or len(weights) != len(codes) or codes.min() < 0 or codes.max() >= 8192:
        raise ValueError('complete in-range target hierarchies required')
    result = np.zeros((641, 8192), dtype=np.int64)
    positions = np.arange(641)[None, :]
    for start in range(0, len(codes), 512):
        rows = codes[start:start+512]
        np.add.at(result, (positions, rows), np.asarray(weights[start:start+512], dtype=np.int64)[:, None])
    if not np.all(result.sum(-1) == sum(weights)): raise ValueError('every target position must have the same exposure')
    return result


def metrics(hist, held):
    total = int(hist[0].sum()); offset = 0; result = []
    for level in LEVELS:
        part = hist[offset:offset+level]
        probabilities = part/total
        logs = np.zeros_like(probabilities); np.log2(probabilities, out=logs, where=probabilities > 0)
        entropy = -(probabilities*logs).sum(-1)
        present = part[np.arange(level)[None, :], held[:, offset:offset+level]] > 0
        result.append({'positions': level, 'mean_position_marginal_entropy_bits': float(entropy.mean()),
            'positionwise_majority_code_fraction': int(part.max(-1).sum())/(total*level),
            'mean_distinct_codes_per_position': float((part > 0).sum(-1).mean()),
            'reserved_target_codes_seen': int(present.sum()), 'reserved_target_code_positions': present.size,
            'reserved_all_codes_seen_cases': int(present.all(-1).sum())})
        offset += level
    return result


def main():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('fresh CPU-only diagnostic required')
    proofs = inputs.Proofs()
    targets = proofs.completed(inputs.qualified.CODES)
    balanced = proofs.completed(inputs.schedule.balancing.ROOT)
    proofs.completed(DIAGNOSTIC)
    bindings = proofs.read(inputs.BINDINGS)
    anchor_root = Path(bindings['targets']['full']); proofs.completed(anchor_root)
    anchors = proofs.read(anchor_root/'cases.json')
    if len(anchors) != 2384 or targets['frames'] != 43894: raise ValueError('complete target populations required')
    schedule_path = inputs.schedule.balancing.ROOT/'planned_examples.jsonl'; proofs.bind(schedule_path)
    recovered_weights, anchor_weights = Counter(), Counter()
    with schedule_path.open() as f:
        for row in map(json.loads, f):
            (recovered_weights if row['population'] == 'recovered' else anchor_weights)[row['id']] += 1
    if sum(recovered_weights.values())+sum(anchor_weights.values()) != balanced['presentations']:
        raise ValueError('entire qualified exposure stream required')
    index = inputs.lines(targets['index']); proofs.bind(targets['index'])
    codes = np.memmap(targets['binary'], dtype='<u2', mode='r', shape=(43894, 641))
    weights = [sum(recovered_weights[k] for k in row['pair_ids']) for row in index]
    if len(index) != 43894 or any(len(row['pair_ids']) != 1 for row in index) or min(weights) < 1:
        raise ValueError('complete unique-frame and target-exposure mapping required')
    reserved = [r for r in proofs.read(DIAGNOSTIC/'outputs.json') if r['arm'] == 'parent_codec_only' and r['split'] == 'reserved_component']
    if len(reserved) != 64 or len({r['id'] for r in reserved}) != 64: raise ValueError('fixed64 reserved component diagnostic required')
    held = np.asarray([r['codes'] for r in reserved], dtype=np.int64)
    gold = np.asarray([r['oracle_codes'] for r in anchors], dtype=np.int64)
    a_weights = [anchor_weights[r['id']] for r in anchors]
    proof = {'schema': 'nal.ncm-recovered-code-coverage.v1', 'classification': 'descriptive_target_aware_component_diagnostic',
        'config': {'populations': ['reviewed_anchors', 'recovered_targets', 'planned_mixed_exposure'],
            'levels': LEVELS, 'reserved_component_cases': 64, 'code_vocabulary': 8192, 'model_calls': 0,
            'joint_information_estimated': False, 'training_recipe_changed': False},
        'source_hashes': {str(p): proofs.sha(p) for p in (Path(__file__), Path(inputs.__file__))},
        'input_hashes': dict(proofs.inputs)}
    ROOT.mkdir(); (ROOT/'producer.py').write_bytes(Path(__file__).read_bytes())
    def write(name, value):
        with (ROOT/name).open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    write('producer.json', proof)
    summary = {}
    hist = histogram(gold, [1]*len(gold)); summary['reviewed_anchors'] = metrics(hist, held)
    hist = histogram(codes, [1]*len(codes)); summary['recovered_targets'] = metrics(hist, held)
    hist = histogram(codes, weights); hist += histogram(gold, a_weights)
    summary['planned_mixed_exposure'] = metrics(hist, held)
    proofs.verify(proof['source_hashes']); proofs.verify(proof['input_hashes'])
    result = {'schema': proof['schema'], 'classification': proof['classification'],
        'recovered_targets': 43894, 'reviewed_anchors': 2384, 'planned_presentations': balanced['presentations'],
        'level_statistics': summary, 'task_accuracy': None, 'model_calls': 0, 'new_judge_calls': 0,
        'final_test_opened': False, 'training_recipe_changed': False,
        'interpretation_limit': 'Marginal label support and majority-code frequency do not measure source-conditioned prediction accuracy, joint entropy, semantic quality, or a hard accuracy ceiling.',
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts': {str(p): proofs.sha(p) for p in ROOT.iterdir() if p.is_file()}}
    write('result.json', result)
    print(json.dumps({'status': 'complete_target_code_statistics', 'coarsest_level': {k: v[0] for k, v in summary.items()},
        'finest_level': {k: v[-1] for k, v in summary.items()}, 'peak_RSS_bytes': result['peak_RSS_bytes'], 'task_accuracy': None}), flush=True)


if __name__ == '__main__': main()
