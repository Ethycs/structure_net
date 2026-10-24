"""Independently replay the two TRAIN-case history diagnostic from raw logits."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_native_history_parity as run
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = run.ROOT / '20261024_native655_history_path_parity_audit_v1'
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-native-history-parity.md')
LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)


def replay(raw):
    expected = {'packed', 'sequential', 'oracle_codes', 'free_codes',
        'generated_history_logits', 'swapped_source_logits'}
    if set(raw) != expected:
        raise ValueError('complete original six-tensor diagnostic required')
    for key, tensor in raw.items():
        shape = (1, 641) if key.endswith('codes') else (1, 641, 8192)
        dtype = torch.long if key.endswith('codes') else torch.float32
        if tensor.shape != shape or tensor.dtype != dtype or not torch.isfinite(tensor).all():
            raise ValueError('complete finite original diagnostic tensors required')
        if key.endswith('codes') and ((tensor < 0).any() or (tensor >= 8192).any()):
            raise ValueError('native concept vocabulary required')
    rows, start = [], 0
    for size in LEVELS:
        end = start + size
        a, b = raw['packed'][:, start:end], raw['sequential'][:, start:end]
        gold = raw['oracle_codes'][:, start:end]
        rollout = raw['generated_history_logits'][:, start:end]
        rows.append({'positions': size, 'max_abs_logit_difference': (a-b).abs().max().item(),
            'allclose_1e4': torch.allclose(a, b, atol=1e-4, rtol=1e-4),
            'packed_sequential_argmax_disagreements': (a.argmax(-1) != b.argmax(-1)).sum().item(),
            'clean_history_code_CE': F.cross_entropy(a.reshape(-1, 8192), gold.reshape(-1)).item(),
            'generated_history_code_CE': F.cross_entropy(rollout.reshape(-1, 8192), gold.reshape(-1)).item(),
            'clean_history_correct_codes': (a.argmax(-1) == gold).sum().item(),
            'free_correct_codes': (raw['free_codes'][:, start:end] == gold).sum().item()})
        start = end
    return {'levels': rows, 'all_levels_aligned_1e4': all(r['allclose_1e4'] for r in rows),
        'root_clean_generated_max_abs': (raw['packed'][:, :1] - raw['generated_history_logits'][:, :1]).abs().max().item(),
        'source_swap_root_max_abs': (raw['packed'][:, :1] - raw['swapped_source_logits'][:, :1]).abs().max().item(),
        'source_swap_argmax_changes': (raw['packed'].argmax(-1) != raw['swapped_source_logits'].argmax(-1)).sum().item()}


def main():
    run.checkpoint.configure()
    if DEST.exists() or REPORT.exists() or torch.cuda.is_initialized():
        raise ValueError('fresh CPU replay and report required')
    receipt = run.q.read(run.DEST / 'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed diagnostic required')
    producer, result = run.q.read(run.DEST / 'producer.json'), run.q.read(run.DEST / 'result.json')
    run.native.vendor.verify(producer)
    selected = run.checkpoint.selected
    cases = run.q.read(selected.training.DATA / 'cases.json')
    manifest = run.q.read(selected.training.DATA / 'manifest.json')
    longest = min(cases, key=lambda r: (-manifest[r['id']]['source_tokens'], r['id']))
    ids = [cases[0]['id'], longest['id']]
    if ids != producer['config']['TRAIN_case_ids'] or len(set(ids)) != 2:
        raise ValueError('unchanged first and longest TRAIN cases required')
    expected_codes = {r['id']: r['codes'] for r in run.q.read(selected.training.training.CACHE / 'outputs.json')}
    path = run.DEST / 'logits.pt'
    if run.file_hash(path) != result['logits_sha256']:
        raise ValueError('raw logit checksum changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if set(saved) != set(ids):
        raise ValueError('exact original two TRAIN cases required')
    reports, totals = {}, {}
    for case_id in ids:
        raw = saved[case_id]
        if raw['oracle_codes'].tolist() != [expected_codes[case_id]]:
            raise ValueError('original TRAIN target codes differ')
        report = replay(raw)
        if report != result['reports'][case_id]:
            raise ValueError('raw-logit metric replay differs')
        reports[case_id] = report
        totals[case_id] = {'gold_history_correct_codes': sum(r['clean_history_correct_codes'] for r in report['levels']),
            'free_generated_correct_codes': sum(r['free_correct_codes'] for r in report['levels']),
            'total_codes': 641, 'generated_history_argmax_disagreements_with_free_codes':
                (raw['generated_history_logits'].argmax(-1) != raw['free_codes']).sum().item()}
    if (not result['model_unchanged'] or result['peak_allocated_bytes'] > 2_000_000_000
            or result['source_only_task_accuracy'] is not None or result['final_test_opened']
            or not result['no_training_or_semantic_judgments']):
        raise ValueError('frozen training-only diagnostic boundaries differ')
    classification = ('two_TRAIN_packed_sequential_paths_aligned_at_identical_histories'
        if all(v['all_levels_aligned_1e4'] for v in reports.values()) else 'two_TRAIN_history_path_difference_detected')
    artifacts = {str(p): run.file_hash(p) for p in [Path(__file__), run.DEST / 'producer.json',
        run.DEST / 'result.json', run.DEST / 'actual_tool_receipt.json', path]}
    campaign = {'hypothesis_id': run.HYPOTHESIS, 'classification': classification,
        'reports': reports, 'TRAIN_code_totals': totals, 'cases': 2,
        'source_only_task_accuracy': None, 'student_trained': False, 'final_test_opened': False,
        'semantic_judgments': False, 'speed_tested': False, 'confirmed': False,
        'not_a_generalization_or_semantic_accuracy_measure': True, 'artifacts': artifacts}
    DEST.mkdir()
    run.write_json(DEST / 'campaign_results.json', campaign)
    table = ['| TRAIN case | Gold-history codes correct | Free-generated codes correct |',
        '| --- | ---: | ---: |']
    for key, value in totals.items():
        table.append(f"| {key[:12]} | {value['gold_history_correct_codes']}/641 | {value['free_generated_correct_codes']}/641 |")
    REPORT.write_text('# Native NCM packed and sequential history diagnostic\n\n'
        f'Hypothesis `{run.HYPOTHESIS}`. Classification **{classification}**.\n\n'
        'The old native655 checkpoint was frozen. The first and longest original TRAIN cases '
        'were selected before execution; source-only DEV and sealed final sources were not used. '
        'With identical oracle histories, packed and sequential inference agree within1e-4 '
        'at every level and have zero argmax disagreements across both641-code sequences. '
        'The largest absolute logit difference is '
        f"{max(r['max_abs_logit_difference'] for v in reports.values() for r in v['levels']):.9g}.\n\n"
        + '\n'.join(table) + '\n\n'
        'Generated histories increase later-level code loss. Swapping the two source memories '
        'changes452and504packed code choices, respectively, so this trained head is sensitive '
        'to source conditioning on these probes. This supports investigating accumulated history '
        'errors; it does not establish their share of held-out semantic failures or prove source '
        'understanding. Two TRAIN cases cannot rule out all implementation bugs. '
        'These code-match counts are not whole-summary accuracy. The independent CPU audit '
        'replayed every metric from the retained raw logits and original TRAIN targets.\n')
    print(json.dumps({k: v for k, v in campaign.items() if k not in ('reports', 'artifacts')}), flush=True)


def register():
    campaign = DEST / 'campaign_results.json'
    if run.q.read(DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual independent CPU replay required before registration')
    result = run.q.read(campaign)
    register_records(campaign, REPORT, DEST, {run.HYPOTHESIS: {
        'name': 'Two TRAIN cases: native packed versus sequential history paths',
        'description': 'Frozen old native655, identical oracle histories, all12levels and641codes; retained raw logits independently replayed on CPU; compare clean and generated histories plus source swap. No source-only semantic accuracy or generalization claim.',
        'classification': result['classification'], 'diagnostic_only': True,
        'student_task_accuracy_measured': False, 'cases': 2}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('audit', 'register'))
    args = parser.parse_args()
    {'audit': main, 'register': register}[args.stage]()
