"""CPU-only source tokenization/alignment for all10000 reserved TRAIN cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_additional_training_sources as sources
from experiments.structure_net import ncm_pinned_date_development as pinned
from experiments.structure_net import ncm_lm1b_grounded_training as old_training
from experiments.structure_net import ncm_lm1b_codec as native
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

development, base, q, ROOT = pinned.development, pinned.base, pinned.q, sources.ROOT
HYPOTHESIS = 'ncm-additional10000-source-tokenization-v1'
MODULE = 'experiments.structure_net.ncm_additional_source_metadata'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-additional-source-metadata-plan.md')
TEST = Path('tests/structure_net/test_ncm_additional_source_metadata.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_additional_source_metadata.py')
WORKFLOW = ROOT / '20261024_additional10000_source_metadata_workflow_v1'
PILOT = ROOT / '20261024_additional10000_source_metadata_pilot_v1'
DATA = ROOT / '20261024_additional10000_source_metadata_v1'
SOURCE_SHA = '3e07a05b0cc1f1409fade64335dc995f83edfead733a1e6f9922409385d6f29d'
FORBIDDEN = development.FORBIDDEN | {'native_word_ids', 'target_ids', 'codes', 'teacher_latent', 'text'}
GEOMETRY = ('context_ids', 'source_ids', 'alignment', 'alignment_sha256', 'native_prompt_sha256', 'source_frame')


def require_sources():
    if (q.read(sources.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0
            or file_hash(sources.DEST / 'cases.json') != SOURCE_SHA):
        raise ValueError('actual completed immutable10000 reserved TRAIN sources required')
    sources.reservation.verify(q.read(sources.DEST / 'producer.json'))
    sources.metadata()
    rows = q.read(sources.DEST / 'cases.json')
    if (len(rows) != 10000 or len({r['id'] for r in rows}) != 10000
            or len({r['source_component'] for r in rows}) != 10000
            or any(r['split'] != 'additional_train' or r['source_truncated'] or FORBIDDEN & r.keys() for r in rows)):
        raise ValueError('all10000complete distinct source-only TRAIN cases required')
    return rows


def tokenizers():
    native_tokenizer, gpt, storage = development.tokenizers()
    return pinned.PinnedDateTokenizer(native_tokenizer, date='23 Oct 2026'), gpt, storage


def prepare_one(row, native_tokenizer, gpt, storage):
    if FORBIDDEN & row.keys():
        raise ValueError('source-only preparation rejects every target/code field before tokenizer access')
    return development.prepare_row(row, row['split'], native_tokenizer, gpt, storage)


def parity(native_tokenizer, gpt, storage):
    rows = q.read(old_training.DATA / 'cases.json')
    manifest = q.read(old_training.DATA / 'manifest.json')
    probes = [rows[0], max(rows, key=lambda r: manifest[r['id']]['source_tokens'])]
    fields = ('id', 'source_id', 'group_id', 'source_sha256', 'source', 'spans', 'split', 'source_cl100k_tokens')
    proof = []
    for row in probes:
        prepared = prepare_one({k: row[k] for k in fields}, native_tokenizer, gpt, storage)
        if any(prepared[k] != row[k] for k in GEOMETRY):
            raise ValueError('original first/longest TRAIN tokenizer/date/alignment geometry changed')
        proof.append({'id': row['id'], 'native_prompt_sha256': row['native_prompt_sha256'],
            'alignment_sha256': row['alignment_sha256'], 'source_positions': len(row['source_ids']), 'all_geometry_exact': True})
    return proof


def pilot_ids(rows):
    selected = []
    for band in range(4):
        selected.append(min((r for r in rows if r['bin'] == band),
            key=lambda r: (text_hash(HYPOTHESIS + ':pilot:' + r['id']), r['id']))['id'])
    longest = min(rows, key=lambda r: (-r['source_cl100k_tokens'], r['id']))
    if longest['bin'] != 4 or longest['id'] in selected:
        raise ValueError('complete five-band metadata pilot requires distinct longest source')
    return [*selected, longest['id']]


def producing(root, stage, inputs=()):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(sources.__file__), Path(pinned.__file__),
        Path(development.__file__), Path(base.__file__)]
    assets = [p for parent in (base.MODEL_PATH, base.GPT2) for p in parent.iterdir()
        if p.is_file() and p.name in {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
            'chat_template.jinja', 'vocab.json', 'merges.txt', 'added_tokens.json'}]
    common = [sources.DEST / n for n in ('producer.json', 'cases.json', 'result.json')]
    common += [sources.WORKFLOW / 'tool_terminal_receipt.json', old_training.DATA / 'cases.json',
        old_training.DATA / 'manifest.json']
    p = native.vendor.freeze(root, stage, [*inputs, *common, *files, *assets], {'hypothesis_id': HYPOTHESIS,
        'CPU_only': True, 'source_only': True, 'target_labels_used': False, 'body_loaded': False,
        'new_teacher_label_outputs_read': False, 'original_reference_target_columns_ignored': True,
        'parent_body_hash_for_future_features': base.EXPECTED_BODY, 'parent_layers': 28,
        'original_MLPs_updated': False, 'date_string': '23 Oct 2026', 'native_system': base.SYSTEM,
        'source_tokenizer': 'unchanged GPT2byteBPE', 'alignment': 'original exact character-overlap weights',
        'full_source_retained': True, 'maximum_source_frame': 8192, 'source_positions_or_inputs_clipped': False,
        'feature_values_computed': False, 'feature_storage_precision_selected': False,
        'no_new_semantic_judgments': True, 'final_sources_opened_or_hashed': False, 'task_accuracy': None})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'source_metadata_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def prepare(pilot=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('source metadata worker must remain CPU-only')
    rows = require_sources()
    ids = pilot_ids(rows)
    native_tokenizer, gpt, storage = tokenizers()
    proof = parity(native_tokenizer, gpt, storage)
    retained = {}
    extra = []
    if not pilot:
        native.vendor.terminal(PILOT)
        if q.read(PILOT / 'result.json')['status'] != 'five_band_source_metadata_pilot_passed':
            raise ValueError('actual five-band tokenizer/alignment pilot required')
        retained = {r['id']: r for r in q.read(PILOT / 'index.json')}
        extra = [PILOT / n for n in ('index.json', 'parity.json', 'result.json', 'worker_terminal_receipt.json')]
    chosen = [r for r in rows if r['id'] in ids] if pilot else rows
    root = PILOT if pilot else DATA
    p = producing(root, 'five_band_CPU_tokenizer_pilot' if pilot else 'all10000_CPU_source_tokenization_alignment', extra)
    (root / 'cases').mkdir()
    index = []
    for i, row in enumerate(chosen, 1):
        if row['id'] in retained:
            entry = retained[row['id']]
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('retained metadata pilot changed')
            index.append(entry)
            continue
        prepared = prepare_one(row, native_tokenizer, gpt, storage)
        path = root / 'cases' / (row['id'] + '.json')
        write_json(path, prepared)
        index.append({'id': row['id'], 'source_id': row['source_id'], 'group_id': row['group_id'],
            'source_component': row['source_component'], 'source_sha256': row['source_sha256'], 'bin': row['bin'],
            'path': str(path), 'sha256': file_hash(path), 'source_positions': len(prepared['source_ids']),
            'native_prompt_positions': len(prepared['context_ids']), 'source_frame': prepared['source_frame'],
            'alignment_sha256': prepared['alignment_sha256'], 'native_prompt_sha256': prepared['native_prompt_sha256']})
        if pilot or i % 128 == 0 or i == len(chosen):
            write_json(root / 'progress.json', {'prepared': i, 'cases': len(chosen), 'CPU_only': True})
            print(json.dumps({'prepared_source_metadata': i, 'cases': len(chosen)}), flush=True)
    write_json(root / 'index.json', index)
    write_json(root / 'parity.json', proof)
    source_positions = sum(r['source_positions'] for r in index)
    result = {'status': 'five_band_source_metadata_pilot_passed' if pilot else 'all10000_source_metadata_prepared_no_features_or_labels',
        'cases': len(index), 'source_positions': source_positions, 'max_source_positions': max(r['source_positions'] for r in index),
        'max_native_prompt_positions': max(r['native_prompt_positions'] for r in index),
        'estimated_raw_FP32_feature_bytes': source_positions * 3072 * 4,
        'estimated_BF16_feature_bytes': source_positions * 3072 * 2,
        'index_sha256': file_hash(root / 'index.json'), 'CPU_only': True, 'feature_values_computed': False,
        'teacher_labels_used_for_preparation': False, 'task_accuracy': None, 'all_final_sources_closed': True}
    native.vendor.verify(p)
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)


def workflow():
    require_sources()
    if any(p.exists() for p in (WORKFLOW, PILOT, DATA)):
        raise ValueError('fresh append-only source metadata roots required')
    p = producing(WORKFLOW, 'supervise_CPU_only_additional_source_metadata')
    receipts = []
    for stage, root in (('pilot', PILOT), ('prepare', DATA)):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', MODULE, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'child_pid': child.pid, 'observer_pid': os.getpid(), 'observer': 'subprocess.wait', 'command': command}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('source metadata stage failed:' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'all10000_CPU_source_metadata_workflow_completed', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('pilot', 'prepare', 'workflow'))
    args = parser.parse_args()
    {'pilot': lambda: prepare(True), 'prepare': prepare, 'workflow': workflow}[args.stage]()
