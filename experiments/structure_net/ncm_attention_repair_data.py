"""Prepare frozen-code targets and real NCM errors for a larger repair run."""
import argparse
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_core_cached_target_admission as core
from experiments.structure_net import ncm_engineering_summary_inventory_v2 as engineering
from experiments.structure_net import ncm_engineering_source_metadata as metadata
from experiments.structure_net import ncm_engineering_source_features as cache
from experiments.structure_net import ncm_engineering_target_codes as targets
from experiments.structure_net import ncm_level_conditioning_checkpoint as checkpoint
from experiments.structure_net import ncm_lm1b_native_development as rendering
from experiments.structure_net import ncm_streaming_source_features as streaming
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

native, q, ROOT = core.native, checkpoint.q, core.ROOT
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-attention-repair-data-plan.md')
TEST = Path('tests/structure_net/test_ncm_attention_repair_data.py')
AUDIT = ROOT / '20261025_engineering_target_codes_replay_v1'
CORE_CODES = ROOT / '20261025_core4949_TRAIN_target_codes_v1'
PREDICTIONS = {kind: ROOT / ('20261025_' + kind + '_actual_NCM_errors_v1') for kind in ('core', 'engineering')}


def completed(root):
    if q.read(root / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual independent tool completion required: ' + str(root))
    native.vendor.verify(q.read(root / 'producer.json'))


def corpus(kind, need_features=False):
    if kind == 'core':
        completed(core.DEST)
        state = q.read(core.DEST / 'result.json')
        for path, digest in state['artifacts'].items():
            if file_hash(path) != digest:
                raise ValueError('replayed core data changed')
        rows = q.read(core.DEST / 'cases.json')
        index = q.read(core.DEST / 'source_index.json')
        manifest = q.read(core.DEST / 'feature_manifest.json')
    elif kind == 'engineering':
        completed(engineering.DEST)
        if file_hash(engineering.DEST / 'cases.json') != q.read(engineering.DEST / 'result.json')['cases_sha256']:
            raise ValueError('unchanged engineering source/target data required')
        rows = q.read(engineering.DEST / 'cases.json')['train']
        completed(metadata.DEST)
        if file_hash(metadata.DEST / 'index.json') != q.read(metadata.DEST / 'result.json')['index_sha256']:
            raise ValueError('source-only engineering geometry changed')
        index = [r for r in q.read(metadata.DEST / 'index.json') if r['split'] == 'train']
        manifest = None
        if need_features:
            if q.read(cache.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
                raise ValueError('actually completed full engineering feature workflow required')
            native.vendor.terminal(cache.CACHE)
            whole = q.read(cache.CACHE / 'manifest.json')
            if file_hash(cache.CACHE / 'manifest.json') != q.read(cache.CACHE / 'result.json')['manifest_sha256']:
                raise ValueError('complete engineering source features changed')
            manifest = {r['id']: whole[r['id']] for r in rows}
    else:
        raise ValueError('fixed core or engineering TRAIN partition required')
    expected = 4949 if kind == 'core' else 732
    if (len(rows) != expected or len({r['id'] for r in rows}) != expected
            or {r['id'] for r in rows} != {r['id'] for r in index}
            or manifest is not None and set(manifest) != {r['id'] for r in rows}):
        raise ValueError('complete unchanged TRAIN partition required')
    return rows, index, manifest


def producing(root, stage, extra, config):
    files = [Path(__file__), PLAN, TEST, Path(core.__file__), Path(engineering.__file__),
        Path(metadata.__file__), Path(cache.__file__), Path(targets.__file__),
        Path(checkpoint.__file__), Path(rendering.__file__), Path(streaming.__file__),
        Path(checkpoint.training.lifecycle.head.__file__), Path(checkpoint.training.lifecycle.head.concepts.__file__)]
    root.mkdir()
    (root / 'producer_source').mkdir()
    for i, path in enumerate(files):
        (root / 'producer_source' / f'{i}_{path.name}').write_bytes(path.read_bytes())
    p = {'schema': 'nal.ncm-actual-error-training-data.v1', 'stage': stage,
        'source_hashes': {str(p): file_hash(p) for p in files},
        'input_hashes': {str(p): file_hash(p) for p in extra},
        'source_scope': 'separate historical core TRAIN and new engineering TRAIN; reserved evaluation excluded',
        'config': {**config, 'physical_GPU': 2, 'precision': 'FP32', 'TF32': False,
            'parent_loaded': False, 'all_parent_MLPs_frozen': True, 'model_training': False,
            'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False}}
    write_json(root / 'producer.json', p)
    return p


def frame(row, tokenizer):
    lexical = tokenizer.encode(row['target_text'], add_special_tokens=False)
    if (text_hash(row['target_text']) != row['target_sha256'] or not 0 < len(lexical) <= 126
            or set(lexical) & set(tokenizer.all_special_ids)
            or row['native_word_ids'] != [101, *lexical, 101] + [0] * (126-len(lexical))):
        raise ValueError('unaltered complete native target frame required')
    return row['native_word_ids']


def target_codes(kind):
    native.vendor.corpus.quality.dependencies()
    native.require_gpu()
    rows, _, _ = corpus(kind)
    audit = kind == 'engineering'
    root = AUDIT if audit else CORE_CODES
    if root.exists():
        raise ValueError('fresh target-code preparation required')
    existing = None
    extra = [engineering.DEST / n for n in ('cases.json', 'result.json', 'actual_tool_receipt.json')] if audit else [core.DEST / n for n in ('cases.json', 'result.json', 'actual_tool_receipt.json')]
    if audit:
        completed(targets.DEST)
        value = q.read(targets.DEST / 'result.json')
        if file_hash(targets.DEST / 'outputs.json') != value['outputs_sha256']:
            raise ValueError('original target-code output changed')
        existing = q.read(targets.DEST / 'outputs.json')
        if [r['id'] for r in existing] != [r['id'] for r in rows]:
            raise ValueError('every original engineering TRAIN code required')
        extra += [targets.DEST / n for n in ('outputs.json', 'result.json', 'actual_tool_receipt.json')]
    extra += [native.RELEASE / 'files/lm1b/tokenizer.pt', native.RELEASE / 'files/lm1b/small/config.yaml']
    p = producing(root, 'independent732_target_code_replay' if audit else 'all4949_TRAIN_target_codes', extra,
        {'cases': len(rows), 'kind': kind, 'heldout_targets_encoded': False, 'native_encoder_and_VQ_frozen': True})
    codec, tokenizer = native.load_codec(), native.tokenizer()
    before = native.vendor.module_hash(codec)
    first = rows[0]['id']
    longest = min(rows, key=lambda r: (-r['native_lexical_tokens'], r['id']))['id']
    outputs = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i, row in enumerate(rows):
            words = torch.tensor([frame(row, tokenizer)], dtype=torch.long, device='cuda')
            codes = codec.encode(words)
            if codes.shape != (1, 641) or codes.min() < 0 or codes.max() >= 8192:
                raise ValueError('complete in-range native hierarchy required')
            if row['id'] in (first, longest) and not torch.equal(codes, codec.encode(words)):
                raise ValueError('real complete target encoding must repeat')
            value = {'id': row['id'], 'source_sha256': row['source_sha256'], 'target_sha256': row['target_sha256'],
                'source_component': row['source_component'], 'native_word_ids': row['native_word_ids'], 'codes': codes.cpu().tolist()[0]}
            if audit and value != existing[i]:
                raise ValueError('independent raw target/code replay differs')
            outputs.append(value)
            if (i+1) % 128 == 0 or i+1 == len(rows):
                if not audit: write_json(root / 'outputs.json', outputs)
                print(json.dumps({'target_codes_replayed_or_encoded': i+1, 'cases': len(rows), 'kind': kind}), flush=True)
    if native.vendor.module_hash(codec) != before or any(v.grad is not None for v in codec.parameters()):
        raise ValueError('native target geometry changed')
    peak = torch.cuda.max_memory_allocated()
    if peak > 3_000_000_000: raise ValueError('target stage exceeds3GB allocation')
    native.vendor.verify(p)
    write_json(root / 'result.json', {'status': 'all732_engineering_codes_independently_replayed' if audit else 'all4949_core_TRAIN_codes_encoded',
        'cases': len(rows), 'codec_sha256': before, 'peak_allocated_bytes': peak,
        'outputs_sha256': file_hash(targets.DEST / 'outputs.json') if audit else file_hash(root / 'outputs.json'),
        'heldout_targets_encoded': False, 'task_accuracy': None, 'model_training': False})


@torch.no_grad()
def generate(model, features, tokenizer):
    """No target fields or oracle history exist in this prediction API."""
    source = features.unsqueeze(0).cuda()
    valid = torch.ones(source.shape[:2], dtype=torch.bool, device='cuda')
    value = model.generate(source, valid)
    if value['codes'].shape != (1, 641) or value['raw_word_ids'].shape != (1, 128):
        raise ValueError('complete free native generation required')
    codes, words = value['codes'].cpu().tolist()[0], value['raw_word_ids'].cpu().tolist()[0]
    if any(type(v) is not int or not 0 <= v < 8192 for v in codes):
        raise ValueError('full native concept vocabulary required')
    return {'codes': codes, 'raw_word_ids': words, **rendering.decode(words, tokenizer)}


def baseline(kind):
    native.vendor.corpus.quality.dependencies()
    checkpoint.training.require_gpu()
    rows, index, manifest = corpus(kind, need_features=True)
    root = PREDICTIONS[kind]
    if root.exists(): raise ValueError('fresh actual-error generation root required')
    data_root = core.DEST if kind == 'core' else engineering.DEST
    extra = [data_root / n for n in ('cases.json', 'result.json', 'actual_tool_receipt.json')]
    extra += [checkpoint.training.TRAIN / n for n in ('model.pt', 'result.json', 'diagnostic_outputs.json')]
    extra += [core.DEST / n for n in ('source_index.json', 'feature_manifest.json')] if kind == 'core' else [metadata.DEST / 'index.json', cache.CACHE / 'manifest.json']
    p = producing(root, 'complete_source_only_frozen_generator_errors_' + kind, extra,
        {'cases': len(rows), 'kind': kind, 'checkpoint_sha256': '3911eab74db6ad13104f87765196e4212893a3e179ba0283e103e65e78273bec',
            'target_fields_in_prediction': False, 'oracle_codes_in_prediction': False,
            'host_feature_cache_budget_bytes': 2*1024**3, 'raw_word_positions': 128, 'code_positions': 641})
    (root / 'cases').mkdir()
    model, state = checkpoint.load_selected()
    before = native.vendor.module_hash(model)
    tokenizer = native.tokenizer()
    # Exact source-only replay of an existing original TRAIN output.
    references = q.read(checkpoint.training.TRAIN / 'diagnostic_outputs.json')
    selected = [references[0], min(references, key=lambda r: (-r['source_positions'], r['id']))]
    ref_rows = q.read(checkpoint.training.DATA / 'cases.json')
    reference_manifest = q.read(checkpoint.training.DATA / 'manifest.json')
    parity = []
    for reference in selected:
        ref = next(r for r in ref_rows if r['id'] == reference['id'])
        source = {k: ref[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256', 'split')}
        check = rendering.predict(model, source, reference_manifest, tokenizer)
        if check != reference: raise ValueError('frozen selected model differs from original TRAIN output')
        parity.append({'id': reference['id'], 'complete_output_exact': True})
    write_json(root / 'reference_parity.json', parity)
    loader = streaming.FeatureCache(index, manifest, max_bytes=2*1024**3)
    outputs = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for i, row in enumerate(rows, 1):
            feature = loader.get(row)
            value = generate(model, feature, tokenizer)
            if i == 1 and value != generate(model, feature, tokenizer):
                raise ValueError('real new source generation does not repeat exactly')
            value.update(id=row['id'], source_sha256=row['source_sha256'], source_component=row['source_component'],
                alignment_sha256=row['alignment_sha256'] if 'alignment_sha256' in row else q.read(next(x['path'] for x in index if x['id']==row['id']))['alignment_sha256'],
                source_positions=len(feature), feature_payload_sha256=manifest[row['id']]['payload_sha256'])
            write_json(root / 'cases' / (row['id']+'.json'), value)
            outputs.append(value)
            if i % 16 == 0 or i == len(rows):
                write_json(root / 'progress.json', {'completed': i, 'cases': len(rows), 'kind': kind})
                write_json(root / 'outputs.json', outputs)
                print(json.dumps({'actual_NCM_errors_prepared': i, 'cases': len(rows), 'kind': kind}), flush=True)
        if native.vendor.module_hash(model) != before or any(v.grad is not None for v in model.parameters()):
            raise ValueError('frozen baseline changed')
        peak = torch.cuda.max_memory_allocated()
        if peak > 3_000_000_000: raise ValueError('baseline generation exceeds3GB allocation')
    finally:
        model.close()
    native.vendor.verify(p)
    write_json(root / 'result.json', {'status': 'all_actual_frozen_generator_errors_preserved_' + kind,
        'cases': len(outputs), 'checkpoint_sha256': state['checkpoint_sha256'], 'model_sha256': before,
        'outputs_sha256': file_hash(root / 'outputs.json'), 'peak_allocated_bytes': peak,
        'source_only': True, 'oracle_codes_in_prediction': False, 'model_training': False,
        'feature_cache': loader.metrics(), 'task_accuracy': None, 'final_test_opened': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('audit-engineering-codes', 'core-codes', 'predict-core', 'predict-engineering'))
    stage = parser.parse_args().stage
    {'audit-engineering-codes': lambda: target_codes('engineering'), 'core-codes': lambda: target_codes('core'),
        'predict-core': lambda: baseline('core'), 'predict-engineering': lambda: baseline('engineering')}[stage]()
