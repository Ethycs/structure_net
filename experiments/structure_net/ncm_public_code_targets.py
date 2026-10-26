"""Prepare every admitted public target with the unchanged native FP32 codec."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch

from experiments.structure_net import ncm_public_training_catalog as catalog
from experiments.structure_net import ncm_indexed_jsonl as indexed
from experiments.structure_net import ncm_public_xsum_codec_recovery as codec_source

data = catalog.data
ROOTS = {s: data.ROOT/f'20261026_public_code_targets_{s}_v1' for s in ('pilot', 'full')}
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-code-targets-plan.md')
TEST = Path('tests/structure_net/test_ncm_public_code_targets.py')
HYPOTHESIS = 'ncm-public-code-targets-v1'
BOUNDARIES = {'model_training': False, 'new_judge_requests': 0, 'task_accuracy': None,
              'final_test_opened': False, 'targets_admitted_for_training': False}


def pilot_positions(entries):
    if len(entries) < 16 or len({r['id'] for r in entries}) != len(entries):
        raise ValueError('complete unique public catalog required')
    selected = [0]
    for length in (1, 16, 32, 48, 64, 96, 126):
        selected.append(min((i for i in range(len(entries)) if i not in selected),
            key=lambda i: (abs(entries[i]['native_lexical_tokens']-length), entries[i]['id'])))
    selected += sorted((i for i in range(len(entries)) if i not in selected), key=lambda i: entries[i]['id'])[:16-len(selected)]
    return selected


def bind(row, codes):
    if (row['split'] != 'TRAIN' or len(codes) != 641
            or any(type(c) is not int or not 0 <= c < 8192 for c in codes)):
        raise ValueError('complete native TRAIN hierarchy required')
    return {**{k: row[k] for k in ('id', 'source_sha256', 'target_sha256', 'native_frame_sha256')},
            'codec_sha256': codec_source.CODEC_SHA, 'codes': codes}


def ready():
    from experiments.structure_net import ncm_pixi
    training, environment = ncm_pixi.configure()
    admitted = data.completed(catalog.AUDIT)
    if not admitted['training_data_admitted'] or admitted['counts']['retained_pairs'] != 162353:
        raise ValueError('complete admitted public reference catalog required')
    state = data.completed(catalog.DEST)
    with (catalog.DEST/'training_index.jsonl').open() as stream:
        entries = list(map(json.loads, stream))
    dataset = indexed.IndexedJSONL(catalog.DEST/'training.jsonl', entries,
                                  state['artifacts'][str(catalog.DEST/'training.jsonl')])
    return training, environment, dataset


def freeze(root, stage, environment, training, positions):
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    paths = [catalog.DEST/name for name in ('result.json', 'actual_tool_receipt.json', 'training.jsonl', 'training_index.jsonl')]
    paths += [catalog.AUDIT/name for name in ('result.json', 'actual_tool_receipt.json')]
    paths += [codec_source.BROAD/name for name in ('model.pt', 'result.json', 'actual_tool_receipt.json')]
    paths += [data.ROOT/'20261025_broad_generator_prepare_v1'/name for name in ('cases.json', 'selection.json')]
    paths += [training.native.RELEASE/'files/lm1b/tokenizer.pt', training.native.RELEASE/'files/lm1b/small/config.yaml',
              Path('pyproject.toml'), Path('pixi.lock')]
    if stage == 'full':
        paths += [ROOTS['pilot']/name for name in ('result.json', 'actual_tool_receipt.json', 'codes.jsonl')]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': 'nal.ncm-public-code-targets.v1', 'hypothesis_id': HYPOTHESIS, 'stage': stage,
        'environment': environment, 'pilot_positions': positions,
        'config': {'codec_sha256': codec_source.CODEC_SHA, 'physical_GPU': 1, 'precision': 'FP32',
                   'TF32': False, 'batch_size': 1, 'repeated_encode_every_record': True, **BOUNDARIES},
        'input_hashes': {str(p): data.sha(p) for p in paths}, 'source_hashes': {str(p): data.sha(p) for p in sorted(files)}}
    data.write(root/'producer.json', proof)
    return proof


def run(stage):
    root = ROOTS[stage]
    if root.exists():
        raise ValueError('fresh append-only code target root required')
    training, environment, dataset = ready()
    from experiments.structure_net.ncm_cpu_adamw import require_rtx2060
    require_rtx2060()
    if data.sha(codec_source.BROAD/'model.pt') != codec_source.CHECKPOINT_SHA:
        raise ValueError('exact selected BROAD checkpoint required')
    pilot = {}
    if stage == 'full':
        state = data.completed(ROOTS['pilot'])
        if state['cases'] != 16 or not state['all_repeated_encodes_exact']:
            raise ValueError('completed 16-case public codec pilot required')
        with (ROOTS['pilot']/'codes.jsonl').open() as stream:
            pilot = {r['id']: r for r in map(json.loads, stream)}
    positions = pilot_positions(dataset.entries)
    proof = freeze(root, stage, environment, training, positions)
    codec = training.native.load_codec().float().eval().requires_grad_(False)
    checkpoint = torch.load(codec_source.BROAD/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    codec.load_state_dict(codec_source.codec_state(checkpoint['model'], codec.state_dict()), strict=True)
    if training.native.vendor.module_hash(codec) != codec_source.CODEC_SHA:
        raise ValueError('unchanged repaired native codec required')
    del checkpoint
    control_root = data.ROOT/'20261025_broad_generator_prepare_v1'
    controls = {r['id']: r for r in data.read(control_root/'cases.json')}
    identities = data.read(control_root/'selection.json')['diagnostic_ids']
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for identity in identities:
            row = controls[identity]
            codes = codec.encode(torch.tensor([row['native_word_ids']], device='cuda', dtype=torch.long))[0].cpu().tolist()
            if codes != row['oracle_codes']:
                raise ValueError('original TRAIN codec control differs')
        if len(identities) != 16:
            raise ValueError('all 16 original TRAIN codec controls required')
        data.write(root/'original_TRAIN_controls.json', {'ids': identities, 'all_code_hierarchies_exact': True})
        selected = positions if stage == 'pilot' else range(len(dataset))
        count = repeated = 0
        started = time.monotonic()
        with (root/'codes.jsonl').open('xb') as stream, (root/'index.jsonl').open('x') as offsets:
            for position in selected:
                row = dataset[position]
                gold = torch.tensor([row['native_word_ids']], device='cuda', dtype=torch.long)
                codes = codec.encode(gold)
                if not torch.equal(codes, codec.encode(gold)):
                    raise ValueError('repeated native codec encode differs')
                value = bind(row, codes[0].cpu().tolist())
                if row['id'] in pilot:
                    if value != pilot[row['id']]:
                        raise ValueError('pilot/full public code targets differ')
                    repeated += 1
                indexed.write_record(stream, offsets, value, catalog_position=position,
                    target_sha256=row['target_sha256'], native_frame_sha256=row['native_frame_sha256'])
                count += 1
                if stage == 'pilot' or count % 512 == 0:
                    print(json.dumps({'prepared': count, 'total': len(selected), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    dataset.close()
    if count != len(selected) or (stage == 'full' and repeated != 16):
        raise ValueError('all selected code targets and public controls required')
    if training.native.vendor.module_hash(codec) != codec_source.CODEC_SHA or any(p.grad is not None for p in codec.parameters()):
        raise ValueError('frozen codec changed or received gradients')
    if torch.cuda.max_memory_allocated() > 4_900_000_000:
        raise ValueError('codec allocation budget exceeded')
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    value = {'status': 'public_code_targets_'+stage+'_completed', 'hypothesis_id': HYPOTHESIS,
        'stage': stage, 'cases': count, 'pilot_repetitions_exact': repeated,
        'all_repeated_encodes_exact': True, 'original_TRAIN_controls_exact': 16,
        'codec_sha256': codec_source.CODEC_SHA, 'codec_unchanged': True,
        'elapsed_seconds': time.monotonic()-started, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts': {str(p): data.sha(p) for p in root.iterdir() if p.is_file()}, **BOUNDARIES}
    data.write(root/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


def audit(stage):
    root = ROOTS[stage]
    if (root/'audit_result.json').exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('fresh CPU-only code-target audit required')
    saved = data.completed(root)
    proof = data.read(root/'producer.json')
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    _, _, dataset = ready()
    positions = pilot_positions(dataset.entries) if stage == 'pilot' else range(len(dataset))
    with (root/'index.jsonl').open() as stream:
        entries = list(map(json.loads, stream))
    codes = indexed.IndexedJSONL(root/'codes.jsonl', entries, saved['artifacts'][str(root/'codes.jsonl')])
    if len(codes) != len(positions) or len(codes) != saved['cases']:
        raise ValueError('complete code target population required')
    for i, position in enumerate(positions):
        value, source = codes[i], dataset[position]
        if (value != bind(source, value['codes']) or entries[i]['catalog_position'] != position
                or entries[i]['target_sha256'] != source['target_sha256']
                or entries[i]['native_frame_sha256'] != source['native_frame_sha256']):
            raise ValueError('complete original target/code binding differs')
    codes.close(); dataset.close()
    result = {'status': 'all_public_'+stage+'_target_bindings_replayed', 'cases': len(positions),
        'all_bindings_replayed': True, 'independent_GPU_reencoding': False, **BOUNDARIES,
        'targets_admitted_for_training': stage == 'full',
        'artifacts': {str(root/'result.json'): data.sha(root/'result.json')}}
    data.write(root/'audit_result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('encode', 'audit'))
    parser.add_argument('stage', choices=tuple(ROOTS))
    args = parser.parse_args()
    existed = ROOTS[args.stage].exists()
    try:
        (run if args.command == 'encode' else audit)(args.stage)
    except Exception as error:
        if args.command == 'encode' and not existed and ROOTS[args.stage].exists():
            data.write(ROOTS[args.stage]/'failure.json', {'exception': type(error).__name__, 'message': str(error), **BOUNDARIES})
        raise
