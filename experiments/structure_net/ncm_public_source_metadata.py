"""Stream all admitted public sources through the original prompt/alignment path."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from experiments.structure_net import ncm_public_training_catalog as catalog
from experiments.structure_net import ncm_indexed_jsonl as indexed

data = catalog.data
HYPOTHESIS = 'ncm-public-source-metadata-v1'
ROOTS = {stage: data.ROOT/f'20261026_public_source_metadata_{stage}_v1' for stage in ('pilot', 'full')}
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-source-metadata-plan.md')
TESTS = [Path('tests/structure_net/test_ncm_indexed_jsonl.py'),
         Path('tests/structure_net/test_ncm_public_source_metadata.py')]
BOUNDARIES = {'model_training': False, 'body_loaded': False, 'features_computed': False,
    'target_fields_in_body_input': False, 'new_judge_requests': 0,
    'task_accuracy': None, 'final_test_opened': False, 'full_metadata_audited': False}


def pilot_positions(entries):
    if len(entries) < 9 or len({r['id'] for r in entries}) != len(entries):
        raise ValueError('complete unique admitted index required')
    selected = [0]
    for tokens in (192, 256, 384, 512, 768, 1024, 1536, 2048):
        selected.append(min((i for i in range(len(entries)) if i not in selected),
            key=lambda i: (abs(entries[i]['source_cl100k_tokens']-tokens), entries[i]['id'])))
    return selected


def prepare_source(pair, tokenizers, metadata):
    source = catalog.source_only(pair)
    prepared = metadata.prepare_one(source, *tokenizers)
    if (metadata.FORBIDDEN & prepared.keys() or prepared['source'] != pair['source']
            or prepared['source_sha256'] != pair['source_sha256']
            or prepared['source_component'] != pair['source_component'] or prepared['source_truncated']
            or not 1 <= len(prepared['source_ids']) <= 8192
            or len(prepared['alignment']) != len(prepared['source_ids'])):
        raise ValueError('complete unaltered source-only prompt/alignment required')
    return prepared


def ready():
    from experiments.structure_net import ncm_pixi
    from experiments.structure_net import ncm_additional_source_metadata as metadata
    training, environment = ncm_pixi.configure()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or training.torch.cuda.is_initialized():
        raise ValueError('CPU-only public source preparation required')
    admitted = data.completed(catalog.AUDIT)
    campaign = data.read(catalog.AUDIT/'campaign_results.json')
    data.verify(campaign['artifacts'])
    if (not admitted['training_data_admitted'] or admitted['admission_scope'] != 'public_reference_conditional_pretraining_only'
            or not campaign['all_remaining_pairs_retained'] or not campaign['all_byte_offsets_replayed']):
        raise ValueError('complete independently admitted public paired catalog required')
    with (catalog.DEST/'training_index.jsonl').open() as stream:
        entries = [json.loads(line) for line in stream]
    if len(entries) != campaign['counts']['retained_pairs']:
        raise ValueError('complete admitted index required')
    expected = campaign['artifacts'][str(catalog.DEST/'training.jsonl')]
    records = indexed.IndexedJSONL(catalog.DEST/'training.jsonl', entries, expected)
    return training, environment, metadata, records


def freeze(root, stage, environment, metadata, positions):
    sources = {Path(__file__).resolve(), PLAN.resolve(), *[p.resolve() for p in TESTS]}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()): sources.add(p)
    paths = [catalog.DEST/name for name in ('training.jsonl', 'training_index.jsonl', 'result.json', 'actual_tool_receipt.json')]
    paths += [catalog.AUDIT/name for name in ('campaign_results.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [metadata.old_training.DATA/name for name in ('cases.json', 'manifest.json')]
    paths += [Path('pyproject.toml'), Path('pixi.lock')]
    for directory in (metadata.base.MODEL_PATH, metadata.base.GPT2):
        paths += [p for p in directory.iterdir() if p.is_file() and p.name in
            {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'chat_template.jinja', 'vocab.json', 'merges.txt', 'added_tokens.json'}]
    if stage == 'full':
        paths += [ROOTS['pilot']/name for name in ('metadata.jsonl', 'index.jsonl', 'result.json', 'actual_tool_receipt.json')]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(sources)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    value = {'schema': 'nal.ncm-public-source-metadata.v1', 'hypothesis_id': HYPOTHESIS, 'stage': stage,
        'environment': environment, 'config': {'date_string': '23 Oct 2026', 'maximum_source_frame': 8192,
            'full_source_and_prompt_retained': True, 'indexed_JSONL': True, 'pilot_positions': positions,
            'future_original_body_sha256': metadata.base.EXPECTED_BODY, **BOUNDARIES},
        'source_hashes': {str(p): data.sha(p) for p in sorted(sources)},
        'input_hashes': {str(p): data.sha(p) for p in paths}}
    data.write(root/'producer.json', value)
    return value


def run(stage):
    root = ROOTS[stage]
    if root.exists(): raise ValueError('fresh append-only public metadata root required')
    training, environment, metadata, records = ready()
    positions = pilot_positions(records.entries)
    old = {}
    if stage == 'full':
        pilot = data.completed(ROOTS['pilot'])
        if not pilot['original_TRAIN_geometry_equal'] or pilot['cases'] != 9:
            raise ValueError('actual passed nine-source pilot required')
        with (ROOTS['pilot']/'metadata.jsonl').open() as stream:
            old = {r['id']: r for r in map(json.loads, stream)}
    proof = freeze(root, stage, environment, metadata, positions)
    tokens = metadata.tokenizers()
    parity = metadata.parity(*tokens)
    if len(parity) != 2 or not all(r['all_geometry_exact'] for r in parity):
        raise ValueError('both original TRAIN geometry controls must reproduce')
    data.write(root/'original_TRAIN_parity.json', parity)
    selected = positions if stage == 'pilot' else range(len(records))
    started = time.monotonic()
    count = source_positions = context_positions = max_source = max_context = repeated = 0
    with (root/'metadata.jsonl').open('xb') as out, (root/'index.jsonl').open('x') as idx:
        for position in selected:
            prepared = prepare_source(records[position], tokens, metadata)
            if prepared['id'] in old:
                if prepared != old[prepared['id']]: raise ValueError('pilot/full source metadata differs')
                repeated += 1
            n, c = len(prepared['source_ids']), len(prepared['context_ids'])
            indexed.write_record(out, idx, prepared, catalog_position=position,
                source_sha256=prepared['source_sha256'], source_component=prepared['source_component'],
                source_positions=n, native_prompt_positions=c, source_frame=prepared['source_frame'],
                alignment_sha256=prepared['alignment_sha256'], native_prompt_sha256=prepared['native_prompt_sha256'])
            count += 1; source_positions += n; context_positions += c
            max_source = max(max_source, n); max_context = max(max_context, c)
            if stage == 'pilot' or count % 1024 == 0:
                print(json.dumps({'stage': stage, 'prepared': count, 'total': len(selected),
                    'elapsed_seconds': time.monotonic()-started}), flush=True)
    records.close()
    if count != len(selected) or (stage == 'full' and repeated != 9):
        raise ValueError('all source positions and pilot repetitions required')
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    value = {'status': 'public_source_metadata_'+stage+'_completed_audit_pending', 'hypothesis_id': HYPOTHESIS,
        'cases': count, 'source_positions': source_positions, 'native_prompt_positions': context_positions,
        'max_source_positions': max_source, 'max_native_prompt_positions': max_context,
        'estimated_full_FP32_feature_bytes': source_positions*3072*4,
        'original_TRAIN_geometry_equal': True, 'pilot_repetitions_exact': repeated,
        'elapsed_seconds': time.monotonic()-started,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts': {str(p): data.sha(p) for p in sorted(root.iterdir()) if p.is_file()}, **BOUNDARIES}
    data.write(root/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=tuple(ROOTS))
    stage = parser.parse_args().stage
    existed = ROOTS[stage].exists()
    try:
        run(stage)
    except Exception as error:
        if not existed and ROOTS[stage].exists():
            data.write(ROOTS[stage]/'failure.json', {'exception': type(error).__name__, 'message': str(error), **BOUNDARIES})
        raise
