"""Prepare complete source-only geometry for the new engineering partitions."""
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_engineering_summary_inventory_v2 as data
from experiments.structure_net import ncm_additional_source_metadata as geometry
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

ROOT, native = data.ROOT, data.native
DEST = ROOT / '20261025_engineering_source_metadata_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-engineering-source-metadata-plan.md')
TEST = Path('tests/structure_net/test_ncm_engineering_source_metadata.py')


def source_only(row, split):
    if text_hash(row['source']) != row['source_sha256']:
        raise ValueError('unaltered complete source required')
    keep = ('id', 'source', 'source_sha256', 'source_id', 'group_id', 'source_component', 'spans')
    return {**{k: row[k] for k in keep}, 'source_cl100k_tokens': row['input_tokens'],
        'source_truncated': False, 'split': split, 'bin': data.old.reservation.graph.bucket(row['input_tokens'])}


def main():
    if DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only engineering source metadata required')
    native.vendor.corpus.quality.dependencies()
    if data.old.reservation.read(data.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed engineering candidate reservation required')
    native.vendor.verify(data.old.reservation.read(data.DEST / 'producer.json'))
    result = data.old.reservation.read(data.DEST / 'result.json')
    if (result['candidate_pairs'] != {'train': 732, 'test': 156}
            or file_hash(data.DEST / 'cases.json') != result['cases_sha256']):
        raise ValueError('complete unchanged candidate partitions required')
    parts = data.old.reservation.read(data.DEST / 'cases.json')
    # Targets are deliberately removed before any tokenizer/model interface.
    rows = [source_only(row, split) for split in ('train', 'test') for row in parts[split]]
    tokenizers = geometry.tokenizers()
    parity = geometry.parity(*tokenizers)
    files = [Path(__file__), PLAN, TEST, Path(data.__file__), Path(geometry.__file__),
        Path(geometry.pinned.__file__), Path(geometry.development.__file__), Path(geometry.base.__file__)]
    assets = [p for parent in (geometry.base.MODEL_PATH, geometry.base.GPT2) for p in parent.iterdir()
        if p.is_file() and p.name in {'config.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'chat_template.jinja', 'vocab.json', 'merges.txt', 'added_tokens.json'}]
    inputs = [data.DEST / n for n in ('producer.json', 'result.json', 'cases.json', 'actual_tool_receipt.json')]
    inputs += [geometry.old_training.DATA / n for n in ('cases.json', 'manifest.json')]
    inputs += assets
    DEST.mkdir()
    (DEST / 'cases').mkdir()
    (DEST / 'producer_source').mkdir()
    for i, path in enumerate(files):
        (DEST / 'producer_source' / f'{i}_{path.name}').write_bytes(path.read_bytes())
    producer = {'schema': 'nal.ncm-engineering-source-geometry.v1',
        'source_hashes': {str(p): file_hash(p) for p in files},
        'input_hashes': {str(p): file_hash(p) for p in inputs},
        'source_scope': 'source-only engineering TRAIN and engineering heldout; not a public-LME proof',
        'config': {'date_string': '23 Oct 2026', 'native_system': geometry.base.SYSTEM,
            'source_only': True, 'target_fields_in_tokenizer_or_model_arguments': False,
            'parent_loaded': False, 'full_source_retained': True, 'source_frame_maximum': 8192,
            'original_parent_and_MLPs_frozen': True, 'feature_values_computed': False,
            'task_accuracy': None, 'sealed_final_sources_opened_or_hashed': False}}
    write_json(DEST / 'producer.json', producer)
    index = []
    for i, row in enumerate(rows, 1):
        prepared = geometry.prepare_one(row, *tokenizers)
        path = DEST / 'cases' / (row['id'] + '.json')
        write_json(path, prepared)
        index.append({k: prepared[k] for k in ('id', 'source_id', 'group_id', 'source_component',
            'source_sha256', 'split', 'alignment_sha256', 'native_prompt_sha256', 'source_frame')})
        index[-1].update(path=str(path), sha256=file_hash(path), source_positions=len(prepared['source_ids']),
            native_prompt_positions=len(prepared['context_ids']))
        if i % 128 == 0 or i == len(rows):
            print(json.dumps({'source_metadata_prepared': i, 'total': len(rows)}), flush=True)
    write_json(DEST / 'index.json', index)
    write_json(DEST / 'original_training_geometry_parity.json', parity)
    native.vendor.verify(producer)
    result = {'status': 'all_engineering_source_only_tokenization_and_alignment_prepared',
        'cases': len(index), 'partitions': {k: sum(r['split'] == k for r in index) for k in parts},
        'original_training_geometry_parity': True, 'targets_in_tokenizer_or_model_arguments': False,
        'source_positions': sum(r['source_positions'] for r in index),
        'max_source_positions': max(r['source_positions'] for r in index),
        'max_native_prompt_positions': max(r['native_prompt_positions'] for r in index),
        'estimated_raw_FP32_feature_bytes': sum(r['source_positions'] for r in index) * 3072 * 4,
        'index_sha256': file_hash(DEST / 'index.json'), 'parent_loaded': False,
        'feature_values_computed': False, 'new_model_training': False, 'task_accuracy': None,
        'sealed_final_sources_opened_or_hashed': False}
    write_json(DEST / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
