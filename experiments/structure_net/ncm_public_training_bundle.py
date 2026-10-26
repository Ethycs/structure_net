"""Admit and open all audited public source features and fixed code targets."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from experiments.structure_net import ncm_public_feature_store as feature
from experiments.structure_net import ncm_public_code_targets as target
from experiments.structure_net import ncm_public_training_stream as streaming

data, catalog = feature.data, target.catalog
ROOT = data.ROOT/'20261026_public_training_bundle_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-training-bundle-plan.md')
TEST = Path('tests/structure_net/test_ncm_public_training_bundle.py')
HYPOTHESIS = 'ncm-public-training-bundle-v1'
BODY_SHA = feature.original.base.EXPECTED_BODY
CODEC_SHA = target.codec_source.CODEC_SHA


def completed_with_audit(root):
    receipt = data.read(root/'actual_tool_receipt.json')
    audit_receipt = data.read(root/'audit_actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0 or audit_receipt['actual_tool_exit_code'] != 0:
        raise ValueError('actual successful producer and CPU audit required')
    data.verify(receipt['artifacts']); data.verify(audit_receipt['artifacts'])
    result, audited = data.read(root/'result.json'), data.read(root/'audit_result.json')
    if (receipt['artifacts'].get(str(root/'result.json')) != data.sha(root/'result.json')
            or audit_receipt['artifacts'].get(str(root/'audit_result.json')) != data.sha(root/'audit_result.json')
            or audited['artifacts'].get(str(root/'result.json')) != data.sha(root/'result.json')):
        raise ValueError('audit must bind the exact completed producer result')
    data.verify(audited['artifacts'])
    return result, audited


def validate_stages(metadata, features, feature_audit, targets, target_audit):
    values = (metadata, features, feature_audit, targets, target_audit)
    if any(v['cases'] != 162353 for v in values):
        raise ValueError('all162353 original public pairs required at every stage')
    if (tuple(v['status'] for v in values) != ('all_public_full_source_metadata_replayed',
            'public_feature_store_full_completed', 'public_feature_store_full_disk_replay_passed',
            'public_code_targets_full_completed', 'all_public_full_target_bindings_replayed')
            or not metadata['full_metadata_audited'] or not metadata['all_records_replayed']
            or metadata['source_positions'] != 91344717 or features['source_positions'] != 91344717
            or feature_audit['source_positions'] != 91344717
            or features['body_sha256'] != BODY_SHA or not features['all_encoded_roundtrips_exact']
            or features['prior_raw_hash_controls'] != 16
            or not feature_audit['all_disk_records_replayed'] or not feature_audit['all_shards_completely_covered']
            or targets['codec_sha256'] != CODEC_SHA or not targets['codec_unchanged']
            or not targets['all_repeated_encodes_exact'] or targets['pilot_repetitions_exact'] != 16
            or targets['original_TRAIN_controls_exact'] != 16 or not target_audit['all_bindings_replayed']
            or not target_audit['targets_admitted_for_training']):
        raise ValueError('complete exact original-body features and frozen-codec labels required')
    if any(v.get('model_training') or v.get('new_judge_requests') or v.get('final_test_opened')
           or v.get('task_accuracy') is not None for v in values):
        raise ValueError('data-only preparation boundaries required')


def lines(path):
    with Path(path).open() as stream:
        return list(map(json.loads, stream))


def dataset_for(manifest):
    data.verify(manifest['index_hashes'])
    pair_entries = lines(manifest['pair_index'])
    code_entries = lines(manifest['code_index'])
    metadata = lines(manifest['source_index'])
    feature_entries = lines(manifest['feature_index'])
    pairs = target.indexed.IndexedJSONL(manifest['pairs'], pair_entries, manifest['pair_payload_sha256'])
    codes = target.indexed.IndexedJSONL(manifest['codes'], code_entries, manifest['code_payload_sha256'])
    return streaming.PublicTrainingDataset(pairs, codes, metadata, feature_entries,
                                           body_sha256=BODY_SHA, codec_sha256=CODEC_SHA)


def prepare():
    if ROOT.exists():
        raise ValueError('fresh append-only public training bundle required')
    from experiments.structure_net import ncm_pixi
    training, environment = ncm_pixi.configure()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or training.torch.cuda.is_initialized():
        raise ValueError('CPU-only training input admission required')
    admitted = data.completed(catalog.AUDIT)
    if not admitted['training_data_admitted'] or admitted['admission_scope'] != 'public_reference_conditional_pretraining_only':
        raise ValueError('original full public paired catalog admission required')
    source = data.completed(feature.audited.ROOTS['full'])
    features, feature_audit = completed_with_audit(feature.ROOTS['full'])
    targets, target_audit = completed_with_audit(target.ROOTS['full'])
    validate_stages(source, features, feature_audit, targets, target_audit)
    source_result = data.read(feature.metadata.ROOTS['full']/'result.json')
    catalog_result = data.read(catalog.DEST/'result.json')
    source_index = feature.metadata.ROOTS['full']/'index.jsonl'
    feature_index = feature.STORAGE/'full/index.jsonl'
    code_index = target.ROOTS['full']/'index.jsonl'
    pair_index = catalog.DEST/'training_index.jsonl'
    index_hashes = {str(source_index): source_result['artifacts'][str(source_index)],
        str(feature_index): features['artifacts'][str(feature_index)],
        str(code_index): targets['artifacts'][str(code_index)],
        str(pair_index): catalog_result['artifacts'][str(pair_index)]}
    evidence = [catalog.AUDIT/name for name in ('result.json', 'actual_tool_receipt.json', 'campaign_results.json')]
    evidence += [catalog.DEST/name for name in ('result.json', 'actual_tool_receipt.json', 'protection.json', 'public_development.jsonl')]
    evidence += [feature.audited.ROOTS['full']/name for name in ('result.json', 'actual_tool_receipt.json')]
    evidence += [feature.metadata.ROOTS['full']/name for name in ('result.json', 'actual_tool_receipt.json')]
    for root in (feature.ROOTS['full'], target.ROOTS['full']):
        evidence += [root/name for name in ('result.json', 'actual_tool_receipt.json', 'audit_result.json', 'audit_actual_tool_receipt.json', 'producer.json')]
    evidence += [feature.ROOTS['full']/'shards.json', Path('pyproject.toml'), Path('pixi.lock')]
    files = [Path(__file__), Path(streaming.__file__), Path(target.indexed.__file__),
             Path(feature.store.__file__), Path(feature.store.codec.__file__), PLAN, TEST,
             Path('tests/structure_net/test_ncm_public_training_stream.py')]
    manifest = {'schema': 'nal.ncm-public-training-bundle.v1', 'hypothesis_id': HYPOTHESIS,
        'cases': 162353, 'source_positions': 91344717, 'body_sha256': BODY_SHA, 'codec_sha256': CODEC_SHA,
        'pairs': str(catalog.DEST/'training.jsonl'), 'pair_index': str(pair_index),
        'codes': str(target.ROOTS['full']/'codes.jsonl'), 'code_index': str(code_index),
        'source_index': str(source_index), 'feature_index': str(feature_index),
        'pair_payload_sha256': catalog_result['artifacts'][str(catalog.DEST/'training.jsonl')],
        'code_payload_sha256': targets['artifacts'][str(target.ROOTS['full']/'codes.jsonl')],
        'index_hashes': index_hashes, 'evidence_hashes': {str(p): data.sha(p) for p in evidence},
        'source_hashes': {str(p): data.sha(p) for p in files}}
    started = time.monotonic()
    dataset = dataset_for(manifest)
    if len(dataset) != 162353:
        raise ValueError('complete training Dataset required')
    shards = data.read(feature.ROOTS['full']/'shards.json')
    if (len(shards) != features['shards'] or len({r['path'] for r in shards}) != len(shards)
            or any(Path(r['path']).stat().st_size != r['bytes'] or features['artifacts'][r['path']] != r['sha256'] for r in shards)):
        raise ValueError('all exact declared feature shards required')
    dataset.pairs.close(); dataset.codes.close()
    data.verify(manifest['index_hashes']); data.verify(manifest['evidence_hashes']); data.verify(manifest['source_hashes'])
    ROOT.mkdir()
    data.write(ROOT/'manifest.json', manifest)
    value = {'status': 'complete_public_training_inputs_admitted', 'hypothesis_id': HYPOTHESIS,
        'cases': 162353, 'all_original_ordered_index_joins_verified': True, 'all_source_positions_retained': True,
        'source_positions': 91344717, 'features_admitted_for_training': True, 'targets_admitted_for_training': True,
        'admission_scope': 'public_reference_conditional_pretraining_only', 'label_factual_quality_verified': False,
        'full_feature_payloads_rehashed_at_admission': False, 'prior_exhaustive_payload_audit_required': True,
        'requested_payloads_verified_before_training': True, 'feature_matrices_loaded_during_admission': 0,
        'body_sha256': BODY_SHA, 'codec_sha256': CODEC_SHA, 'public_development_reserved': 320,
        'environment': environment, 'elapsed_seconds': time.monotonic()-started,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'model_training': False, 'new_judge_requests': 0, 'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(ROOT/'manifest.json'): data.sha(ROOT/'manifest.json')}}
    data.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k not in ('artifacts', 'environment')}), flush=True)


def open_dataset():
    admitted = data.completed(ROOT)
    if (admitted['status'] != 'complete_public_training_inputs_admitted' or admitted['cases'] != 162353
            or not admitted['features_admitted_for_training'] or not admitted['targets_admitted_for_training']
            or admitted['admission_scope'] != 'public_reference_conditional_pretraining_only'):
        raise ValueError('actual complete public training bundle admission required')
    manifest = data.read(ROOT/'manifest.json')
    data.verify(manifest['evidence_hashes']); data.verify(manifest['source_hashes'])
    return dataset_for(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare',))
    parser.parse_args()
    prepare()
