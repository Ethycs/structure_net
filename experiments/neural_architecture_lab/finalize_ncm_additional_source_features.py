"""Conservatively register the completed larger original-body source cache."""
import argparse
import json
from pathlib import Path

from neural_architecture_lab.ncm_source_data_hypotheses import CONTROLS, SourceDataEvidence
from experiments.structure_net import ncm_additional_source_features as run
from experiments.neural_architecture_lab import audit_ncm_additional_source_features as audit
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = run.ROOT / '20261024_additional10000_source_features_results_v1'
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-additional-source-features.md')


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError('fresh larger source-cache aggregate/report required')
    receipt = audit.DEST / 'actual_tool_terminal_receipt.json'
    if run.q.read(receipt)['exit_code'] != 0:
        raise ValueError('actual completed independent all10000payload audit required')
    result = run.q.read(audit.DEST / 'result.json')
    expected = CONTROLS[run.HYPOTHESIS]
    if (result['status'] != expected[0] or result['cases'] != 10000 or not result['FP32_bytes_preserved']
            or result['original_body_hash_unchanged'] != run.base.EXPECTED_BODY
            or result['raw_bytes'] != 95_563_960_320 or result['model_speed_measured']
            or not result['all_final_sources_closed']):
        raise ValueError('complete unchanged original-body lossless FP32 cache required')
    for path, expected_hash in result['artifacts'].items():
        if run.file_hash(path) != expected_hash:
            raise ValueError('audited source-cache evidence changed:' + path)
    inputs = [audit.DEST / 'result.json', receipt, run.FEATURES / 'manifest.json', run.FEATURES / 'result.json',
        run.WORKFLOW / 'tool_terminal_receipt.json', run.metadata.DATA / 'index.json',
        Path(__file__), Path('src/neural_architecture_lab/ncm_source_data_hypotheses.py')]
    p = run.native.vendor.freeze(DEST, 'aggregate_audited_all10000_original_FP32_source_features', inputs,
        {'hypothesis_id': run.HYPOTHESIS, 'data_preparation_only': True, 'source_cases': 10000,
            'source_positions': 7777015, 'model_weights_changed': False, 'new_teacher_labels_read': False,
            'canonical_firsts_changed': False, 'final_test_opened': False, 'model_speed_measured': False})
    campaign = {'hypothesis_id': run.HYPOTHESIS, 'classification': result['status'],
        'source_cases': 10000, 'measurements': 10000, 'source_positions': 7777015,
        'raw_bytes': result['raw_bytes'], 'stored_bytes': result['stored_bytes'],
        'stored_over_raw': result['stored_bytes'] / result['raw_bytes'], 'FP32_bytes_preserved': True,
        'original_body_hash_unchanged': result['original_body_hash_unchanged'], 'student_task_accuracy': None,
        'model_speed_measured': False, 'canonical_firsts_changed': False, 'final_test_opened': False,
        'confirmed': False, 'artifacts': {str(path): run.file_hash(path) for path in inputs}}
    campaign['artifacts'][str(DEST / 'producer.json')] = run.file_hash(DEST / 'producer.json')
    run.native.vendor.verify(p)
    path = DEST / 'campaign_results.json'
    run.write_json(path, campaign)
    REPORT.write_text('# Larger original frozen-body source feature cache\n\n'
        + f"Status **{result['status']}**. Data preparation only. Hypothesis `{run.HYPOTHESIS}`.\n"
        + f"Campaign `{path}`, SHA256 `{run.file_hash(path)}`.\n\n"
        'All10,000reserved source-disjoint TRAIN sources retain every source position. '
        'The unchanged28layer BF16Llama body produced7,777,015character-aligned FP32source positions, each with3072channels. '
        'Original first/longest TRAINfeature references matched exactly before each worker; every pilot was reused. '
        'All10,000stored payloads were independently decompressed and verified against source/prompt/alignment/body bindings and originalFP32checksums.\n\n'
        + f"Actual raw features: {result['raw_bytes']:,}bytes. Actual stored headers+payloads: {result['stored_bytes']:,}bytes "
        + f"({campaign['stored_over_raw'] * 100:.2f}%of raw). Byte-shuffled zlib1is reversible storage compression, not quantization.\n\n"
        'This compresses data storage, not model weights or inference computation. '
        'Teacher-label creation and future model training are separate. The native655student still failed0/128DEV; '
        'this cache establishes neither95%summary quality nor a speedup. No new canonical semantic judgments were made and final320sources remain closed/unhashed.\n')
    print(json.dumps({k: v for k, v in campaign.items() if k != 'artifacts'}), flush=True)


def register():
    path = DEST / 'campaign_results.json'
    result = run.q.read(path)
    evidence = SourceDataEvidence(run.HYPOTHESIS, result['classification'], result['source_cases'], result['measurements'],
        str(path), run.file_hash(path), str(REPORT), run.file_hash(REPORT)).record()
    register_records(path, REPORT, DEST, {run.HYPOTHESIS: evidence})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('finalize', 'register'))
    args = parser.parse_args()
    {'finalize': finalize, 'register': register}[args.stage]()
