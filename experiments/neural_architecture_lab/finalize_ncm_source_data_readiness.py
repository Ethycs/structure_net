"""Register completed source metadata and lossless storage controls conservatively."""
import argparse
import json
from pathlib import Path

from neural_architecture_lab.ncm_source_data_hypotheses import CONTROLS, SourceDataEvidence
from experiments.structure_net import ncm_additional_source_metadata as metadata
from experiments.structure_net import ncm_lossless_feature_storage as storage
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT, q, native = metadata.ROOT, metadata.q, metadata.native
META_AUDIT = ROOT / '20261024_additional10000_source_metadata_audit_v1'
STORE_AUDIT = ROOT / '20261024_lossless_FP32_feature_storage_audit_v1'
RESULTS = ROOT / '20261024_source_data_readiness_results_v1'
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-source-data-readiness.md')


def finalize():
    if RESULTS.exists() or REPORT.exists():
        raise ValueError('fresh append-only source-data aggregate required')
    controls = {}
    artifacts = {}
    for hyp, root, cases, count in ((metadata.HYPOTHESIS, META_AUDIT, 10000, 10000),
            (storage.HYPOTHESIS, STORE_AUDIT, 16, 32)):
        receipt_path = root / 'actual_tool_terminal_receipt.json'
        if q.read(receipt_path)['exit_code'] != 0:
            raise ValueError('actual independent completed data-control audit required')
        result = q.read(root / 'result.json')
        expected = CONTROLS[hyp]
        if result['status'] != expected[0] or result.get('cases', result.get('source_cases')) != cases:
            raise ValueError('actual complete data-only control scope differs')
        for path, digest in result['artifacts'].items():
            if file_hash(path) != digest:
                raise ValueError('completed data-control evidence changed:' + path)
        artifacts.update(result['artifacts'])
        artifacts[str(root / 'result.json')] = file_hash(root / 'result.json')
        artifacts[str(receipt_path)] = file_hash(receipt_path)
        controls[hyp] = {'classification': result['status'], 'source_cases': cases, 'measurements': count}
    index = q.read(metadata.DATA / 'index.json')
    meta = q.read(META_AUDIT / 'result.json')
    metrics = q.read(STORE_AUDIT / 'result.json')['metrics']
    projected = sum(sum(r['source_positions'] for r in index if r['bin'] == band) * 3072 * 4
        * metrics['byte_shuffled']['by_band'][str(band)]['stored_bytes']
        / metrics['byte_shuffled']['by_band'][str(band)]['raw_bytes'] for band in range(5))
    sources = [Path(__file__), Path('src/neural_architecture_lab/ncm_source_data_hypotheses.py')]
    p = native.vendor.freeze(RESULTS, 'aggregate_completed_source_metadata_and_lossless_storage_controls',
        [Path(path) for path in artifacts] + sources, {'hypothesis_ids': list(controls), 'student_task_accuracy_measured': False,
            'model_speed_measured': False, 'feature_values_for10000sources_computed': False,
            'projection_is_not_measured_full_corpus_compression': True, 'final_test_opened': False})
    artifacts[str(RESULTS / 'producer.json')] = file_hash(RESULTS / 'producer.json')
    campaign = {'classification': 'source_data_controls_completed_student_quality_unproven', 'controls': controls,
        'source_positions': meta['source_positions'], 'projected_raw_FP32_feature_bytes': meta['estimated_raw_FP32_feature_bytes'],
        'measured_fixed16_feature_storage': metrics, 'projected_full_corpus_stored_bytes_from_band_samples': projected,
        'projection_is_not_measured_full_corpus_compression': True, 'student_task_accuracy': None,
        'full_corpus_feature_values_computed': False, 'model_speed_measured': False,
        'canonical_firsts_changed': False, 'final_test_opened': False, 'confirmed': False, 'artifacts': artifacts}
    native.vendor.verify(p)
    path = RESULTS / 'campaign_results.json'
    write_json(path, campaign)
    sample = metrics['byte_shuffled']
    REPORT.write_text(
        '# Larger TRAIN source metadata and lossless FP32 storage controls\n\n'
        'Status **completed data controls; student quality unproven**.\n'
        + f"Hypotheses `{metadata.HYPOTHESIS}` and `{storage.HYPOTHESIS}`.\n"
        + f"Campaign `{path}`, SHA256 `{file_hash(path)}`.\n\n"
        'All 10,000 reserved public TRAIN sources were independently retokenized and\n'
        'their full character alignments rebuilt on CPU. Original first/longest\n'
        'TRAIN prompt/date/alignment geometry matched exactly. No new teacher\n'
        'labels or selected final sources entered preparation.\n\n'
        + f"The corpus has {meta['source_positions']:,} GPT-2 source positions: "
        + f"{meta['estimated_raw_FP32_feature_bytes'] / 1e9:.2f} GB of raw FP32 features.\n\n"
        'Both lossless layouts were independently decoded on all 16 fixed TRAIN\n'
        'feature tensors. Every one of 32 payloads reproduced the original FP32 bytes,\n'
        'including every source position/channel; no quantization or arithmetic.\n\n'
        + f"Byte-shuffled zlib at level 1 stored {sample['stored_bytes']:,} bytes versus "
        + f"{sample['raw_bytes']:,} raw bytes ({sample['stored_over_raw'] * 100:.2f}%).\n"
        + f"Applying measured per-band sample ratios projects {projected / 1e9:.2f} GB for the new corpus. "
        'That is an estimate; its actual full feature cache has not been computed.\n\n'
        'This compresses training-data storage. It does not shrink model weights,\n'
        'measure model token throughput or establish 95% summary accuracy. Current\n'
        'student training on 655 examples and its held-out evaluation remain separate.\n'
        'Full feature extraction is gated after its completed DEV and runs only\n'
        'if further training is needed. Final 320 sources stay closed and unhashed.\n')
    print(json.dumps({k: v for k, v in campaign.items() if k not in ('artifacts', 'measured_fixed16_feature_storage')}), flush=True)


def register():
    path = RESULTS / 'campaign_results.json'
    state = q.read(path)
    records = {}
    for hyp, value in state['controls'].items():
        records[hyp] = SourceDataEvidence(hyp, value['classification'], value['source_cases'], value['measurements'],
            str(path), file_hash(path), str(REPORT), file_hash(REPORT)).record()
    register_records(path, REPORT, RESULTS, records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('finalize', 'register'))
    args = parser.parse_args()
    {'finalize': finalize, 'register': register}[args.stage]()
