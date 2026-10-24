"""CPU replay of the read-only native655 feature systems diagnostic."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_native_feature_pipeline_check as run

DEST = run.ROOT / 'audit'
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-native-feature-pipeline-check.md')


def main():
    if DEST.exists() or REPORT.exists() or torch.cuda.is_initialized():
        raise ValueError('fresh CPU replay/report required')
    artifacts = {str(Path(__file__)): run.features.file_hash(Path(__file__))}
    for stage in ('cpu', 'gpu'):
        root = run.ROOT / stage
        if run.q.read(root / 'actual_tool_terminal_receipt.json')['exit_code'] != 0:
            raise ValueError('actual successful CPU and GPU tool exits required')
        run.native.vendor.verify(run.q.read(root / 'producer.json'))
        for name in ('producer.json', 'result.json', 'actual_tool_terminal_receipt.json'):
            artifacts[str(root / name)] = run.features.file_hash(root / name)
    cpu = run.q.read(run.ROOT / 'cpu/result.json')
    rows = run.q.read(run.ROOT / 'cpu/cases.json')
    if (cpu['source_cases'] != 783 or len(rows) != 783 or len({r['id'] for r in rows}) != 783
            or cpu['metadata_mismatches'] != sum(bool(r['metadata_differences']) for r in rows)
            or cpu['cases_sha256'] != run.features.file_hash(run.ROOT / 'cpu/cases.json')):
        raise ValueError('complete783source CPU diagnostic differs')
    artifacts[str(run.ROOT / 'cpu/cases.json')] = cpu['cases_sha256']
    gpu = run.q.read(run.ROOT / 'gpu/result.json')
    expected = {(cohort, row['id']) for cohort, cases, manifest in run.populations() for row in run.select(cases, manifest)}
    if len(gpu['cases']) != 6 or {(r['cohort'], r['id']) for r in gpu['cases']} != expected:
        raise ValueError('six fixed first/longest TRAIN/shortDEV/longDEV probes required')
    pairs = []
    for item in gpu['cases']:
        for name in ('reference', 'recomputed'):
            path, digest = item[name + '_path'], item[name + '_sha256']
            if run.features.file_hash(path) != digest:
                raise ValueError('stored GPU/reference feature changed')
            artifacts[path] = digest
        before = torch.load(item['reference_path'], map_location='cpu', weights_only=True)['features']
        after = torch.load(item['recomputed_path'], map_location='cpu', weights_only=True)['features']
        exact = (before.dtype == after.dtype == torch.float32 and before.shape == after.shape
            and torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8)))
        if exact != item['FP32_bytes_exact']:
            raise ValueError('independent CPU byte replay differs')
        pairs.append({'id': item['id'], 'cohort': item['cohort'], 'FP32_bytes_exact': exact})
    passed = cpu['metadata_mismatches'] == 0 and all(r['FP32_bytes_exact'] for r in pairs)
    result = {'classification': 'six_probe_pipeline_parity_passed' if passed else 'source_pipeline_mismatch_detected',
        'systems_readiness_only_not_quality_evidence': True, 'CPU_source_cases': 783,
        'metadata_mismatches': cpu['metadata_mismatches'], 'GPU_pairs_replayed': pairs,
        'peak_allocated_bytes': gpu['peak_allocated_bytes'], 'task_accuracy': None,
        'final_sources_opened_or_hashed': False, 'artifacts': artifacts}
    DEST.mkdir()
    run.features.write_json(DEST / 'result.json', result)
    REPORT.write_text('# Native655 source feature pipeline diagnostic\n\n'
        + f"Status **{result['classification']}**. Read-only systems check after the recorded0/128DEVfailure.\n"
        + f"Artifact `{DEST / 'result.json'}`, SHA256 `{run.features.file_hash(DEST / 'result.json')}`.\n\n"
        + f"All783source records (655TRAIN,64shortDEV,64longDEV) were independently rebuilt: {cpu['metadata_mismatches']}"
        ' prompt/tokenization/date/alignment differences. Every cached feature had its source/prompt/alignment binding, fullFP32shape and file hash checked.\n\n'
        + f"An unchanged28layer frozen Llama body recomputed six fixed first/longest sources. CPUreplay found "
        + f"{sum(r['FP32_bytes_exact'] for r in pairs)}/6 byte-exactFP32tensors. Original BF16body and FP32alignment were retained; "
        + f"peak GPUallocation {gpu['peak_allocated_bytes']:,}bytes. No training or new semantic judging occurred.\n\n"
        'Feature-scale means (TRAIN/shortDEV/longDEV) were1.588/1.592/1.599RMS and1.359/1.357/1.411centeredRMS. '
        'These descriptive values and the six probes provide no evidence of the suspected feature mismatch; they do not rule out every possible integration error.\n\n'
        'The adapted checkpoint still failed all128complete-summary judgments. Fluent TRAINoutputs alongside unreadable DEVoutputs remain consistent with memorization and poor conditional generalization. '
        'The LM1Brelease is an unconditional language generator; the new frozen-body source-conditioning path was trained on only655pairs. '
        'It is not a matched replication of the paper\'s conditional summarization training. Larger, higher-quality paired training and a corrected level-dependent conditioning-corruption experiment are next hypotheses, not demonstrated fixes. '
        'Final320sources remain closed and unhashed;95%quality and a speedup are unestablished.\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
