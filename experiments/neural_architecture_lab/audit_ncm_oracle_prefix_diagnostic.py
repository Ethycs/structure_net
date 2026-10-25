"""Independent CPU binding, prefix and rendering audit for privileged diagnostics."""
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_oracle_prefix_diagnostic as run

DEST = run.ROOT / '20261025_level_conditioning_oracle_prefix_audit_v1'


def main():
    if (DEST.exists() or torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or run.q.read(run.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('fresh CPU audit after actual diagnostic exit 0 required')
    producer = run.q.read(run.DEST / 'producer.json')
    run.native.vendor.verify(producer)
    result = run.q.read(run.DEST / 'result.json')
    values = run.q.read(run.DEST / 'outputs.json')
    if result['outputs_sha256'] != run.component.file_hash(run.DEST / 'outputs.json'):
        raise ValueError('complete diagnostic outputs changed')
    rows = run.component.inputs()
    chosen = [rows[0], min(rows, key=lambda r: (-r['feature']['source_tokens'], r['id']))]
    if [r['id'] for r in values] != [r['id'] for r in chosen] or result['cases'] != 2:
        raise ValueError('predeclared first and longest-source pair differs')
    tokenizer = run.native.tokenizer()
    summaries = []
    for value, row in zip(values, chosen):
        if (value['oracle_codes'] != row['oracle_codes'] or value['oracle_word_ids'] != row['native_word_ids']
                or value['source_sha256'] != row['source_sha256']
                or value['feature_payload_sha256'] != row['feature']['payload_sha256']
                or value['source_positions'] != row['feature']['source_tokens']):
            raise ValueError('original source/feature/target bindings differ')
        for count in run.FORCED_LEVELS:
            arm = value['arms'][str(count)]
            cut = sum(run.native.LEVELS[:count])
            if (arm['forced_positions'] != cut or len(arm['codes']) != 641
                    or arm['codes'][:cut] != row['oracle_codes'][:cut]
                    or [t['positions'] for t in arm['level_trace']] != list(run.native.LEVELS)
                    or [t['forced'] for t in arm['level_trace']] != [i < count for i in range(12)]):
                raise ValueError('complete native forced prefix or trace differs')
            active = [i for i, w in enumerate(row['native_word_ids']) if w]
            matches = sum(arm['raw_word_ids'][i] == row['native_word_ids'][i] for i in active)
            if arm['active_words'] != len(active) or arm['active_word_matches'] != matches:
                raise ValueError('word diagnostic count differs')
            decoded = run.component.selected.decode(arm['raw_word_ids'], tokenizer)
            if any(decoded[k] != arm[k] for k in ('text', 'predicted_BOS', 'predicted_EOS_position', 'missing_EOS')):
                raise ValueError('native raw rendering changed')
        if value['arms']['12']['oracle_latent_MSE'] != 0.:
            raise ValueError('all-oracle latent diagnostic differs')
        source_off = value['oracle_decode_without_source']
        decoded = run.component.selected.decode(source_off['raw_word_ids'], tokenizer)
        if any(decoded[k] != source_off[k] for k in ('text', 'predicted_BOS', 'predicted_EOS_position', 'missing_EOS')):
            raise ValueError('source-off rendering differs')
        summaries.append({'id': row['id'], 'conditions': {str(k): {
            key: value['arms'][str(k)][key] for key in
            ('active_word_matches', 'active_words', 'word_cross_entropy', 'oracle_latent_MSE')}
            for k in run.FORCED_LEVELS}})
    if (result['checkpoint_sha256'] != '3911eab74db6ad13104f87765196e4212893a3e179ba0283e103e65e78273bec'
            or not result['model_unchanged'] or not result['source_contexts_restored']
            or result['task_accuracy'] is not None or result['final_test_opened']
            or result['peak_allocated_bytes'] > 3_000_000_000):
        raise ValueError('frozen new checkpoint or limited interpretation differs')
    paths = [Path(__file__), run.DEST / 'producer.json', run.DEST / 'outputs.json',
        run.DEST / 'result.json', run.DEST / 'actual_tool_receipt.json']
    DEST.mkdir()
    audit = {'status': 'oracle_prefix_bindings_counts_and_rendering_replayed_not_task_accuracy',
        'cases': 2, 'diagnostics': summaries, 'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(p): run.component.file_hash(p) for p in paths}}
    run.component.write_json(DEST / 'result.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
