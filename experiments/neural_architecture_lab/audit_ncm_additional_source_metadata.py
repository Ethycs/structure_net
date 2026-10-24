"""Rebuild all10000source-only tokenizations and character alignments on CPU."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_additional_source_metadata as run

DEST = run.ROOT / '20261024_additional10000_source_metadata_audit_v1'


def main():
    if DEST.exists() or torch.cuda.is_initialized() or run.q.read(run.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0:
        raise ValueError('fresh CPU audit and actual whole source metadata completion required')
    for root in (run.PILOT, run.DATA):
        run.native.vendor.terminal(root)
    rows = run.require_sources()
    index = run.q.read(run.DATA / 'index.json')
    if len(index) != 10000 or [r['id'] for r in index] != [r['id'] for r in rows]:
        raise ValueError('all10000original ordered source entries required')
    native_tokenizer, gpt, storage = run.tokenizers()
    proof = run.parity(native_tokenizer, gpt, storage)
    if run.q.read(run.DATA / 'parity.json') != proof or run.q.read(run.PILOT / 'parity.json') != proof:
        raise ValueError('original TRAIN first/longest prompt/date/alignment parity differs')
    for i, (row, entry) in enumerate(zip(rows, index), 1):
        prepared = run.prepare_one(row, native_tokenizer, gpt, storage)
        if (run.file_hash(entry['path']) != entry['sha256'] or run.q.read(entry['path']) != prepared
                or entry['source_positions'] != len(prepared['source_ids'])
                or entry['native_prompt_positions'] != len(prepared['context_ids'])
                or entry['source_frame'] != prepared['source_frame']
                or entry['alignment_sha256'] != prepared['alignment_sha256']
                or entry['native_prompt_sha256'] != prepared['native_prompt_sha256']):
            raise ValueError('complete actual source tokenization/alignment replay differs')
        if i % 512 == 0:
            print(json.dumps({'audited_source_metadata': i, 'total': 10000}), flush=True)
    by_id = {r['id']: r for r in index}
    pilot = run.q.read(run.PILOT / 'index.json')
    if {r['id'] for r in pilot} != set(run.pilot_ids(rows)) or any(by_id[r['id']] != r for r in pilot):
        raise ValueError('all five first metadata pilot rows must remain exact')
    state = run.q.read(run.DATA / 'result.json')
    positions = sum(r['source_positions'] for r in index)
    if (state['cases'] != 10000 or state['source_positions'] != positions
            or state['estimated_raw_FP32_feature_bytes'] != positions * 3072 * 4
            or state['estimated_BF16_feature_bytes'] != positions * 3072 * 2
            or state['feature_values_computed'] or state['teacher_labels_used_for_preparation']):
        raise ValueError('actual source population/feature storage projection differs')
    artifacts = {str(Path(__file__)): run.file_hash(Path(__file__))}
    for root in (run.WORKFLOW, run.PILOT, run.DATA):
        for path in root.rglob('*'):
            if path.is_file() and 'producer_source' not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    result = {'status': 'all10000_source_tokenization_alignment_replayed_no_features_or_labels',
        'cases': 10000, 'source_positions': positions, 'estimated_raw_FP32_feature_bytes': positions * 3072 * 4,
        'CPU_only': True, 'task_accuracy': None, 'all_final_sources_closed': True, 'artifacts': artifacts}
    DEST.mkdir()
    run.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
