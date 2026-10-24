"""Independent CPU reconstruction of every selected raw-text native frame."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_expanded_codec_corpus as run

DEST = run.ROOT / '20261024_expanded9936_codec_corpus_audit_v1'


def main():
    run.component.configure()
    if DEST.exists() or torch.cuda.is_initialized() or run.q.read(run.DATA / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('fresh CPU audit after actual complete corpus preparation required')
    run.native.vendor.verify(run.q.read(run.DATA / 'producer.json'))
    state = run.q.read(run.DATA / 'result.json')
    for path, digest in state['artifacts'].items():
        if run.file_hash(path) != digest:
            raise ValueError('corpus artifact changed')
    rows, held, proof = run.sources()
    split = {'identity_disjointness': proof, 'excluded64_ids': [r['id'] for r in held],
        'excluded64_components': [r['source_component'] for r in held], 'permanent_DEV_disjoint': True}
    if run.q.read(run.DATA / 'split_proof.json') != split:
        raise ValueError('source family or permanent DEV split differs')
    public = run.component.text_control.reference
    if run.q.read(run.DATA / 'public_proof.json') != {'public_file_sha256': public.PUBLIC_SHA256,
            'exact_sources': public.verify_public(rows)}:
        raise ValueError('public source proof differs')
    frames = torch.load(run.DATA / 'frames.pt', map_location='cpu', weights_only=True)
    index = run.q.read(run.DATA / 'index.json')
    if frames.ndim != 2 or frames.shape[1] != 128 or frames.dtype != torch.long or len(index) != 9936:
        raise ValueError('full native TRAIN frame tensor required')
    tokenizer = run.native.tokenizer()
    tokenizer.model_max_length = 1_000_000
    offset, lexical, unknown = 0, 0, 0
    for row, entry in zip(rows, index):
        ids = tokenizer.encode(row['source'], add_special_tokens=False, split_special_tokens=True)
        if not ids or any(v in (0, 101) or not 0 < v < 30522 for v in ids):
            raise ValueError('complete valid lexical source required')
        chunks = [ids[i:i+126] for i in range(0, len(ids), 126)]
        expected = torch.tensor([[101, *c, 101, *([0] * (126-len(c)))] for c in chunks], dtype=torch.long)
        identity = {k: row[k] for k in ('id', 'source_id', 'group_id', 'source_sha256', 'source_component')}
        identity.update(frame_start=offset, frame_count=len(chunks), lexical_tokens=len(ids),
            unknown_tokens=ids.count(100), text_sha256=run.text_hash(row['source']))
        if entry != identity or not torch.equal(frames[offset:offset+len(chunks)], expected):
            raise ValueError('complete original source framing or offsets differ')
        offset += len(chunks)
        lexical += len(ids)
        unknown += ids.count(100)
    if (offset != len(frames) or state['raw_frames'] != offset or state['lexical_tokens'] != lexical
            or state['unknown_tokens'] != unknown or state['source_cases'] != 9936
            or state['summary_targets_used'] or state['student_training_admitted']
            or not state['all_lexical_positions_retained'] or state['final_test_opened']):
        raise ValueError('corpus counts or data-only boundaries differ')
    artifacts = {str(run.DATA / n): run.file_hash(run.DATA / n) for n in
        ('producer.json', 'result.json', 'actual_tool_receipt.json', 'frames.pt', 'index.json', 'split_proof.json', 'public_proof.json')}
    artifacts[str(Path(__file__))] = run.file_hash(__file__)
    DEST.mkdir()
    result = {'status': 'all9936_sources_complete_native_frames_and_disjointness_replayed',
        'source_cases': 9936, 'raw_frames': offset, 'lexical_tokens': lexical, 'unknown_tokens': unknown,
        'all_lexical_positions_retained': True, 'component64_excluded': True, 'permanent_DEV_disjoint': True,
        'summary_targets_used': False, 'student_training_admitted': False, 'student_task_accuracy': None,
        'final_test_opened': False, 'artifacts': artifacts}
    run.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
