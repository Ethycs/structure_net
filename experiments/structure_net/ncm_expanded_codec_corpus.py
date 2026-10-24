"""Complete raw-text codec corpus, excluding all64 component source families."""
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_sol64_codec_component as component
from experiments.structure_net import ncm_lm1b_domain_decoder as domain
from experiments.structure_net import ncm_sol_evidence_teacher_recovery as teacher
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

native, q, ROOT = component.native, component.q, component.ROOT
DATA = ROOT / '20261024_expanded9936_codec_corpus_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-expanded-codec-corpus-plan.md')
TEST = Path('tests/structure_net/test_ncm_expanded_codec_corpus.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_expanded_codec_corpus.py')
HYPOTHESIS = 'ncm-expanded9936-source-disjoint-codec-corpus-v1'


def split(rows, held):
    if len(rows) != 10000 or len(held) != 64:
        raise ValueError('complete10000 reserved sources and64 component controls required')
    blocked = {r['source_component'] for r in held}
    result = [r for r in rows if r['source_component'] not in blocked]
    excluded = [r for r in rows if r['source_component'] in blocked]
    if (len(blocked) != 64 or len(result) != 9936 or len(excluded) != 64
            or {r['id'] for r in excluded} != {r['id'] for r in held}):
        raise ValueError('all64 whole component source families must be excluded')
    a, b = domain.identities(result), domain.identities(held)
    for name in a:
        if a[name] & b[name]:
            raise ValueError('TRAIN/component identity overlap: ' + name)
    return result, {name: {'train_unique': len(a[name]), 'held_unique': len(b[name]), 'overlap': 0}
        for name in a}


def sources():
    source = teacher.data.DEST / 'cases.json'
    labels = teacher.GENERATE / 'labels.json'
    if file_hash(source) != teacher.base.original.old.SOURCE_SHA:
        raise ValueError('pinned10000-source corpus changed')
    if file_hash(labels) != '2a48d4d7925dba2368b4a4732f89a8bc578c7b7e180b61f3bb4b129d950643e3':
        raise ValueError('all64 unchanged component source bindings required')
    rows, held = q.read(source), q.read(labels)
    selected, proof = split(rows, held)
    development = q.read(component.text_control.development.SOURCE / 'cases.json')
    dev = [r for cohort in development.values() for r in cohort]
    for key in ('id', 'source_id', 'group_id', 'source_sha256'):
        if {r[key] for r in selected} & {r[key] for r in dev}:
            raise ValueError('codec raw TRAIN corpus overlaps permanent DEV: ' + key)
    return selected, held, proof


def frame_text(text, tokenizer):
    # Literal strings such as [CLS] are ordinary source text. Split them into
    # lexical pieces rather than silently interpreting them as frame boundaries.
    ids = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
    if not ids or any(type(v) is not int or not 0 < v < 30522 or v == 101 for v in ids):
        raise ValueError('nonempty complete native lexical sequence required')
    _, frames = component.framed.pack(ids)
    reconstructed = []
    for frame in frames:
        end = frame.index(101, 1)
        reconstructed.extend(frame[1:end])
    if reconstructed != ids:
        raise ValueError('complete lexical coverage without cropping required')
    return ids, frames


def producing(inputs):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(domain.__file__), Path(component.framed.__file__),
        Path(teacher.__file__), Path(teacher.data.__file__)]
    common = [teacher.data.DEST / 'cases.json', teacher.GENERATE / 'labels.json',
        component.text_control.development.SOURCE / 'cases.json', domain.AUTHOR_DATA / 'frames.pt',
        domain.AUTHOR_DATA / 'result.json', domain.AUTHOR_DATA / 'worker_terminal_receipt.json',
        *[p for p in sorted((native.INVENTORY / 'bert-tokenizer').iterdir()) if p.is_file()]]
    p = native.vendor.freeze(DATA, 'complete9936_raw_source_lexical_frames', list(dict.fromkeys([*inputs, *common, *files])), {
        'hypothesis_id': HYPOTHESIS, 'raw_source_cases': 9936, 'component_families_excluded': 64,
        'summary_target_text_used': False, 'new_teacher_requests': False, 'native_context': 128,
        'frame': 'CLS101 + atmost126 lexical + CLS101 + PAD0',
        'literal_special_strings': 'split_special_tokens=True for raw source text only',
        'all_lexical_positions_retained': True, 'unknown_word_ID_allowed': 100,
        'author_TRAIN_replay_pool': 16384, 'author_validation_used': False,
        'permanent_DEV_disjoint': True, 'student_training_admitted': False,
        'student_task_accuracy': None, 'canonical_semantic_cache_changed': False,
        'final_test_opened': False, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (DATA / 'producer_source' / f'corpus_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(DATA / 'producer.json', p)
    return p


def main():
    component.configure()
    if DATA.exists() or torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only expanded codec corpus required')
    native.vendor.terminal(domain.AUTHOR_DATA)
    rows, held, proof = sources()
    public = component.text_control.reference
    public_proof = public.verify_public(rows)
    p = producing([public.PUBLIC_FILE])
    tokenizer = native.tokenizer()
    tokenizer.model_max_length = 1_000_000
    frames, index, lexical_count, unknown_count = [], [], 0, 0
    for i, row in enumerate(rows, 1):
        ids, packed = frame_text(row['source'], tokenizer)
        index.append({**{k: row[k] for k in ('id', 'source_id', 'group_id', 'source_sha256', 'source_component')},
            'frame_start': len(frames), 'frame_count': len(packed), 'lexical_tokens': len(ids),
            'unknown_tokens': ids.count(100), 'text_sha256': text_hash(row['source'])})
        frames.extend(packed)
        lexical_count += len(ids)
        unknown_count += ids.count(100)
        if i % 1024 == 0:
            print(json.dumps({'raw_source_cases_framed': i, 'population': 9936}), flush=True)
    torch.save(torch.tensor(frames, dtype=torch.long), DATA / 'frames.pt')
    write_json(DATA / 'index.json', index)
    write_json(DATA / 'split_proof.json', {'identity_disjointness': proof,
        'excluded64_ids': [r['id'] for r in held], 'excluded64_components': [r['source_component'] for r in held],
        'permanent_DEV_disjoint': True})
    write_json(DATA / 'public_proof.json', {'public_file_sha256': public.PUBLIC_SHA256, 'exact_sources': public_proof})
    author = torch.load(domain.AUTHOR_DATA / 'frames.pt', map_location='cpu', weights_only=True)
    if author.shape != (16384, 128) or author.dtype != torch.long:
        raise ValueError('complete existing author TRAIN replay pool required')
    native.vendor.verify(p)
    result = {'status': 'complete9936_raw_codec_corpus_prepared_independent_replay_pending',
        'source_cases': 9936, 'raw_frames': len(frames), 'lexical_tokens': lexical_count,
        'unknown_tokens': unknown_count, 'all_lexical_positions_retained': True,
        'summary_targets_used': False, 'author_TRAIN_replay_frames': len(author),
        'student_training_admitted': False, 'student_task_accuracy': None,
        'canonical_semantic_cache_changed': False, 'final_test_opened': False,
        'artifacts': {str(DATA / n): file_hash(DATA / n) for n in ('frames.pt', 'index.json', 'split_proof.json', 'public_proof.json')}}
    write_json(DATA / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
