"""Replay first schema-valid draft/review responses; no semantic quality claim."""
import json
from pathlib import Path

from experiments.structure_net import ncm_evidence_teacher_pilot as run

DEST = run.ROOT / '20261024_evidence_teacher64_raw_audit_v1'


def replay(row, stage, observation, encoding, draft=None):
    expected_body = run.payload(row, stage, draft)
    found = None
    attempts = observation['attempts']
    if not 1 <= len(attempts) <= 3:
        raise ValueError('exact one-to-three ordered attempts required')
    for i, name in enumerate(attempts):
        path = Path(name)
        if path.name != f'attempt_{i}.json' or found is not None:
            raise ValueError('ordered attempts must stop at first schema-valid response')
        raw = run.q.read(path)
        tokens = len(encoding.encode(json.dumps(expected_body['messages'], ensure_ascii=False), disallowed_special=())) + 16
        if raw['request'] != expected_body or raw['prompt_tokens'] != tokens or tokens > 6300:
            raise ValueError('complete exact source/draft/prompt binding differs')
        parsed = None
        try:
            if (raw.get('http_status') != 200 or raw['response'].get('model') != run.MODELS[stage]
                    or raw['response']['choices'][0]['finish_reason'] != 'stop'):
                raise ValueError('invalid transport/model/termination')
            parsed = run.parse(raw['response']['choices'][0]['message']['content'], row)
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        if raw['valid'] != (parsed is not None):
            raise ValueError('raw first validity replay differs')
        if parsed is not None:
            found = parsed
    if observation != {'status': 'valid' if found is not None else 'invalid', 'value': found, 'attempts': attempts}:
        raise ValueError('first schema-valid response or preserved failure differs')
    return found


def main():
    if DEST.exists() or run.q.read(run.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0:
        raise ValueError('fresh audit and actual whole-tool completion required')
    run.native.vendor.corpus.quality.dependencies()
    for root in (run.PREP, run.PILOT, run.GENERATE):
        run.native.vendor.terminal(root)
    rows = run.select(run.q.read(run.data.DEST / 'cases.json'), run.q.read(run.old.PREP / 'cases.json'))
    if rows != run.q.read(run.PREP / 'cases.json'):
        raise ValueError('fixed metadata-only fresh64 selection differs')
    if run.q.read(run.PREP / 'public_proof.json') != {'public_file_sha256': run.qualification.reference.PUBLIC_SHA256,
            'exact_sources': run.qualification.reference.verify_public(rows)}:
        raise ValueError('exact public source proof differs')
    tokenizer, encoding = run.native.tokenizer(), run.data.storage_encoding()
    values = run.q.read(run.GENERATE / 'labels.json')
    if [r['id'] for r in values] != [r['id'] for r in rows]:
        raise ValueError('all64 first candidates required')
    for source, value in zip(rows, values):
        if any(source[k] != value[k] for k in source):
            raise ValueError('complete reserved source binding differs')
        draft = replay(source, 'draft', value['draft'], encoding)
        review = replay(source, 'review', value['review'], encoding, draft) if draft is not None else None
        valid = review is not None
        text = review['summary'] if valid else ''
        lexical = tokenizer.encode(text, add_special_tokens=False)
        stored = len(encoding.encode(text, disallowed_special=()))
        fits = bool(text.strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0, 101}
        words = [101, *lexical, 101] + [0] * (126 - len(lexical)) if fits else None
        if (value['target_text'] != text or value['teacher_valid'] != valid or value['capacity_fit'] != fits
                or value['stored_tokens'] != stored or value['native_lexical_tokens'] != len(lexical)
                or value['native_word_ids'] != words or value['semantic_qualified']):
            raise ValueError('final first candidate/capacity/word-ID replay differs')
    by_id = {r['id']: r for r in values}
    if any(by_id[r['id']] != r for r in run.q.read(run.PILOT / 'labels.json')):
        raise ValueError('original two-case pilot candidates changed')
    artifacts = {str(Path(__file__)): run.file_hash(Path(__file__))}
    for root in (run.WORKFLOW, run.PREP, run.PILOT, run.GENERATE):
        for path in root.rglob('*'):
            if path.is_file() and 'producer_source' not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    result = {'status': 'fresh64_first_draft_review_candidates_replayed_quality_pending', 'cases': 64,
        'teacher_valid': sum(r['teacher_valid'] for r in values), 'capacity_fit': sum(r['capacity_fit'] for r in values),
        'task_accuracy': None, 'student_training_admitted': False, 'semantic_qualification_pending': True,
        'canonical_semantic_cache_changed': False, 'artifacts': artifacts}
    DEST.mkdir()
    run.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
