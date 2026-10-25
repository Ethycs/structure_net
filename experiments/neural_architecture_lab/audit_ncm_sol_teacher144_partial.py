"""Replay the drained teacher outage without rebinding completed producers."""
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_sol_teacher_continuation as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = run.ROOT / '20261025_sol_teacher144_partial_inventory_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-teacher-outage-inventory-plan.md')
TEST = Path('tests/structure_net/test_ncm_sol_teacher144_partial.py')


def replay_stage(row, stage, observation, encoding, models, systems, draft=None):
    """Reconstruct the frozen request and first valid result, never send it."""
    original = run.previous.original
    content = {'source_lines': original.source_lines(row)}
    if stage == 'review' and draft is not None:
        content['untrusted_draft'] = draft
    elif stage != 'draft' or draft is not None:
        raise ValueError('exact draft then review contract required')
    body = {'model': models[stage], 'temperature': 0,
        'max_tokens': 768 if stage == 'draft' else 512,
        'messages': [{'role': 'system', 'content': systems[stage]},
            {'role': 'user', 'content': json.dumps(content, ensure_ascii=False)}]}
    tokens = len(encoding.encode(json.dumps(body['messages'], ensure_ascii=False), disallowed_special=())) + 16
    paths, found = observation['attempts'], None
    if not 1 <= len(paths) <= 3 or tokens > 6300:
        raise ValueError('original bounded attempt and prompt allowances required')
    for i, name in enumerate(paths):
        path = Path(name)
        if path.name != f'attempt_{i}.json' or found is not None:
            raise ValueError('ordered attempts stop at the first valid response')
        raw = run.q.read(path)
        if raw['request'] != body or raw['prompt_tokens'] != tokens:
            raise ValueError('exact original request or token binding differs')
        parsed = None
        try:
            if (raw.get('http_status') != 200 or raw['response'].get('model') != models[stage]
                    or raw['response']['choices'][0]['finish_reason'] != 'stop'):
                raise ValueError('invalid transport, identity or termination')
            parsed = run.previous.teacher.base.recipe.parse(
                raw['response']['choices'][0]['message']['content'], row)
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        if raw['valid'] != (parsed is not None):
            raise ValueError('raw response validity differs on independent replay')
        if parsed is not None:
            found = parsed
    if (observation != {'status': 'valid' if found is not None else 'invalid',
            'value': found, 'attempts': paths} or (found is None and len(paths) != 3)):
        raise ValueError('the first result or exhausted allowance differs')
    return found


def main():
    if DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only audit required')
    run.native.vendor.corpus.quality.dependencies()
    receipts = [run.WORKFLOW / n for n in ('actual_tool_receipt.json', 'generate_receipt.json')]
    if (run.q.read(receipts[0])['actual_tool_exit_code'] != 1
            or run.q.read(receipts[1])['exit_code'] != 1
            or run.q.read(run.GENERATE / 'worker_terminal_receipt.json')['exit_code'] != 1):
        raise ValueError('actually failed terminal workflow and worker required')
    run.native.vendor.terminal(run.PREP)
    for root in (run.WORKFLOW, run.GENERATE):
        run.native.vendor.verify(run.q.read(root / 'producer.json'))
    config = run.q.read(run.GENERATE / 'producer.json')['config']
    models, systems = config['models'], config['systems']
    if models != run.previous.teacher.MODELS or systems != run.previous.original.SYSTEMS:
        raise ValueError('unchanged previously qualified recipe required')
    rows = run.sources()
    partial = run.q.read(run.GENERATE / 'partial_firsts.json')
    progress = run.q.read(run.GENERATE / 'progress.json')
    failures = run.q.read(run.GENERATE / 'failures.json')
    if (rows != run.q.read(run.PREP / 'cases.json') or len(rows) != 8631
            or len(partial) != 144 or len({r['id'] for r in partial}) != 144
            or progress['pending_jobs'] != 0 or not progress['submissions_stopped']
            or progress['completed_firsts'] != 144 or progress['population'] != 8631
            or progress['failures'] != failures
            or {p.stem for p in (run.GENERATE / 'cases').glob('*.json')} != {r['id'] for r in partial}):
        raise ValueError('all144 drained firsts and terminal submission frontier required')
    by_id = {r['id']: r for r in rows}
    values = {v['id']: v for v in partial}
    if partial != [values[r['id']] for r in rows if r['id'] in values]:
        raise ValueError('original source order must be preserved')
    encoding, tokenizer = run.previous.data.storage_encoding(), run.native.tokenizer()
    attempts, exhausted, schema, fatal = set(), [], [], []
    for identity, value in values.items():
        row = by_id[identity]
        if any(value[k] != v for k, v in row.items()):
            raise ValueError('complete original source binding differs')
        draft = replay_stage(row, 'draft', value['draft'], encoding, models, systems)
        if draft is None and value['review'] is not None:
            raise ValueError('review cannot follow exhausted draft')
        review = replay_stage(row, 'review', value['review'], encoding, models, systems, draft) if draft is not None else None
        text = review['summary'] if review is not None else ''
        lexical = tokenizer.encode(text, add_special_tokens=False)
        stored = len(encoding.encode(text, disallowed_special=()))
        fit = bool(text.strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0, 101}
        expected = {'target_text': text, 'teacher_valid': review is not None, 'capacity_fit': fit,
            'stored_tokens': stored, 'native_lexical_tokens': len(lexical), 'semantic_qualified': False,
            'native_word_ids': [101, *lexical, 101] + [0] * (126-len(lexical)) if fit else None}
        if (any(value[k] != v for k, v in expected.items())
                or run.q.read(run.GENERATE / 'cases' / (identity + '.json')) != value):
            raise ValueError('saved first, target or capacity binding differs')
        if not value['teacher_valid']:
            (fatal if run.fatal_exhaustion(value) else schema).append(identity)
        for stage in ('draft', 'review'):
            observation = value[stage]
            if observation is None:
                continue
            paths = [run.GENERATE / 'requests' / identity / stage / f'attempt_{i}.json'
                for i in range(len(observation['attempts']))]
            if [Path(p) for p in observation['attempts']] != paths or sorted(paths[0].parent.glob('attempt_*.json')) != paths:
                raise ValueError('original ordered physical stage attempts required')
            attempts.update(str(p) for p in paths)
            if observation['status'] == 'invalid':
                exhausted.append({'id': identity, 'stage': stage, 'attempts': [str(p) for p in paths],
                    'fatal_exhaustion': identity in fatal})
    actual = {str(p) for p in (run.GENERATE / 'requests').rglob('attempt_*.json')}
    if (actual != attempts or len(exhausted) != 9 or len(schema) != 1 or len(fatal) != 8
            or set(schema) != set(failures['schema_exhausted']) or set(fatal) != set(failures['fatal'])):
        raise ValueError('all attempts and all nine original failures must remain')
    remaining = [r['id'] for r in rows if r['id'] not in values]
    if len(remaining) != 8487:
        raise ValueError('exact unattempted frontier differs')
    files = [Path(__file__), PLAN, TEST, Path(run.__file__), Path(run.previous.__file__),
        Path(run.previous.original.__file__), Path(run.previous.teacher.base.recipe.__file__),
        Path(run.previous.teacher.base.recipe.recovery.__file__)]
    inputs = [*files, *receipts, run.WORKFLOW / 'producer.json',
        run.PREP / 'producer.json', run.PREP / 'cases.json', run.PREP / 'worker_terminal_receipt.json',
        *[run.GENERATE / n for n in ('producer.json', 'partial_firsts.json', 'progress.json', 'failures.json', 'worker_terminal_receipt.json')],
        *[Path(p) for p in sorted(attempts)], *sorted((run.GENERATE / 'cases').glob('*.json'))]
    producer = run.native.vendor.freeze(DEST, 'replay144_drained_firsts_after_transport_outage', inputs,
        {'new_requests': 0, 'completed_producer_globals_rebound': False, 'invalid_firsts_rerolled': False,
         'student_training_admitted': False, 'task_accuracy': None, 'final_test_opened': False})
    write_json(DEST / 'first_candidates.json', partial)
    write_json(DEST / 'unattempted_ids.json', remaining)
    write_json(DEST / 'exhausted_firsts.json', exhausted)
    run.native.vendor.verify(producer)
    result = {'status': 'all144_drained_teacher_firsts_independently_replayed',
        'observed': 144, 'teacher_valid': sum(v['teacher_valid'] for v in partial),
        'capacity_fit': sum(v['capacity_fit'] for v in partial), 'invalid_firsts': 9,
        'schema_exhausted_firsts': len(schema), 'fatal_exhausted_firsts': len(fatal),
        'raw_attempts': len(attempts), 'unattempted': len(remaining), 'all_inflight_drained': True,
        'prior_observed_firsts_preserved': 1369, 'total_observed_firsts': 1513,
        'total_preserved_invalid_firsts': 73, 'new_requests': 0,
        'student_training_admitted': False, 'semantic_accuracy_measured': False, 'task_accuracy': None,
        'canonical_semantic_cache_changed': False, 'final_test_opened': False,
        'artifacts': {str(p): file_hash(p) for p in [*inputs, DEST / 'producer.json',
            DEST / 'first_candidates.json', DEST / 'unattempted_ids.json', DEST / 'exhausted_firsts.json']}}
    write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
