"""Preserve and replay all188 firsts after a single exhausted support-syntax case."""
import json
from pathlib import Path

from experiments.structure_net import ncm_sol_teacher_expansion as run
from experiments.neural_architecture_lab.audit_ncm_evidence_teacher_pilot import replay

DEST = run.ROOT / '20261024_sol_teacher188_partial_inventory_v1'


def main():
    run.configure()
    run.require_teacher()
    if DEST.exists() or run.q.read(run.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 1:
        raise ValueError('fresh inventory of actually stopped first collection required')
    if run.q.read(run.WORKFLOW / 'generate_receipt.json')['exit_code'] != 1:
        raise ValueError('actual failed generation stage required')
    for root in (run.PREP, run.PILOT):
        run.native.vendor.terminal(root)
    for root in (run.WORKFLOW, run.GENERATE):
        run.native.vendor.verify(run.q.read(root / 'producer.json'))
    if run.q.read(run.GENERATE / 'worker_terminal_receipt.json')['exit_code'] != 1:
        raise ValueError('drained failed child must be terminal')
    rows = run.sources()
    if rows != run.q.read(run.PREP / 'cases.json'):
        raise ValueError('unchanged original8819 source reservation required')
    by_id = {r['id']: r for r in rows}
    pilot = run.q.read(run.PILOT / 'labels.json')
    partial = run.q.read(run.GENERATE / 'partial_firsts.json')
    if (len(pilot) != 8 or len(partial) != 180 or len({r['id'] for r in (*pilot, *partial)}) != 188
            or {p.stem for p in (run.GENERATE / 'cases').glob('*.json')} != {r['id'] for r in partial}):
        raise ValueError('all eight pilot plus180 drained firsts required')
    progress = run.q.read(run.GENERATE / 'progress.json')
    if progress['pending_jobs'] != 0 or not progress['submissions_stopped'] or progress['completed_firsts'] != 180:
        raise ValueError('actual drained submission frontier required')
    values = {r['id']: r for r in (*pilot, *partial)}
    encoding, tokenizer = run.data.storage_encoding(), run.native.tokenizer()
    attempts, exhausted = set(), []
    for identity, value in values.items():
        row = by_id[identity]
        if any(value[k] != v for k, v in row.items()):
            raise ValueError('original complete source binding changed')
        draft = replay(row, 'draft', value['draft'], encoding)
        review = replay(row, 'review', value['review'], encoding, draft) if draft is not None else None
        text = review['summary'] if review is not None else ''
        lexical = tokenizer.encode(text, add_special_tokens=False)
        stored = len(encoding.encode(text, disallowed_special=()))
        fit = bool(text.strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0,101}
        expected = {'target_text': text, 'teacher_valid': review is not None,
            'capacity_fit': fit, 'stored_tokens': stored, 'native_lexical_tokens': len(lexical),
            'native_word_ids': [101,*lexical,101]+[0]*(126-len(lexical)) if fit else None,
            'semantic_qualified': False}
        if any(value[k] != v for k,v in expected.items()):
            raise ValueError('raw first summary, validity or capacity changed')
        owner = run.PILOT if identity in {r['id'] for r in pilot} else run.GENERATE
        if run.q.read(owner / 'cases' / (identity+'.json')) != value:
            raise ValueError('saved first differs from pipeline result')
        for stage in ('draft','review'):
            observation = value[stage]
            if observation is None:
                continue
            expected_paths = [owner/'requests'/identity/stage/f'attempt_{i}.json' for i in range(len(observation['attempts']))]
            if [Path(p) for p in observation['attempts']] != expected_paths or sorted(expected_paths[0].parent.glob('attempt_*.json')) != expected_paths:
                raise ValueError('raw first attempt order or allowance changed')
            attempts.update(observation['attempts'])
            if observation['status'] == 'invalid':
                if len(observation['attempts']) != 3:
                    raise ValueError('exhausted first cannot receive a new allowance')
                exhausted.append({'id':identity,'stage':stage,'attempts':observation['attempts']})
    actual_attempts = {str(p) for root in (run.PILOT,run.GENERATE) for p in (root/'requests').rglob('attempt_*.json')}
    if attempts != actual_attempts or len(exhausted) != 1:
        raise ValueError('every raw attempt and the single exhausted first must remain')
    observed = [values[r['id']] for r in rows if r['id'] in values]
    remaining = [r['id'] for r in rows if r['id'] not in values]
    if len(remaining) != 8631:
        raise ValueError('exact remaining8631 unattempted IDs required')
    inputs = [Path(__file__), run.WORKFLOW/'actual_tool_receipt.json', run.WORKFLOW/'generate_receipt.json',
        run.PREP/'cases.json', run.PILOT/'labels.json', run.GENERATE/'partial_firsts.json',
        run.GENERATE/'progress.json', run.GENERATE/'worker_terminal_receipt.json',
        *[Path(p) for p in sorted(attempts)], *[p for root in (run.PILOT,run.GENERATE) for p in (root/'cases').glob('*.json')]]
    producer = run.native.vendor.freeze(DEST, 'partial188_firsts_replay_after_actual1', inputs,
        {'data_only':True,'all188firsts_preserved':True,'invalid_first_not_rerolled':True,
            'new_requests':0,'canonical_semantic_cache_changed':False,'student_training_admitted':False,'final_test_opened':False})
    run.write_json(DEST/'first_candidates.json', observed)
    run.write_json(DEST/'unattempted_ids.json', remaining)
    run.write_json(DEST/'exhausted_firsts.json', exhausted)
    run.native.vendor.verify(producer)
    result = {'status':'all188_completed_firsts_and_single_exhausted_pipeline_replayed', 'observed':188,
        'teacher_valid':sum(r['teacher_valid'] for r in observed), 'capacity_fit':sum(r['capacity_fit'] for r in observed),
        'invalid_firsts':sum(not r['teacher_valid'] for r in observed),'unattempted':8631,
        'all_inflight_drained':True,'new_requests':0,'student_training_admitted':False,
        'canonical_semantic_cache_changed':False,'final_test_opened':False,
        'artifacts':{str(p):run.file_hash(p) for p in [*inputs,DEST/'producer.json',DEST/'first_candidates.json',DEST/'unattempted_ids.json',DEST/'exhausted_firsts.json']}}
    run.write_json(DEST/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


if __name__ == '__main__':
    main()
