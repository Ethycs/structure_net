"""Replay interrupted teacher firsts from raw files; never make a model request."""
import json
from pathlib import Path

from experiments.structure_net import ncm_evidence_teacher_bulk as run
from experiments.neural_architecture_lab.audit_ncm_evidence_teacher_pilot import replay

DEST = run.ROOT / '20261024_teacher10000_partial_firsts_inventory_v1'
INTERRUPT = run.ROOT / '20261024_teacher10000_quota_interruption_v1'
JOURNAL = run.ROOT / '20261024_evidence_teacher10000_followthrough_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-evidence-teacher-quota-interruption.md')
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-teacher-quota-interruption.md')
TEST = Path('tests/structure_net/test_ncm_partial_teacher_inventory.py')


def descriptor(row, stage, root):
    paths = sorted((root / 'requests' / row['id'] / stage).glob('attempt_*.json'))
    if not paths:
        return None
    if len(paths) > 3 or [p.name for p in paths] != [f'attempt_{i}.json' for i in range(len(paths))]:
        raise ValueError('original ordered maximum-three stage attempts required')
    value = None
    for index, path in enumerate(paths):
        raw = run.q.read(path)
        if raw['valid']:
            if index != len(paths) - 1:
                raise ValueError('later attempt after valid first response')
            value = run.original.parse(raw['response']['choices'][0]['message']['content'], row)
    if value is None and len(paths) < 3:
        return None  # unfinished; preserve it, never invent an exhausted failure
    return {'status': 'valid' if value is not None else 'invalid', 'value': value, 'attempts': [str(p) for p in paths]}


def candidate(row, encoding, tokenizer):
    draft = descriptor(row, 'draft', run.GENERATE)
    if draft is None:
        return None
    first_draft = replay(row, 'draft', draft, encoding)
    review = descriptor(row, 'review', run.GENERATE) if first_draft is not None else None
    if first_draft is not None and review is None:
        return None
    first_review = replay(row, 'review', review, encoding, first_draft) if review is not None else None
    text = first_review['summary'] if first_review is not None else ''
    lexical = tokenizer.encode(text, add_special_tokens=False)
    stored = len(encoding.encode(text, disallowed_special=()))
    fits = bool(text.strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0, 101}
    return {**row, 'draft': draft, 'review': review, 'target_text': text,
        'teacher_valid': first_review is not None, 'stored_tokens': stored, 'native_lexical_tokens': len(lexical),
        'capacity_fit': fits, 'native_word_ids': [101, *lexical, 101] + [0] * (126 - len(lexical)) if fits else None,
        'semantic_qualified': False}


def main():
    if (DEST.exists() or REPORT.exists() or run.q.read(JOURNAL / 'bulk_actual_tool_result.json')['exit_code'] != 1
            or run.q.read(run.WORKFLOW / 'generate_receipt.json')['exit_code'] not in (-2, 130)):
        raise ValueError('actual interrupted terminal teacher producer and fresh inventory required')
    run.recipe.configure()
    run.native.vendor.corpus.quality.dependencies()
    rows = run.q.read(run.PREP / 'cases.json')
    retained = [*run.q.read(run.PREP / 'retained64.json'), *run.q.read(run.PILOT / 'labels.json')]
    if len(rows) != 10000 or len(retained) != 72 or len({r['id'] for r in retained}) != 72:
        raise ValueError('original10000source universe and72distinct retained firsts required')
    original = run.q.read(run.data.DEST / 'cases.json')
    if rows != original:
        raise ValueError('reserved source population changed')
    raw_ids = {p.name for p in (run.GENERATE / 'requests').iterdir() if p.is_dir()}
    saved_ids = {p.stem for p in (run.GENERATE / 'cases').glob('*.json')}
    ids = {r['id'] for r in rows}
    if not saved_ids <= raw_ids or not raw_ids <= ids or raw_ids & {r['id'] for r in retained}:
        raise ValueError('raw/saved/retained source identities overlap or leave reserved population')
    inputs = [Path(__file__), PLAN, TEST, Path(run.__file__), Path(run.recipe.__file__),
        Path(run.original.__file__), Path(replay.__code__.co_filename), run.PREP / 'cases.json',
        run.PREP / 'retained64.json', run.PILOT / 'labels.json', run.data.DEST / 'cases.json',
        JOURNAL / 'bulk_actual_tool_result.json', run.WORKFLOW / 'generate_receipt.json',
        INTERRUPT / 'before_signal.json', INTERRUPT / 'signal_sent.json']
    inputs += sorted((run.GENERATE / 'requests').rglob('attempt_*.json'))
    inputs += sorted((run.GENERATE / 'cases').glob('*.json'))
    p = run.native.vendor.freeze(DEST, 'read_only_first_candidate_replay_after_external_quota_interruption', inputs,
        {'source_population': 10000, 'no_network_requests': True, 'no_label_rerolls': True,
            'all_firsts_and_invalids_preserved': True, 'student_training_admitted': False,
            'canonical_firsts_changed': False, 'final_sources_opened_or_hashed': False})
    tokenizer, encoding = run.native.tokenizer(), run.data.storage_encoding()
    observed = {r['id']: r for r in retained}
    unfinished, recovered = [], []
    for row in rows:
        if row['id'] not in raw_ids:
            continue
        value = candidate(row, encoding, tokenizer)
        if value is None:
            unfinished.append(row['id'])
            continue
        if row['id'] in saved_ids:
            if run.q.read(run.GENERATE / 'cases' / (row['id'] + '.json')) != value:
                raise ValueError('reconstructed first differs from original saved candidate')
        else:
            recovered.append(row['id'])
        observed[row['id']] = value
    if not saved_ids <= set(observed):
        raise ValueError('an original saved candidate was lost')
    candidates = [observed[r['id']] for r in rows if r['id'] in observed]
    unattempted = [r['id'] for r in rows if r['id'] not in raw_ids and r['id'] not in observed]
    if len(candidates) + len(unfinished) + len(unattempted) != 10000:
        raise ValueError('complete original10000denominator required')
    run.write_json(DEST / 'first_candidates.json', candidates)
    run.write_json(DEST / 'unfinished_ids.json', unfinished)
    run.write_json(DEST / 'unattempted_ids.json', unattempted)
    run.native.vendor.verify(p)
    result = {'status': 'interrupted_teacher_firsts_replayed_no_new_requests', 'original_source_population': 10000,
        'original_saved_new_candidates': len(saved_ids), 'retained_firsts': 72,
        'recovered_raw_only_firsts': len(recovered), 'recovered_ids': recovered,
        'completed_observed_firsts': len(candidates), 'unfinished_sources': len(unfinished),
        'unattempted_sources': len(unattempted), 'schema_valid_firsts': sum(r['teacher_valid'] for r in candidates),
        'capacity_fit_firsts': sum(r['capacity_fit'] for r in candidates),
        'invalid_firsts_preserved': sum(not r['teacher_valid'] for r in candidates),
        'new_requests': 0, 'new_semantic_judgments': 0, 'student_training_admitted': False,
        'canonical_firsts_changed': False, 'final_sources_opened_or_hashed': False,
        'first_candidates_sha256': run.file_hash(DEST / 'first_candidates.json')}
    run.write_json(DEST / 'result.json', result)
    REPORT.write_text('# Teacher quota interruption: preserved first candidates\n\n'
        'The Sonnet reviewer returned a weekly-limit message (resetOct26at4pmAmerica/Los_Angeles). '
        'The bulk generator was interrupted after existing requests drained; its actualwhole-tool exit was1, childSIGINT(-2). '
        'This is an external resource interruption, not a semantic failure. No old artifact or valid reply was replaced.\n\n'
        + f"Read-only replay verified{len(saved_ids)}saved new candidates, retained72earlierfirsts and recovered{len(recovered)}"
        ' additional firsts from completed raw tasks that the interrupted parent had not saved. '
        + f"Total observed firsts:{len(candidates)}; schema-valid:{result['schema_valid_firsts']}; capacity-fit:{result['capacity_fit_firsts']}. "
        + f"Invalid firsts retained:{result['invalid_firsts_preserved']}; unfinished:{len(unfinished)}; unattempted:{len(unattempted)}. "
        'All10,000reserved sources remain accounted for. No new request, semantic judging or retry was performed.\n\n'
        + f"Inventory `{DEST / 'result.json'}`; candidateSHA256 `{result['first_candidates_sha256']}`. "
        'This is not a completed10,000label dataset and admits no new student training or95%claim. '
        'Resuming unattempted or unfinished work requires a fresh prospective recovery protocol; exhausted original stages do not receive a new attempt allowance. '
        'Canonical3222judgments and final320remain unchanged. GPUfeature extraction and the independentFP16student trial continue.\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
