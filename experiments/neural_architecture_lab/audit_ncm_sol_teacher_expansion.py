"""Replay every unchanged first target and source binding in the8819 collection."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_sol_teacher_expansion as run
from experiments.neural_architecture_lab.audit_ncm_evidence_teacher_pilot import replay
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = run.ROOT / '20261024_sol_teacher8819_raw_audit_v1'
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-sol-teacher-expansion.md')


def main():
    run.configure()
    run.require_teacher()
    if DEST.exists() or REPORT.exists() or run.q.read(run.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('fresh independent audit after actual whole collection exit0 required')
    for root in (run.PREP, run.PILOT, run.GENERATE):
        run.native.vendor.terminal(root)
    run.native.vendor.verify(run.q.read(run.WORKFLOW / 'producer.json'))
    rows = run.sources()
    if rows != run.q.read(run.PREP / 'cases.json') or run.pilot_ids(rows) != run.q.read(run.PREP / 'pilot_ids.json'):
        raise ValueError('complete original reservation and metadata pilot must remain')
    reference = run.qualification.reference
    if run.q.read(run.PREP / 'public_proof.json') != {'public_file_sha256': reference.PUBLIC_SHA256,
            'exact_sources': reference.verify_public(rows)}:
        raise ValueError('complete unchanged public source binding differs')
    labels = run.q.read(run.GENERATE / 'labels.json')
    if len(labels) != 8819 or [r['id'] for r in labels] != [r['id'] for r in rows]:
        raise ValueError('all8819 ordered first targets required')
    pilot = {r['id']: r for r in run.q.read(run.PILOT / 'labels.json')}
    if set(pilot) != set(run.pilot_ids(rows)) or any(r != next(v for v in labels if v['id'] == key) for key, r in pilot.items()):
        raise ValueError('all eight first pilot candidates must remain unchanged')
    encoding, tokenizer = run.data.storage_encoding(), run.native.tokenizer()
    paths = set()
    for source, value in zip(rows, labels):
        if any(value[k] != v for k, v in source.items()):
            raise ValueError('an original complete source changed')
        draft = replay(source, 'draft', value['draft'], encoding)
        if draft is None or value['review'] is None:
            raise ValueError('complete collection cannot contain exhausted first pipelines')
        review = replay(source, 'review', value['review'], encoding, draft)
        if review is None:
            raise ValueError('complete collection cannot contain exhausted review')
        owner = run.PILOT if source['id'] in pilot else run.GENERATE
        for stage in ('draft', 'review'):
            attempts = value[stage]['attempts']
            expected = [owner / 'requests' / source['id'] / stage / f'attempt_{i}.json' for i in range(len(attempts))]
            if [Path(p) for p in attempts] != expected or sorted(expected[0].parent.glob('attempt_*.json')) != expected:
                raise ValueError('a first attempt was replaced, omitted or given a new allowance')
            paths.update(attempts)
        text = review['summary']
        lexical = tokenizer.encode(text, add_special_tokens=False)
        count = len(encoding.encode(text, disallowed_special=()))
        fit = bool(text.strip()) and 0 < count <= 128 and len(lexical) <= 126 and not set(lexical) & {0, 101}
        words = [101, *lexical, 101] + [0] * (126 - len(lexical)) if fit else None
        expected = {'target_text': text, 'teacher_valid': True, 'stored_tokens': count,
            'native_lexical_tokens': len(lexical), 'capacity_fit': fit, 'native_word_ids': words,
            'semantic_qualified': False}
        if any(value[k] != v for k, v in expected.items()) or run.q.read(owner / 'cases' / (source['id'] + '.json')) != value:
            raise ValueError('unaltered first summary or complete native capacity replay differs')
    request_paths = {str(p) for root in (run.PILOT, run.GENERATE) for p in (root / 'requests').rglob('attempt_*.json')}
    if paths != request_paths:
        raise ValueError('every raw attempt must belong to an unchanged first pipeline')
    values = run.q.read(run.GENERATE / 'result.json')
    if (values['cases'] != 8819 or values['valid'] != 8819 or values['retained_pilot_firsts'] != 8
            or values['new_candidates'] != 8811 or values['labels_sha256'] != run.file_hash(run.GENERATE / 'labels.json')
            or values['capacity_fit'] != sum(r['capacity_fit'] for r in labels)
            or values['student_training_admitted'] or values['canonical_semantic_cache_changed']):
        raise ValueError('complete first-target population and measured counts differ')
    artifacts = {str(Path(__file__)): run.file_hash(__file__)}
    for root in (run.WORKFLOW, run.PREP, run.PILOT, run.GENERATE):
        for path in root.rglob('*'):
            if path.is_file() and 'producer_source' not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    campaign = {'hypothesis_id': run.HYPOTHESIS, 'classification': 'all8819_first_targets_source_payload_and_capacity_replayed',
        'cases': 8819, 'teacher_valid': 8819, 'capacity_fit': values['capacity_fit'],
        'original1181firsts_preserved': True, 'old63invalids_not_retried': True,
        'component64_excluded_from_future_gradients': True, 'student_training_admitted': False,
        'individual_semantic_quality_not_measured': True, 'student_task_accuracy': None,
        'canonical_semantic_cache_changed': False, 'final_test_opened': False, 'speed_tested': False,
        'confirmed': False, 'artifacts': artifacts}
    DEST.mkdir()
    run.write_json(DEST / 'campaign_results.json', campaign)
    REPORT.write_text('# Remaining TRAIN teacher first targets\n\n'
        f"Classification **{campaign['classification']}**. All8819new first pipelines completed, including the retained8casepilot. "
        f"{values['capacity_fit']}/8819fit both unchanged target bounds; all capacity failures remain stored.\n\n"
        'The qualified Terra-draft/Sol-review recipe and complete original sources were unchanged. '
        'An independent CPU audit replayed every raw prompt, first eligible response and native frame. '
        'Original1117firsts,63exhausted firsts and the newer64component cases were never rerolled. '
        'This is data collection, not a measure of student accuracy or individual label semantics. '
        'Future training must exclude component64and use a separate frozen training protocol. '
        'Permanent DEV and final320boundaries remain unchanged; final sources remain unopened and unhashed.\n')
    print(json.dumps({k: v for k, v in campaign.items() if k != 'artifacts'}), flush=True)


def register():
    if run.q.read(DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed independent CPU replay required')
    path = DEST / 'campaign_results.json'
    state = run.q.read(path)
    register_records(path, REPORT, DEST, {run.HYPOTHESIS: {
        'name': 'Remaining8819 TRAIN first teacher targets',
        'description': 'Unchanged jointly qualified Terra draft and Sol review on only reserved unattempted TRAIN sources; bounded8workers with failure-stop; every raw first replayed;1181oldfirsts preserved; no student training or semantic accuracy claim.',
        'classification': state['classification'], 'data_preparation_only': True,
        'student_task_accuracy_measured': False, 'cases': 8819}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('audit', 'register'))
    args = parser.parse_args()
    {'audit': main, 'register': register}[args.stage]()
