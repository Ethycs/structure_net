"""Replay all10000 original-source bindings and first draft/review candidates."""
import json
from pathlib import Path

from experiments.structure_net import ncm_evidence_teacher_bulk as run
from experiments.neural_architecture_lab.audit_ncm_evidence_teacher_pilot import replay

DEST = run.ROOT / '20261024_evidence_teacher10000_raw_audit_v1'


def main():
    if DEST.exists() or run.q.read(run.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0:
        raise ValueError('fresh audit and actual whole bulk workflow completion required')
    run.recipe.configure()
    run.native.vendor.corpus.quality.dependencies()
    run.require_teacher()
    for root in (run.PREP, run.PILOT, run.GENERATE):
        run.native.vendor.terminal(root)
    rows = run.q.read(run.data.DEST / 'cases.json')
    values = run.q.read(run.GENERATE / 'labels.json')
    if (run.q.read(run.PREP / 'cases.json') != rows or len(values) != 10000
            or [r['id'] for r in rows] != [v['id'] for v in values]):
        raise ValueError('complete original ordered10000 TRAIN sources/candidates required')
    proof = run.qualification.reference.verify_public(rows)
    if run.q.read(run.PREP / 'public_proof.json') != {'public_file_sha256': run.qualification.reference.PUBLIC_SHA256, 'exact_sources': proof}:
        raise ValueError('complete exact public10000source proof differs')
    encoding, tokenizer = run.data.storage_encoding(), run.native.tokenizer()
    for source, value in zip(rows, values):
        if any(value[k] != v for k, v in source.items()):
            raise ValueError('full source metadata/component binding differs')
        draft = replay(source, 'draft', value['draft'], encoding)
        review = replay(source, 'review', value['review'], encoding, draft) if draft is not None else None
        text = review['summary'] if review is not None else ''
        lexical = tokenizer.encode(text, add_special_tokens=False)
        stored = len(encoding.encode(text, disallowed_special=()))
        fits = bool(text.strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0, 101}
        words = [101, *lexical, 101] + [0] * (126 - len(lexical)) if fits else None
        if (value['target_text'] != text or value['teacher_valid'] != (review is not None)
                or value['capacity_fit'] != fits or value['stored_tokens'] != stored
                or value['native_lexical_tokens'] != len(lexical) or value['native_word_ids'] != words
                or value['semantic_qualified']):
            raise ValueError('first final candidate/native words/capacity replay differs')
    by_id = {v['id']: v for v in values}
    retained = run.q.read(run.recipe.GENERATE / 'labels.json')
    pilot = run.q.read(run.PILOT / 'labels.json')
    if any(by_id[r['id']] != r for r in [*retained, *pilot]):
        raise ValueError('all64qualified firsts and8pilot firsts must remain exact, including failures')
    if run.q.read(run.PREP / 'pilot_ids.json') != run.pilot_ids(rows, {r['id']: r for r in retained}):
        raise ValueError('metadata-only8case systems selection differs')
    artifacts = {str(Path(__file__)): run.file_hash(Path(__file__))}
    for root in (run.WORKFLOW, run.PREP, run.PILOT, run.GENERATE):
        for path in root.rglob('*'):
            if path.is_file() and 'producer_source' not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    result = {'status': 'all10000_first_teacher_candidates_independently_replayed_student_quality_pending',
        'cases': 10000, 'valid': sum(v['teacher_valid'] for v in values), 'capacity_fit': sum(v['capacity_fit'] for v in values),
        'teacher_recipe_qualified': True, 'individual_label_semantic_quality_not_assumed': True,
        'all64_original_and8pilot_firsts_exact': True, 'canonical_cache_changed': False,
        'task_accuracy': None, 'student_trained': False, 'speed_tested': False, 'artifacts': artifacts}
    DEST.mkdir()
    run.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
