"""Fresh TRAIN-only qualification candidates for Terra draft / Sol review."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_evidence_teacher_pilot_v3 as recipe
from experiments.structure_net import ncm_sol_judgment_lineage as lineage
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.neural_architecture_lab.audit_ncm_evidence_teacher_pilot import replay

original, q, ROOT = recipe.original, recipe.q, recipe.ROOT
native, data = original.native, original.data
HYPOTHESIS = 'ncm-sol-evidence-first-fresh64-teacher-generation-v1'
MODULE = 'experiments.structure_net.ncm_sol_evidence_teacher'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol-teacher64-plan.md')
TEST = Path('tests/structure_net/test_ncm_sol_evidence_teacher.py')
PREP = ROOT / '20261024_sol_teacher64_prepare_v1'
PILOT = ROOT / '20261024_sol_teacher64_systems_pilot_v1'
GENERATE = ROOT / '20261024_sol_teacher64_candidates_v1'
WORKFLOW = ROOT / '20261024_sol_teacher64_workflow_v1'
AUDIT = ROOT / '20261024_sol_teacher64_raw_audit_v1'
INVENTORY = ROOT / '20261024_teacher10000_partial_firsts_inventory_v1'
FIRSTS_SHA = 'ebbeb4bf90c1c1f7ac7ef36358bc221d82dfafe9f3c43d3c6c1a853d21f47bfd'
QUOTAS = {0: 32, 1: 8, 2: 8, 3: 8, 4: 8}
MODELS = {'draft': lineage.SECONDARY, 'review': lineage.qualification.MODEL}


def configure():
    lineage.require_judges()
    native.vendor.corpus.quality.dependencies()
    original.MODELS = dict(MODELS)
    original.parse = recipe.parse


def select(rows, unattempted, firsts):
    if (len(rows) != 10000 or len({r['id'] for r in rows}) != 10000
            or len(unattempted) != 8883 or len(set(unattempted)) != 8883
            or len(firsts) != 1117 or len({r['id'] for r in firsts}) != 1117):
        raise ValueError('complete original10000 =1117firsts+8883unattempted required')
    untouched = set(unattempted)
    observed = {r['id'] for r in firsts}
    if observed & untouched or observed | untouched != {r['id'] for r in rows}:
        raise ValueError('original observed and unattempted populations must partition10000')
    blocked = {r['source_component'] for r in firsts}
    result = []
    for band, count in QUOTAS.items():
        values = sorted((r for r in rows if r['id'] in untouched and r['bin'] == band
            and r['source_component'] not in blocked),
            key=lambda r: (text_hash(HYPOTHESIS + ':case:' + r['id']), r['id']))
        if len(values) < count:
            raise ValueError('fresh complete64 source bin quotas unavailable')
        result.extend(values[:count])
    if len(result) != 64 or len({r['source_component'] for r in result}) != 64:
        raise ValueError('64 fresh distinct TRAIN source components required')
    return result


def pilot_ids(rows):
    selected = [rows[0]['id'], min(rows, key=lambda r: (-r['source_cl100k_tokens'], r['id']))['id']]
    selected += [r['id'] for r in rows if r['id'] not in selected][:6]
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError('eight distinct first/longest concurrency cases required')
    return selected


def inventory_inputs():
    return [INVENTORY / n for n in ('first_candidates.json', 'unattempted_ids.json', 'unfinished_ids.json',
        'result.json', 'actual_tool_receipt.json')]


def producing(root, stage, inputs=()):
    files = [Path(__file__), PLAN, TEST, Path(recipe.__file__), Path(recipe.recovery.__file__),
        Path(original.__file__), Path(lineage.__file__), Path(replay.__code__.co_filename),
        lineage.qualification.PLAN, Path(lineage.qualification.__file__)]
    p = native.vendor.freeze(root, stage, [*inputs, *inventory_inputs(), *lineage.judge_inputs(), *files],
        {'hypothesis_id': HYPOTHESIS, 'models': MODELS, 'systems': original.SYSTEMS,
         'single_source_calls': True, 'draft_max_tokens': 768, 'review_max_tokens': 512,
         'max_prompt_tokens': 6300, 'workers': 8, 'max_attempts': 3, 'semantic_retries': False,
         'no_capacity_retries': True, 'original_firsts_preserved': 1117, 'unattempted_before_selection': 8883,
         'qualification_cases': 64, 'quotas': QUOTAS, 'student_training_admitted': False,
         'semantic_qualification_pending': True, 'canonical_semantic_cache_changed': False,
         'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'sol_teacher_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def sources():
    if (file_hash(data.DEST / 'cases.json') != original.old.SOURCE_SHA
            or file_hash(INVENTORY / 'first_candidates.json') != FIRSTS_SHA
            or q.read(INVENTORY / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or q.read(INVENTORY / 'unfinished_ids.json')):
        raise ValueError('audited complete reserved sources and preserved original firsts required')
    return select(q.read(data.DEST / 'cases.json'), q.read(INVENTORY / 'unattempted_ids.json'),
        q.read(INVENTORY / 'first_candidates.json'))


def prepare():
    configure()
    rows = sources()
    public = original.qualification.reference
    proof = public.verify_public(rows)
    p = producing(PREP, 'metadata_only_fresh64_unattempted_TRAIN_selection', [data.DEST / 'cases.json', public.PUBLIC_FILE])
    write_json(PREP / 'cases.json', rows)
    write_json(PREP / 'pilot_ids.json', pilot_ids(rows))
    write_json(PREP / 'public_proof.json', {'public_file_sha256': public.PUBLIC_SHA256, 'exact_sources': proof})
    native.vendor.verify(p)
    write_json(PREP / 'result.json', {'status': 'fresh64_sol_teacher_sources_prepared', 'cases': 64,
        'cases_sha256': file_hash(PREP / 'cases.json'), 'task_accuracy': None})


def generate(pilot=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU gateway workflow only')
    configure()
    native.vendor.terminal(PREP)
    all_rows = q.read(PREP / 'cases.json')
    ids = q.read(PREP / 'pilot_ids.json')
    prior = {}
    inputs = [PREP / n for n in ('cases.json', 'pilot_ids.json', 'public_proof.json', 'worker_terminal_receipt.json')]
    if not pilot:
        native.vendor.terminal(PILOT)
        if q.read(PILOT / 'result.json')['status'] != 'eight_case_sol_teacher_systems_passed':
            raise ValueError('actual eight-worker systems pilot required')
        prior = {r['id']: r for r in q.read(PILOT / 'labels.json')}
        inputs += [PILOT / n for n in ('labels.json', 'result.json', 'worker_terminal_receipt.json')]
    rows = [r for r in all_rows if r['id'] in ids] if pilot else [r for r in all_rows if r['id'] not in prior]
    root = PILOT if pilot else GENERATE
    p = producing(root, 'eight_worker_systems_only' if pilot else 'fresh64_first_candidates_semantic_quality_pending', inputs)
    encoding, tokenizer = data.storage_encoding(), native.tokenizer()
    generated = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(original.candidate, row, root, encoding, tokenizer) for row in rows]
        for future in as_completed(jobs):
            value = future.result()
            generated[value['id']] = value
            write_json(root / 'cases' / (value['id'] + '.json'), value)
            print(json.dumps({'completed': len(generated), 'new_total': len(rows), 'pilot': pilot}), flush=True)
    labels = [prior[r['id']] if r['id'] in prior else generated[r['id']] for r in (rows if pilot else all_rows)]
    write_json(root / 'labels.json', labels)
    valid, capacity = sum(r['teacher_valid'] for r in labels), sum(r['capacity_fit'] for r in labels)
    status = ('eight_case_sol_teacher_systems_passed' if valid == capacity == 8 else 'eight_case_sol_teacher_systems_failed') if pilot else 'fresh64_sol_teacher_candidates_completed_quality_pending'
    native.vendor.verify(p)
    result = {'status': status, 'cases': len(labels), 'valid': valid, 'capacity_fit': capacity,
        'labels_sha256': file_hash(root / 'labels.json'), 'student_training_admitted': False,
        'semantic_qualification_pending': True, 'canonical_semantic_cache_changed': False, 'task_accuracy': None}
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)
    if pilot and (valid != 8 or capacity != 8):
        raise ValueError('systems failure preserved; no full generation admission')


def workflow():
    configure()
    if any(p.exists() for p in (WORKFLOW, PREP, PILOT, GENERATE, AUDIT)):
        raise ValueError('fresh Sol teacher roots required')
    p = producing(WORKFLOW, 'supervise_fresh64_sol_teacher_candidates')
    receipts = []
    for stage, root in (('prepare', PREP), ('pilot', PILOT), ('generate', GENERATE)):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', MODULE, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'command': command, 'child_pid': child.pid,
            'observer_pid': os.getpid(), 'observer': 'subprocess.wait', 'exit_code': code}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('Sol teacher stage failed: ' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'fresh64_sol_teacher_workflow_completed_quality_pending', 'stages': receipts})


def audit():
    configure()
    if AUDIT.exists() or q.read(WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('fresh audit after actual whole-tool completion required')
    for root in (WORKFLOW, PREP, PILOT, GENERATE):
        native.vendor.verify(q.read(root / 'producer.json'))
    for root in (PREP, PILOT, GENERATE):
        native.vendor.terminal(root)
    rows = sources()
    if rows != q.read(PREP / 'cases.json') or pilot_ids(rows) != q.read(PREP / 'pilot_ids.json'):
        raise ValueError('metadata-only fresh selection changed')
    public = original.qualification.reference
    if q.read(PREP / 'public_proof.json') != {'public_file_sha256': public.PUBLIC_SHA256, 'exact_sources': public.verify_public(rows)}:
        raise ValueError('public complete source proof changed')
    values = q.read(GENERATE / 'labels.json')
    if [r['id'] for r in values] != [r['id'] for r in rows]:
        raise ValueError('complete64 first candidate denominator required')
    encoding, tokenizer = data.storage_encoding(), native.tokenizer()
    for source, value in zip(rows, values):
        if any(value[k] != v for k, v in source.items()):
            raise ValueError('original full source binding changed')
        draft = replay(source, 'draft', value['draft'], encoding)
        review = replay(source, 'review', value['review'], encoding, draft) if draft is not None else None
        text = review['summary'] if review is not None else ''
        lexical = tokenizer.encode(text, add_special_tokens=False)
        stored = len(encoding.encode(text, disallowed_special=()))
        fits = bool(text.strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0, 101}
        words = [101, *lexical, 101] + [0] * (126 - len(lexical)) if fits else None
        if (value['teacher_valid'] != (review is not None) or value['target_text'] != text
                or value['capacity_fit'] != fits or value['stored_tokens'] != stored
                or value['native_lexical_tokens'] != len(lexical) or value['native_word_ids'] != words
                or value['semantic_qualified']):
            raise ValueError('raw first candidate and capacity replay differs')
    by_id = {r['id']: r for r in values}
    if any(by_id[r['id']] != r for r in q.read(PILOT / 'labels.json')):
        raise ValueError('pilot first candidates changed')
    paths = [p for root in (WORKFLOW, PREP, PILOT, GENERATE) for p in root.rglob('*')
        if p.is_file() and 'producer_source' not in p.parts]
    AUDIT.mkdir()
    result = {'status': 'fresh64_sol_teacher_firsts_replayed_quality_pending', 'cases': 64,
        'teacher_valid': sum(r['teacher_valid'] for r in values), 'capacity_fit': sum(r['capacity_fit'] for r in values),
        'task_accuracy': None, 'student_training_admitted': False, 'canonical_semantic_cache_changed': False,
        'artifacts': {str(p): file_hash(p) for p in paths}}
    write_json(AUDIT / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'generate', 'workflow', 'audit'))
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: generate(True), 'generate': generate, 'workflow': workflow, 'audit': audit}[args.stage]()
