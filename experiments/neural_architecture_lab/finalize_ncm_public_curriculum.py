"""Verify committed public training and complete-summary semantic milestones."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys

import torch

from experiments.structure_net import ncm_public_development_lineage as development
from experiments.neural_architecture_lab import audit_ncm_public_training as audit
from experiments.neural_architecture_lab.finalize_ncm_broad_generator import classify

study, data, pipeline = development.study, development.data, development.study.pipeline
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-milestone-reporting-plan.md')
TEST = Path('tests/structure_net/test_ncm_public_milestone_reporting.py')
REGISTRAR = Path('experiments/neural_architecture_lab/register_ncm_public_curriculum.py')
PREFIX = data.ROOT/'20261026_public_training_prefix64_audit_v1'


def roots(epoch):
    if epoch not in study.CONFIG.milestones:
        raise ValueError('fixed public milestone required')
    return {s: data.ROOT/f'20261026_public_curriculum_epoch{epoch}_{s}_v1'
            for s in ('training_audit', 'semantic_replay', 'results')}


def report_path(epoch):
    roots(epoch)
    return Path(f'docs/08 - Analysis/2026-10-26_ncm-public-epoch{epoch}.md')


def receipt(root, names=('producer.json', 'result.json'), *, failed=False):
    filename = 'failure_actual_tool_receipt.json' if failed else 'actual_tool_receipt.json'
    value = data.read(root/filename)
    code = value['actual_tool_exit_code']
    if type(code) is not int or (code == 0) == failed:
        raise ValueError('actual terminal tool state differs')
    if any(value['artifacts'].get(str(root/n)) != data.sha(root/n) for n in names):
        raise ValueError('actual receipt does not bind the required artifacts')
    data.verify(value['artifacts'])
    return root/filename


def source_files():
    files = {Path(__file__).resolve(), TEST.resolve(), REGISTRAR.resolve(), PLAN.resolve(), audit.PLAN.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):
                files.add(p)
    return sorted(files)


def bindings(paths, initial):
    data.verify(initial)
    return {**initial, **{str(p): data.sha(p) for p in dict.fromkeys([*source_files(), *paths])}}


def capture_prefix(path, count):
    if type(count) is not int or count < 1:
        raise ValueError('positive published update count required')
    rows = []
    with Path(path).open('rb') as stream:
        for step in range(1, count+1):
            line = stream.readline()
            if not line.endswith(b'\n') or json.loads(line)['step'] != step:
                raise ValueError('complete ordered published prefix required')
            rows.append(line)
    return b''.join(rows)


def prefix(epoch, attempt):
    if (epoch, attempt) != (1, 1) or PREFIX.exists():
        raise ValueError('fresh fixed first64-update systems qualification required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only prefix replay required')
    initial = {str(p): data.sha(p) for p in source_files()}
    receipt(audit.PILOT); qualified = data.completed(audit.PILOT)
    data.verify(data.read(audit.PILOT/'producer.json')['input_hashes'])
    if qualified['status'] != 'public_training_auditor_real_two_updates_passed':
        raise ValueError('qualified training-record auditor required')
    training, environment = pipeline.broad.runtime.configure()
    source = study.root_for(1, 1)
    proof = data.read(source/'producer.json')
    training.native.vendor.verify(proof)
    snapshot = capture_prefix(source/'trace.jsonl', 64)
    sources = pipeline.ready(training)
    PREFIX.mkdir(); (PREFIX/'trace_prefix.jsonl').write_bytes(snapshot)
    paths = [source/'producer.json', source/'segment.json', PREFIX/'trace_prefix.jsonl',
             *[audit.PILOT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')],
             *[pipeline.bundle.ROOT/n for n in ('manifest.json', 'result.json', 'actual_tool_receipt.json')]]
    hashes = bindings(paths, initial)
    data.write(PREFIX/'producer.json', {'schema': 'nal.ncm-public-training-prefix-audit.v1',
        'environment': environment, 'input_hashes': hashes, 'training_fingerprint': proof['fingerprint'],
        'config': {'recorded_updates': 64, 'source_trace': str(source/'trace.jsonl'),
                   'source_prefix_bytes': len(snapshot), 'complete_epoch': False, 'checkpoint_commit_verified': False}})
    result = audit.replay(map(json.loads, snapshot.splitlines()), audit.Labels(sources), 0, 64)
    if capture_prefix(source/'trace.jsonl', 64) != snapshot:
        raise ValueError('published source prefix changed during audit')
    data.verify(hashes)
    sources.public.pairs.close(); sources.public.codes.close()
    value = {'status': 'published_training_prefix64_replayed', **result, 'systems_only': True,
        'complete_epoch': False, 'checkpoint_commit_verified': False, 'feature_matrices_loaded': 0,
        'model_training': False, 'new_judge_requests': 0, 'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(PREFIX/n): data.sha(PREFIX/n) for n in ('producer.json', 'trace_prefix.jsonl')}}
    data.write(PREFIX/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


def checkpoint_owner(path):
    try:
        relative = Path(path).resolve().relative_to(study.STORAGE.resolve())
    except ValueError as error:
        raise ValueError('checkpoint must belong to this public training study') from error
    match = re.fullmatch(r'epoch(1|2|4|8)_attempt([1-9][0-9]*)/step_([0-9]{6})\.pt', relative.as_posix())
    if not match:
        raise ValueError('committed checkpoint path contract differs')
    epoch, attempt, step = map(int, match.groups())
    if not 0 < step <= epoch*study.CONFIG.updates_per_epoch:
        raise ValueError('checkpoint exceeds its declared milestone')
    return study.root_for(epoch, attempt), step


def check_exposure(epoch, rows):
    cursor, public, anchors, noisy = 0, 0, 0, 0
    for row in rows:
        if row['first_completed_update'] != cursor or row['last_completed_update'] <= cursor:
            raise ValueError('committed trajectory has a gap or repeated updates')
        cursor = row['last_completed_update']
        public += row['presentations']['public']; anchors += row['presentations']['anchor']
        noisy += row['noisy_presentations']
    if (cursor != epoch*study.CONFIG.updates_per_epoch or public != epoch*study.CONFIG.public_cases
            or anchors != epoch*study.CONFIG.anchors_per_epoch
            or noisy != epoch*(study.CONFIG.presentations_per_epoch//2)):
        raise ValueError('complete public epochs, reviewed anchors and noise population required')
    return {'updates': cursor, 'public_presentations': public, 'anchor_presentations': anchors,
            'noisy_presentations': noisy}


def audit_training(epoch, attempt):
    destination = roots(epoch)['training_audit']
    if destination.exists():
        raise ValueError('fresh training-chain audit required')
    initial = {str(p): data.sha(p) for p in source_files()}
    training, environment = pipeline.broad.runtime.configure()
    pipeline.determinism.configure_runtime()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only training-chain audit required')
    receipt(audit.PILOT)
    qualification = data.completed(audit.PILOT)
    data.verify(data.read(audit.PILOT/'producer.json')['input_hashes'])
    if qualification['status'] != 'public_training_auditor_real_two_updates_passed':
        raise ValueError('qualified real-record auditor required')
    selected_root = study.root_for(epoch, attempt)
    receipt(selected_root)
    selected = data.completed(selected_root)
    expected_config = training.q.digest(asdict(study.CONFIG))
    parent = data.read(pipeline.parent.PARENT/'result.json')
    if (selected['epoch'] != epoch or selected['updates'] != epoch*study.CONFIG.updates_per_epoch
            or selected['public_cases'] != 162353 or selected['reviewed_cases'] != 1376
            or not selected['strict_reload_passed'] or not selected['oracle_unchanged']
            or selected['frozen_hashes'] != parent['frozen_hashes']
            or training.q.digest(selected['config']) != expected_config
            or selected['task_accuracy'] is not None or selected['final_test_opened'] or selected['speed_tested']):
        raise ValueError('complete fixed milestone and unchanged reconstruction required')
    identity = selected['training_identity']
    cursor = Path(selected_root/'model.pt').resolve()
    if checkpoint_owner(cursor) != (selected_root, selected['updates']):
        raise ValueError('selected model must be this completed attempt checkpoint')
    paths, chain, seen = [], [], set()
    expected_sidecar = None
    while True:
        owner, end = checkpoint_owner(cursor)
        if owner in seen:
            raise ValueError('cyclic checkpoint ancestry')
        seen.add(owner)
        sidecar = data.read(cursor.with_suffix('.json'))
        if expected_sidecar is not None and sidecar != expected_sidecar:
            raise ValueError('recorded resume differs from its exact checkpoint sidecar')
        if (sidecar['path'] != str(cursor) or sidecar['sha256'] != data.sha(cursor)
                or sidecar['completed_updates'] != end or sidecar['training_identity'] != identity):
            raise ValueError('checkpoint sidecar, bytes or cursor differs')
        saved = torch.load(cursor, map_location='cpu', weights_only=True, mmap=True)
        if (saved['completed_updates'] != end or saved['producer_fingerprint'] != identity
                or training.q.digest(saved['curriculum']) != expected_config):
            raise ValueError('committed checkpoint metadata differs')
        del saved
        proof, segment = [data.read(owner/n) for n in ('producer.json', 'segment.json')]
        training.native.vendor.verify(proof)
        if (proof['fingerprint'] != identity or segment['training_identity'] != identity
                or proof['fingerprint'] != training.q.digest({k: v for k, v in proof.items() if k != 'fingerprint'})
                or proof['environment'] != environment):
            raise ValueError('training identity or runtime changed across segments')
        success_receipt = owner/'actual_tool_receipt.json'
        failure_receipt = owner/'failure_actual_tool_receipt.json'
        if success_receipt.exists() and failure_receipt.exists():
            raise ValueError('contradictory terminal receipts')
        complete = success_receipt.exists()
        terminal = receipt(owner, ('producer.json', 'result.json') if complete else ('producer.json', 'trace.jsonl'), failed=not complete)
        if complete:
            result = data.completed(owner)
            if (result['updates'] != end or result['training_identity'] != identity
                    or result['checkpoint_sha256'] != sidecar['sha256']):
                raise ValueError('completed segment differs from resumed checkpoint')
        start = segment['first_completed_update']
        if not 0 <= start < end <= segment['target_completed_update']:
            raise ValueError('committed segment interval differs')
        chain.append((owner, start, end, complete))
        paths += [cursor, cursor.with_suffix('.json'), terminal, *[owner/n for n in ('producer.json', 'segment.json', 'trace.jsonl')]]
        if start == 0:
            break
        resumed = data.read(owner/'resume.json')
        if resumed['completed_updates'] != start or resumed['training_identity'] != identity:
            raise ValueError('resume cursor or training identity differs')
        paths.append(owner/'resume.json')
        expected_sidecar = resumed
        cursor = Path(resumed['path']).resolve()
    sources = pipeline.ready(training)
    paths += [*[selected_root/n for n in ('result.json', 'actual_tool_receipt.json')],
              *[audit.PILOT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]]
    hashes = bindings(paths, initial)
    labels, results = audit.Labels(sources), []
    for owner, start, end, complete in reversed(chain):
        with (owner/'trace.jsonl').open() as stream:
            result = audit.replay(map(json.loads, stream), labels, start, end, complete_trace=complete)
        results.append({**result, 'root': str(owner), 'completed_segment': complete})
    totals = check_exposure(epoch, results)
    sources.public.pairs.close(); sources.public.codes.close()
    data.verify(hashes)
    value = {'schema': 'nal.ncm-public-training-chain-audit.v1', 'status': 'complete_committed_training_chain_replayed',
        'epoch': epoch, 'attempt': attempt, 'training_identity': identity, 'segments': results, **totals,
        'checkpoint_sha256': selected['checkpoint_sha256'], 'frozen_hashes': selected['frozen_hashes'],
        'feature_matrices_loaded': 0, 'new_judge_requests': 0, 'task_accuracy': None, 'final_test_opened': False,
        'artifacts': hashes}
    destination.mkdir(); data.write(destination/'result.json', value)
    print(json.dumps({k: value[k] for k in ('status', 'epoch', 'updates', 'public_presentations', 'anchor_presentations')}), flush=True)


def selection(milestones):
    epochs = [r['epoch'] for r in milestones]
    if not epochs or epochs != list(study.CONFIG.milestones[:len(epochs)]):
        raise ValueError('complete ordered milestone prefix required')
    candidate = None
    for row in milestones:
        if candidate is not None:
            raise ValueError('training continued beyond the first qualifying milestone')
        expected = classify(row['valid'], row['passed'], row['primary_admitted'], row['joint'])
        if (row['classification'] != expected or row['DEV_qualified'] != (expected == 'DEV_qualified')
                or row['task_accuracy'] != (row['passed']/128 if row['valid'] == 128 else None)):
            raise ValueError('semantic counts or qualification differ')
        joint_accuracy = sum(v['joint_passed'] for v in row['joint'].values())/128 if row['joint'] and all(v['all_valid'] for v in row['joint'].values()) else None
        if row['joint_task_accuracy'] != joint_accuracy:
            raise ValueError('joint accuracy differs from same-case decisions')
        if expected == 'DEV_qualified': candidate = row['epoch']
        if row is not milestones[-1] and expected == 'DEV_transport_invalid':
            raise ValueError('invalid milestone cannot authorize further training')
    return candidate


def replay_semantics(epoch, attempt):
    from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
    from experiments.neural_architecture_lab.finalize_ncm_lm1b_native_development import replay_outputs
    destination, checked = roots(epoch)['semantic_replay'], roots(epoch)['training_audit']
    if destination.exists(): raise ValueError('fresh semantic replay required')
    initial = {str(p): data.sha(p) for p in source_files()}
    receipt(checked, ('result.json',))
    audited = data.completed(checked)
    if audited['epoch'] != epoch or audited['attempt'] != attempt:
        raise ValueError('selected audited training attempt required')
    receipt(development.PREFLIGHT)
    data.verify(data.read(development.PREFLIGHT/'producer.json')['input_hashes'])
    training, evaluation, environment = development.configure(epoch, attempt)
    q, old, lineage, stages = training.q, evaluation.old, evaluation.lineage, development.roots(epoch)
    lineage.configure()
    paths = [checked/'result.json', checked/'actual_tool_receipt.json']
    for root in stages.values(): paths += [p for p in root.rglob('*.json') if 'producer_source' not in p.parts]
    hashes = bindings(paths, initial)
    trained = data.completed(study.root_for(epoch, attempt))
    if trained['checkpoint_sha256'] != audited['checkpoint_sha256']:
        raise ValueError('selected checkpoint differs from full training audit')
    for stage, root in stages.items():
        receipt(root); data.completed(root)
        proof = q.read(root/'producer.json')
        training.native.vendor.verify(proof)
        if (proof['schema'] != 'nal.ncm-public-curriculum-dev.v2' or proof['epoch'] != epoch
                or proof['stage'] != stage or proof['environment'] != environment
                or proof['fingerprint'] != q.digest({k: v for k, v in proof.items() if k != 'fingerprint'})
                or q.read(root/'result.json')['producer_fingerprint'] != proof['fingerprint']):
            raise ValueError('fixed evaluation producer or runtime differs')
    cohorts, manifest, split = evaluation.sources()
    rows = [r for c in old.COHORTS for r in cohorts[c]]
    outputs, probes = [q.read(stages[s]/'outputs.json') for s in ('outputs', 'inference_pilot')]
    tokenizer = training.native.tokenizer()
    replay_outputs(outputs, rows, manifest, tokenizer)
    replay_outputs(probes, old.inference_probes(cohorts, manifest), manifest, tokenizer)
    by_id = {r['id']: r for r in outputs}
    if any(by_id[r['id']] != r for r in probes): raise ValueError('full outputs differ from inference pilot')
    for stage, count in (('outputs', 128), ('inference_pilot', len(probes))):
        state = q.read(stages[stage]/'result.json')
        if (state['cases'] != count or state['model_sha256'] != trained['final_model_sha256']
                or state['outputs_sha256'] != data.sha(stages[stage]/'outputs.json')):
            raise ValueError('selected checkpoint, output bytes or population differs')
    cases = {c: [{**r, 'texts': {'student': by_id[r['id']]['text']}} for r in cohorts[c]] for c in old.COHORTS}
    packets = {c: {'student': [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts']['student']}]) for r in cases[c]]} for c in old.COHORTS}
    inherited, cache, pilot_cache = lineage.base_cache(), q.read(stages['grade']/'judgment_cache.json'), q.read(stages['judge_pilot']/'judgment_cache.json')
    pilot_allowed = {old.oracle.first.packet_key(p, 'secondary') for c in old.COHORTS for p in packets[c]['student'][:2]}
    development.previous.preserve_firsts(inherited, pilot_cache, allowed_new=pilot_allowed)
    development.previous.preserve_firsts(pilot_cache, cache); lineage.audit_cache(cache)
    for stage in ('judge_pilot', 'grade'):
        proof = q.read(stages[stage]/'producer.json')
        if (q.read(stages[stage]/'cases.json') != cases or q.read(stages[stage]/'exact_packets.json') != packets
                or proof['config']['inherited_first_judgments'] != len(inherited)
                or proof['config']['rubric'] != q.RUBRIC or proof['config']['judges'] != q.JUDGES):
            raise ValueError('exact source-only packets or first lineage differs')
    metrics, scores, pilot_metrics = {c: {} for c in old.COHORTS}, {c: {} for c in old.COHORTS}, {c: {} for c in old.COHORTS}

    def decisions(stage, cohort, role, count, firsts):
        expected = replay_scores(cases[cohort][:count], {'student': packets[cohort]['student'][:count]}, firsts, 'student', role)
        directory = stages[stage]/cohort
        reported = q.read(directory/f'{role}_student_result.json')
        if (q.read(directory/f'{role}_student_case_scores.json') != expected
                or any(reported.get(k) != v for k, v in old.task_metric(expected).items())):
            raise ValueError('raw first responses and reported semantic scores differ')
        return expected, reported

    for c in old.COHORTS:
        pilot_scores, pilot_metrics[c]['secondary'] = decisions('judge_pilot', c, 'secondary', 2, pilot_cache)
        if not all(r['valid'] for r in pilot_scores): raise ValueError('valid four-case judge pilot required')
        scores[c]['secondary'], metrics[c]['secondary'] = decisions('grade', c, 'secondary', 64, cache)
    pilot_result = q.read(stages['judge_pilot']/'result.json')
    if (not pilot_result['valid_pilot'] or pilot_result['metrics'] != pilot_metrics or pilot_result['primary_admitted']
            or pilot_result['joint'] is not None or pilot_result['task_accuracy'] is not None or pilot_result['DEV_qualified']
            or pilot_result['canonical_first_judgments'] != len(pilot_cache)
            or pilot_result['inherited_first_judgments'] != len(inherited)):
        raise ValueError('judge pilot aggregate differs')
    admitted = old.secondary_admitted({c: metrics[c]['secondary'] for c in old.COHORTS})
    joints = None
    if admitted:
        joints = {}
        for c in old.COHORTS:
            scores[c]['primary'], metrics[c]['primary'] = decisions('grade', c, 'primary', 64, cache)
            joints[c] = old.joint_gate(scores[c]['secondary'], scores[c]['primary'])
    elif any((stages['grade']/c/'primary_student_case_scores.json').exists() for c in old.COHORTS):
        raise ValueError('primary judge ran without admission')
    roles = ('secondary', 'primary') if admitted else ('secondary',)
    allowed = {old.oracle.first.packet_key(p, role) for c in old.COHORTS for p in packets[c]['student'] for role in roles}
    development.previous.preserve_firsts(inherited, cache, allowed_new=allowed)
    valid, passed = (sum(metrics[c]['secondary'][k] for c in old.COHORTS) for k in ('valid', 'passed'))
    classification = classify(valid, passed, admitted, joints)
    expected = {'metrics': metrics, 'primary_admitted': admitted, 'joint': joints,
        'DEV_qualified': classification == 'DEV_qualified', 'task_accuracy': passed/128 if valid == 128 else None,
        'canonical_first_judgments': len(cache), 'inherited_first_judgments': len(inherited),
        'final_test_opened': False, 'speed_tested': False, 'metric': 'complete_summary_semantic_pass',
        'task_accuracy_judge': q.JUDGES['secondary']}
    grade = q.read(stages['grade']/'result.json')
    if any(grade.get(k) != v for k, v in expected.items()):
        raise ValueError('semantic aggregate or qualification differs')
    data.verify(hashes)
    value = {'schema': 'nal.ncm-public-semantic-replay.v1', 'epoch': epoch, 'attempt': attempt,
        'classification': classification, 'cases': 128, 'valid': valid, 'passed': passed, **expected,
        'joint_task_accuracy': sum(v['joint_passed'] for v in joints.values())/128 if joints and all(v['all_valid'] for v in joints.values()) else None,
        'training_updates': trained['updates'], 'public_presentations': audited['public_presentations'],
        'anchor_presentations': audited['anchor_presentations'], 'split_proof': split,
        'checkpoint_sha256': trained['checkpoint_sha256'], 'training_identity': trained['training_identity'],
        'frozen_hashes': trained['frozen_hashes'], 'source_only': True, 'final_task_goal_achieved': False,
        'artifacts': hashes}
    destination.mkdir(); data.write(destination/'scores.json', scores); data.write(destination/'result.json', value)
    print(json.dumps({k: value[k] for k in ('epoch', 'classification', 'valid', 'passed', 'task_accuracy', 'DEV_qualified')}), flush=True)


def report(epoch, attempt):
    destination, document = roots(epoch)['results'], report_path(epoch)
    if destination.exists() or document.exists(): raise ValueError('fresh milestone report required')
    initial = {str(p): data.sha(p) for p in source_files()}
    milestones, paths, inherited = [], [], 4435
    for number in study.CONFIG.milestones:
        if number > epoch: break
        root = roots(number)['semantic_replay']
        receipt(root, ('result.json', 'scores.json'))
        state = data.completed(root)
        if state['epoch'] != number or state['inherited_first_judgments'] != inherited:
            raise ValueError('full sequential milestone lineage required')
        if number == epoch and state['attempt'] != attempt: raise ValueError('selected attempt differs')
        if milestones and any(state[k] != milestones[0][k] for k in ('training_identity', 'frozen_hashes', 'split_proof')):
            raise ValueError('milestones changed training identity, reconstruction or evaluation population')
        inherited = state['canonical_first_judgments']
        milestones.append({k: v for k, v in state.items() if k != 'artifacts'})
        paths += [root/n for n in ('result.json', 'scores.json', 'actual_tool_receipt.json')]
    candidate = selection(milestones)
    hashes = bindings(paths, initial)
    value = {'schema': 'nal.ncm-public-milestone-results.v1', 'hypothesis_id': f'ncm-public-curriculum-epoch{epoch}-v1',
        'parent_hypothesis_id': 'ncm-public-curriculum-v1', 'epoch': epoch, 'attempt': attempt,
        'classification': milestones[-1]['classification'], 'milestones': milestones, 'selected_candidate_epoch': candidate,
        'metric': 'complete_summary_semantic_pass', 'cases_per_milestone': 128,
        'conformance': 'EXPLORATORY_UNDERPOWERED', 'independent_training_trajectories': 1,
        'canonical_first_judgments': inherited, 'final_test_opened': False, 'final_task_goal_achieved': False,
        'speed_tested': False, 'artifacts': hashes}
    destination.mkdir(); data.write(destination/'campaign_results.json', value)
    lines = [f'# Public NCM curriculum: epoch{epoch}', '', f"Classification: `{value['classification']}`. Final320 remains sealed.", '',
        '| Epoch | Terra semantic passes | Valid judgments | Joint Terra/Sol accuracy | DEV qualified |',
        '|---|---:|---:|---:|---|']
    for row in milestones:
        accuracy = 'unresolved' if row['task_accuracy'] is None else f"{row['passed']}/128 ({100*row['task_accuracy']:.2f}%)"
        joint = 'unmeasured or unresolved' if row['joint_task_accuracy'] is None else f"{100*row['joint_task_accuracy']:.2f}%"
        lines.append(f"| {row['epoch']} | {accuracy} | {row['valid']}/128 | {joint} | {row['DEV_qualified']} |")
    lines += ['', 'Accuracy measures complete readable, faithful, salient summaries within128 stored tokens; paraphrases are accepted.',
        'Training loss, code agreement and codec reconstruction are separate diagnostics and are not task accuracy.',
        'Each epoch visits all162,353 public pairs and54,118 reviewed-anchor presentations. The original Llama body and reconstruction path remain fixed.',
        'This is one continuing training trajectory evaluated at fixed milestones, not independent model replications.',
        f'Earliest candidate for the separate sealed-final protocol: `{candidate}`. DEV qualification does not establish the final95% goal.',
        'Inference speed remains unmeasured.', '', f'Detailed cohort decisions and artifact hashes: `{destination}/campaign_results.json`.']
    document.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'epoch': epoch, 'classification': value['classification'], 'selected_candidate_epoch': candidate}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prefix', 'audit', 'replay', 'report'))
    parser.add_argument('--epoch', type=int, choices=study.CONFIG.milestones, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    {'prefix': prefix, 'audit': audit_training, 'replay': replay_semantics, 'report': report}[args.stage](args.epoch, args.attempt)
