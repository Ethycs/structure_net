"""Independent replay of the complete final source/feature/output/first-decision chain."""
import argparse
import json
from pathlib import Path

import torch

from neural_architecture_lab.ncm_native_final_hypothesis import NativeFinalEvidence
from experiments.structure_net import ncm_lm1b_native_final as run
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def finalize():
    admitted = run.require_admission()
    run.development.configure()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError('fresh append-only complete final aggregate required')
    roots = (run.DATA, run.FEATURE_PILOT, run.FEATURES, run.PILOT, run.OUTPUTS, run.PACKETS, run.JUDGE_PILOT, run.GRADE)
    for root in roots:
        run.native.vendor.terminal(root)
    q = run.q
    cases = run.sources()
    ordered = [r for values in cases.values() for r in values]
    selected, seals = run.reservation.sealed_metadata()
    # Rebind the exact originally sealed identities after valid admission.
    for source_split, cohort in run.reservation.SOURCE_SPLITS.items():
        path = run.reservation.partitions.DEST / (source_split + '_sources.json')
        if run.file_hash(path) != seals[source_split]['creation_sha256']:
            raise ValueError('original sealed source creation bytes differ')
        originals = q.read(path)
        if [run.reservation.partitions.metadata(r) for r in originals] != selected[cohort]:
            raise ValueError('all originally reserved source identities required')
        tokens = run.body.metadata.tokenizers()
        expected = [run.body.metadata.development.prepare_row(r, cohort, *tokens) for r in originals]
        if expected != cases[cohort]:
            raise ValueError('complete final prompt/date/tokenization/alignment reconstruction differs')
    run.exclusion(ordered)
    public = {'public_file_sha256': run.reference.PUBLIC_SHA256, 'exact_sources': run.reference.verify_public(ordered)}
    if q.read(run.DATA / 'public_proof.json') != public or run.file_hash(run.DATA / 'cases.json') != q.read(run.DATA / 'result.json')['cases_sha256']:
        raise ValueError('exact final source metadata/public proof differs')
    manifest = q.read(run.FEATURES / 'manifest.json')
    feature_pilot = q.read(run.FEATURE_PILOT / 'manifest.json')
    if set(manifest) != {r['id'] for r in ordered}:
        raise ValueError('all320 original source features required')
    for row in ordered:
        entry = manifest[row['id']]
        if run.file_hash(entry['path']) != entry['sha256']:
            raise ValueError('actual original-body feature file changed')
        value = torch.load(entry['path'], map_location='cpu', weights_only=True, mmap=True)
        if (any(value[k] != row[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256'))
                or value['features'].dtype != torch.float32 or value['features'].shape != (len(row['source_ids']), 3072)
                or entry['source_tokens'] != len(row['source_ids']) or not torch.isfinite(value['features']).all()):
            raise ValueError('complete finite source-only original FP32 feature binding differs')
        if row['id'] in feature_pilot and entry != feature_pilot[row['id']]:
            raise ValueError('original feature pilot must be reused exactly')
    for root, n in ((run.FEATURE_PILOT, len(feature_pilot)), (run.FEATURES, 320)):
        value = q.read(root / 'result.json')
        if (value['cases'] != n or value['original_body_hash_unchanged'] != run.body.base.EXPECTED_BODY
                or not value['all_source_positions_retained'] or value['peak_allocated_bytes'] > 8_600_000_000
                or value['manifest_sha256'] != run.file_hash(root / 'manifest.json')):
            raise ValueError('actual unchanged original body/full-source extraction differs')
        parity = q.read(root / 'reference_parity.json')
        old = q.read(run.body.metadata.old_training.DATA / 'manifest.json')
        if len(parity) != 2 or any(not p['FP32_bytes_exact'] or run.file_hash(old[p['id']]['path']) != p['reference_sha256'] for p in parity):
            raise ValueError('actual original first/longest TRAIN feature parity required')
    state, saved = run.development.checkpoint_contract()
    model = run.development.training.lifecycle.mechanism.enable(run.development.head.from_domain(device='cpu'))
    model.load_state_dict(saved['model'], strict=True)
    run.development.model_contract(model, state)
    model_hash = run.native.vendor.module_hash(model)
    model.close()
    if model_hash != q.read(run.development.PILOT / 'result.json')['model_sha256']:
        raise ValueError('same entire trained checkpoint as admitted DEV required')
    outputs, pilot = q.read(run.OUTPUTS / 'outputs.json'), q.read(run.PILOT / 'outputs.json')
    if [r['id'] for r in outputs] != [r['id'] for r in ordered]:
        raise ValueError('every ordered final output must remain in denominator')
    tokenizer = run.native.tokenizer()
    by_id = {v['id']: v for v in outputs}
    for value, row in zip(outputs, ordered):
        decoded = run.development.decode(value['raw_word_ids'], tokenizer)
        if (any(value[k] != v for k, v in decoded.items()) or value['source_sha256'] != row['source_sha256']
                or value['split'] != row['split'] or value['source_positions'] != len(row['source_ids'])
                or value['feature_sha256'] != manifest[row['id']]['sha256'] or len(value['codes']) != 641
                or any(type(c) is not int or not 0 <= c < 8192 for c in value['codes'])):
            raise ValueError('complete native hard codes/words/source decoding replay differs')
    if any(by_id[r['id']] != r for r in pilot) or {r['id'] for r in pilot} != set(q.read(run.DATA / 'selection.json')['inference_ids']):
        raise ValueError('both original final inference pilot outputs must remain exact')
    for root, n in ((run.PILOT, 2), (run.OUTPUTS, 320)):
        value = q.read(root / 'result.json')
        if (value['cases'] != n or value['checkpoint_sha256'] != state['checkpoint_sha256']
                or value['model_sha256'] != model_hash or not value['model_unchanged']
                or value['outputs_sha256'] != run.file_hash(root / 'outputs.json')):
            raise ValueError('actual full selected checkpoint/source-only inference state differs')
    packet_cases = {c: [{**r, 'texts': {'student': by_id[r['id']]['text']}} for r in rows] for c, rows in cases.items()}
    packets = {c: {'student': [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts']['student']}])
        for r in rows]} for c, rows in packet_cases.items()}
    if (q.read(run.PACKETS / 'cases.json') != packet_cases or q.read(run.PACKETS / 'packets.json') != packets
            or q.read(run.PACKETS / 'public_proof.json') != public):
        raise ValueError('all320 anonymous source-only semantic packets differ')
    old = run.base_cache()
    pilot_cache, cache = q.read(run.JUDGE_PILOT / 'judgment_cache.json'), q.read(run.GRADE / 'judgment_cache.json')
    if any(pilot_cache.get(k) != v for k, v in old.items()) or any(cache.get(k) != v for k, v in pilot_cache.items()):
        raise ValueError('a preceding canonical first changed')
    run.semantic.audit_cache(cache)
    metrics, scores = {c: {} for c in run.COUNTS}, {}

    def replay(root, cohort, role, count, decisions):
        values = replay_scores(packet_cases[cohort][:count], {'student': packets[cohort]['student'][:count]}, decisions, 'student', role)
        metric = {**run.development.task_metric(values), 'judge': q.JUDGES[role]}
        if (q.read(root / cohort / f'{role}_student_case_scores.json') != values
                or q.read(root / cohort / f'{role}_student_result.json') != metric):
            raise ValueError('actual final raw first decisions/complete-summary metrics differ')
        return values, metric

    pilot_metrics = {}
    for cohort, count in run.COUNTS.items():
        probes, metric = replay(run.JUDGE_PILOT, cohort, 'secondary', 2, pilot_cache)
        if not all(r['valid'] for r in probes):
            raise ValueError('actual four-case validity-only pilot required')
        pilot_metrics[cohort] = {'secondary': metric}
        scores[cohort, 'secondary'], metrics[cohort]['secondary'] = replay(run.GRADE, cohort, 'secondary', count, cache)
    primary_admitted = all(run.qualifies(metrics[c]['secondary'], n) for c, n in run.COUNTS.items())
    if primary_admitted:
        for cohort, count in run.COUNTS.items():
            scores[cohort, 'primary'], metrics[cohort]['primary'] = replay(run.GRADE, cohort, 'primary', count, cache)
    elif any((run.GRADE / c / 'primary_student_case_scores.json').exists() for c in run.COUNTS):
        raise ValueError('primary ran before both complete secondary cohort gates')
    joints = {}
    for cohort, count in run.COUNTS.items():
        a, b = scores[cohort, 'secondary'], scores.get((cohort, 'primary'))
        if b is not None and [r['id'] for r in a] != [r['id'] for r in b]:
            raise ValueError('same final sources under both judges required')
        # Independently compute actual intersection before using common formatting.
        if b is not None:
            expected_valid = sum(x['valid'] and y['valid'] for x, y in zip(a, b))
            expected_passed = sum(x['passed'] is True and y['passed'] is True for x, y in zip(a, b))
            if run.consensus(a, b)['valid_both'] != expected_valid or run.consensus(a, b)['joint_passed'] != expected_passed:
                raise ValueError('same-case final intersection differs')
        joints[cohort] = run.consensus(a, b)
    valid = all(r['valid'] for values in scores.values() for r in values)
    qualified = all(joints[c]['qualified'] for c in run.COUNTS)
    classification = 'native655_sealed320_quality_qualified' if qualified else 'native655_sealed320_judgments_invalid' if not valid else 'native655_sealed320_quality_below_95'
    aggregate = q.read(run.GRADE / 'result.json')
    if (aggregate['status'] != classification or aggregate['metrics'] != metrics or aggregate['joint'] != joints
            or aggregate['primary_admitted'] != primary_admitted or aggregate['quality_gate_passed'] != qualified):
        raise ValueError('locked final quality decision differs')
    pilot_state = q.read(run.JUDGE_PILOT / 'result.json')
    if pilot_state['status'] != 'four_final_judge_firsts_valid_only' or pilot_state['metrics'] != pilot_metrics:
        raise ValueError('actual final judge pilot state differs')
    for root, decisions, prior in ((run.JUDGE_PILOT, pilot_cache, old), (run.GRADE, cache, pilot_cache)):
        s = q.read(root / 'result.json')
        if s['canonical_first_judgments'] != len(decisions) or s['inherited_first_judgments'] != len(prior):
            raise ValueError('full final first-judgment lineage counts differ')
    allowed = {run.reference.first.packet_key(p, role) for c in run.COUNTS for p in packets[c]['student']
        for role in (('secondary', 'primary') if primary_admitted else ('secondary',))}
    pilot_allowed = {run.reference.first.packet_key(p, 'secondary') for c in run.COUNTS for p in packets[c]['student'][:2]}
    if set(cache) - set(old) - allowed or set(pilot_cache) - set(old) - pilot_allowed:
        raise ValueError('new firsts outside complete final population appeared')
    joint_valid = sum(v['valid_both'] for v in joints.values())
    joint_passed = sum(v['joint_passed'] for v in joints.values())
    task_accuracy = joint_passed / 320 if joint_valid == 320 else None
    secondary_passed = sum(metrics[c]['secondary']['passed'] for c in run.COUNTS)
    secondary_valid = sum(metrics[c]['secondary']['valid'] for c in run.COUNTS)
    joint_upper = sum(joints[c]['joint_accuracy_upper_bound'] * n for c, n in run.COUNTS.items()) / 320
    artifacts = {}
    for root in (*roots, run.WORKFLOW):
        for p in root.rglob('*'):
            if p.is_file() and 'producer_source' not in p.parts:
                artifacts[str(p)] = run.file_hash(p)
    artifacts[str(run.development.GRADE / 'judgment_cache.json')] = run.file_hash(run.development.GRADE / 'judgment_cache.json')
    artifacts[str(run.development.training.TRAIN / 'model.pt')] = state['checkpoint_sha256']
    for entry in cache.values():
        for p in entry['row']['attempts']:
            artifacts[p] = run.file_hash(p)
    campaign = {'hypothesis_id': run.HYPOTHESIS, 'classification': classification, 'quality_gate_passed': qualified,
        'counts': run.COUNTS, 'metrics': metrics, 'joint': joints, 'joint_valid': joint_valid, 'joint_passed': joint_passed,
        'task_accuracy': task_accuracy, 'primary_admitted': primary_admitted, 'canonical_first_judgments': len(cache),
        'secondary_passed': secondary_passed, 'secondary_valid': secondary_valid,
        'secondary_task_accuracy': secondary_passed / 320 if secondary_valid == 320 else None,
        'joint_accuracy_upper_bound': joint_upper,
        'canonical_parent_firsts': len(old), 'source_prediction': True, 'oracle_target_codes': False,
        'checkpoint_sha256': state['checkpoint_sha256'], 'model_sha256': model_hash,
        'original_body_hash': run.body.base.EXPECTED_BODY, 'final_test_opened': True, 'speed_tested': False,
        'single_adaptation_seed': True, 'confirmed': False, 'artifacts': artifacts}
    run.RESULTS.mkdir()
    path = run.RESULTS / 'campaign_results.json'
    run.write_json(path, campaign)
    table = ['| Cohort | Terra passes | Sonnet passes | Same-case joint |', '| --- | ---: | ---: | ---: |']
    for c, n in run.COUNTS.items():
        primary = f"{metrics[c]['primary']['passed']}/{n}" if primary_admitted else 'not admitted'
        table.append(f"| {c} | {metrics[c]['secondary']['passed']}/{n} | {primary} | {joints[c]['joint_passed']}/{n} |")
    headline = (f"Complete-summary joint pass rate: **{joint_passed}/320 ({task_accuracy * 100:.2f}%)**.\n"
        if task_accuracy is not None else f"Joint accuracy is unmeasured; both-judge-valid cases: {joint_valid}/320. "
        f"Terra passed {secondary_passed}/320; primary admitted: {primary_admitted}. "
        f"Joint accuracy upper bound: {joint_upper * 100:.2f}%.\n")
    run.REPORT.write_text('# Full-native655 source-only sealed final quality\n\n'
        + f"Classification **{classification}**; hypothesis `{run.HYPOTHESIS}`.\n"
        + f"Campaign `{path}`, SHA256 `{run.file_hash(path)}`.\n\n"
        + headline + f"Predeclared per-cohort quality gate passed: **{qualified}**.\n\n"
        + '\n'.join(table) + '\n\n'
        'The same fixed final 1,024-update checkpoint that passed both source-disjoint DEV cohorts\n'
        'was tested once on the original sealed 256 short and 64 long sources. Raw source\n'
        'content was opened only after actual DEV completion, registration, same-case\n'
        'quality gates and independent backend readback. All prior known source\n'
        'and DEV exclusions were rechecked. Original body/MLPs and codec geometry\n'
        'stayed frozen; no training, target codes, output repair or fallback ran.\n\n'
        'The calibrated judges assessed supported meaning, main content and readable\n'
        'prose, with nonempty output and a 128-token storage cap. Exact reference matches\n'
        'were not used. All 320 cases and all original first judgments remain in the\n'
        'denominator. Qualification requires 244/256 and 61/64 of the SAME cases to\n'
        'pass both judges with every decision valid. This is one adaptation seed\n'
        'on a public source-disjoint evaluation; model speed was not measured.\n')
    print(json.dumps({k: v for k, v in campaign.items() if k not in ('artifacts', 'metrics')}), flush=True)


def register():
    path = run.RESULTS / 'campaign_results.json'
    s = run.q.read(path)
    evidence = NativeFinalEvidence(s['classification'], s['quality_gate_passed'], str(path), run.file_hash(path),
        str(run.REPORT), run.file_hash(run.REPORT), s['checkpoint_sha256'], s['joint_valid'])
    register_records(path, run.REPORT, run.RESULTS, {run.HYPOTHESIS: evidence.record()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('finalize', 'register'))
    args = parser.parse_args()
    {'finalize': finalize, 'register': register}[args.stage]()
