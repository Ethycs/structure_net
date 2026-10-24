"""Single-use sealed320 quality test of the same full-native655 DEV checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_lm1b_native_development as development
from experiments.structure_net import ncm_clean_sealed_final as reservation
from experiments.structure_net import ncm_additional_source_features as body
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = development.native, development.q, development.ROOT
reference, semantic, oracle = development.reference, development.semantic, development.oracle
HYPOTHESIS = 'ncm-lm1b-native655-source-only-sealed320-v1'
MODULE = 'experiments.structure_net.ncm_lm1b_native_final'
AUDITOR_MODULE = 'experiments.neural_architecture_lab.finalize_ncm_lm1b_native_final'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-lm1b-native-final-plan.md')
TEST = Path('tests/structure_net/test_ncm_lm1b_native_final.py')
AUDITOR = Path('experiments/neural_architecture_lab/finalize_ncm_lm1b_native_final.py')
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-lm1b-native-final.md')
WORKFLOW = ROOT / '20261024_lm1b_native655_final_workflow_v1'
DATA = ROOT / '20261024_lm1b_native655_final_data_v1'
FEATURE_PILOT = ROOT / '20261024_lm1b_native655_final_feature_pilot_v1'
FEATURES = ROOT / '20261024_lm1b_native655_final_features_v1'
PILOT = ROOT / '20261024_lm1b_native655_final_inference_pilot_v1'
OUTPUTS = ROOT / '20261024_lm1b_native655_final_outputs_v1'
PACKETS = ROOT / '20261024_lm1b_native655_final_packets_v1'
JUDGE_PILOT = ROOT / '20261024_lm1b_native655_final_judge_pilot_v1'
GRADE = ROOT / '20261024_lm1b_native655_final_grade_v1'
RESULTS = ROOT / '20261024_lm1b_native655_final_results_v1'
COUNTS = {'short_final': 256, 'long_final': 64}
minimum, qualifies, consensus = reservation.minimum, reservation.qualifies, reservation.consensus


def development_gate(state):
    if (state['classification'] != 'full128_source_only_DEV_qualified' or not state['DEV_qualified']
            or not state['primary_admitted'] or not state['all128_cases_retained']
            or not state['source_prediction'] or state['oracle_target_codes'] or state['final_test_opened']):
        raise ValueError('complete source-only native655 DEV success required before sealed access')
    if set(state['metrics']) != set(development.COHORTS) or set(state['joint'] or {}) != set(development.COHORTS):
        raise ValueError('both complete DEV cohorts required')
    for cohort in development.COHORTS:
        joint = state['joint'][cohort]
        if (not all(qualifies(state['metrics'][cohort].get(role), 64) for role in ('secondary', 'primary'))
                or joint['cases'] != 64 or not joint['all_valid'] or joint['joint_passed'] < 61 or not joint['qualified']):
            raise ValueError('same61of64 cases must pass both judges in each DEV cohort')


def require_admission():
    # This method never reads or hashes either sealed source file.
    path = development.RESULTS / 'campaign_results.json'
    state = q.read(path)
    development_gate(state)
    receipt = q.read(development.RESULTS / (development.HYPOTHESIS + '_registry_readback.json'))
    if (receipt['status'] != 'registered_and_read_back' or receipt['record']['campaign_sha256'] != file_hash(path)
            or receipt['exact']['campaign_sha256'] != file_hash(path)
            or q.read(development.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0
            or q.read(development.RESULTS / 'backend_exact_readback.json')['status'] != 'independent_backend_exact_readback_passed'):
        raise ValueError('actual whole DEV completion, registry and independent backend readback required')
    native.vendor.verify(q.read(development.WORKFLOW / 'producer.json'))
    paths = [development.training.TRAIN / 'model.pt', development.training.TRAIN / 'result.json',
        development.GRADE / 'judgment_cache.json', development.GRADE / 'result.json',
        development.PILOT / 'result.json', development.PILOT / 'outputs.json', development.METADATA / 'cases.json']
    if any(state['artifacts'].get(str(p)) != file_hash(p) for p in paths):
        raise ValueError('same final checkpoint/DEV firsts/source evidence required')
    if (file_hash(development.SOURCE / 'cases.json') != development.SOURCE_SHA
            or file_hash(development.FEATURES / 'manifest.json') != development.FEATURE_SHA):
        raise ValueError('unchanged complete DEV source/feature identities required')
    development.checkpoint_contract()
    return state


def base_cache():
    admitted = require_admission()
    path = development.GRADE / 'judgment_cache.json'
    cache = q.read(path)
    if len(cache) != admitted['canonical_first_judgments']:
        raise ValueError('all latest completed DEV firsts required')
    semantic.audit_cache(cache)
    return cache


def producing(root, stage, inputs=(), extra=None):
    admitted = require_admission()
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(development.__file__), development.PLAN, development.AUDITOR_FILE,
        Path(reservation.__file__), Path(body.__file__), Path(body.base.__file__), Path(body.base.source.__file__),
        Path(body.metadata.__file__), Path(reference.first.__file__), reference.first.AMENDMENT,
        Path(reference.profile.__file__), reference.profile.PLAN,
        Path('src/neural_architecture_lab/ncm_native_final_hypothesis.py'),
        Path('tests/neural_architecture_lab/test_ncm_native_final_hypothesis.py')]
    common = [development.RESULTS / 'campaign_results.json', development.RESULTS / (development.HYPOTHESIS + '_registry_readback.json'),
        development.RESULTS / 'backend_exact_readback.json', development.WORKFLOW / 'tool_terminal_receipt.json',
        development.GRADE / 'judgment_cache.json', development.training.TRAIN / 'model.pt', development.training.TRAIN / 'result.json']
    common += [body.base.MODEL_PATH / n for n in ('config.json', 'model.safetensors.index.json',
        'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors')]
    common += [reservation.partitions.DEST / n for n in ('selection_metadata.json', 'seal_receipts.json', 'public_inventory.json')]
    p = native.vendor.freeze(root, stage, [*inputs, *common, *files], {**(extra or {}), 'hypothesis_id': HYPOTHESIS,
        'counts': COUNTS, 'joint_minimum': {k: minimum(v) for k, v in COUNTS.items()},
        'source_only': True, 'oracle_target_codes': False, 'all_source_positions_retained': True,
        'original_body_and_MLPs_frozen': True, 'body_precision': 'BF16unchanged', 'aligned_features': 'FP32unchanged',
        'head_precision': 'FP32unchanged', 'TF32': False, 'stored_token_cap': 128,
        'raw_word_positions': 128, 'native_concept_codes': 641, 'word_vocabulary': 30528, 'code_vocabulary': 8192,
        'output_policy': 'unchanged nativeCLSrenderer,full-vocabulary hardargmax,no repairs/masks/ARfallback',
        'canonical_parent_firsts': admitted['canonical_first_judgments'], 'judges': q.JUDGES, 'rubric': q.RUBRIC,
        'no_semantic_rerolls_or_checkpoint_selection': True, 'training_executed': False,
        'final_access_admitted': True, 'task_accuracy': None, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'native_final_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def exclusion(rows):
    known, paths = reservation.known_sources()
    dev = [r for cohort in development.COHORTS for r in q.read(development.SOURCE / 'cases.json')[cohort]]
    inventory = q.read(reservation.partitions.DEST / 'public_inventory.json')
    reservation.development.source_guard([*known, *dev], rows, {k: v['tokens'] for k, v in inventory['turns'].items()})
    return paths


def prepare():
    require_admission()
    if (WORKFLOW / 'final_access_intent.json').exists():
        raise ValueError('single-use final source access already attempted; preserve it')
    selected, seals = reservation.sealed_metadata()  # Metadata/stat only.
    state = q.read(development.training.TRAIN / 'result.json')
    write_json(WORKFLOW / 'final_access_intent.json', {'DEV_admitted': True,
        'same_checkpoint_sha256': state['checkpoint_sha256'], 'source_contents_may_be_opened_after_this_record': True})
    parts, source_paths = {}, []
    for source_split, cohort in reservation.SOURCE_SPLITS.items():
        path = reservation.partitions.DEST / (source_split + '_sources.json')
        if file_hash(path) != seals[source_split]['creation_sha256']:
            raise ValueError('sealed final source changed from its creation bytes')
        rows = q.read(path)
        if [reservation.partitions.metadata(r) for r in rows] != selected[cohort]:
            raise ValueError('unchanged final source identities/order/components required')
        parts[cohort] = rows
        source_paths.append(path)
    all_rows = [r for values in parts.values() for r in values]
    if ({k: len(v) for k, v in parts.items()} != COUNTS or len({r['source_component'] for r in all_rows}) != 320
            or any(development.FORBIDDEN & r.keys() for r in all_rows)
            or any(r['bin'] != 0 for r in parts['short_final'])
            or [sum(r['bin'] == b for r in parts['long_final']) for b in range(1, 5)] != [16] * 4):
        raise ValueError('complete original256short+64long source-only final population required')
    known_paths = exclusion(all_rows)
    public = reference.verify_public(all_rows)
    tokenizers = body.metadata.tokenizers()
    p = producing(DATA, 'single_use_all320_source_metadata', [*source_paths, *known_paths, reference.PUBLIC_FILE])
    cases = {cohort: [body.metadata.development.prepare_row(r, cohort, *tokenizers) for r in values] for cohort, values in parts.items()}
    rows = [r for values in cases.values() for r in values]
    first = rows[0]
    other = [r for r in rows if r['id'] != first['id']]
    feature_ids = list(dict.fromkeys([first['id'], min(other, key=lambda r: (-len(r['context_ids']), r['id']))['id'],
        min(other, key=lambda r: (-len(r['source_ids']), r['id']))['id']]))
    if len(feature_ids) < 2:
        raise ValueError('distinct full-range final feature probes required')
    write_json(DATA / 'cases.json', cases)
    write_json(DATA / 'selection.json', {'feature_ids': feature_ids,
        'inference_ids': [first['id'], min(other, key=lambda r: (-len(r['source_ids']), r['id']))['id']]})
    write_json(DATA / 'public_proof.json', {'public_file_sha256': reference.PUBLIC_SHA256, 'exact_sources': public})
    native.vendor.verify(p)
    write_json(DATA / 'result.json', {'status': 'all320_original_final_source_metadata_prepared', 'counts': COUNTS,
        'cases_sha256': file_hash(DATA / 'cases.json'), 'source_disjoint': True, 'final_test_opened': True, 'task_accuracy': None})


def sources():
    require_admission()
    native.vendor.terminal(DATA)
    cases = q.read(DATA / 'cases.json')
    if ({k: len(v) for k, v in cases.items()} != COUNTS or file_hash(DATA / 'cases.json') != q.read(DATA / 'result.json')['cases_sha256']
            or any(development.FORBIDDEN & r.keys() for values in cases.values() for r in values)):
        raise ValueError('unchanged complete source-only final population required')
    return cases


def features(pilot=False):
    cases = sources()
    development.training.training.require_gpu()
    rows = [r for values in cases.values() for r in values]
    ids = q.read(DATA / 'selection.json')['feature_ids']
    chosen = [r for r in rows if r['id'] in ids] if pilot else rows
    root = FEATURE_PILOT if pilot else FEATURES
    retained, extra = {}, []
    if not pilot:
        native.vendor.terminal(FEATURE_PILOT)
        retained = q.read(FEATURE_PILOT / 'manifest.json')
        extra = [FEATURE_PILOT / n for n in ('manifest.json', 'result.json', 'reference_parity.json', 'worker_terminal_receipt.json')]
    p = producing(root, 'final_source_feature_pilot' if pilot else 'all320_original_body_full_FP32_features',
        [DATA / n for n in ('cases.json', 'selection.json', 'public_proof.json', 'worker_terminal_receipt.json')] + extra, {'pilot': pilot})
    (root / 'features').mkdir()
    from transformers import AutoModelForCausalLM
    parent = AutoModelForCausalLM.from_pretrained(body.base.MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    if parent.config.num_hidden_layers != 28 or body.base.module_hash(parent.model) != body.base.EXPECTED_BODY:
        raise ValueError('same original frozen body required')
    torch.cuda.reset_peak_memory_stats()
    parity = body.reference_parity(parent)
    manifest = {}
    for i, row in enumerate(chosen, 1):
        if row['id'] in retained:
            entry = retained[row['id']]
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('retained final feature pilot changed')
        else:
            value = body.extract(parent, row)
            if pilot and not torch.equal(value.view(torch.uint8), body.extract(parent, row).contiguous().view(torch.uint8)):
                raise ValueError('repeated final feature extraction differs')
            path = root / 'features' / (row['id'] + '.pt')
            torch.save({**{k: row[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')}, 'features': value}, path)
            entry = {'path': str(path), 'sha256': file_hash(path), 'source_tokens': len(value)}
        manifest[row['id']] = entry
        if pilot or i % 16 == 0 or i == len(chosen):
            print(json.dumps({'final_features': i, 'cases': len(chosen)}), flush=True)
    peak = torch.cuda.max_memory_allocated()
    if body.base.module_hash(parent.model) != body.base.EXPECTED_BODY or any(p.grad is not None for p in parent.parameters()) or peak > 8_600_000_000:
        raise ValueError('frozen original body state or feature memory budget differs')
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'reference_parity.json', parity)
    native.vendor.verify(p)
    write_json(root / 'result.json', {'status': 'final_original_body_features_completed', 'cases': len(manifest),
        'original_body_hash_unchanged': body.base.EXPECTED_BODY, 'all_source_positions_retained': True,
        'peak_allocated_bytes': peak, 'manifest_sha256': file_hash(root / 'manifest.json'), 'task_accuracy': None, 'final_test_opened': True})


def generate(pilot=False):
    cases = sources()
    development.training.training.require_gpu()
    native.vendor.terminal(FEATURES)
    manifest = q.read(FEATURES / 'manifest.json')
    if file_hash(FEATURES / 'manifest.json') != q.read(FEATURES / 'result.json')['manifest_sha256']:
        raise ValueError('complete unchanged final source feature manifest required')
    rows = [r for values in cases.values() for r in values]
    ids = q.read(DATA / 'selection.json')['inference_ids']
    chosen = [r for r in rows if r['id'] in ids] if pilot else rows
    root = PILOT if pilot else OUTPUTS
    retained, extra = {}, []
    if not pilot:
        native.vendor.terminal(PILOT)
        retained = {r['id']: r for r in q.read(PILOT / 'outputs.json')}
        extra = [PILOT / n for n in ('outputs.json', 'result.json', 'worker_terminal_receipt.json')]
    p = producing(root, 'final_head_inference_pilot' if pilot else 'all320_final_source_only_native_predictions',
        [FEATURES / 'manifest.json', DATA / 'cases.json', *[Path(manifest[r['id']]['path']) for r in chosen], *extra])
    model, selected = development.load_selected()
    before = native.vendor.module_hash(model)
    if before != q.read(development.PILOT / 'result.json')['model_sha256']:
        raise ValueError('same complete trained model as admitted DEV required')
    tokenizer, outputs = native.tokenizer(), []
    torch.cuda.reset_peak_memory_stats()
    for i, row in enumerate(chosen, 1):
        value = development.predict(model, row, manifest, tokenizer)
        if (pilot and development.predict(model, row, manifest, tokenizer) != value) or (row['id'] in retained and retained[row['id']] != value):
            raise ValueError('same deterministic native inference required across fresh pilot/full runs')
        outputs.append(value)
        write_json(root / 'outputs.json', outputs)
        if pilot or i % 16 == 0:
            print(json.dumps({'final_generated': i, 'cases': len(chosen)}), flush=True)
    if native.vendor.module_hash(model) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('selected complete native model changed during final inference')
    model.close()
    native.vendor.verify(p)
    write_json(root / 'result.json', {'status': 'final_source_only_native_predictions_completed', 'cases': len(outputs),
        'checkpoint_sha256': selected['checkpoint_sha256'], 'model_sha256': before, 'model_unchanged': True,
        'outputs_sha256': file_hash(root / 'outputs.json'), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'task_accuracy': None, 'final_test_opened': True})


def packets():
    cases = sources()
    native.vendor.terminal(OUTPUTS)
    outputs = q.read(OUTPUTS / 'outputs.json')
    ordered = [r for values in cases.values() for r in values]
    if [r['id'] for r in outputs] != [r['id'] for r in ordered]:
        raise ValueError('every original320source prediction required')
    by_id = {r['id']: r for r in outputs}
    combined = {cohort: [{**r, 'texts': {'student': by_id[r['id']]['text']}} for r in values] for cohort, values in cases.items()}
    payloads = {cohort: {'student': [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts']['student']}])
        for r in values]} for cohort, values in combined.items()}
    public = reference.verify_public(ordered)
    p = producing(PACKETS, 'all320_anonymous_complete_source_semantic_packets', [DATA / 'cases.json', OUTPUTS / 'outputs.json', reference.PUBLIC_FILE])
    write_json(PACKETS / 'cases.json', combined)
    write_json(PACKETS / 'packets.json', payloads)
    write_json(PACKETS / 'public_proof.json', {'public_file_sha256': reference.PUBLIC_SHA256, 'exact_sources': public})
    native.vendor.verify(p)
    write_json(PACKETS / 'result.json', {'status': 'all320_final_public_packets_prepared', 'cases': 320, 'task_accuracy': None})


def grade(pilot=False):
    development.configure()
    semantic.require_judges()
    native.vendor.terminal(PACKETS)
    cases, payloads = q.read(PACKETS / 'cases.json'), q.read(PACKETS / 'packets.json')
    cache = base_cache()
    extra = []
    if not pilot:
        native.vendor.terminal(JUDGE_PILOT)
        prior = q.read(JUDGE_PILOT / 'judgment_cache.json')
        if any(prior.get(k) != v for k, v in cache.items()) or q.read(JUDGE_PILOT / 'result.json')['status'] != 'four_final_judge_firsts_valid_only':
            raise ValueError('unchanged canonical firsts and actual validity-only final judge pilot required')
        cache = prior
        extra = [JUDGE_PILOT / n for n in ('result.json', 'judgment_cache.json', 'worker_terminal_receipt.json')]
    inherited = dict(cache)
    root = JUDGE_PILOT if pilot else GRADE
    p = producing(root, 'four_case_final_judge_validity_pilot' if pilot else 'all320_same_case_final_semantic_quality',
        [PACKETS / n for n in ('cases.json', 'packets.json', 'public_proof.json', 'worker_terminal_receipt.json')] + extra)
    write_json(root / 'judgment_cache.json', cache)
    metrics, scores = {c: {} for c in COUNTS}, {}

    def judge(cohort, role):
        directory = root / cohort
        directory.mkdir(exist_ok=True)
        chosen = cases[cohort][:2] if pilot else cases[cohort]
        oracle.grade_arm(directory, 'student', role, chosen, payloads[cohort], cache)
        values = q.read(directory / f'{role}_student_case_scores.json')
        value = {**development.task_metric(values), 'judge': q.JUDGES[role]}
        write_json(directory / f'{role}_student_result.json', value)
        write_json(root / 'judgment_cache.json', cache)
        metrics[cohort][role], scores[cohort, role] = value, values

    for cohort in COUNTS:
        judge(cohort, 'secondary')
    admitted = not pilot and all(qualifies(metrics[c]['secondary'], COUNTS[c]) for c in COUNTS)
    if admitted:
        for cohort in COUNTS:
            judge(cohort, 'primary')
    joints = {} if pilot else {c: consensus(scores[c, 'secondary'], scores.get((c, 'primary'))) for c in COUNTS}
    qualified = not pilot and all(joints[c]['qualified'] for c in COUNTS)
    valid = all(s['valid'] for values in scores.values() for s in values)
    status = ('four_final_judge_firsts_valid_only' if valid else 'final_judge_pilot_invalid') if pilot else (
        'native655_sealed320_quality_qualified' if qualified else 'native655_sealed320_judgments_invalid' if not valid else 'native655_sealed320_quality_below_95')
    if any(cache.get(k) != v for k, v in inherited.items()):
        raise ValueError('a preceding final/DEV first judgment changed')
    semantic.audit_cache(cache)
    native.vendor.verify(p)
    write_json(root / 'result.json', {'status': status, 'metrics': metrics, 'joint': joints,
        'primary_admitted': admitted, 'quality_gate_passed': qualified, 'canonical_first_judgments': len(cache),
        'inherited_first_judgments': len(inherited), 'cases': 4 if pilot else 320, 'source_prediction': True,
        'oracle_target_codes': False, 'final_test_opened': True, 'speed_tested': False})
    if pilot and not valid:
        raise ValueError('invalid final judge pilot preserved; no new semantic first attempts')


def workflow():
    development.configure()
    require_admission()
    base_cache()
    roots = (WORKFLOW, DATA, FEATURE_PILOT, FEATURES, PILOT, OUTPUTS, PACKETS, JUDGE_PILOT, GRADE, RESULTS)
    if any(p.exists() for p in roots):
        raise ValueError('fresh single-use final workflow required')
    p = producing(WORKFLOW, 'supervise_single_use_source_only_sealed320_quality')
    receipts = []
    stages = [(MODULE, 'prepare', DATA, ''), (MODULE, 'feature-pilot', FEATURE_PILOT, '0'), (MODULE, 'features', FEATURES, '0'),
        (MODULE, 'inference-pilot', PILOT, '0'), (MODULE, 'generate', OUTPUTS, '0'), (MODULE, 'packets', PACKETS, ''),
        (MODULE, 'judge-pilot', JUDGE_PILOT, ''), (MODULE, 'grade', GRADE, ''),
        (AUDITOR_MODULE, 'finalize', RESULTS, ''), (AUDITOR_MODULE, 'register', None, '')]
    for module, stage, root, gpu in stages:
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', module, stage]
        child = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu})
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'child_pid': child.pid, 'observer_pid': os.getpid(), 'observer': 'subprocess.wait', 'command': command}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root is not None and root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('final320stage failed; all evidence retained:' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'native655_sealed320_quality_workflow_completed', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'feature-pilot', 'features', 'inference-pilot', 'generate', 'packets', 'judge-pilot', 'grade', 'workflow'))
    args = parser.parse_args()
    {'prepare': prepare, 'feature-pilot': lambda: features(True), 'features': features, 'inference-pilot': lambda: generate(True),
        'generate': generate, 'packets': packets, 'judge-pilot': lambda: grade(True), 'grade': grade, 'workflow': workflow}[args.stage]()
