"""Source-grounded semantic comparison of four frozen XSum probe outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

from experiments.structure_net import ncm_xsum_output_distillation as parent
from experiments.structure_net import ncm_sol_judgment_lineage as qualified
from experiments.structure_net import ncm_source_readout_grade as first

data, base, q = parent.data, parent.base, qualified.q
HYPOTHESIS = 'ncm-xsum-semantic-probe-v1'
SCHEMA = 'nal.ncm-xsum-semantic-probe.v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-xsum-semantic-probe-plan.md')
TEST = Path('tests/structure_net/test_ncm_xsum_semantic_probe.py')
ANALYSIS = Path('experiments/neural_architecture_lab/ncm_xsum_semantic_probe_results.py')
ROOTS = {s: data.ROOT/f'20261027_xsum_semantic_probe_{s}_v1' for s in ('prepare', 'pilot', 'grade', 'audit', 'results')}
ARMS = ('native', 'initial', 'mse', 'distill')
PRIOR = data.ROOT/'20261026_public_curriculum_epoch2_DEV_grade_v1'
PRIOR_SHA = 'af236453163020508089734901b7ce247da602cf53a3e4472ddb6d442f49e908'
FRONTIER = data.ROOT/'20261025_grounded_semantic_stop_inventory_v1/request_frontier.json'


def configure():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU/network only')
    qualified.configure()
    if q.JUDGES != {'primary': 'codex_sdk/gpt-5.6-sol', 'secondary': 'codex_sdk/gpt-5.6-terra'}:
        raise ValueError('current qualified judges required')
    if q.RUBRIC != qualified.qualification.profile.RUBRIC: raise ValueError('qualified 128-token rubric required')


def preserve(prior, current):
    if any(current.get(k) != v for k, v in prior.items()): raise ValueError('inherited first judgment changed or lost')


def prior_cache():
    if data.sha(PRIOR/'judgment_cache.json') != PRIOR_SHA: raise ValueError('4691-first parent changed')
    receipt = data.read(PRIOR/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(PRIOR/'judgment_cache.json')) != PRIOR_SHA:
        raise ValueError('actual parent grade receipt required')
    data.verify(receipt['artifacts'])
    cache = data.read(PRIOR/'judgment_cache.json')
    if len(cache) != 4691 or data.read(PRIOR/'result.json')['canonical_first_judgments'] != 4691:
        raise ValueError('complete parent first-decision lineage required')
    qualified.audit_cache(cache)
    return cache


def packets_for(cases):
    return {a: [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts'][a]}], role='secondary')
                for r in cases] for a in ARMS}


def verify_source(row, selected, entry, tokenizer):
    source = row['source']; digest = hashlib.sha256(source.encode()).hexdigest()
    if (row['id'] != selected['id'] or entry['id'] != selected['id']
            or digest != selected['source_sha256'] or digest != row['source_sha256']
            or digest != entry['source_sha256'] or row['source_truncated']
            or row['source_component'] != selected['source_component']
            or row['public_provenance'] != entry['public_provenance']
            or row['public_provenance']['dataset'] != 'EdinburghNLP/xsum'
            or row['public_provenance']['split'] != 'train'
            or row['public_provenance']['revision'] != '7d4d486c2f8ef850b1a11aead99b894ff3dd7da9'
            or tokenizer.encode(source, add_special_tokens=False) != selected['source_ids']
            or tokenizer.decode(selected['source_ids']) != source):
        raise ValueError('complete exact public source/feature/provenance binding differs')


def freeze(root, stage, inputs):
    if root.exists(): raise ValueError('fresh append-only stage required')
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve(), ANALYSIS.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()): files.add(p)
    inputs = [*inputs, *qualified.judge_inputs(), PRIOR/'result.json', PRIOR/'actual_tool_receipt.json',
              PRIOR/'judgment_cache.json', FRONTIER, Path('pixi.lock'), Path('pyproject.toml')]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)): (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS, 'stage': stage,
        'config': {'arms': ARMS, 'probe_cases': 128, 'pilot_cases': 4, 'workers': 4,
            'judges': dict(q.JUDGES), 'rubric': q.RUBRIC, 'inherited_first_judgments': 4691,
            'stored_token_cap': 128, 'primary_gate': 122, 'directional_minimum_extra_passes': 7,
            'final_test_opened': False, 'speed_tested': False, 'public_reused_probe': True},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in dict.fromkeys(inputs)}}
    proof['fingerprint'] = base.fingerprint(proof); data.write(root/'producer.json', proof)
    return proof


def finish(root, proof, value, files):
    base.verify_proof(proof)
    data.write(root/'result.json', {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS, **value,
        'application_task_accuracy': None, 'final_test_opened': False, 'speed_tested': False,
        'model_training': False, 'NCM_success_established': False,
        'artifacts': {str(p): data.sha(p) for p in [root/'producer.json', *files]}})
    print(json.dumps({'stage': proof['stage'], 'status': value['status'],
                      'canonical_first_judgments': value.get('canonical_first_judgments')}), flush=True)


def tokenizer():
    from transformers import AutoTokenizer
    from experiments.structure_net.ncm_llama import GPT2
    value = AutoTokenizer.from_pretrained(GPT2, local_files_only=True)
    value.add_special_tokens({'pad_token': '<|pad|>'}); return value


def prepare():
    configure(); qualified.require_judges(); inherited = prior_cache()
    selected = parent.parent(); base.require(parent.ROOTS['evaluate']); base.require(parent.ROOTS['audit'])
    receipt = data.read(parent.ROOTS['results']/'backend_exact_readback.json')
    campaign = parent.ROOTS['results']/'campaign_results.json'
    if receipt['campaign_sha256'] != data.sha(campaign): raise ValueError('registered parent comparison changed')
    rows = data.read(parent.ROOTS['evaluate']/'outputs.json')
    if [r['id'] for r in rows] != [r['id'] for r in selected['probe']]: raise ValueError('all128 unchanged outputs required')
    if set(r['source_sha256'] for r in selected['fit']) & set(r['source_sha256'] for r in selected['probe']):
        raise ValueError('adapter fit/probe source overlap')
    manifest = data.read(base.bundle.ROOT/'manifest.json'); data.verify(manifest['index_hashes'])
    index = {r['id']: r for r in base.bundle.lines(manifest['pair_index'])}
    gpt = tokenizer(); cases = []; from experiments.structure_net.ncm_llama import decode_ids
    with Path(manifest['pairs']).open('rb') as source_file:
        for chosen, prediction in zip(selected['probe'], rows):
            entry = index[chosen['id']]; source_file.seek(entry['offset_bytes']); raw = source_file.read(entry['length_bytes'])
            if hashlib.sha256(raw).hexdigest() != entry['record_sha256']: raise ValueError('catalog source record changed')
            row = json.loads(raw); verify_source(row, chosen, entry, gpt)
            decoded = {a: decode_ids(prediction[a]['words'], gpt) for a in ARMS}
            cases.append({'id': row['id'], 'source': row['source'], 'source_sha256': row['source_sha256'],
                'source_component': row['source_component'], 'public_provenance': row['public_provenance'],
                'catalog_entry': entry, 'source_ids': chosen['source_ids'],
                'texts': {a: decoded[a][0] for a in ARMS}, 'termination': {a: decoded[a][1] for a in ARMS},
                'raw_words': {a: prediction[a]['words'] for a in ARMS}})
    packets = packets_for(cases)
    uncertain = {r['key'] for r in data.read(FRONTIER) if r['status'] == 'unknown_inflight_or_unstarted'}
    if any(first.packet_key(p, role) in uncertain for ps in packets.values() for p in ps for role in ('secondary', 'primary')):
        raise ValueError('an uncertain prior request cannot receive a fresh attempt budget')
    root = ROOTS['prepare']
    inputs = [parent.ROOTS['evaluate']/n for n in ('result.json', 'actual_tool_receipt.json', 'outputs.json')]
    inputs += [parent.ROOTS['results']/n for n in ('campaign_results.json', 'backend_exact_readback.json', 'backend_actual_tool_receipt.json')]
    inputs += [base.ROOTS['prepare']/'cases.json', base.bundle.ROOT/'manifest.json', Path(manifest['pair_index']), Path(manifest['pairs'])]
    proof = freeze(root, 'prepare', inputs)
    data.write(root/'cases.json', cases); data.write(root/'exact_packets.json', packets)
    data.write(root/'public_payload_proof.json', {'dataset': 'EdinburghNLP/xsum',
        'revision': '7d4d486c2f8ef850b1a11aead99b894ff3dd7da9', 'split': 'train',
        'case_ids': [r['id'] for r in cases], 'source_sha256': {r['id']: r['source_sha256'] for r in cases},
        'source_fields_public_exact': True, 'source_truncated': False, 'private_data_in_packets': False,
        'candidate_texts_locally_derived': True, 'exact_packets_sha256': data.sha(root/'exact_packets.json'),
        'judges': dict(q.JUDGES), 'rubric_sha256': hashlib.sha256(q.RUBRIC.encode()).hexdigest(),
        'gateway': 'https://central-dev.zt:4000/v1/chat/completions', 'source_files': manifest['pairs']})
    finish(root, proof, {'status': 'public_semantic_probe_prepared', 'cases': 128, 'arm_cases': 512,
        'inherited_first_judgments': len(inherited), 'new_judge_requests': 0},
        [root/'cases.json', root/'exact_packets.json', root/'public_payload_proof.json'])


def prepared():
    base.require(ROOTS['prepare']); base.verify_proof(data.read(ROOTS['prepare']/'producer.json'))
    cases = data.read(ROOTS['prepare']/'cases.json'); packets = data.read(ROOTS['prepare']/'exact_packets.json')
    proof = data.read(ROOTS['prepare']/'public_payload_proof.json')
    if len(cases) != 128 or packets != packets_for(cases) or proof['exact_packets_sha256'] != data.sha(ROOTS['prepare']/'exact_packets.json'):
        raise ValueError('fixed complete public packets changed')
    for case in cases:
        if hashlib.sha256(case['source'].encode()).hexdigest() != case['source_sha256']:
            raise ValueError('source text changed')
    return cases, packets


def metric(scores):
    if not scores or len({s['id'] for s in scores}) != len(scores): raise ValueError('complete unique scored population required')
    n = len(scores); valid = sum(s['valid'] for s in scores); passed = sum(s['passed'] is True for s in scores)
    if any((not s['valid'] and s['passed'] is not None) or (s['valid'] and type(s['passed']) is not bool) for s in scores):
        raise ValueError('invalid decisions must remain unresolved')
    return {'cases': n, 'valid': valid, 'invalid': n-valid, 'passed': passed,
        'semantic_pass_rate': passed/n if valid == n else None,
        'all_case_pass_lower_bound': passed/n, 'all_case_pass_upper_bound': (passed+n-valid)/n,
        'wilson_95': q.wilson(passed, n) if valid == n else None,
        'components': {k: sum(s['valid'] and s['evaluations'][0][k] is True for s in scores) for k in ('faithful','salient','readable')},
        'stored_overflow': sum(s['tokens'] > 128 for s in scores), 'empty': sum(not s['text'].strip() for s in scores)}


def qualifies(value):
    return value['cases'] == 128 and value['valid'] == 128 and value['passed'] >= 122


def score_rows(cases, packets, cache, arm, role):
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base'); scores = []
    for case, packet in zip(cases, packets):
        key = first.packet_key(packet, role); entry = cache[key]; first.audit_row(packet, role, entry['row'])
        row = entry['row']; text = case['texts'][arm]; n = len(enc.encode(text, disallowed_special=()))
        valid = row['status'] == 'valid'
        scores.append({'id': case['id'], 'text': text, 'tokens': n, 'valid': valid,
            'passed': q.passes(row['evaluations'][0], text, n) if valid else None,
            'evaluations': row.get('evaluations'), 'packet_key': key, 'first_origin': entry['origin']})
    return scores


def grade_arm(root, cases, packets, cache, arm, role, journal):
    missing = {first.packet_key(p, role): p for p in packets if first.packet_key(p, role) not in cache}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {}
        for key, packet in missing.items():
            request_root = root/'requests'/role/key
            (request_root/'attempts').mkdir(parents=True, exist_ok=False)
            jobs[pool.submit(q.request, packet, role, request_root)] = (key, packet, request_root)
        for i, future in enumerate(as_completed(jobs), 1):
            key, packet, request_root = jobs[future]; row = future.result()
            data.write(request_root/'row.json', row)
            first.audit_row(packet, role, row)
            entry = {'packet': packet, 'role': role, 'row': row, 'origin': str(request_root)}
            first.remember(cache, key, entry)
            journal.write(json.dumps({'key': key, 'row_path': str(request_root/'row.json'), 'row_sha256': data.sha(request_root/'row.json')})+'\n')
            journal.flush(); os.fsync(journal.fileno())
            if i % 8 == 0 or i == len(missing): print(json.dumps({'arm': arm, 'role': role, 'new_completed': i, 'new_total': len(missing)}), flush=True)
    scores = score_rows(cases, packets, cache, arm, role); value = metric(scores)
    data.write(root/f'{role}_{arm}_scores.json', scores)
    return {**value, 'judge': q.JUDGES[role], 'new_requests': len(missing), 'reused_first_judgments': len(cases)-len(missing)}


def grade(pilot=False):
    configure(); qualified.require_judges(); cases, packets = prepared(); inherited = prior_cache(); cache = dict(inherited)
    inputs = [ROOTS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json','exact_packets.json','public_payload_proof.json')]
    if not pilot:
        state = base.require(ROOTS['pilot'])
        if state['status'] != 'systems_semantic_pilot_passed': raise ValueError('all pilot judgments must be valid')
        pilot_cache = data.read(ROOTS['pilot']/'judgment_cache.json'); preserve(inherited, pilot_cache)
        for key, entry in pilot_cache.items(): first.remember(cache, key, entry)
        inputs += [ROOTS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','judgment_cache.json')]
    root = ROOTS['pilot' if pilot else 'grade']; proof = freeze(root, 'pilot' if pilot else 'grade', inputs)
    count = 4 if pilot else 128; cases = cases[:count]; metrics = {}
    with (root/'new_firsts.jsonl').open('x') as journal:
        for arm in ARMS:
            metrics[arm] = {'secondary': grade_arm(root, cases, packets[arm][:count], cache, arm, 'secondary', journal), 'primary': None}
        if not pilot:
            for arm in ARMS:
                if qualifies(metrics[arm]['secondary']):
                    metrics[arm]['primary'] = grade_arm(root, cases, packets[arm], cache, arm, 'primary', journal)
    preserve(inherited, cache); data.write(root/'judgment_cache.json', cache)
    status = ('systems_semantic_pilot_passed' if all(v['secondary']['valid'] == 4 for v in metrics.values())
              else 'systems_semantic_pilot_failed') if pilot else 'semantic_probe_grade_completed'
    finish(root, proof, {'status': status, 'metrics': metrics, 'inherited_first_judgments': 4691,
        'canonical_first_judgments': len(cache), 'new_judge_requests': sum(v[r]['new_requests'] for v in metrics.values() for r in v if v[r]),
        'pilot': pilot, 'cases_per_arm': count}, [root/'judgment_cache.json',root/'new_firsts.jsonl',
            *sorted(root.glob('*_scores.json')), *sorted((root/'requests').rglob('*.json'))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=('prepare','pilot','grade'))
    stage = parser.parse_args().stage; root = ROOTS[stage]; existed = root.exists()
    try: prepare() if stage == 'prepare' else grade(stage == 'pilot')
    except Exception as error:
        if root.exists() and not existed: data.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
