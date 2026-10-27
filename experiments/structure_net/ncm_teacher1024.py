"""Collect a fixed fresh task-summary batch, preserving every prior teacher first."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from experiments.structure_net import ncm_broad_teacher_catalog as catalog
from experiments.structure_net import ncm_sol_teacher_continuation as scheduler
from experiments.structure_net import ncm_xsum_semantic_probe as provenance
from experiments.neural_architecture_lab.audit_ncm_evidence_teacher_pilot import replay

data, base = provenance.data, provenance.base
previous, teacher = scheduler.previous, scheduler.previous.original
HYPOTHESIS = 'ncm-fresh-task-teacher1024-v1'
SCHEMA = 'nal.ncm-fresh-task-teacher.v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-teacher1024-plan.md')
TEST = Path('tests/structure_net/test_ncm_teacher1024.py')
SERVICE = Path('experiments/neural_architecture_lab/ncm_teacher1024_service.py')
ROOTS = {s: data.ROOT/f'20261027_teacher1024_{s}_v1'
         for s in ('prepare', 'pilot', 'pilot_audit', 'generate', 'audit')}
INVENTORY = data.ROOT/'20261025_sol_teacher144_partial_inventory_v1'
UNUSED_SHA = '69d8dca0de1d9894ddbc5e1ea6bcc7cad35fb1284170ef6205835c12c086ce1e'
QUOTAS = {0: 512, 1: 128, 2: 128, 3: 128, 4: 128}


def configure():
    training, environment = catalog.configure()
    previous.configure()
    return training, environment


def select(rows, unused, observed, quotas=QUOTAS):
    ids, untouched, old = {r['id'] for r in rows}, set(unused), {r['id'] for r in observed}
    if (len(ids) != len(rows) or len(untouched) != len(unused) or len(old) != len(observed)
            or untouched & old or untouched | old != ids):
        raise ValueError('original sources must partition into unique observed and untouched IDs')
    blocked = {r['source_component'] for r in observed}
    chosen = []
    for band, count in quotas.items():
        values = sorted((r for r in rows if r['id'] in untouched and r['bin'] == band
                         and r['source_component'] not in blocked),
                        key=lambda r: (catalog.text_hash(HYPOTHESIS+':case:'+r['id']), r['id']))
        if len(values) < count:
            raise ValueError('complete fixed source-bin quotas unavailable')
        chosen.extend(values[:count])
    if len({r['source_component'] for r in chosen}) != sum(quotas.values()):
        raise ValueError('one fresh source component per selected example required')
    return chosen


def pilot_ids(rows):
    chosen = [rows[0]['id'], min(rows, key=lambda r: (-r['source_cl100k_tokens'], r['id']))['id']]
    chosen += [next(r['id'] for r in rows if r['bin'] == band) for band in QUOTAS]
    chosen = list(dict.fromkeys(chosen))
    chosen += [r['id'] for r in rows if r['id'] not in chosen][:8-len(chosen)]
    if len(chosen) != 8 or len(set(chosen)) != 8:
        raise ValueError('eight distinct first/longest/all-bin pilot cases required')
    return chosen


def sources():
    previous.require_teacher()
    state = data.completed(INVENTORY)
    if (state['status'] != 'all144_drained_teacher_firsts_independently_replayed'
            or state['unattempted'] != 8487 or not state['all_inflight_drained']
            or state['total_observed_firsts'] != 1513 or state['total_preserved_invalid_firsts'] != 73
            or data.sha(INVENTORY/'unattempted_ids.json') != UNUSED_SHA):
        raise ValueError('complete drained1513-first history and exact8487 unused sources required')
    data.verify(state['artifacts'])
    observed, paths = [], [catalog.SOURCE, catalog.INDEX, catalog.MANIFEST,
        INVENTORY/'unattempted_ids.json', INVENTORY/'result.json', INVENTORY/'actual_tool_receipt.json']
    for dirname, count, digest in catalog.INVENTORIES:
        root = data.ROOT/dirname; p = root/'first_candidates.json'
        if data.sha(p) != digest:
            raise ValueError('historical first inventory changed')
        data.completed(root)
        rows = data.read(p)
        if len(rows) != count:
            raise ValueError('complete prior first population required')
        observed += rows
        paths += [p, root/'result.json', root/'actual_tool_receipt.json']
    component = previous.teacher.GENERATE/'labels.json'
    observed += data.read(component); paths.append(component)
    if len(observed) != 1513 or sum(not r['teacher_valid'] for r in observed) != 73:
        raise ValueError('all observed successes and failures must remain excluded')
    all_rows = data.read(catalog.SOURCE)
    if len(all_rows) != 10000:
        raise ValueError('complete original10000 TRAIN source reservation required')
    selected = select(all_rows, data.read(INVENTORY/'unattempted_ids.json'), observed)
    paths += [previous.qualification.RESULTS/'campaign_results.json', previous.qualification.RESULTS/'backend_actual_tool_receipt.json']
    return selected, paths


def freeze(root, stage, inputs, environment):
    if root.exists():
        raise ValueError('fresh append-only teacher stage required')
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve(), SERVICE.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):
                files.add(p)
    inputs = list(dict.fromkeys([*inputs, Path('pixi.lock'), Path('pyproject.toml')]))
    if any(not p.is_file() for p in (*files, *inputs)):
        raise ValueError('every producer input must exist before stage creation')
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS, 'stage': stage,
        'environment': environment, 'config': {'cases': 1024, 'quotas': QUOTAS, 'workers': 8,
            'models': dict(teacher.MODELS), 'systems': dict(teacher.SYSTEMS),
            'max_attempts_per_stage': 3, 'no_semantic_or_capacity_retries': True,
            'reuse_pilot_firsts': True, 'preserve_prior_observed': 1513, 'preserve_prior_invalid': 73,
            'stop_on_fatal_or_three_consecutive_schema_failures': True, 'drain_inflight': True,
            'feature_cache_bytes': 256*1024**2, 'new_semantic_judgments': False,
            'student_training_admitted': False, 'final_test_opened': False, 'task_accuracy': None},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in inputs}}
    proof['fingerprint'] = base.fingerprint(proof); data.write(root/'producer.json', proof)
    return proof


def finish(root, proof, value, paths):
    base.verify_proof(proof)
    result = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS, **value,
        'student_training_admitted': False, 'semantic_accuracy_measured': False,
        'task_accuracy': None, 'canonical_semantic_cache_changed': False, 'final_test_opened': False,
        'artifacts': {str(p): data.sha(p) for p in [root/'producer.json', *paths]}}
    data.write(root/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}), flush=True)


def require(stage):
    root = ROOTS[stage]; value = base.require(root)
    base.verify_proof(data.read(root/'producer.json'))
    return value


def prepare():
    training, environment = configure(); rows, inputs = sources()
    from experiments.structure_net import ncm_shared_summary_inventory as inventory
    from experiments.structure_net import ncm_shared_summary_source_guard as guard
    from experiments.structure_net.ncm_streaming_source_features import FeatureCache
    protected, guard_paths, counts = inventory.protected_metadata()
    cohorts, _ = guard.development.source_inputs()
    held = [r for name in guard.development.COHORTS for r in cohorts[name]] + data.read(inventory.COMPONENT)
    if len(held) != 192 or counts['reserved_components'] != 512:
        raise ValueError('all512 protected metadata components and192 open sources required')
    text_index = guard.TextIndex([r['source'] for r in held]); held_ids = {r['id'] for r in held}
    encoding = previous.data.storage_encoding()
    for row in rows:
        if (row['split'] != 'train' or row['source_truncated']
                or catalog.text_hash(row['source']) != row['source_sha256']
                or len(encoding.encode(row['source'], disallowed_special=())) != row['source_cl100k_tokens']
                or not 192 <= row['source_cl100k_tokens'] <= 2048
                or catalog.protected_reasons(row, protected, inventory.reservation.graph.guard.source_family)
                or row['id'] in held_ids or text_index.matches(row['source'])):
            raise ValueError('complete source or held-out source guard failed')
    public = previous.qualification.reference
    public_proof = public.verify_public(rows)
    selected = {r['id'] for r in rows}
    index = [r for r in data.read(catalog.INDEX) if r['id'] in selected]
    manifest = {k:v for k,v in data.read(catalog.MANIFEST).items() if k in selected}
    if len(index) != 1024 or set(manifest) != selected:
        raise ValueError('complete selected existing source features required')
    inputs += [*guard_paths, guard.development.SOURCE/'cases.json', public.PUBLIC_FILE]
    inputs += [Path(r['path']) for r in index]
    inputs += [Path(v[k]) for v in manifest.values() for k in ('header_path', 'payload_path')]
    root = ROOTS['prepare']; proof = freeze(root, 'guard_sources_and_replay_features', inputs, environment)
    cache = FeatureCache(index, manifest, max_bytes=256*1024**2)
    total = positions = 0
    with (root/'payloads.jsonl').open('x') as log:
        for i, row in enumerate(rows, 1):
            value = cache.get(row)
            if str(value.dtype) != 'torch.float32' or not training.torch.isfinite(value).all():
                raise ValueError('complete finite original FP32 source feature required')
            size = value.numel()*value.element_size(); total += size; positions += len(value)
            log.write(json.dumps({'id': row['id'], 'raw_bytes': size, 'positions': len(value),
                'payload_sha256': manifest[row['id']]['payload_sha256']})+'\n')
            if i % 128 == 0: print(json.dumps({'features_replayed': i, 'total': 1024}), flush=True)
    for name, value in (('cases.json', rows), ('pilot_ids.json', pilot_ids(rows)),
                        ('source_index.json', index), ('feature_manifest.json', manifest),
                        ('public_proof.json', public_proof)):
        data.write(root/name, value)
    finish(root, proof, {'status': 'fresh1024_sources_guarded_features_replayed', 'cases': 1024,
        'source_bins': dict(Counter(r['bin'] for r in rows)), 'protected_metadata': counts,
        'open_protected_sources': 192, 'raw_feature_bytes_replayed': total, 'source_positions': positions,
        'feature_cache': cache.metrics(), 'new_requests': 0},
        [root/n for n in ('cases.json', 'pilot_ids.json', 'source_index.json', 'feature_manifest.json', 'public_proof.json', 'payloads.jsonl')])


def generate(pilot=False):
    _, environment = configure(); require('prepare')
    all_rows = data.read(ROOTS['prepare']/'cases.json')
    ids = set(data.read(ROOTS['prepare']/'pilot_ids.json')); inherited = {}
    inputs = [ROOTS['prepare']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    if not pilot:
        state = require('pilot_audit')
        if not state['generation_admitted']:
            raise ValueError('all eight original pilot firsts must qualify before further submission')
        inherited = {r['id']:r for r in data.read(ROOTS['pilot']/'labels.json')}
        inputs += [ROOTS['pilot_audit']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
        inputs.append(ROOTS['pilot']/'labels.json')
    selected = [r for r in all_rows if r['id'] in ids] if pilot else [r for r in all_rows if r['id'] not in inherited]
    root = ROOTS['pilot' if pilot else 'generate']; proof = freeze(root, 'pilot' if pilot else 'generate', inputs, environment)
    (root/'cases').mkdir()
    values = scheduler.candidate_jobs(selected, root, previous.data.storage_encoding(), previous.native.tokenizer())
    observed = {r['id']:r for r in values}; observed.update(inherited)
    labels = [observed[r['id']] for r in (selected if pilot else all_rows)]
    data.write(root/'labels.json', labels)
    valid, capacity = sum(r['teacher_valid'] for r in labels), sum(r['capacity_fit'] for r in labels)
    finish(root, proof, {'status': 'teacher_pilot_completed' if pilot else 'all1024_first_candidates_completed',
        'cases': len(labels), 'teacher_valid': valid, 'capacity_fit': capacity,
        'pilot': pilot, 'new_candidates': len(values), 'raw_replay_pending': True},
        [root/'labels.json', root/'partial_firsts.json', root/'failures.json', *sorted((root/'requests').rglob('*.json'))])


def replay_candidate(source, value, encoding, tokenizer):
    if any(value.get(k) != v for k,v in source.items()):
        raise ValueError('teacher source changed')
    draft = replay(source, 'draft', value['draft'], encoding)
    if draft is None and value['review'] is not None:
        raise ValueError('review cannot precede a valid draft')
    review = replay(source, 'review', value['review'], encoding, draft) if draft is not None else None
    text = review['summary'] if review is not None else ''
    lexical = tokenizer.encode(text, add_special_tokens=False)
    stored = len(encoding.encode(text, disallowed_special=()))
    fits = bool(text.strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0,101}
    expected = {'target_text':text, 'teacher_valid':review is not None, 'capacity_fit':fits,
        'stored_tokens':stored, 'native_lexical_tokens':len(lexical),
        'native_word_ids':[101,*lexical,101]+[0]*(126-len(lexical)) if fits else None, 'semantic_qualified':False}
    if any(value[k] != v for k,v in expected.items()):
        raise ValueError('first teacher response or complete capacity replay differs')


def audit(pilot=False):
    _, environment = configure(); require('prepare')
    stage = 'pilot' if pilot else 'generate'; source_state = require(stage)
    all_rows = data.read(ROOTS['prepare']/'cases.json'); ids = set(data.read(ROOTS['prepare']/'pilot_ids.json'))
    rows = [r for r in all_rows if r['id'] in ids] if pilot else all_rows
    values = data.read(ROOTS[stage]/'labels.json')
    if len(values) != (8 if pilot else 1024) or [r['id'] for r in values] != [r['id'] for r in rows]:
        raise ValueError('complete fixed original source order required')
    root = ROOTS['pilot_audit' if pilot else 'audit']
    proof = freeze(root, 'replay_pilot' if pilot else 'replay_all1024',
        [ROOTS[stage]/n for n in ('producer.json','result.json','actual_tool_receipt.json','labels.json')], environment)
    encoding, tokenizer = previous.data.storage_encoding(), previous.native.tokenizer()
    for source, value in zip(rows, values): replay_candidate(source, value, encoding, tokenizer)
    if not pilot:
        require('pilot_audit'); by_id = {r['id']:r for r in values}
        if any(by_id[r['id']] != r for r in data.read(ROOTS['pilot']/'labels.json')):
            raise ValueError('pilot firsts replaced')
    valid, capacity = sum(r['teacher_valid'] for r in values), sum(r['capacity_fit'] for r in values)
    if source_state['teacher_valid'] != valid or source_state['capacity_fit'] != capacity:
        raise ValueError('candidate counts differ from raw replay')
    finish(root, proof, {'status':'teacher_pilot_firsts_replayed' if pilot else 'all1024_teacher_firsts_replayed',
        'cases':len(values),'teacher_valid':valid,'capacity_fit':capacity,
        'generation_admitted':pilot and valid == capacity == 8,'new_requests':0,
        'preserved_prior_firsts':1513,'preserved_prior_invalid_firsts':73}, [])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=tuple(ROOTS))
    stage = parser.parse_args().stage; root = ROOTS[stage]; existed = root.exists()
    try:
        if stage == 'prepare': prepare()
        elif stage in ('pilot', 'generate'): generate(stage == 'pilot')
        else: audit(stage == 'pilot_audit')
    except Exception as error:
        if root.exists() and not existed:
            data.write(root/'failure.json', {'error_type':type(error).__name__, 'error':str(error),
                'preserve_firsts':True,'retry_same_sources_prohibited':True})
        raise
