"""Admit unchanged new teacher firsts alongside the exact existing TRAIN catalog."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from experiments.structure_net import ncm_teacher1024_recovery as collection
from experiments.structure_net import ncm_broad_teacher_catalog as old
from experiments.structure_net import ncm_shared_summary_inventory as inventory
from experiments.structure_net import ncm_shared_summary_source_guard as guard
from experiments.structure_net.ncm_streaming_source_features import FeatureCache

data, base = collection.data, collection.base
HYPOTHESIS = 'ncm-expanded-reviewed-catalog-v1'
SCHEMA = 'nal.ncm-expanded-reviewed-catalog.v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-expanded-teacher-catalog-plan.md')
TEST = Path('tests/structure_net/test_ncm_expanded_teacher_catalog.py')
ROOTS = {s:data.ROOT/f'20261027_expanded_teacher_catalog_{s}_v1' for s in ('pilot','prepare','audit')}
OLD_CASES_SHA = 'cd3d2a8a1fe0d398d1ca951e0df382d05c47cefe4d07892107b23d6e1c7e5914'
MIN_NEW_FOR_TRAINING = 900


def training_admission(observed, admitted, *, pilot):
    if any(type(n) is not int for n in (observed,admitted)) or not 0 <= admitted <= observed <= 1024:
        raise ValueError('complete bounded observation and admission counts required')
    return not pilot and observed == 1024 and admitted >= MIN_NEW_FOR_TRAINING


def configured():
    collection.install()
    return collection.configure()


def require_old():
    state, features = data.completed(old.DEST), data.completed(old.AUDIT)
    proof = data.read(old.DEST/'producer.json')
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    data.verify(state['artifacts'])
    if (data.sha(old.DEST/'cases.json') != OLD_CASES_SHA or state['retained'] != 1376
            or state['observed'] != 1449 or state['original_invalid_firsts'] != 73
            or features['status'] != 'all_catalog_feature_payloads_replayed'
            or features['cases'] != 1376 or features['raw_bytes_replayed'] != 14234284032
            or features['catalog_sha256'] != data.sha(old.DEST/'result.json')
            or features['payload_log_sha256'] != data.sha(old.AUDIT/'payloads.jsonl')):
        raise ValueError('complete unchanged1376-pair catalog and feature audit required')
    return [old.DEST/n for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json',
        'source_index.json','feature_manifest.json','observed_firsts.json','excluded.json')] + [old.AUDIT/n for n in
        ('producer.json','result.json','actual_tool_receipt.json','payloads.jsonl')]


def merge_bindings(old_index, old_manifest, new_index, new_manifest, admitted_ids):
    old_ids, new_ids = {r['id'] for r in old_index}, {r['id'] for r in new_index}
    if (len(old_ids) != len(old_index) or len(new_ids) != len(new_index)
            or old_ids != set(old_manifest) or new_ids != set(new_manifest) or old_ids & new_ids
            or not old_ids <= admitted_ids or not admitted_ids <= old_ids | new_ids):
        raise ValueError('disjoint complete old/new feature identities and preserved old population required')
    index = [r for r in (*old_index,*new_index) if r['id'] in admitted_ids]
    manifest = {k:v for k,v in {**old_manifest,**new_manifest}.items() if k in admitted_ids}
    return index, manifest


def materialize(previous, sources, candidates, protected, held, tokenizer, encoding, origin):
    old_ids = {r['id'] for r in previous}; ids = [r['id'] for r in sources]
    if (len(old_ids) != len(previous) or len(set(ids)) != len(ids)
            or [r['id'] for r in candidates] != ids or old_ids & set(ids)):
        raise ValueError('complete ordered disjoint first candidates required')
    family = inventory.reservation.graph.guard.source_family
    text_index = guard.TextIndex([r['source'] for r in held]); held_ids = {r['id'] for r in held}
    seen_sources = {r['source_sha256'] for r in previous}
    seen_components = {r['source_component'] for r in previous}
    rows, excluded, observed = list(previous), [], []
    for position,(source,value) in enumerate(zip(sources,candidates)):
        if (any(value.get(k) != v for k,v in source.items()) or source['split'] != 'additional_train'
                or source['source_truncated'] or old.text_hash(source['source']) != source['source_sha256']):
            raise ValueError('unaltered complete reserved source required')
        collection.original.replay_candidate(source,value,encoding,tokenizer)
        lexical = tokenizer.encode(value['target_text'],add_special_tokens=False)
        stored = len(encoding.encode(value['target_text'],disallowed_special=()))
        reasons = old.numeric_reasons(value,lexical,set(tokenizer.all_special_ids),stored)
        reasons += old.protected_reasons(value,protected,family)
        if value['id'] in held_ids:reasons.append('reserved_component_or_DEV_id')
        conflicts = text_index.matches(value['source'])
        if conflicts:reasons.append('open_heldout_text_overlap')
        if value['source_sha256'] in seen_sources:reasons.append('exact_source_duplicate')
        if value['source_component'] in seen_components:reasons.append('source_component_duplicate')
        provenance = {'inventory':str(origin),'inventory_position':position,'original_id':value['id']}
        observed.append({'id':value['id'],'teacher_valid':value['teacher_valid'],
            'capacity_fit':value['capacity_fit'],'admitted':not reasons,'reasons':reasons,**provenance})
        if reasons:
            excluded.append({'id':value['id'],'reasons':reasons,**provenance,
                'text_conflicts':[{'heldout_id':held[i]['id'],**p} for i,p in conflicts]})
            continue
        seen_sources.add(value['source_sha256']);seen_components.add(value['source_component'])
        rows.append({**value,'split':'TRAIN','target_sha256':old.text_hash(value['target_text']),
            'teacher_first_provenance':provenance,'target_origin':'unchanged_reviewed_complete_summary_first',
            'semantic_qualified':False})
    if rows[:len(previous)] != previous:
        raise ValueError('historical TRAIN rows must remain exactly unchanged')
    return rows, excluded, observed


def inputs_and_catalog(pilot):
    paths = require_old()
    collector = collection.original
    ready = collector.require('pilot_audit' if pilot else 'audit')
    expected = 8 if pilot else 1024
    if (ready['status'] != ('teacher_pilot_firsts_replayed' if pilot else 'all1024_teacher_firsts_replayed')
            or ready['cases'] != expected or ready['new_requests'] != 0
            or (pilot and not ready['generation_admitted'])):
        raise ValueError('complete independently replayed first-candidate population required')
    collector.require('prepare')
    root = collection.ROOTS['pilot' if pilot else 'generate']
    collector.require('pilot' if pilot else 'generate')
    sources = data.read(collection.ROOTS['prepare']/'cases.json')
    if pilot:
        selected = set(data.read(collection.ROOTS['prepare']/'pilot_ids.json'))
        sources = [r for r in sources if r['id'] in selected]
    candidates = data.read(root/'labels.json')
    if len(candidates) != expected:
        raise ValueError('partial collection cannot be substituted for the fixed population')
    protected, guard_paths, counts = inventory.protected_metadata()
    cohorts,_ = guard.development.source_inputs()
    held = [r for name in guard.development.COHORTS for r in cohorts[name]] + data.read(inventory.COMPONENT)
    if counts['reserved_components'] != 512 or len(held) != 192:
        raise ValueError('complete protected metadata and open-source text guards required')
    tokenizer = collection.previous.native.tokenizer();encoding = collection.previous.data.storage_encoding()
    previous = data.read(old.DEST/'cases.json')
    rows, excluded, observed = materialize(previous,sources,candidates,protected,held,tokenizer,encoding,root/'labels.json')
    old_index, old_manifest = [data.read(old.DEST/n) for n in ('source_index.json','feature_manifest.json')]
    new_index, new_manifest = [data.read(collection.ROOTS['prepare']/n) for n in ('source_index.json','feature_manifest.json')]
    index, manifest = merge_bindings(old_index,old_manifest,new_index,new_manifest,{r['id'] for r in rows})
    lookup = {r['id']:r for r in index}
    for row in rows:
        entry = lookup[row['id']];feature = manifest[row['id']]
        if (any(entry[k] != row[k] for k in ('source_id','source_sha256','source_component','group_id'))
                or feature['source_tokens'] != entry['source_positions']
                or feature['raw_bytes'] != entry['source_positions']*3072*4):
            raise ValueError('complete original FP32 feature/source binding differs')
    stages = ('prepare','pilot','pilot_audit') if pilot else ('prepare','pilot','pilot_audit','generate','audit')
    paths += [collection.ROOTS[s]/n for s in stages for n in ('producer.json','result.json','actual_tool_receipt.json')]
    paths += [collection.ROOTS['prepare']/n for n in ('cases.json','source_index.json','feature_manifest.json')]
    paths += [root/'labels.json',*guard_paths,guard.development.SOURCE/'cases.json']
    paths += [Path(p) for v in candidates for stage in ('draft','review') if v[stage]
              for p in v[stage]['attempts']]
    arms = {'existing':[r['id'] for r in previous],'expanded':[r['id'] for r in rows],
            'new_admitted':[r['id'] for r in rows[len(previous):]]}
    return {'cases.json':rows,'excluded_new.json':excluded,'observed_new_firsts.json':observed,
        'source_index.json':index,'feature_manifest.json':manifest,'arm_ids.json':arms,
        'protected_metadata.json':counts},paths


def freeze(root,stage,environment,paths):
    if root.exists():raise ValueError('fresh append-only catalog stage required')
    files = {Path(__file__).resolve(),PLAN.resolve(),TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):files.add(p)
    paths = list(dict.fromkeys([*paths,Path('pixi.lock'),Path('pyproject.toml')]))
    hashes = {str(p):data.sha(p) for p in paths}
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'stage':stage,'environment':environment,
        'config':{'old_rows_unchanged':1376,'old_invalid_firsts_preserved':73,'new_candidates':1024,
            'pilot_new_candidates':8,'minimum_new_training_pairs':MIN_NEW_FOR_TRAINING,
            'individual_semantic_gold':False,'final_test_opened':False,'new_judge_requests':0},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},'input_hashes':hashes}
    proof['fingerprint'] = base.fingerprint(proof);data.write(root/'producer.json',proof)
    return proof


def finish(root,proof,value,paths):
    base.verify_proof(proof)
    data.write(root/'result.json',{'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,**value,
        'new_judge_requests':0,'model_training':False,'task_accuracy':None,
        'individual_semantic_gold':False,'final_test_opened':False,
        'artifacts':{str(p):data.sha(p) for p in [root/'producer.json',*paths]}})
    print(json.dumps(value),flush=True)


def prepare(pilot=False):
    training,environment = configured()
    if not pilot:
        status = base.require(ROOTS['pilot']);base.verify_proof(data.read(ROOTS['pilot']/'producer.json'))
        if status['status'] != 'expanded_catalog_eight_first_lifecycle_passed':raise ValueError('real first-candidate catalog pilot required')
    values, paths = inputs_and_catalog(pilot)
    if not pilot:paths += [ROOTS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    stage = 'pilot' if pilot else 'prepare';root = ROOTS[stage];proof = freeze(root,stage,environment,paths)
    for name,value in values.items():data.write(root/name,value)
    rows = values['cases.json'];new = values['arm_ids.json']['new_admitted']
    feature_status = None
    if pilot:
        old_ids = values['arm_ids.json']['existing'];lookup = {r['id']:r for r in rows}
        longest = max((r for r in values['source_index.json'] if r['id'] in set(old_ids)),key=lambda r:r['source_positions'])['id']
        selected = list(dict.fromkeys([*old_ids[:15],longest,*new]))
        cache = FeatureCache(values['source_index.json'],values['feature_manifest.json'],max_bytes=256*1024**2)
        for identity in selected:
            value = cache.get(lookup[identity])
            if str(value.dtype) != 'torch.float32' or not training.torch.isfinite(value).all():raise ValueError('finite full original feature required')
        feature_status = {'cases_replayed':len(selected),'all_new_admitted_replayed':len(new),'cache':cache.metrics()}
    finish(root,proof,{'status':'expanded_catalog_eight_first_lifecycle_passed' if pilot else 'expanded_catalog_materialized_audit_pending',
        'previous_cases':1376,'new_observed':len(values['observed_new_firsts.json']),'new_admitted':len(new),
        'new_excluded':len(values['excluded_new.json']),'total_pairs':len(rows),'pilot':pilot,
        'training_data_admitted':False,'feature_lifecycle':feature_status},[root/name for name in values])


def audit():
    training,environment = configured();root = ROOTS['audit'];source = ROOTS['prepare']
    base.require(source);base.verify_proof(data.read(source/'producer.json'))
    values, paths = inputs_and_catalog(False)
    for name,expected in values.items():
        if data.read(source/name) != expected:raise ValueError('independent full catalog reconstruction differs: '+name)
    paths += [source/n for n in ('producer.json','result.json','actual_tool_receipt.json',*values)]
    proof = freeze(root,'audit',environment,paths)
    cache = FeatureCache(values['source_index.json'],values['feature_manifest.json'],max_bytes=256*1024**2)
    total = positions = 0
    with (root/'payloads.jsonl').open('x') as log:
        for i,row in enumerate(values['cases.json'],1):
            value = cache.get(row)
            if str(value.dtype) != 'torch.float32' or not training.torch.isfinite(value).all():raise ValueError('finite full original feature required')
            size = value.numel()*value.element_size();total += size;positions += len(value)
            log.write(json.dumps({'id':row['id'],'source_sha256':row['source_sha256'],'raw_bytes':size,
                'positions':len(value),'payload_sha256':values['feature_manifest.json'][row['id']]['payload_sha256']})+'\n')
            if i%128 == 0:print(json.dumps({'union_feature_payloads_replayed':i,'total':len(values['cases.json'])}),flush=True)
    new = len(values['arm_ids.json']['new_admitted'])
    admitted = training_admission(len(values['observed_new_firsts.json']),new,pilot=False)
    finish(root,proof,{'status':'expanded_catalog_independently_replayed','previous_cases':1376,'new_observed':1024,
        'new_admitted':new,'new_excluded':len(values['excluded_new.json']),'total_pairs':len(values['cases.json']),
        'training_data_admitted':admitted,'admission_reason':'complete_fixed_batch_and_minimum900_new_pairs' if admitted else 'insufficient_new_pairs',
        'source_components':len({r['source_component'] for r in values['cases.json']}),
        'raw_feature_bytes_replayed':total,'source_positions':positions,'feature_cache':cache.metrics(),
        'exclusion_reasons':dict(Counter(x for r in values['excluded_new.json'] for x in r['reasons'])),
        'native_target_tokens':sum(r['native_lexical_tokens'] for r in values['cases.json']),
        'stored_target_tokens':sum(r['stored_tokens'] for r in values['cases.json'])},[root/'payloads.jsonl'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=tuple(ROOTS))
    stage = parser.parse_args().stage;root = ROOTS[stage];existed = root.exists()
    try:
        audit() if stage == 'audit' else prepare(stage == 'pilot')
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'error_type':type(error).__name__,'error':str(error)})
        raise
