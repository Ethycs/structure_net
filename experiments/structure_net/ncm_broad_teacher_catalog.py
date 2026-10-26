"""Consolidate task-correct teacher firsts; preserve failures and held-out sources."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import resource

ROOT=Path('data/experiments/ncm_paper_regime')
DEST=ROOT/'20261025_broad_teacher_catalog_v1'
AUDIT=ROOT/'20261025_broad_teacher_feature_audit_v1'
PLAN=Path('docs/07 - Status Reports/2026-10-25_ncm-broad-teacher-catalog-plan.md')
TEST=Path('tests/structure_net/test_ncm_broad_teacher_catalog.py')
INVENTORIES=(
 ('20261024_teacher10000_partial_firsts_inventory_v1',1117,'ebbeb4bf90c1c1f7ac7ef36358bc221d82dfafe9f3c43d3c6c1a853d21f47bfd'),
 ('20261024_sol_teacher188_partial_inventory_v1',188,'3314194e8dd025a2b8d0e7f8e622ee0bd5a7401c496dc2e62f71ab86e3962a3e'),
 ('20261025_sol_teacher144_partial_inventory_v1',144,'a8b91fa92c852eade0e34d9c1e937b3e5d0ed27b1a2f9a8cf2219cd70231e567'))
SOURCE=ROOT/'20261024_additional10000_training_sources_v1/cases.json'
INDEX=ROOT/'20261024_additional10000_source_metadata_v1/index.json'
MANIFEST=ROOT/'20261024_additional10000_source_features_v1/manifest.json'
QUALIFICATIONS=[ROOT/name/'campaign_results.json' for name in (
 '20261024_evidence_teacher64_qualification_results_v1','20261024_sol_teacher64_semantic_results_v1')]


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def text_hash(text):return hashlib.sha256(text.encode()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def numeric_reasons(row,lexical,specials,stored):
    reasons=[]
    if not row['teacher_valid']:reasons.append('invalid_first_preserved')
    if not row['capacity_fit']:reasons.append('original_capacity_failed')
    if not row['target_text'].strip() or not 0<stored<=128:reasons.append('stored_target_empty_or_over128')
    if not 0<len(lexical)<=126:reasons.append('target_not_complete_native128')
    if set(lexical)&specials:reasons.append('native_special_or_unknown')
    if not 192<=row['source_cl100k_tokens']<=2048:reasons.append('source_outside_application_range')
    return reasons


def protected_reasons(row,protected,family):
    families={family(v) for v in [row['source_id'],*[s['source_id'] for s in row['spans']]]}
    reasons=[]
    for key,hit in (
        ('source_family',families&protected['families']),
        ('group',row['group_id'] in protected['groups']),
        ('exact_source',row['source_sha256'] in protected['hashes']),
        ('informative_turn',{s['turn_text_sha256'] for s in row['spans']}&protected['turns'])):
        if hit:reasons.append('protected_'+key)
    return reasons


def replay_valid(row,teacher,parser):
    """Verify ordered raw stages; exhausted failures remain in the input catalog."""
    paths=[]
    previous=None
    for stage in ('draft','review'):
        observation=row[stage]
        if observation['status']!='valid':raise ValueError('valid candidate needs both valid stages')
        attempts=observation['attempts']
        if not 1<=len(attempts)<=3:raise ValueError('original bounded attempts required')
        parsed=None
        for i,name in enumerate(attempts):
            path=Path(name);raw=read(path);paths.append(path)
            if path.name!=f'attempt_{i}.json' or parsed is not None:
                raise ValueError('first valid reply cannot be replaced')
            request=raw['request']
            body=json.loads(request['messages'][1]['content'])
            if body['source_lines']!=teacher.source_lines(row):raise ValueError('complete source prompt differs')
            if stage=='review' and body['untrusted_draft']!=previous:raise ValueError('review draft binding differs')
            if raw['valid']:
                response=raw['response']
                if raw['http_status']!=200 or response['model']!=request['model'] or response['choices'][0]['finish_reason']!='stop':
                    raise ValueError('valid raw transport/model/termination differs')
                parsed=parser(response['choices'][0]['message']['content'],row)
        if parsed!=observation['value']:raise ValueError('first candidate value differs from raw response')
        previous=parsed
    if previous['summary']!=row['target_text']:raise ValueError('target differs from reviewed first')
    return paths


def configure():
    from experiments.structure_net import ncm_pixi
    training,environment=ncm_pixi.configure()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or training.torch.cuda.is_initialized():
        raise ValueError('CPU-only catalog and feature verification required')
    return training,environment


def prepare():
    training,environment=configure()
    from experiments.structure_net import ncm_shared_summary_inventory as inventory
    from experiments.structure_net import ncm_shared_summary_source_guard as guard
    from experiments.structure_net import ncm_evidence_teacher_pilot as teacher
    from experiments.structure_net import ncm_evidence_teacher_pilot_v3 as recipe
    import tiktoken
    if DEST.exists():raise ValueError('fresh append-only catalog required')
    originals={r['id']:r for r in read(SOURCE)}
    if len(originals)!=10000:raise ValueError('complete original TRAIN reservation required')
    protected,guard_paths,protected_counts=inventory.protected_metadata()
    cohorts,_=guard.development.source_inputs()
    held=[r for name in guard.development.COHORTS for r in cohorts[name]]+read(inventory.COMPONENT)
    if len(held)!=192:raise ValueError('all open protected sources required')
    held_ids={r['id'] for r in held}
    text_index=guard.TextIndex([r['source'] for r in held])
    for path in QUALIFICATIONS:
        q=read(path)
        if not q['teacher_labels_qualified'] or not q['joint']['qualified']:
            raise ValueError('qualified complete-summary recipe required')
    index_all={r['id']:r for r in read(INDEX)}
    manifest_all=read(MANIFEST)
    tokenizer,encoding=training.native.tokenizer(),tiktoken.get_encoding('cl100k_base')
    rows,excluded,observations,inputs=[],[],[],[SOURCE,INDEX,MANIFEST,*guard_paths,*QUALIFICATIONS,guard.development.SOURCE/'cases.json']
    seen_ids,seen_sources=set(),set()
    for dirname,count,expected in INVENTORIES:
        root=ROOT/dirname;path=root/'first_candidates.json';state=read(root/'result.json')
        if sha(path)!=expected or read(root/'actual_tool_receipt.json')['actual_tool_exit_code']!=0:
            raise ValueError('completed frozen first-candidate inventory required')
        expected_again=state.get('first_candidates_sha256',state.get('artifacts',{}).get(str(path)))
        if expected_again!=expected:raise ValueError('independent inventory binding differs')
        values=read(path)
        if len(values)!=count:raise ValueError('whole first-candidate inventory required')
        inputs.extend([path,root/'result.json',root/'actual_tool_receipt.json'])
        for position,row in enumerate(values):
            if row['id'] in seen_ids:raise ValueError('a source received another candidate instead of its first')
            seen_ids.add(row['id'])
            original=originals[row['id']]
            if any(row.get(k)!=v for k,v in original.items()) or row['source_truncated'] or text_hash(row['source'])!=row['source_sha256']:
                raise ValueError('original complete source or reservation binding differs')
            lexical=tokenizer.encode(row['target_text'],add_special_tokens=False)
            stored=len(encoding.encode(row['target_text'],disallowed_special=()))
            if stored!=row['stored_tokens'] or len(lexical)!=row['native_lexical_tokens']:
                raise ValueError('native/application token count differs')
            reasons=numeric_reasons(row,lexical,set(tokenizer.all_special_ids),stored)
            reasons+=protected_reasons(row,protected,inventory.reservation.graph.guard.source_family)
            if row['id'] in held_ids:reasons.append('reserved_component_or_DEV_id')
            matches=text_index.matches(row['source'])
            if matches:reasons.append('open_heldout_text_overlap')
            if row['source_sha256'] in seen_sources:reasons.append('exact_source_duplicate')
            if row['teacher_valid']:
                inputs.extend(replay_valid(row,teacher,recipe.parse))
            provenance={'inventory':str(path),'inventory_position':position,'original_id':row['id']}
            observations.append({'id':row['id'],'teacher_valid':row['teacher_valid'],'capacity_fit':row['capacity_fit'],**provenance})
            if reasons:
                excluded.append({'id':row['id'],'reasons':reasons,**provenance,
                                 'text_conflicts':[{'heldout_id':held[i]['id'],**p} for i,p in matches]})
                continue
            frame=[101,*lexical,101]+[0]*(126-len(lexical))
            if frame!=row['native_word_ids']:raise ValueError('unaltered native frame required')
            entry=index_all[row['id']]
            if any(entry[k]!=row[k] for k in ('source_id','source_sha256','source_component','group_id')):
                raise ValueError('source feature metadata identity differs')
            feature=manifest_all[row['id']]
            if feature['source_tokens']!=entry['source_positions'] or feature['raw_bytes']!=entry['source_positions']*3072*4:
                raise ValueError('full FP32 source shape differs')
            seen_sources.add(row['source_sha256'])
            rows.append({**row,'split':'TRAIN','target_sha256':text_hash(row['target_text']),
                         'teacher_first_provenance':provenance,'target_origin':'unchanged_reviewed_complete_summary_first',
                         'semantic_qualified':False})
    if len(observations)!=1449 or sum(not x['teacher_valid'] for x in observations)!=73:
        raise ValueError('all observed firsts and original73 failures required')
    del originals,index_all,manifest_all
    gc.collect()
    accepted_ids={r['id'] for r in rows}
    index=[r for r in read(INDEX) if r['id'] in accepted_ids]
    manifest={k:v for k,v in read(MANIFEST).items() if k in accepted_ids}
    if len(index)!=len(rows) or set(manifest)!=accepted_ids:raise ValueError('complete feature bindings required')
    files=[Path(__file__),PLAN,TEST,Path(inventory.__file__),Path(guard.__file__),Path(teacher.__file__),
           Path(recipe.__file__),Path(recipe.recovery.__file__),Path('pixi.lock')]
    DEST.mkdir();(DEST/'producer_source').mkdir()
    for p in files[:-1]:(DEST/'producer_source'/p.name).write_bytes(p.read_bytes())
    proof={'hypothesis_id':'ncm-broad-teacher-catalog-v1','environment':environment,
           'source_hashes':{str(p):sha(p) for p in files},'input_hashes':{str(p):sha(p) for p in dict.fromkeys(inputs)},
           'config':{'targets_unchanged':True,'all_firsts_preserved':True,'new_requests':0,'final_source_text_opened_or_hashed':False,
                     'open_protected_sources':192,'metadata_protected_components':512,'model_training':False,
                     'source_only_task_accuracy':None,'training_hyperparameters_fixed':False}}
    write(DEST/'producer.json',proof)
    for name,value in [('cases.json',rows),('excluded.json',excluded),('observed_firsts.json',observations),
                       ('source_index.json',index),('feature_manifest.json',manifest),('protected_metadata.json',protected_counts)]:
        write(DEST/name,value)
    result={'status':'complete_summary_firsts_guarded_and_bound_features_pending',
            'observed':len(observations),'original_invalid_firsts':73,'retained':len(rows),'excluded':len(excluded),
            'exclusion_reasons':dict(Counter(x for r in excluded for x in r['reasons'])),
            'source_components':len({r['source_component'] for r in rows}),
            'source_sessions':len({r['source_id'] for r in rows}),
            'source_bins':dict(Counter(r['bin'] for r in rows)),
            'native_target_min':min(r['native_lexical_tokens'] for r in rows),
            'native_target_max':max(r['native_lexical_tokens'] for r in rows),
            'label_schema_and_split_admitted':True,'individual_semantic_gold':False,'feature_payload_audit_pending':True,
            'model_training':False,'new_requests':0,'source_only_task_accuracy':None,'final_test_opened':False,
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'artifacts':{str(p):sha(p) for p in DEST.glob('*.json')}}
    write(DEST/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


def features():
    training,environment=configure()
    from experiments.structure_net.ncm_streaming_source_features import FeatureCache
    if AUDIT.exists() or read(DEST/'actual_tool_receipt.json')['actual_tool_exit_code']!=0:
        raise ValueError('fresh audit after actual successful catalog completion required')
    state=read(DEST/'result.json')
    for path,digest in state['artifacts'].items():
        if sha(path)!=digest:raise ValueError('catalog artifact changed')
    rows,index,manifest=[read(DEST/n) for n in ('cases.json','source_index.json','feature_manifest.json')]
    cache=FeatureCache(index,manifest,max_bytes=256*1024**2)
    AUDIT.mkdir();write(AUDIT/'producer.json',{'source_hashes':{str(Path(__file__)):sha(__file__)},'environment':environment,
           'input_hashes':{str(DEST/'result.json'):sha(DEST/'result.json')},'CPU_only':True,'model_training':False})
    total=0;raw_bytes=0
    with (AUDIT/'payloads.jsonl').open('x') as output:
        for row in rows:
            value=cache.get(row)
            if str(value.dtype)!='torch.float32' or not training.torch.isfinite(value).all():
                raise ValueError('complete finite original FP32 feature required')
            size=value.numel()*value.element_size();raw_bytes+=size;total+=len(value)
            output.write(json.dumps({'id':row['id'],'source_sha256':row['source_sha256'],'positions':len(value),
                                     'raw_bytes':size,'payload_sha256':manifest[row['id']]['payload_sha256']})+'\n')
            if cache.loads%128==0:print(json.dumps({'feature_payloads_replayed':cache.loads,'total':len(rows)}),flush=True)
    result={'status':'all_catalog_feature_payloads_replayed','cases':len(rows),'source_positions':total,
            'raw_bytes_replayed':raw_bytes,'feature_cache':cache.metrics(),
            'model_training':False,'new_requests':0,'source_only_task_accuracy':None,'final_test_opened':False,
            'catalog_sha256':sha(DEST/'result.json'),'payload_log_sha256':sha(AUDIT/'payloads.jsonl'),
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    write(AUDIT/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('prepare','features'))
    {'prepare':prepare,'features':features}[p.parse_args().stage]()
