"""Source-disjoint semantic evaluation for the repaired full732 NCM."""
import argparse
import json
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_sequential_full_training_v2 as training
from experiments.structure_net import ncm_lm1b_native_development as old
from experiments.structure_net import ncm_after_grounded_stop_judgment_lineage as lineage
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, preparation, q, ROOT = training.native, training.preparation, training.q, training.ROOT
RUNS = {s: ROOT / ('20261025_sequential_full732_DEV_' + s + '_v2') for s in ('inference_pilot','outputs','judge_pilot','grade')}
TEST = Path('tests/structure_net/test_ncm_sequential_development_v2.py')


def split_guard(train, cohorts):
    old.validate_sources(cohorts)
    rows = [r for c in old.COHORTS for r in cohorts[c]]
    result = {}
    for key in ('id','source_id','group_id','source_sha256','source_component'):
        overlap = {r[key] for r in train} & {r[key] for r in rows}
        if overlap: raise ValueError('TRAIN/DEV overlap in ' + key)
        result[key] = {'TRAIN_unique':len({r[key] for r in train}),'DEV_unique':len({r[key] for r in rows}),'overlap':0}
    return result


def freeze(root,stage,extra):
    files={Path(__file__),TEST,training.PLAN}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_file() and path.suffix=='.py' and (path.is_relative_to(Path('experiments').resolve()) or path.is_relative_to(native.vendor.VENDOR.resolve())):
                files.add(path)
    inputs=[old.SOURCE/'cases.json',old.FEATURES/'manifest.json',training.base.DATA/'cases.json']
    inputs += [training.RUNS['train']/n for n in ('producer.json','result.json','actual_tool_receipt.json','model.pt')]
    inputs += extra
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof={'schema':'nal.ncm-sequential-full732-dev.v2','stage':stage,'hypothesis_id':'ncm-sequential-full732-DEV-v2',
        'config':{'source_only':True,'target_fields_in_prediction':False,'full_source_features':True,
            'source_disjoint_DEV_cases':128,'cohorts':old.COHORTS,'final_test_opened':False,'speed_tested':False,
            'rubric':q.RUBRIC,'judges':q.JUDGES,'inherited_judgments':lineage.COUNT,
            'task_accuracy_metric':'whole_summary_semantic_pass','stored_token_cap':128,'exact_match_task_metric':False},
        'source_hashes':{str(p):file_hash(p) for p in sorted(files)},
        'input_hashes':{str(p):file_hash(p) for p in inputs}}
    proof['fingerprint']=q.digest(proof);write_json(root/'producer.json',proof)
    return proof


def sources():
    cohorts,manifest=old.source_inputs()
    guard=split_guard(q.read(training.base.DATA/'cases.json'),cohorts)
    # Existing inherited native655 source split proof is part of the parent lineage.
    return cohorts,manifest,guard


@torch.no_grad()
def predict(model,row,manifest,tokenizer):
    if old.FORBIDDEN & row.keys():raise ValueError('source-only input required')
    entry=manifest[row['id']]
    if file_hash(entry['path'])!=entry['sha256']:raise ValueError('DEV feature bytes changed')
    value=torch.load(entry['path'],map_location='cpu',weights_only=True)
    if (any(value[k]!=row[k] for k in ('id','source_sha256','alignment_sha256','native_prompt_sha256'))
            or value['features'].shape!=(entry['source_tokens'],3072) or value['features'].dtype!=torch.float32
            or not torch.isfinite(value['features']).all()):raise ValueError('full source feature contract differs')
    features=value['features'].unsqueeze(0).cuda()
    mask=torch.ones(features.shape[:2],dtype=torch.bool,device=features.device)
    codes=training.prior.mechanism.generate(model,features,mask)
    words=model.word_logits(features,mask,codes).argmax(-1).cpu().tolist()[0]
    if model._word_bound is not None or model.concepts._bound is not None:raise ValueError('stale source context')
    return {'id':row['id'],'source_sha256':row['source_sha256'],'split':row['split'],
        'feature_sha256':entry['sha256'],'source_positions':entry['source_tokens'],
        'codes':codes.cpu().tolist()[0],'raw_word_ids':words,**old.decode(words,tokenizer)}


def generate(pilot):
    native.vendor.corpus.quality.dependencies();lineage.configure()
    training.offload.require_rtx2060()
    preparation.completed(training.RUNS['train'])
    state=q.read(training.RUNS['train']/'result.json')
    if not state['strict_reload_passed'] or file_hash(training.RUNS['train']/'model.pt')!=state['checkpoint_sha256']:
        raise ValueError('completed fixed2048 generator checkpoint required')
    cohorts,manifest,guard=sources()
    chosen=old.inference_probes(cohorts,manifest) if pilot else [r for c in old.COHORTS for r in cohorts[c]]
    stage='inference_pilot' if pilot else 'outputs';root=RUNS[stage]
    extra=[Path(manifest[r['id']]['path']) for r in chosen]
    before={}
    if not pilot:
        preparation.completed(RUNS['inference_pilot'])
        before={r['id']:r for r in q.read(RUNS['inference_pilot']/'outputs.json')}
        extra += [RUNS['inference_pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json')]
    model=training.load_initial()
    saved=torch.load(training.RUNS['train']/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    expected=native.vendor.module_hash(model)
    if expected!=state['final_model_sha256']:raise ValueError('exact selected generator differs')
    proof=freeze(root,stage,extra);write_json(root/'split_proof.json',guard)
    tokenizer=native.tokenizer();outputs=[]
    for row in chosen:
        output=predict(model,row,manifest,tokenizer)
        if pilot and predict(model,row,manifest,tokenizer)!=output:raise ValueError('inference does not repeat')
        if row['id'] in before and before[row['id']]!=output:raise ValueError('full outputs differ from pilot')
        outputs.append(output);write_json(root/'outputs.json',outputs)
        if pilot or len(outputs)%16==0:print(json.dumps({'generated':len(outputs),'total':len(chosen)}),flush=True)
    if native.vendor.module_hash(model)!=expected:raise ValueError('inference changed model')
    native.vendor.verify(proof)
    write_json(root/'result.json',{'status':'complete_source_only_'+stage,'cases':len(outputs),'outputs_sha256':file_hash(root/'outputs.json'),
        'model_sha256':expected,'source_disjoint':True,'task_accuracy':None,'final_test_opened':False,
        'speed_tested':False,'producer_fingerprint':proof['fingerprint']})
    model.close()


def grade(pilot):
    native.vendor.corpus.quality.dependencies();lineage.configure();lineage.require_judges()
    preparation.completed(RUNS['outputs'])
    cohorts,_,_=sources()
    output=q.read(RUNS['outputs']/'outputs.json')
    if file_hash(RUNS['outputs']/'outputs.json')!=q.read(RUNS['outputs']/'result.json')['outputs_sha256']:
        raise ValueError('fixed model outputs changed')
    rows=[r for c in old.COHORTS for r in cohorts[c]]
    if [r['id'] for r in rows]!=[r['id'] for r in output]:raise ValueError('complete ordered128 outputs required')
    old.reference.verify_public(rows)
    cases={c:[{**r,'texts':{'student':o['text']}} for r,o in zip(rows,output) if r['split']==c] for c in old.COHORTS}
    packets={c:{'student':[q.packet(r['id'],r['source'],[{'candidate_id':'summary','text':r['texts']['student']}]) for r in cases[c]]} for c in old.COHORTS}
    cache=lineage.base_cache();inherited=dict(cache)
    unresolved=set(lineage.uncertain_allowances())
    for c in old.COHORTS:
        for packet in packets[c]['student']:
            if any(old.oracle.first.packet_key(packet,role) in unresolved for role in ('primary','secondary')):
                raise ValueError('candidate overlaps preserved uncertain historical request; do not reset attempts')
    extra=[*lineage.inputs(),old.reference.PUBLIC_FILE,RUNS['outputs']/'outputs.json']
    if not pilot:
        preparation.completed(RUNS['judge_pilot'])
        if not q.read(RUNS['judge_pilot']/'result.json')['valid_pilot']:
            raise ValueError('valid semantic transport pilot required')
        cache=q.read(RUNS['judge_pilot']/'judgment_cache.json')
        if any(cache.get(k)!=v for k,v in inherited.items()):raise ValueError('prior first judgment changed')
        extra += [RUNS['judge_pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','judgment_cache.json')]
    stage='judge_pilot' if pilot else 'grade';root=RUNS[stage]
    proof=freeze(root,stage,extra)
    write_json(root/'exact_packets.json',packets);write_json(root/'cases.json',cases);write_json(root/'judgment_cache.json',cache)
    metrics={c:{} for c in old.COHORTS}
    def judge(cohort,role):
        target=root/cohort;target.mkdir(exist_ok=True)
        values=cases[cohort][:2] if pilot else cases[cohort]
        result=old.oracle.grade_arm(target,'student',role,values,packets[cohort],cache)
        scores=q.read(target/f'{role}_student_case_scores.json')
        value={**{k:v for k,v in result.items() if k not in old.oracle.metric(scores)},**old.task_metric(scores)}
        write_json(target/f'{role}_student_result.json',value)
        write_json(root/'judgment_cache.json',cache);metrics[cohort][role]=value
    for cohort in old.COHORTS:judge(cohort,'secondary')
    admitted=not pilot and old.secondary_admitted({c:metrics[c]['secondary'] for c in old.COHORTS})
    joints=None
    if admitted:
        for cohort in old.COHORTS:judge(cohort,'primary')
        joints={c:old.joint_gate(q.read(root/c/'secondary_student_case_scores.json'),q.read(root/c/'primary_student_case_scores.json')) for c in old.COHORTS}
    if any(cache.get(k)!=v for k,v in inherited.items()):raise ValueError('historical first judgment changed')
    lineage.audit_cache(cache)
    valid=all(metrics[c]['secondary']['valid']==(2 if pilot else 64) for c in old.COHORTS)
    qualified=admitted and all(joints[c]['qualified'] for c in old.COHORTS)
    native.vendor.verify(proof)
    result={'schema':proof['schema'],'status':('systems_judge_pilot_valid' if valid else 'systems_judge_pilot_invalid') if pilot else ('DEV_qualified' if qualified else 'DEV_quality_failed'),
        'metrics':metrics,'valid_pilot':valid if pilot else None,'primary_admitted':admitted,'joint':joints,
        'DEV_qualified':qualified,'canonical_first_judgments':len(cache),'inherited_first_judgments':len(inherited),
        'task_accuracy':sum(metrics[c]['secondary']['passed'] for c in old.COHORTS)/128 if not pilot and valid else None,
        'task_accuracy_judge':q.JUDGES['secondary'],'metric':'complete_summary_semantic_pass',
        'final_test_opened':False,'speed_tested':False,'producer_fingerprint':proof['fingerprint']}
    write_json(root/'result.json',result);print(json.dumps(result),flush=True)
    if pilot and not valid:raise ValueError('semantic pilot invalid; firsts retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=tuple(RUNS))
    stage=parser.parse_args().stage
    generate(stage=='inference_pilot') if stage in ('inference_pilot','outputs') else grade(stage=='judge_pilot')
