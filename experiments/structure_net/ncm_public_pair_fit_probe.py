"""Read-only fit diagnostic on fixed public TRAIN pairs, separate from task accuracy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_public_training_probe as reference

data, broad, study = reference.data, reference.broad, reference.study
bundle = study.pipeline.bundle
ROOTS = {s:data.ROOT/f'20261027_public_pair_fit_{s}_v1' for s in ('pilot','run','audit')}
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-public-pair-fit-probe.md')
TEST = Path('tests/structure_net/test_ncm_public_pair_fit_probe.py')
HYPOTHESIS = 'ncm-public-pair-fit-probe-v1'
LIMIT = 3_500_000_000
QUALIFICATION = data.ROOT/'20261027_public_pair_fit_CPU_qualification_v1.json'
UNITS = {
    'structure-net-ncm-public-epoch4-attempt1.service':('4a547d3283d743998a3e2843b17a4992',
        data.ROOT/'20261026_public_curriculum_epoch4_attempt1_v1/trace.jsonl'),
    'structure-net-ncm-task-supervision-existing-train-v3.service':('140877e4e84d43599852fd0af853cd63',
        data.ROOT/'20261027_task_supervision_existing_train_v3/trace.jsonl')}


def select(entries, count=32):
    if len(entries)<count or len({r['id'] for r in entries})!=len(entries):
        raise ValueError('complete unique public index required')
    return sorted(range(len(entries)),key=lambda i:(hashlib.sha256(
        ('public-pair-fit-271142:'+entries[i]['id']).encode()).hexdigest(),entries[i]['id']))[:count]


def pilot_ids(cases, lengths):
    if len(cases)!=32 or set(lengths)!={r['id'] for r in cases}:
        raise ValueError('all32 selected public cases and lengths required')
    longest=max(cases,key=lambda r:(lengths[r['id']],r['id']))['id']
    return list(dict.fromkeys([cases[0]['id'],longest,*[r['id'] for r in cases]]))[:2]


def live():
    out={}
    for unit,(identity,trace) in UNITS.items():
        raw=subprocess.check_output(['systemctl','--user','show',unit,
            '--property=MainPID,ActiveState,SubState,InvocationID,NRestarts'],text=True)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if (state['InvocationID']!=identity or state['NRestarts']!='0'
                or state['MainPID']=='0' or state['SubState']!='running'
                or not trace.is_file() or not trace.stat().st_size):
            raise ValueError('both unchanged training workers must have completed real updates')
        out[unit]=state
    return out


def prepared():
    dataset=bundle.open_dataset()
    if len(dataset)!=162353:raise ValueError('complete admitted public population required')
    positions=select(dataset.pairs.entries)
    cases=[];lengths={}
    for position in positions:
        pair,code=dataset.pairs[position],dataset.codes[position]
        if pair['split']!='TRAIN' or pair['id']!=code['id']:
            raise ValueError('public TRAIN-only paired identity required')
        cases.append({**pair,'oracle_codes':code['codes'],'catalog_position':position})
        lengths[pair['id']]=dataset.metadata[position]['source_positions']
    return dataset,cases,lengths


class Loader:
    def __init__(self,dataset):self.dataset=dataset
    def get(self,row):
        value=self.dataset[row['catalog_position']]
        if any(value[k]!=row[k] for k in ('id','source_sha256','target_sha256','native_word_ids','oracle_codes')):
            raise ValueError('complete source/target/features differ from selected TRAIN pair')
        return value['features']


def require(root):
    reference.receipt(root)
    value=data.completed(root)
    reference.study.pipeline.broad.runtime.configure()[0].native.vendor.verify(data.read(root/'producer.json'))
    return value


def freeze(root,stage,training,environment,cases,extra=()):
    files={Path(__file__).resolve(),PLAN.resolve(),TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_file() and p.suffix=='.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):files.add(p)
    paths=[bundle.ROOT/n for n in ('manifest.json','result.json','actual_tool_receipt.json')]
    paths += [p/n for p in reference.PARENTS.values() for n in ('producer.json','result.json','actual_tool_receipt.json','model.pt')]
    paths += [Path('pixi.lock'),Path('pyproject.toml'),QUALIFICATION,*extra]
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(root/'cases.json',cases)
    proof={'schema':'nal.ncm-public-pair-fit.v1','hypothesis_id':HYPOTHESIS,'stage':stage,
        'environment':environment,'config':{'population':162353,'sample':32,'physical_GPU':2,
            'precision':'FP32','TF32':False,'peak_limit_bytes':LIMIT,'model_updates':0,
            'new_judge_requests':0,'final_test_opened':False,'task_accuracy':None},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},
        'input_hashes':{str(p):data.sha(p) for p in [*paths,root/'cases.json']}}
    proof['fingerprint']=training.q.digest(proof);data.write(root/'producer.json',proof)
    return proof


def finish(root,proof,value,paths):
    data.verify(proof['source_hashes']);data.verify(proof['input_hashes'])
    result={'schema':proof['schema'],'hypothesis_id':HYPOTHESIS,**value,'model_updates':0,
        'new_judge_requests':0,'task_accuracy':None,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):data.sha(p) for p in [root/'producer.json',root/'cases.json',*paths]}}
    data.write(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


def diagnostics(training,model,cases,loader,tokenizer,shuffled=False):
    with reference.capture_teacher(model.concepts) as captured:
        rows=training.prior.diagnostics(model,cases,loader,tokenizer,shuffle=shuffled)
    if len(rows)!=len(captured):raise ValueError('all teacher predictions must be preserved')
    return [{**row,'teacher_codes':codes} for row,codes in zip(rows,captured)]


def run(stage):
    root=ROOTS[stage]
    if root.exists():raise ValueError('fresh append-only diagnostic required')
    qualified=data.read(QUALIFICATION)
    if qualified['actual_tool_exit_code']!=0:raise ValueError('successful CPU contracts required')
    data.verify(qualified['artifacts'])
    training,environment=broad.runtime.configure();study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'
            or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 2060 SUPER'):
        raise ValueError('physicalGPU2 RTX2060SUPER required')
    if torch.cuda.mem_get_info()[0]<LIMIT:raise ValueError('insufficient free memory for qualified pilot')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    before=live()
    for parent in reference.PARENTS.values():reference.receipt(parent);data.completed(parent)
    extra=[]
    if stage=='run':
        if require(ROOTS['pilot'])['status']!='public_pair_fit_pilot_passed':raise ValueError('real pilot required')
        extra=[ROOTS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json')]
    dataset,all_cases,lengths=prepared();ids=pilot_ids(all_cases,lengths)
    cases=[r for identity in ids for r in all_cases if r['id']==identity] if stage=='pilot' else all_cases
    proof=freeze(root,stage,training,environment,cases,extra)
    data.write(root/'selection.json',{'all_ids':[r['id'] for r in all_cases],'lengths':lengths,'pilot_ids':ids})
    loader=Loader(dataset);tokenizer=training.native.tokenizer();torch.cuda.reset_peak_memory_stats()
    model=training.load_initial();outputs={};scores={}
    for arm,parent in reference.PARENTS.items():
        state=data.read(parent/'result.json');saved=torch.load(parent/'model.pt',map_location='cpu',weights_only=True,mmap=True)
        if data.sha(parent/'model.pt')!=state['checkpoint_sha256']:raise ValueError('checkpoint bytes changed')
        model.load_state_dict(saved['model'],strict=True);del saved
        expected=state['final_model_sha256']
        if training.native.vendor.module_hash(model)!=expected:raise ValueError('checkpoint tensors changed')
        outputs[arm]={};scores[arm]={}
        for shuffled in ((False,) if stage=='pilot' else (False,True)):
            mode='shuffled' if shuffled else 'matched'
            rows=diagnostics(training,model,cases,loader,tokenizer,shuffled)
            scores[arm][mode]=reference.score_rows(rows,cases,shuffled)
            if stage=='pilot' and diagnostics(training,model,cases,loader,tokenizer)!=rows:
                raise ValueError('pilot outputs did not repeat exactly')
            if stage=='run' and not shuffled:
                prior=data.read(ROOTS['pilot']/'outputs.json')[arm]['matched'];lookup={r['id']:r for r in rows}
                if any(lookup[r['id']]!=r for r in prior):raise ValueError('full outputs differ from pilot')
            if (training.native.vendor.module_hash(model)!=expected or any(p.grad is not None for p in model.parameters())
                    or torch.cuda.max_memory_allocated()>LIMIT):raise ValueError('mutation, gradient or memory failure')
            outputs[arm][mode]=rows;data.write(root/f'outputs_{arm}_{mode}.json',rows)
            print(json.dumps({'arm':arm,'mode':mode,'cases':len(rows)}),flush=True)
    after=live()
    if before!=after:raise ValueError('training service identities changed during read-only diagnostic')
    data.write(root/'outputs.json',outputs);data.write(root/'scores.json',scores)
    data.write(root/'services.json',{'before':before,'after':after})
    finish(root,proof,{'status':'public_pair_fit_pilot_passed' if stage=='pilot' else 'public_pair_fit_completed',
        'cases':len(cases),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'systems_lifecycle_only_not_quality_evidence':stage=='pilot','checkpoint_tensors_unchanged':True},
        [*[root/n for n in ('selection.json','outputs.json','scores.json','services.json')],
         *sorted(root.glob('outputs_*.json'))])
    model.close()


def audit():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU-only replay required')
    root=ROOTS['audit']
    if root.exists():raise ValueError('fresh replay root required')
    training,environment=broad.runtime.configure();result=require(ROOTS['run']);require(ROOTS['pilot'])
    dataset,cases,lengths=prepared()
    if cases!=data.read(ROOTS['run']/'cases.json') or result['cases']!=32:raise ValueError('fixed public selection changed')
    outputs=data.read(ROOTS['run']/'outputs.json')
    if set(outputs)!=set(reference.PARENTS):raise ValueError('both checkpoints required')
    scores={};tokenizer=training.native.tokenizer()
    for arm,conditions in outputs.items():
        if set(conditions)!={'matched','shuffled'}:raise ValueError('both source conditions required')
        for rows in conditions.values():
            for row in rows:
                rendered=training.preparation.rendering.decode(row['raw_word_ids'],tokenizer)
                if any(row[k]!=v for k,v in rendered.items()):raise ValueError('raw word rendering differs')
        scores[arm]={mode:reference.score_rows(rows,cases,mode=='shuffled') for mode,rows in conditions.items()}
    if scores!=data.read(ROOTS['run']/'scores.json'):raise ValueError('raw measurement replay differs')
    paths=[ROOTS[s]/n for s in ('pilot','run') for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json')]
    proof=freeze(root,'audit',training,environment,cases,paths);data.write(root/'scores.json',scores)
    changes={metric:scores['epoch2']['matched'][metric]-scores['BROAD']['matched'][metric]
        for metric in ('teacher_code_agreement','rollout_code_agreement','native_word_agreement')}
    finish(root,proof,{'status':'public_pair_fit_independently_replayed','cases':32,
        'matched_agreement_changes':changes,'undertraining_causally_established':False,
        'semantic_generalization_established':False},[root/'scores.json'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=tuple(ROOTS))
    stage=parser.parse_args().stage;root=ROOTS[stage];existed=root.exists()
    try:audit() if stage=='audit' else run(stage)
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'error_type':type(error).__name__,'error':str(error)})
        raise
