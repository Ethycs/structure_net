"""Real frozen-checkpoint parity and gradient routing for source attention in all blocks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_source_coverage as coverage
from experiments.structure_net import ncm_public_training_probe as reference
from experiments.structure_net import ncm_public_pair_fit_probe as services
from experiments.structure_net import ncm_task_supervision_training as task

data=reference.data
ROOT=data.ROOT/'20261027_source_coverage_pilot_v1'
PLAN=Path('docs/07 - Status Reports/2026-10-27_ncm-source-coverage-pilot.md')
TEST=Path('tests/structure_net/test_ncm_source_coverage.py')
QUALIFICATION=data.ROOT/'20261027_source_coverage_CPU_qualification_v1.json'
HYPOTHESIS='ncm-source-coverage-pilot-v1'
LIMIT=3_500_000_000


def bindings(model):
    return {name:{'dtype':str(value.dtype),'shape':list(value.shape),
        'sha256':hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        for name,value in model.state_dict().items()}


@torch.no_grad()
def outputs(model,cases,loader,training):
    reference.broad.mechanism.configure(model,False)
    rows=[];logits=[]
    for row in cases:
        f,m,gold=training.prior.tensors(row,loader)
        codes=reference.broad.mechanism.generate(model,f,m)
        words=model.word_logits(f,m,codes).argmax(-1)
        logits.append(model.concepts.teacher_logits(f,m,gold).detach().cpu())
        rows.append({'id':row['id'],'codes':codes.cpu().tolist()[0],'raw_word_ids':words.cpu().tolist()[0]})
    return rows,logits


def freeze(training,environment,cases):
    files={Path(__file__).resolve(),Path(coverage.__file__).resolve(),PLAN.resolve(),TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_file() and p.suffix=='.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):files.add(p)
    paths=[QUALIFICATION,Path('pixi.lock'),Path('pyproject.toml')]
    paths += [task.PARENT/n for n in ('producer.json','result.json','actual_tool_receipt.json','model.pt')]
    paths += [reference.ROOTS['run']/n for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json')]
    paths += [reference.broad.RUNS['prepare']/n for n in ('cases.json','selection.json')]
    paths += [reference.broad.catalog.DEST/n for n in ('source_index.json','feature_manifest.json')]
    ROOT.mkdir();(ROOT/'producer_source').mkdir();data.write(ROOT/'cases.json',cases)
    for i,p in enumerate(sorted(files)):(ROOT/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof={'schema':'nal.ncm-source-coverage-pilot.v1','hypothesis_id':HYPOTHESIS,
        'environment':environment,'config':{'physical_GPU':2,'precision':'FP32','TF32':False,
            'peak_limit_bytes':LIMIT,'cases':2,'model_updates':0,'new_judge_requests':0,
            'systems_lifecycle_only_not_quality_evidence':True,'parent_model_sha256':task.PARENT_MODEL_SHA},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},
        'input_hashes':{str(p):data.sha(p) for p in [*paths,ROOT/'cases.json']}}
    proof['fingerprint']=training.q.digest(proof);data.write(ROOT/'producer.json',proof);return proof


def run():
    if ROOT.exists():raise ValueError('fresh append-only source-coverage pilot required')
    qualified=data.read(QUALIFICATION)
    if qualified['actual_tool_exit_code']!=0:raise ValueError('actual successful CPU tests required')
    data.verify(qualified['artifacts'])
    training,environment=reference.broad.runtime.configure();reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'
            or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 2060 SUPER'
            or torch.cuda.mem_get_info()[0]<LIMIT):raise ValueError('physicalGPU2 and adequate free CUDA memory required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    before_services=services.live()
    reference.receipt(task.PARENT);data.completed(task.PARENT)
    if data.sha(task.PARENT/'model.pt')!=task.schedule.PARENT_CHECKPOINT_SHA:
        raise ValueError('fixed epoch2 checkpoint bytes required')
    reference.receipt(reference.ROOTS['run']);data.completed(reference.ROOTS['run'])
    controls,selection=reference.prepared_controls()
    ids=list(dict.fromkeys([controls[0]['id'],selection['longest_source_id'],*[r['id'] for r in controls]]))[:2]
    cases=[next(r for r in controls if r['id']==i) for i in ids]
    proof=freeze(training,environment,cases)
    index,manifest=[data.read(reference.broad.catalog.DEST/n) for n in ('source_index.json','feature_manifest.json')]
    loader=training.preparation.streaming.FeatureCache(index,manifest,max_bytes=256*1024**2)
    torch.cuda.reset_peak_memory_stats();model=task.load_initial(training)
    old=bindings(model);protected=training.prior.protected(model)
    before,before_logits=outputs(model,cases,loader,training)
    archive={r['id']:r for r in data.read(reference.ROOTS['run']/'outputs.json')['epoch2']['matched']}
    if any(any(r[k]!=archive[r['id']][k] for k in ('codes','raw_word_ids')) for r in before):
        raise ValueError('original epoch2 saved codes or words differ on this GPU')
    metadata=coverage.expand(model)
    reference.broad.mechanism.configure(model,False)
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=115939968:
        raise ValueError('exact expanded trainable parameter population required')
    expanded=bindings(model)
    if any(expanded.get(k)!=v for k,v in old.items()):raise ValueError('original tensor changed during expansion')
    after,after_logits=outputs(model,cases,loader,training)
    if before!=after or any(not torch.equal(a,b) for a,b in zip(before_logits,after_logits)):
        raise ValueError('expanded initial function must reproduce every logit, code and word exactly')
    torch.save({'before':before_logits,'after':after_logits},ROOT/'teacher_logits.pt')
    data.write(ROOT/'outputs.json',{'before':before,'after':after})
    gradients=[]
    for row in cases:
        model.zero_grad(set_to_none=True)
        f,m,gold=training.prior.tensors(row,loader)
        logits=model.concepts.teacher_logits(f,m,gold)
        loss=training.objective.losses(logits,gold,model.concepts.generator.levels).mean()
        if not torch.isfinite(loss):raise ValueError('finite full-hierarchy loss required')
        loss.backward();added={}
        for i in metadata['added_source_blocks']:
            module=model.concepts.adapters[str(i)];out=module.out.weight.grad;q=module.q.weight.grad
            if (out is None or not torch.isfinite(out).all() or not out.abs().sum()
                    or q is None or q.count_nonzero()):raise ValueError('added output/query gradient contract failed')
            added[str(i)]={'output_L2':float(out.double().norm()),'query_nonzero':int(q.count_nonzero())}
        if any(p.grad is not None and (not p.requires_grad or not torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('protected or nonfinite gradient')
        gradients.append({'id':row['id'],'loss':float(loss.detach()),'added':added})
        model.zero_grad(set_to_none=True);del f,m,gold,logits,loss
        if torch.cuda.max_memory_allocated()>LIMIT:raise ValueError('pilot CUDA allocation limit exceeded')
    if (bindings(model)!=expanded or training.prior.protected(model)!=protected
            or any(p.grad is not None for p in model.parameters())):raise ValueError('gradient-only pilot changed tensors or left gradients')
    if services.live()!=before_services:raise ValueError('ongoing training service identities changed')
    data.write(ROOT/'coverage.json',metadata);data.write(ROOT/'gradients.json',gradients)
    data.write(ROOT/'tensor_bindings.json',{'original':old,'expanded':expanded,'after_backward':bindings(model)})
    data.write(ROOT/'services.json',{'before':before_services,'after':services.live()})
    data.verify(proof['source_hashes']);data.verify(proof['input_hashes'])
    value={'schema':proof['schema'],'hypothesis_id':HYPOTHESIS,'status':'source_coverage_initial_parity_and_gradients_passed',
        'cases':2,'source_blocks':12,'new_source_blocks':9,'trainable_parameters':115939968,
        'all_initial_teacher_logits_exact':True,'all_initial_codes_and_words_exact':True,
        'all_original_tensors_unchanged':True,'all_added_output_gradients_nonzero':True,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'model_updates':0,'new_judge_requests':0,
        'task_accuracy':None,'final_test_opened':False,'speed_tested':False,
        'systems_lifecycle_only_not_quality_evidence':True,'full_training_lifecycle_qualified':False,
        'artifacts':{str(p):data.sha(p) for p in ROOT.iterdir() if p.is_file()}}
    data.write(ROOT/'result.json',value);model.close()
    print(json.dumps({k:v for k,v in value.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('pilot',));parser.parse_args()
    existed=ROOT.exists()
    try:run()
    except Exception as error:
        if ROOT.exists() and not existed:data.write(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
