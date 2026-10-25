"""Continue the fixed 16-TRAIN-case control, preserving its optimizer and RNG."""
import argparse
from dataclasses import asdict,dataclass
import gc
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_engineering_attention_repair_v3 as original
from experiments.structure_net.ncm_memory_data import file_hash,write_json

ROOT,q,native=original.ROOT,original.q,original.native
PARENT=original.RUNS['capacity']
PLAN=Path('docs/07 - Status Reports/2026-10-25_ncm-repair-continuation-plan.md')
TEST=Path('tests/structure_net/test_ncm_engineering_repair_continuation.py')
RUNS={m:ROOT/('20261025_engineering_repair16_continuation_'+m+'_v1') for m in ('pilot','capacity')}
START_SHA='c7d96bbd906ec74031f5ba105a953246d57377d955de84f3087f5e2c41b41a6c'


@dataclass(frozen=True)
class Config:
    start_step:int=512
    final_step:int=2048
    accumulation:int=4
    learning_rate:float=3e-5


def resume_contract(saved,expected_fingerprint):
    if (saved['step']!=Config().start_step
            or saved['producer_fingerprint']!=expected_fingerprint
            or not saved['optimizer']['param_groups']
            or any(g['lr']!=Config().learning_rate for g in saved['optimizer']['param_groups'])):
        raise ValueError('exact step512 parent, optimizer rate and producing fingerprint required')


def producing(root,mode):
    files=[Path(__file__),PLAN,TEST,Path(original.__file__),Path(original.mechanism.__file__),
        Path(original.objective.__file__),Path(original.preparation.__file__),
        Path(original.lineage.__file__),Path(original.preparation.streaming.__file__)]
    inputs=[original.DATA/n for n in ('producer.json','result.json','cases.json','source_index.json',
        'feature_manifest.json','selection.json','actual_tool_receipt.json')]
    inputs += [PARENT/n for n in ('producer.json','result.json','model.pt','outputs.json','actual_tool_receipt.json')]
    inputs += [original.GATE/'result.json']
    if mode=='capacity':inputs += [RUNS['pilot']/n for n in ('producer.json','result.json','model.pt','actual_tool_receipt.json')]
    root.mkdir();(root/'producer_source').mkdir()
    for i,path in enumerate(files):(root/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    p={'schema':'nal.ncm-repair16-continuation.v1','stage':mode,
        'source_hashes':{str(p):file_hash(p) for p in files},
        'input_hashes':{str(p):file_hash(p) for p in inputs},
        'config':{**asdict(Config()),'pilot_final_step':514,'TRAIN_cases':16,
            'source_scope':'same16TRAINdiagnostics; no new data or heldout input',
            'optimizer_CPU_CUDA_sampler_rng_continued':True,'learning_rate_restart':False,
            'architecture_objective_dropout_unchanged':True,'physical_GPU':2,
            'precision':'FP32','TF32':False,'all_parent_MLP_and_original_NCM_weights_frozen':True,
            'full732_training_admitted':False,'task_accuracy':None,'speed_tested':False}}
    write_json(root/'producer.json',p)
    return p


def save(root,corrector,opt,sampler,step,p):
    torch.save({'model':{k:v.detach().cpu().clone() for k,v in corrector.state_dict().items()},
        'optimizer':opt.state_dict(),'sampler_rng':sampler.get_state(),'CPU_rng':torch.get_rng_state(),
        'CUDA_rng':torch.cuda.get_rng_state_all(),'step':step,'config':asdict(Config()),
        'producer_fingerprint':q.digest(p)},root/'model.pt')


def restore_rng(saved,sampler):
    sampler.set_state(saved['sampler_rng']);torch.set_rng_state(saved['CPU_rng'])
    torch.cuda.set_rng_state_all(saved['CUDA_rng'])


def run(mode):
    native.vendor.corpus.quality.dependencies()
    original.preparation.checkpoint.training.require_gpu()
    original.preparation.completed(original.DATA);original.preparation.completed(PARENT)
    if file_hash(PARENT/'model.pt')!=START_SHA:raise ValueError('exact original512update checkpoint required')
    if mode=='capacity':original.preparation.completed(RUNS['pilot'])
    root=RUNS[mode]
    if root.exists():raise ValueError('fresh continuation root required')
    p=producing(root,mode)
    rows=q.read(original.DATA/'cases.json');lookup={r['id']:r for r in rows}
    ids=q.read(original.DATA/'selection.json')['diagnostic_ids'];selected=[lookup[k] for k in ids]
    loader=original.preparation.streaming.FeatureCache(q.read(original.DATA/'source_index.json'),
        q.read(original.DATA/'feature_manifest.json'),max_bytes=2*1024**3)
    renderer=original.load_renderer();frozen_hash=native.vendor.module_hash(renderer)
    corrector=original.mechanism.AttentionCodeRepair().cuda()
    saved=torch.load(PARENT/'model.pt',map_location='cpu',weights_only=True)
    resume_contract(saved,q.digest(q.read(PARENT/'producer.json')))
    corrector.load_state_dict(saved['model'],strict=True)
    initial_hash=native.vendor.module_hash(corrector)
    if initial_hash!=q.read(PARENT/'result.json')['final_corrector_sha256']:
        raise ValueError('actual trained corrector changed before continuation')
    tokenizer=native.tokenizer()
    initial=original.diagnostics(corrector,renderer,selected,loader,tokenizer)
    if initial!=q.read(PARENT/'outputs.json'):raise ValueError('parent hard outputs must replay before new updates')
    write_json(root/'initial_outputs.json',initial)
    opt=torch.optim.AdamW(corrector.parameters(),lr=Config().learning_rate,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    opt.load_state_dict(saved['optimizer']);sampler=torch.Generator();restore_rng(saved,sampler)
    same=original.preparation.checkpoint.training.lifecycle.same
    if not same(opt.state_dict(),saved['optimizer']):raise ValueError('parent optimizer state differs')
    del saved
    books=[b.weight for b in renderer.codec.vq.embeddings]
    final_step=514 if mode=='pilot' else Config().final_step
    torch.cuda.reset_peak_memory_stats()
    try:
        with (root/'trace.jsonl').open('x') as stream:
            for step in range(Config().start_step+1,final_step+1):
                corrector.train();opt.zero_grad(set_to_none=True);observations=[]
                for group in opt.param_groups:group['lr']=Config().learning_rate
                for micro in range(Config().accumulation):
                    row=selected[int(torch.randint(len(selected),(1,),generator=sampler))]
                    features,mask,predicted,gold,words=original.tensors(row,loader)
                    scores=corrector(predicted,features,mask,books)
                    loss,metrics,hard=original.objective.loss(renderer,scores,gold,words,features,mask)
                    (loss/Config().accumulation).backward()
                    observations.append({'id':row['id'],'metrics':metrics,'source_positions':features.shape[1]})
                    del features,mask,predicted,gold,words,scores,loss,hard
                norm=torch.nn.utils.clip_grad_norm_(corrector.parameters(),1.)
                source_grad=float(corrector.source_kv.weight.grad.abs().sum())
                if not torch.isfinite(norm) or source_grad<=0 or any(v.grad is not None for v in renderer.parameters()):
                    raise ValueError('finite live corrector gradients and frozen renderer required')
                opt.step()
                stream.write(json.dumps({'step':step,'lr':Config().learning_rate,'gradient_norm':float(norm),
                    'source_gradient_L1':source_grad,'microbatches':observations},allow_nan=False)+'\n');stream.flush()
                if step%16==0 or mode=='pilot':
                    write_json(root/'progress.json',{'step':step,'final_step':final_step,'mode':mode})
                    print(json.dumps({'continuation_step':step,'final_step':final_step,'first_losses':observations[0]['metrics']}),flush=True)
                if step==514 and mode=='capacity':
                    pilot=torch.load(RUNS['pilot']/'model.pt',map_location='cpu',weights_only=True)
                    if (not same(corrector.state_dict(),pilot['model']) or not same(opt.state_dict(),pilot['optimizer'])
                            or not torch.equal(sampler.get_state(),pilot['sampler_rng'])
                            or not torch.equal(torch.get_rng_state(),pilot['CPU_rng'])
                            or not same(torch.cuda.get_rng_state_all(),pilot['CUDA_rng'])):
                        raise ValueError('fresh full continuation must exactly replay the two-update pilot prefix')
                    write_json(root/'prefix_replay.json',{'step':514,'all_model_optimizer_and_rng_exact':True})
                    del pilot
        final=original.diagnostics(corrector,renderer,selected,loader,tokenizer)
        final_hash=native.vendor.module_hash(corrector)
        save(root,corrector,opt,sampler,final_step,p)
        del corrector,opt;gc.collect();torch.cuda.empty_cache()
        saved=torch.load(root/'model.pt',map_location='cpu',weights_only=True)
        corrector=original.mechanism.AttentionCodeRepair().cuda();corrector.load_state_dict(saved['model'],strict=True)
        opt=torch.optim.AdamW(corrector.parameters(),lr=Config().learning_rate,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        opt.load_state_dict(saved['optimizer']);restore_rng(saved,sampler)
        if (not same(corrector.state_dict(),saved['model']) or not same(opt.state_dict(),saved['optimizer'])
                or not torch.equal(sampler.get_state(),saved['sampler_rng'])
                or not torch.equal(torch.get_rng_state(),saved['CPU_rng'])
                or not same(torch.cuda.get_rng_state_all(),saved['CUDA_rng'])
                or original.diagnostics(corrector,renderer,selected,loader,tokenizer)!=final
                or native.vendor.module_hash(corrector)!=final_hash or final_hash==initial_hash
                or native.vendor.module_hash(renderer)!=frozen_hash):
            raise ValueError('strict continuation reload or frozen-model integrity failed')
        peak=torch.cuda.max_memory_allocated()
        if peak>4_900_000_000:raise ValueError('continuation exceeds4.9GB allocation')
        write_json(root/'outputs.json',final);native.vendor.verify(p)
        write_json(root/'result.json',{'status':'continued16TRAINcontrol_quality_pending',
            'new_updates':final_step-Config().start_step,'total_updates':final_step,'TRAIN_cases':16,
            'strict_reload_passed':True,'initial_corrector_sha256':initial_hash,'final_corrector_sha256':final_hash,
            'frozen_renderer_sha256':frozen_hash,'checkpoint_sha256':file_hash(root/'model.pt'),
            'peak_allocated_bytes':peak,'full732_training_admitted':False,'task_accuracy':None,'final_test_opened':False})
    finally:renderer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=tuple(RUNS))
    run(parser.parse_args().mode)
