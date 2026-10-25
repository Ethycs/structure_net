"""Fixed-budget native sequential TRAIN control; no held-out accuracy claim."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_engineering_attention_repair_v3 as base
from experiments.structure_net import ncm_sequential_levels as mechanism
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT, native, q, preparation = base.ROOT, base.native, base.q, base.preparation
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sequential-levels-plan.md')
TEST = Path('tests/structure_net/test_ncm_sequential_levels.py')
RUNS = {m: ROOT / ('20261024_sequential_engineering16_' + m + '_v1') for m in ('pilot','capacity')}
CONFIG = dict(updates=512, accumulation=2, warmup=32, max_lr=1e-4, min_lr=1e-5,
    sampler_seed=141142, corruption_seed=142142, dropout_seed=143142,
    optimizer='AdamW', betas=[.9,.95], eps=1e-8, weight_decay=.01, clip=1.,
    precision='FP32weights_FP16training_FP32generation', peak_limit_bytes=4_900_000_000,
    TF32=False, physical_GPU=2, source_only_inference=True, codec_frozen=True,
    task_accuracy=None, final_test_opened=False, speed_tested=False)


def rate(step):
    if not 1 <= step <= CONFIG['updates']:
        raise ValueError('step outside fixed512 budget')
    if step <= 32:
        return CONFIG['max_lr'] * step / 32
    return 1e-5 + .5 * 9e-5 * (1 + math.cos(math.pi * (step-32)/480))


def protected(model):
    return {k:native.vendor.module_hash(v) for k,v in
        (('codec',model.codec),('word_memory',model.word_memory),('word_adapters',model.word_adapters))}


def tensors(row, loader):
    f = loader.get(row).unsqueeze(0).cuda()
    return f, torch.ones(f.shape[:2],dtype=torch.bool,device=f.device), torch.tensor([row['oracle_codes']],device=f.device)


@torch.no_grad()
def diagnostics(model, rows, loader, tokenizer, shuffle=False):
    mechanism.configure(model,False)
    output=[]
    for i,row in enumerate(rows):
        f,m,_=tensors(rows[(i+1)%len(rows)] if shuffle else row,loader)
        gold=torch.tensor([row['oracle_codes']],device=f.device)
        codes=mechanism.generate(model,f,m)
        raw=model.word_logits(f,m,codes).argmax(-1)
        words=torch.tensor([row['native_word_ids']],device=f.device)
        active=words!=0
        logits=model.concepts.teacher_logits(f,m,gold)
        offset,level_metrics=0,[]
        for length in model.concepts.generator.levels:
            scores=logits[:,offset:offset+length].float()
            target=gold[:,offset:offset+length]
            level_metrics.append({'CE':float(F.cross_entropy(scores.flatten(0,1),target.flatten())),
                'teacher_correct':int((scores.argmax(-1)==target).sum()),
                'rollout_correct':int((codes[:,offset:offset+length]==target).sum()),'positions':length})
            offset+=length
        ids=raw.cpu().tolist()[0]
        output.append({'id':row['id'],'source_id_used':rows[(i+1)%len(rows)]['id'] if shuffle else row['id'],
            'codes':codes.cpu().tolist()[0],'raw_word_ids':ids,**preparation.rendering.decode(ids,tokenizer),
            'active_word_matches':int(((raw==words)&active).sum()),'active_words':int(active.sum()),
            'levels':level_metrics,'task_accuracy':None})
    return output


def summarize(rows):
    words=sum(r['active_words'] for r in rows)
    return dict(active_word_matches=sum(r['active_word_matches'] for r in rows),active_words=words,
        active_word_agreement=sum(r['active_word_matches'] for r in rows)/words,
        mean_teacher_CE=sum(x['CE'] for r in rows for x in r['levels'])/(len(rows)*12),
        teacher_correct_codes=sum(x['teacher_correct'] for r in rows for x in r['levels']),
        rollout_correct_codes=sum(x['rollout_correct'] for r in rows for x in r['levels']),
        code_positions=len(rows)*641)


def freeze(root,mode):
    files={Path(__file__),Path(mechanism.__file__),PLAN,TEST}
    for module in tuple(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p:
            p=Path(p).resolve()
            if p.is_file() and p.suffix=='.py' and (p.is_relative_to(Path('experiments').resolve()) or p.is_relative_to(native.vendor.VENDOR.resolve())):
                files.add(p)
    files.update(native.vendor.VENDOR.rglob('*.py'))
    inputs=[base.DATA/n for n in ('cases.json','source_index.json','feature_manifest.json','selection.json','result.json','actual_tool_receipt.json')]
    inputs += [preparation.checkpoint.training.TRAIN/'model.pt']
    if mode=='capacity': inputs += [RUNS['pilot']/n for n in ('result.json','actual_tool_receipt.json','producer.json')]
    root.mkdir()
    (root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof={'schema':'nal.ncm-sequential-engineering.v1','hypothesis_id':'ncm-sequential-native-engineering16-v1',
        'conformance':'EXPLORATORY','stage':mode,'config':{**CONFIG,'actual_updates':2 if mode=='pilot' else 512},
        'source_hashes':{str(p):file_hash(p) for p in sorted(files)},
        'input_hashes':{str(p):file_hash(p) for p in inputs},
        'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()},
        'systems_lifecycle_only_not_quality_evidence':mode=='pilot'}
    proof['fingerprint']=q.digest(proof)
    write_json(root/'producer.json',proof)
    return proof


@torch.no_grad()
def parity(model,row,loader):
    mechanism.configure(model,False)
    f,m,gold=tensors(row,loader)
    native_logits=model.concepts.teacher_logits(f,m,gold)
    cumulative=f.new_zeros(1,128,model.codec.embed_dim)
    previous,offset,outputs=None,0,[]
    for k,n in enumerate(model.concepts.generator.levels):
        outputs.append(mechanism.level_logits(model,f,m,previous,k))
        cumulative,previous=mechanism.advance(model.codec,cumulative,gold[:,offset:offset+n],k)
        offset+=n
    logits=torch.cat(outputs,dim=1)
    same_codes=torch.equal(mechanism.generate(model,f,m),model.concepts.generate_codes(f,m))
    good=torch.allclose(logits,native_logits,rtol=1e-4,atol=1e-4)
    result={'same_history_allclose':good,'same_history_max_abs_difference':float((logits-native_logits).abs().max()),
        'same_history_argmax_disagreements':int((logits.argmax(-1)!=native_logits.argmax(-1)).sum()),
        'source_only_greedy_codes_exact':same_codes}
    if not good or not same_codes: raise ValueError('explicit sequential path differs from native path:'+str(result))
    return result


def save(path,model,opt,scaler,sampler,noise,step,proof):
    if path.exists(): raise ValueError('checkpoint overwrite forbidden')
    torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'sampler_rng':sampler.get_state(),
        'noise_rng':noise.get_state(),'CPU_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state_all(),
        'step':step,'producer_fingerprint':proof['fingerprint']},path)


def run(mode):
    native.vendor.corpus.quality.dependencies()
    preparation.checkpoint.training.require_gpu()
    preparation.completed(base.DATA)
    if mode=='capacity':
        preparation.completed(RUNS['pilot'])
        pilot=q.read(RUNS['pilot']/'result.json')
        if not pilot['strict_reload_passed'] or not pilot['parity_passed']:
            raise ValueError('completed matching CUDA lifecycle required')
    if file_hash(base.DATA/'cases.json')!=q.read(base.DATA/'result.json')['cases_sha256']:
        raise ValueError('TRAIN data changed')
    lookup={r['id']:r for r in q.read(base.DATA/'cases.json')}
    rows=[lookup[i] for i in q.read(base.DATA/'selection.json')['diagnostic_ids']]
    if len(rows)!=16: raise ValueError('same16 TRAIN cases required')
    index,manifest=q.read(base.DATA/'source_index.json'),q.read(base.DATA/'feature_manifest.json')
    loader=preparation.streaming.FeatureCache(index,manifest,max_bytes=1024**3)
    root=RUNS[mode]
    if root.exists(): raise ValueError('fresh sequential experiment root required')
    model=base.load_renderer()
    proof=freeze(root,mode)
    frozen=protected(model)
    initial_hash=native.vendor.module_hash(model)
    lengths={r['id']:r['source_positions'] for r in index}
    longest=max(range(16),key=lambda i:lengths[rows[i]['id']])
    checks=[parity(model,rows[i],loader) for i in dict.fromkeys((0,longest))]
    write_json(root/'parity.json',checks)
    tokenizer=native.tokenizer()
    initial=diagnostics(model,rows,loader,tokenizer)
    write_json(root/'initial_outputs.json',initial)
    if mode=='capacity' and initial!=q.read(RUNS['pilot']/'initial_outputs.json'):
        raise ValueError('capacity initializer differs from pilot')
    write_json(root/'TRAIN_oracle_controls.json',base.oracle_controls(model.eval().requires_grad_(False),rows,loader,tokenizer))
    mechanism.configure(model,True)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=2000)
    sampler=torch.Generator().manual_seed(CONFIG['sampler_seed'])
    noise=torch.Generator().manual_seed(CONFIG['corruption_seed'])
    torch.manual_seed(CONFIG['dropout_seed']); torch.cuda.manual_seed_all(CONFIG['dropout_seed'])
    steps=2 if mode=='pilot' else 512
    permutation=[]
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    with (root/'trace.jsonl').open('x') as trace:
        for step in range(1,steps+1):
            mechanism.configure(model,True)
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups: group['lr']=rate(step)
            metrics=[]
            for micro in range(2):
                if not permutation: permutation=torch.randperm(16,generator=sampler).tolist()
                index=([0,longest][micro] if mode=='pilot' else permutation.pop())
                f,m,gold=tensors(rows[index],loader)
                metric=mechanism.backward_example(model,f,m,gold,noise,scaler,2,noisy=bool(micro))
                metrics.append({'id':rows[index]['id'],**metric})
            scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm): raise ValueError('nonfinite sequential gradients')
            if step==1 and (model.concepts.generator.l0_queries.grad is None or not model.concepts.generator.l0_queries.grad.abs().sum()):
                raise ValueError('native root query must receive gradients')
            if any(p.grad is not None for p in model.codec.parameters()): raise ValueError('frozen codec received gradients')
            scaler.step(opt); scaler.update()
            peak=torch.cuda.max_memory_allocated()
            if peak>CONFIG['peak_limit_bytes']: raise ValueError('sequential training exceeded4.9GB allocation')
            record={'step':step,'lr':rate(step),'gradient_norm':float(norm),'metrics':metrics,'elapsed_seconds':time.monotonic()-started}
            trace.write(json.dumps(record,allow_nan=False)+'\n');trace.flush()
            if step<=2 or step%16==0:
                write_json(root/'progress.json',record)
                print(json.dumps({'step':step,'updates':steps,'mean_CE':sum(r['mean_CE'] for r in metrics)/2,
                    'elapsed_seconds':record['elapsed_seconds'],'peak_bytes':peak}),flush=True)
            if mode=='capacity' and step in (128,256):
                save(root/f'step_{step}.pt',model,opt,scaler,sampler,noise,step,proof)
    final=diagnostics(model,rows,loader,tokenizer)
    shuffled=diagnostics(model,rows,loader,tokenizer,shuffle=True)
    final_hash=native.vendor.module_hash(model)
    save(root/'model.pt',model,opt,scaler,sampler,noise,steps,proof)
    if frozen!=protected(model) or initial_hash==final_hash:
        raise ValueError('codec changed or sequential generator did not learn')
    del opt
    gc.collect();torch.cuda.empty_cache()
    saved=torch.load(root/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True)
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
    sampler.set_state(saved['sampler_rng']);noise.set_state(saved['noise_rng'])
    torch.set_rng_state(saved['CPU_rng']);torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    same=preparation.checkpoint.training.lifecycle.same
    if (not same(model.state_dict(),saved['model']) or not same(opt.state_dict(),saved['optimizer'])
            or not same(scaler.state_dict(),saved['scaler']) or not torch.equal(sampler.get_state(),saved['sampler_rng'])
            or not torch.equal(noise.get_state(),saved['noise_rng']) or not torch.equal(torch.get_rng_state(),saved['CPU_rng'])
            or not same(torch.cuda.get_rng_state_all(),saved['CUDA_rng'])
            or diagnostics(model,rows,loader,tokenizer)!=final or native.vendor.module_hash(model)!=final_hash):
        raise ValueError('strict state/optimizer/scaler/RNG/output reload failed')
    before,after,shuffle=summarize(initial),summarize(final),summarize(shuffled)
    write_json(root/'outputs.json',final);write_json(root/'shuffled_source_outputs.json',shuffled)
    native.vendor.verify(proof)
    result={'status':'sequential_'+mode+'_completed_TRAIN_diagnostic_only','schema':'nal.ncm-sequential-engineering.v1',
        'hypothesis_id':proof['hypothesis_id'],'updates':steps,'train_cases':16,'trainable_parameters':sum(p.numel() for p in params),
        'initial':before,'final':after,'shuffled_source':shuffle,'strict_reload_passed':True,'parity_passed':True,
        'frozen_hashes':frozen,'initial_model_sha256':initial_hash,'final_model_sha256':final_hash,
        'checkpoint_sha256':file_hash(root/'model.pt'),'producer_fingerprint':proof['fingerprint'],
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'training_seconds':time.monotonic()-started,
        'numeric_learnability_pass':after['mean_teacher_CE']<=.5*before['mean_teacher_CE'] and after['active_word_agreement']>=.95,
        'decoded_text_review_pending':True,'full732_training_admitted':False,
        'task_accuracy':None,'final_test_opened':False,'speed_tested':False}
    write_json(root/'result.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)
    model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('pilot','capacity'))
    run(parser.parse_args().stage)
