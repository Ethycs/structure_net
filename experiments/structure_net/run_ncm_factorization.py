"""Calibrate, factorize and locally repair the selected distilled NCM."""
import argparse
import copy
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import random
import shutil
import sys
import time

import torch

from experiments.structure_net.ncm_llama import (
    FIXTURES,LABELS,VENDOR,digest,encode_codes,generator_input,infer_ncm,
    load_ncm,sha256,storage_bytes,text_check,tokenizers,write_json)
from experiments.structure_net.ncm_llama_graph import GraphRuntime
from experiments.structure_net.ncm_factorization import (
    BASE,HYPOTHESIS_ID,SCHEMA_VERSION,PLAN,RANK,NAMES,BiasedFactorLinear,
    boundary_loss,factor_state,set_factors,load_factorized)
from experiments.structure_net.run_ncm_llama_graph import MODEL,CODEC,MODEL_SHA,CODEC_SHA


def read(p):return json.loads(Path(p).read_text())


def freeze(args):
    if args.output.exists():raise ValueError('fresh factorization evidence root required')
    if sha256(MODEL)!=MODEL_SHA or sha256(CODEC)!=CODEC_SHA:raise ValueError('parent checkpoint changed')
    args.output.mkdir(parents=True)
    sources=[Path(__file__),PLAN,Path('experiments/structure_net/ncm_factorization.py'),
        Path('experiments/structure_net/ncm_llama.py'),Path('experiments/structure_net/ncm_llama_graph.py'),
        Path('experiments/structure_net/ncm_capture_generate.py'),
        Path('experiments/structure_net/run_ncm_llama_graph.py'),
        Path('experiments/structure_net/prepare_llama_cotenancy.py'),
        Path('experiments/structure_net/llama_neighbourhood_tensorization.py')]+sorted(VENDOR.rglob('*.py'))
    inputs=[MODEL,CODEC,FIXTURES,LABELS]
    if args.command=='train':inputs.append(args.pilot/'result.json')
    if args.command=='development':inputs.extend([args.training/'result.json',args.training/'initial.pt',args.training/'repaired.pt'])
    if args.command=='evaluate':
        inputs.extend([args.development/'result.json',args.development/'source_review.json'])
        if args.condition!='dense':inputs.append(args.training/(args.condition+'.pt'))
    if args.command=='diagnostic':
        inputs.extend([args.development/'result.json',args.development/'source_review.json',args.training/'repaired.pt'])
    dst=args.output/'producer_source';dst.mkdir()
    for i,path in enumerate(sources):shutil.copyfile(path,dst/f'{i:03d}_{path.name}')
    record={'schema':SCHEMA_VERSION,'hypothesis_id':HYPOTHESIS_ID,'command':sys.argv,
        'stage':args.command,'condition':args.condition,'source_hashes':{str(p):sha256(p) for p in sources},
        'input_hashes':{str(p):sha256(p) for p in inputs},
        'config':{'rank':RANK,'groups':6,'updates':96,'lr':2e-5,'seed':51021,'weight_noise':.01,
            'input_noise':.05,'train_sources':64,'local_development_sources':8,'deployment_factor_dtype':'float16',
            'inference_autocast':'bfloat16','factorization':'input-RMS-scaled randomized SVD, q=rank+16, niter=2'},
        'created_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'physical_gpu':0}
    record['fingerprint']=digest(record);write_json(args.output/'producer.json',record)
    return record


def finish(args,p,result):
    for path,h in {**p['source_hashes'],**p['input_hashes']}.items():
        if sha256(path)!=h:raise ValueError('producer/reference changed')
    result.update(schema=SCHEMA_VERSION,hypothesis_id=HYPOTHESIS_ID,producer_fingerprint=p['fingerprint'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(args.output/'result.json',result);print(json.dumps(result),flush=True)


def pair_forward(blocks,x,net,prefix_ids):
    mask=net.attn_mask_for_masking+net._prefix_pad_bias(prefix_ids.eq(50257),x.shape[1],net.attn_mask_for_masking.dtype,x.device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for b in blocks:x=b(x,attn_mask=mask,rope_cos_sin=(net.rope_cos,net.rope_sin))
    return x


def frozen_digest(*modules):
    import hashlib
    h=hashlib.sha256()
    for m in modules:
        for name,t in m.state_dict().items():
            h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def cache_boundaries(args,vae,net,pilot):
    data=read(FIXTURES)['examples'];labels={e['id']:e for e in read(LABELS)['rows']}
    train=[e for e in data if e['split']=='train'][:2 if pilot else 64]
    dev=[e for e in data if e['split']=='development'][:1 if pilot else 8]
    scales={};counts={};states={};hooks=[];collect=[False]
    for i,b in enumerate(net.transformer.blocks):
        if i%2==0:
            hooks.append(b.register_forward_pre_hook(lambda m,a,g=i//2:states.update({f'{g}.input':a[0].detach().cpu()})))
        else:
            hooks.append(b.register_forward_hook(lambda m,a,y,g=i//2:states.update({f'{g}.target':y.detach().cpu()})))
        for name in NAMES:
            key=f'{i}.{name}'
            def scale_hook(m,a,key=key):
                if collect[0]:
                    x=a[0].detach().flatten(0,1).float()
                    scales[key]=scales.get(key,0)+x.square().sum(0)
                    counts[key]=counts.get(key,0)+x.shape[0]
            hooks.append(getattr(b.mlp,name).register_forward_pre_hook(scale_hook))
    inventory=[]
    for e in train+dev:
        collect[0]=e['split']=='train'
        if labels[e['id']]['source_sha256']!=e['source_sha256']:raise ValueError('label/source mismatch')
        prefix=torch.tensor([e['prefix_ids']],device='cuda')
        target=torch.tensor([labels[e['id']]['target_ids']],device='cuda')
        codes=encode_codes(vae,target)
        with torch.autocast('cuda',dtype=torch.bfloat16):net(generator_input(vae,net,prefix,codes),prefix_pad_mask=prefix.eq(50257))
        path=args.output/'cache'/f"{e['id']}.pt";path.parent.mkdir(exist_ok=True)
        torch.save(states,path);states.clear()
        inventory.append({'id':e['id'],'split':e['split'],'source_sha256':e['source_sha256'],
            'prefix_ids':e['prefix_ids'],'path':str(path),'sha256':sha256(path)})
    for h in hooks:h.remove()
    scales={k:(v/counts[k]).sqrt().cpu() for k,v in scales.items()}
    torch.save(scales,args.output/'scales.pt')
    write_json(args.output/'cache.json',{'rows':inventory,'train_count':len(train),'development_count':len(dev),
        'scales_sha256':sha256(args.output/'scales.pt'),'calibration_training_only':True})
    return inventory,scales


def cached(e,group):
    value=torch.load(e['path'],map_location='cpu',weights_only=True)
    return value[f'{group}.input'].cuda(),value[f'{group}.target'].cuda(),torch.tensor([e['prefix_ids']],device='cuda')


@torch.no_grad()
def validation(blocks,net,examples,group):
    deployed=copy.deepcopy(blocks).requires_grad_(False)
    set_factors(deployed,factor_state(blocks))
    values=[]
    for e in examples:
        x,y,prefix=cached(e,group)
        values.append(float(boundary_loss(pair_forward(deployed,x,net,prefix),y)))
    del deployed
    return sum(values)/len(values)


def assemble_state(net,group_states):
    state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    for group,parts in enumerate(group_states):
        for i in range(2):
            for name in NAMES:
                root=f'transformer.blocks.{group*2+i}.mlp.{name}'
                del state[root+'.weight']
                for part in ('left','right'):state[root+'.'+part]=parts[f'{i}.{name}.{part}'].clone()
    return {'model':state,'rank':RANK,'teacher_model_sha256':MODEL_SHA,'codec_sha256':CODEC_SHA}


def train(args,p):
    pilot=args.command=='pilot'
    if not pilot and read(args.pilot/'result.json')['status']!='passed':raise ValueError('systems pilot required')
    torch.manual_seed(51021)
    vae,net,_=load_ncm(CODEC,MODEL,adaptation=True);net.requires_grad_(False)
    frozen=frozen_digest(vae,net)
    inventory,scales=cache_boundaries(args,vae,net,pilot)
    training=[e for e in inventory if e['split']=='train'];development=[e for e in inventory if e['split']=='development']
    initial_states=[];selected_states=[];records=[]
    for group in range(6):
        start=time.perf_counter()
        dense=net.transformer.blocks[group*2:group*2+2]
        x,y,prefix=cached(development[0],group)
        with torch.no_grad():dense_nmse=float(boundary_loss(pair_forward(dense,x,net,prefix),y))
        if dense_nmse>1e-6:raise RuntimeError('dense neighborhood/cache replay mismatch')
        blocks=copy.deepcopy(dense).eval().requires_grad_(False)
        for i,b in enumerate(blocks):
            for j,name in enumerate(NAMES):
                factor=BiasedFactorLinear.from_linear(getattr(b.mlp,name),scales[f'{2*group+i}.{name}'].cuda(),RANK,51021+1000*(2*group+i)+j)
                setattr(b.mlp,name,factor)
        initial=factor_state(blocks);initial_states.append(initial)
        params=[p for b in blocks for name in NAMES for p in (getattr(b.mlp,name).left,getattr(b.mlp,name).right)]
        should_train=not pilot or group==0
        initial_error=validation(blocks,net,development,group)
        if not should_train:
            selected_states.append(initial);del blocks,params;continue
        rng=torch.Generator(device='cuda').manual_seed(81021+group)
        with torch.no_grad():
            for value in params:value.add_(torch.randn(value.shape,device='cuda',generator=rng)*value.square().mean().sqrt()*.01)
        noisy=factor_state(blocks);torch.save(noisy,args.output/f'group{group}_noisy.pt')
        best=validation(blocks,net,development,group);best_state=noisy;best_step=0
        history=[{'step':0,'development_nmse':best}]
        optimizer=torch.optim.AdamW(params,lr=2e-5,weight_decay=0.,foreach=False)
        order=random.Random(71021+group).choices(training,k=2 if pilot else 96)
        for step,e in enumerate(order,1):
            x,y,prefix=cached(e,group)
            with torch.no_grad():
                rms=x.float().square().mean((0,1),keepdim=True).sqrt()
                changed=(x.float()+torch.randn(x.shape,device='cuda',generator=rng)*rms*.05).to(x.dtype)
                target=pair_forward(dense,changed,net,prefix)
            optimizer.zero_grad(set_to_none=True)
            loss=boundary_loss(pair_forward(blocks,changed,net,prefix),target)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite local objective')
            loss.backward()
            if not all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in params):
                raise RuntimeError('missing or nonfinite factor gradient')
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            row={'step':step,'id':e['id'],'loss':float(loss.detach()),'gradient_norm':float(norm)}
            if step%32==0 or step==len(order):
                error=validation(blocks,net,development,group);row['development_nmse']=error
                if error<best:best,best_state,best_step=error,factor_state(blocks),step
                print(json.dumps({'group':group,**row}),flush=True)
            history.append(row)
        selected_states.append(best_state)
        path=args.output/f'group{group}_selected.pt';torch.save(best_state,path)
        set_factors(blocks,best_state)
        with torch.no_grad():expected=pair_forward(blocks,x,net,prefix)
        set_factors(blocks,torch.load(path,map_location='cpu',weights_only=True))
        with torch.no_grad():actual=pair_forward(blocks,x,net,prefix)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        row={'group':group,'dense_control_nmse':dense_nmse,'initial_development_nmse':initial_error,
            'selected_development_nmse':best,'selected_step':best_step,'history':history,
            'checkpoint_reload_exact':True,'selected_sha256':sha256(path),'seconds':time.perf_counter()-start}
        records.append(row);write_json(args.output/f'group{group}.json',row)
        del blocks,optimizer,params,expected,actual,loss,target,x,y,changed
        gc.collect()
    for name,states in (('initial',initial_states),('repaired',selected_states)):
        torch.save(assemble_state(net,states),args.output/(name+'.pt'))
    if frozen_digest(vae,net)!=frozen or any(x.grad is not None for x in list(vae.parameters())+list(net.parameters())):
        raise RuntimeError('original teacher or codec changed')
    result={'status':'passed' if pilot else 'trained','groups':records,'training_updates':2 if pilot else 576,
        'teacher_and_codec_frozen_sha256':frozen,'checkpoints':{name:sha256(args.output/(name+'.pt')) for name in ('initial','repaired')}}
    if pilot:
        from experiments.structure_net.ncm_llama import infer_ncm
        gpt,_=tokenizers();vae2,net2,_=load_factorized(args.output/'repaired.pt')
        ex=[e for e in read(FIXTURES)['examples'] if e['split']=='development'][0]
        prefix=torch.tensor([ex['prefix_ids']],device='cuda')
        expected=infer_ncm(vae2,net2,prefix,gpt)[0]
        runtime=GraphRuntime(vae2,net2);actual=runtime.generate(ex['prefix_ids'],gpt)
        if expected['native_output_ids']!=actual['native_output_ids']:raise RuntimeError('factor graph/eager parity failed')
        result.update(systems_label='systems_lifecycle_only_not_quality_evidence',graph_eager_exact=True)
    finish(args,p,result)


@torch.inference_mode()
def development(args,p):
    gpt,llama=tokenizers();data=[e for e in read(FIXTURES)['examples'] if e['split']=='development']
    results={}
    for name in ('initial','repaired'):
        vae,net,_=load_factorized(args.training/(name+'.pt'));runtime=GraphRuntime(vae,net);rows=[]
        for e in data:
            r=runtime.generate(e['prefix_ids'],gpt)
            r.update(id=e['id'],source_sha256=e['source_sha256'],diagnostics=text_check(r['text'],e,llama))
            rows.append(r)
        write_json(args.output/(name+'.json'),{'rows':rows})
        results[name]={'field_passes':sum(r['diagnostics']['diagnostic_pass'] for r in rows),
            'unique_runtime_weight_buffer_bytes':storage_bytes(vae,net),'rows_sha256':sha256(args.output/(name+'.json'))}
        del runtime,vae,net;gc.collect();torch.cuda.empty_cache()
    # Full source review occurs after this producer, before promotion.
    finish(args,p,{'status':'development_pending_source_review','conditions':results})


@torch.inference_mode()
def evaluate(args,p):
    decision=read(args.development/'source_review.json')
    if decision['selected'] is None or (args.condition!='dense' and args.condition!=decision['selected']):
        raise ValueError('source-reviewed development promotion required')
    gpt,llama=tokenizers()
    vae,net,_=load_ncm(CODEC,MODEL,adaptation=True) if args.condition=='dense' else load_factorized(args.training/(args.condition+'.pt'))
    runtime=GraphRuntime(vae,net);torch.cuda.reset_peak_memory_stats();rows=[];warm=[]
    for e in [e for e in read(FIXTURES)['examples'] if e['split']=='heldout']:
        for rep in (-1,0,1,2):
            torch.cuda.synchronize();start=time.perf_counter();r=runtime.generate(e['prefix_ids'],gpt);torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            r.update(id=e['id'],repeat=rep,source_sha256=e['source_sha256'],wall_seconds=seconds,
                common_output_tokens=len(llama.encode(r['text'],add_special_tokens=False)),diagnostics=text_check(r['text'],e,llama))
            (rows if rep>=0 else warm).append(r)
        if len(rows)%60==0:print(json.dumps({'condition':args.condition,'cases':len(warm)}),flush=True)
    write_json(args.output/'rows.json',{'rows':rows})
    write_json(args.output/'excluded_warmups.json',{'rows':warm})
    finish(args,p,{'status':'completed_pending_source_review','condition':args.condition,'requests':len(rows),
        'rows_sha256':sha256(args.output/'rows.json'),'warmups_sha256':sha256(args.output/'excluded_warmups.json'),
        'unique_runtime_weight_buffer_bytes':storage_bytes(vae,net),'graph_setup_seconds':runtime.setup_seconds})


@torch.inference_mode()
def diagnostic(args,p):
    if read(args.development/'source_review.json')['selected'] is not None:
        raise ValueError('diagnostic is only for failed development promotion')
    from experiments.neural_architecture_lab.finalize_ncm_llama import summarize
    gpt,llama=tokenizers();results={}
    data=[e for e in read(FIXTURES)['examples'] if e['split']=='development'][:8]
    for condition in ('repaired','dense'):
        vae,net,_=load_ncm(CODEC,MODEL,adaptation=True) if condition=='dense' else load_factorized(args.training/'repaired.pt')
        runtime=GraphRuntime(vae,net);rows=[]
        for e in data:
            runtime.generate(e['prefix_ids'],gpt)
            for rep in range(3):
                torch.cuda.synchronize();start=time.perf_counter();r=runtime.generate(e['prefix_ids'],gpt);torch.cuda.synchronize()
                r.update(id=e['id'],repeat=rep,wall_seconds=time.perf_counter()-start,
                    common_output_tokens=len(llama.encode(r['text'],add_special_tokens=False)),source_sha256=e['source_sha256'])
                rows.append(r)
        write_json(args.output/(condition+'.json'),{'rows':rows})
        results[condition]={'summary':summarize(rows),'model_bytes':storage_bytes(vae,net),
            'rows_sha256':sha256(args.output/(condition+'.json'))}
        del runtime,vae,net;gc.collect();torch.cuda.empty_cache()
    finish(args,p,{'status':'diagnostic_only_after_failed_development','requests':48,'conditions':results,'final_stage_executed':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('pilot','train','development','evaluate','diagnostic'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',type=Path,default=BASE/'20261021_pilot_v1')
    p.add_argument('--training',type=Path,default=BASE/'20261021_training_v1')
    p.add_argument('--development',type=Path,default=BASE/'20261021_development_v1')
    p.add_argument('--condition',choices=('initial','repaired','dense'))
    args=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('select physical GPU0 before Python starts')
    try:
        producer=freeze(args)
        if args.command in ('pilot','train'):train(args,producer)
        else:globals()[args.command](args,producer)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/'result.json').exists():
            write_json(args.output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':main()
