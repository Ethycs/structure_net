"""Lossless fixed-vocabulary validation and matched NCM serving measurements."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from experiments.structure_net.ncm_llama import FIXTURES,VENDOR,digest,load_ncm,sha256,storage_bytes,tokenizers,write_json
from experiments.structure_net.ncm_llama_graph import GraphRuntime,unpack_output
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.run_ncm_replicas import ReplicaPool,REFERENCE,waves,summarize
from experiments.structure_net.run_ncm_batching import BatchRuntime
from experiments.structure_net.run_ncm_llama_graph import MODEL,CODEC,MODEL_SHA,CODEC_SHA

BASE=Path('data/experiments/ncm_host_io')
PLAN=Path('docs/07 - Status Reports/2026-10-21_ncm-host-io-plan.md')
HYPOTHESIS_ID='llama32-3b-ncm-host-io-v1'


def read(p):return json.loads(Path(p).read_text())


def freeze(args):
    if args.output.exists():raise ValueError('fresh host-I/O evidence root required')
    if sha256(MODEL)!=MODEL_SHA or sha256(CODEC)!=CODEC_SHA:raise ValueError('frozen NCM checkpoints differ')
    args.output.mkdir(parents=True);dst=args.output/'producer_source';dst.mkdir()
    sources=[Path(__file__),PLAN]+[Path('experiments/structure_net')/x for x in (
        'ncm_vocabulary.py','run_ncm_replicas.py','run_ncm_batching.py','ncm_llama.py','ncm_llama_graph.py',
        'ncm_capture_generate.py','run_ncm_llama_graph.py','prepare_llama_cotenancy.py')]+sorted(VENDOR.rglob('*.py'))
    for i,path in enumerate(sources):shutil.copyfile(path,dst/f'{i:03d}_{path.name}')
    inputs=[MODEL,CODEC,FIXTURES,REFERENCE]
    if args.command=='run':inputs.append(args.pilot/'result.json')
    p={'schema':'nal.ncm-host-io.v1','hypothesis_id':HYPOTHESIS_ID,'command':sys.argv,'stage':args.command,
        'config':{'mode':args.mode,'size':args.size,'repeats':3,'prefix':512,'canvas':48,
            'immutable_vocabulary_size':50258,'cached_length':True,'training_updates':0},
        'source_hashes':{str(x):sha256(x) for x in sources},'input_hashes':{str(x):sha256(x) for x in inputs},
        'created_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'physical_gpu':0}
    p['fingerprint']=digest(p);write_json(args.output/'producer.json',p);return p


def finish(args,p,result):
    for path,h in {**p['source_hashes'],**p['input_hashes']}.items():
        if sha256(path)!=h:raise ValueError('host-I/O producer/reference changed')
    result.update(hypothesis_id=HYPOTHESIS_ID,producer_fingerprint=p['fingerprint'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(args.output/'result.json',result);print(json.dumps(result),flush=True)


@torch.inference_mode()
def profile(runtime,example,gpt):
    torch.cuda.synchronize();start=time.perf_counter()
    runtime.prefix.copy_(torch.tensor([example['prefix_ids']],device='cuda',dtype=torch.long));torch.cuda.synchronize()
    after_input=time.perf_counter();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    begin.record();runtime.graph.replay();end.record();end.synchronize()
    after_graph=time.perf_counter();values=runtime.output.tolist();after_output=time.perf_counter()
    decoded=unpack_output(values,gpt);finish=time.perf_counter()
    return decoded,{'wall_seconds':finish-start,'input_seconds':after_input-start,
        'gpu_graph_seconds':begin.elapsed_time(end)/1000,'graph_wall_seconds':after_graph-after_input,
        'output_copy_seconds':after_output-after_graph,'validation_decode_seconds':finish-after_output}


@torch.inference_mode()
def execute(args):
    p=freeze(args);gpt,llama=tokenizers();fast=CachedVocabulary(gpt)
    if len(fast)!=50258:raise ValueError('fixed source/output vocabulary identity differs')
    data=read(FIXTURES)['examples'];dev=[e for e in data if e['split']=='development']
    if args.command=='pilot':
        vae,net,_=load_ncm(CODEC,MODEL,adaptation=True);runtime=GraphRuntime(vae,net)
        profiles=[]
        for e in dev[:8]:
            old,a=profile(runtime,e,gpt);new,b=profile(runtime,e,fast)
            if old['native_output_ids']!=new['native_output_ids'] or old['text']!=new['text']:raise RuntimeError('cached vocabulary changed output')
            profiles.append({'id':e['id'],'legacy':a,'cached':b,'parity':True})
        expected={e['id']:runtime.generate(e['prefix_ids'],fast) for e in dev[:9]}
        batched=BatchRuntime(vae,net,9);checks=[]
        for part in (dev[:9],list(reversed(dev[:9])),dev[:9]):
            outputs,_=batched.serve(part,fast)
            checks.extend({'id':e['id'],'parity':r['native_output_ids']==expected[e['id']]['native_output_ids'] and r['text']==expected[e['id']]['text']} for e,r in zip(part,outputs))
        write_json(args.output/'profiles.json',{'rows':profiles});write_json(args.output/'checks.json',{'rows':checks})
        if not all(x['parity'] for x in checks):raise RuntimeError('batched cached-vocabulary parity failed')
        finish(args,p,{'status':'passed','systems_label':'systems_lifecycle_only_not_quality_evidence','parity_checks':35,
            'profiles_sha256':sha256(args.output/'profiles.json'),'checks_sha256':sha256(args.output/'checks.json'),
            'model_bytes':storage_bytes(vae,net)})
        return
    if read(args.pilot/'result.json')['status']!='passed':raise ValueError('host-I/O pilot required')
    if args.mode=='replicas':
        pool=ReplicaPool(args.size);serve=pool.serve;model_bytes=pool.model_bytes
        setup=sum(x.setup_seconds for x in pool.runtimes)
    else:
        vae,net,_=load_ncm(CODEC,MODEL,adaptation=True);runtime=BatchRuntime(vae,net,args.size)
        serve=runtime.serve;model_bytes=storage_bytes(vae,net);setup=runtime.setup_seconds
    torch.cuda.reset_peak_memory_stats()
    warm,seconds=serve(dev[:args.size],fast)
    write_json(args.output/'excluded_warmup.json',{'outputs':warm,'wall_seconds':seconds})
    reference={(x['id'],x['repeat']):x for x in read(REFERENCE)['rows']}
    examples=[e for e in data if e['split']=='heldout'];rows=[];batches=[]
    for rep in range(3):
        for wave,part in enumerate(waves(examples,args.size)):
            outputs,seconds=serve(part,fast)
            batches.append({'repeat':rep,'wave':wave,'ids':[e['id'] for e in part],'wall_seconds':seconds})
            for e,r in zip(part,outputs):
                old=reference[e['id'],rep]
                r.update(id=e['id'],repeat=rep,wave=wave,source_sha256=e['source_sha256'],
                    common_output_tokens=len(llama.encode(r['text'],add_special_tokens=False)),
                    parity=r['native_output_ids']==old['native_output_ids'] and r['text']==old['text'])
                rows.append(r)
            if not all(x['parity'] for x in rows[-len(part):]):raise RuntimeError('full-cohort cached-vocabulary output parity failed')
        write_json(args.output/'rows.json',{'rows':rows,'batches':batches})
        print(json.dumps({'mode':args.mode,'size':args.size,'completed_repeat':rep+1}),flush=True)
    finish(args,p,{'status':'completed','mode':args.mode,'size':args.size,'summary':summarize(rows,batches,args.size),
        'model_bytes':model_bytes,'graph_setup_seconds':setup,'rows_sha256':sha256(args.output/'rows.json'),
        'timed_dummy_outputs':sum(args.size-len(b['ids']) for b in batches) if args.mode=='batch' else 0})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('pilot','run'))
    p.add_argument('--mode',choices=('replicas','batch'),default='batch')
    p.add_argument('--size',type=int,choices=(1,8,9),default=9);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',type=Path,default=BASE/'20261021_pilot_v1');args=p.parse_args()
    if args.mode=='batch' and args.size==8:raise ValueError('fixed batch sizes1/9 required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('select physical GPU0 before Python starts')
    try:execute(args)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/'result.json').exists():
            write_json(args.output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':main()
