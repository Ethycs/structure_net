"""Measure literal frozen NCM replicas on independent streams of one GPU."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from experiments.structure_net.ncm_llama import (
    BASE as PARENT, FIXTURES, VENDOR, digest, load_ncm, sha256,
    storage_bytes, tokenizers, write_json)
from experiments.structure_net.ncm_llama_graph import GraphRuntime, unpack_output
from experiments.structure_net.run_ncm_llama_graph import MODEL, CODEC, MODEL_SHA, CODEC_SHA

BASE = Path('data/experiments/ncm_replicas')
PLAN = Path('docs/07 - Status Reports/2026-10-21_ncm-replicas-plan.md')
HYPOTHESIS_ID = 'llama32-3b-ncm-replicas-v1'
SCHEMA_VERSION = 'nal.ncm-replicas.v1'
ORDER = (1, 9, 2, 8, 4, 7)
REFERENCE = PARENT/'20261021_selected_heldout_v1/rows.json'
TEACHER = PARENT/'20261021_teacher_heldout_v1/result.json'


@dataclass(frozen=True)
class Config:
    copies: int
    repeats: int = 3
    prefix: int = 512
    canvas: int = 48

    def __post_init__(self):
        if type(self.copies) is not int or self.copies not in ORDER or (self.repeats,self.prefix,self.canvas)!=(3,512,48):
            raise ValueError('fixed replica protocol required')


def read(path):
    return json.loads(Path(path).read_text())


def waves(examples, copies):
    if copies not in ORDER:
        raise ValueError('unsupported replica count')
    return [examples[i:i+copies] for i in range(0,len(examples),copies)]


def summarize(rows, batches, copies):
    keys = [(r['id'],r['repeat']) for r in rows]
    if len(keys)!=300 or len(set(keys))!=300 or len({r['id'] for r in rows})!=100:
        raise ValueError('complete unique 100 by 3 request cells required')
    if {r['repeat'] for r in rows}!={0,1,2} or not all(r['parity'] for r in rows):
        raise ValueError('all native output parity required')
    if len(batches)!=3*((100+copies-1)//copies):
        raise ValueError('complete wave schedule required')
    batch_keys = [(b['repeat'],b['wave']) for b in batches]
    if len(set(batch_keys))!=len(batch_keys):
        raise ValueError('unique wave identities required')
    accounted=[]
    for b in batches:
        matched=[r for r in rows if (r['repeat'],r['wave'])==(b['repeat'],b['wave'])]
        if [r['id'] for r in matched]!=b['ids'] or not 0<len(matched)<=copies:
            raise ValueError('wave/request pairing differs')
        if not 0<b['wall_seconds']<float('inf'):
            raise ValueError('positive finite wall time required')
        accounted.extend((r['id'],r['repeat']) for r in matched)
    if sorted(accounted)!=sorted(keys):
        raise ValueError('wave schedule omits requests')
    duration=sum(b['wall_seconds'] for b in batches)
    tokens=sum(r['common_output_tokens'] for r in rows)
    rates=[]
    for rep in range(3):
        t=sum(b['wall_seconds'] for b in batches if b['repeat']==rep)
        n=sum(r['common_output_tokens'] for r in rows if r['repeat']==rep)
        rates.append(n/t)
    return {'copies':copies,'requests':len(rows),'waves':len(batches),
        'wall_seconds':duration,'common_output_tokens':tokens,
        'aggregate_tokens_per_second':tokens/duration,'effective_tokens_per_second_per_copy':tokens/duration/copies,
        'requests_per_second':len(rows)/duration,'repetition_tokens_per_second':rates,
        'mean_wave_seconds':duration/len(batches),'parity_cells':sum(r['parity'] for r in rows)}


class ReplicaPool:
    @torch.inference_mode()
    def __init__(self, copies):
        self.runtimes=[]
        self.streams=[]
        self.modules=[]
        for i in range(copies):
            vae,net,_=load_ncm(CODEC,MODEL,adaptation=True)
            self.modules.extend((vae,net))
            self.runtimes.append(GraphRuntime(vae,net))
            self.streams.append(torch.cuda.Stream())
            print(json.dumps({'loaded_copies':i+1,'requested':copies}),flush=True)
        self.model_bytes=storage_bytes(*self.modules)
        if self.model_bytes!=copies*713294288:
            raise ValueError('independent full model storages required')
        torch.cuda.synchronize()

    @torch.inference_mode()
    def serve(self, examples, gpt):
        if not 1<=len(examples)<=len(self.runtimes):
            raise ValueError('one source per active replica required')
        torch.cuda.synchronize()
        start=time.perf_counter()
        # CPU buffers are fresh per request; retain them until device completion.
        inputs=[torch.tensor([e['prefix_ids']],dtype=torch.long) for e in examples]
        for i,cpu_ids in enumerate(inputs):
            with torch.cuda.stream(self.streams[i]):
                self.runtimes[i].prefix.copy_(cpu_ids,non_blocking=True)
                self.runtimes[i].graph.replay()
        torch.cuda.synchronize()
        outputs=[unpack_output(self.runtimes[i].output.tolist(),gpt) for i in range(len(examples))]
        return outputs,time.perf_counter()-start


def freeze(args):
    if args.output.exists():
        raise ValueError('fresh append-only evidence root required')
    if sha256(MODEL)!=MODEL_SHA or sha256(CODEC)!=CODEC_SHA:
        raise ValueError('frozen checkpoint changed')
    args.output.mkdir(parents=True)
    sources=[Path(__file__),PLAN,Path('experiments/structure_net/ncm_llama.py'),
        Path('experiments/structure_net/ncm_llama_graph.py'),
        Path('experiments/structure_net/ncm_capture_generate.py'),
        Path('experiments/structure_net/run_ncm_llama_graph.py'),
        Path('experiments/structure_net/prepare_llama_cotenancy.py')]+sorted(VENDOR.rglob('*.py'))
    dst=args.output/'producer_source';dst.mkdir()
    for i,path in enumerate(sources):
        shutil.copyfile(path,dst/f'{i:03d}_{path.name}')
    inputs=[MODEL,CODEC,FIXTURES,REFERENCE,TEACHER]
    if args.command=='run':inputs.append(args.pilot/'result.json')
    p={'hypothesis_id':HYPOTHESIS_ID,'schema':SCHEMA_VERSION,'stage':args.command,
        'config':asdict(Config(args.copies)),'source_hashes':{str(x):sha256(x) for x in sources},
        'input_hashes':{str(x):sha256(x) for x in inputs},'command':sys.argv,
        'created_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'physical_gpu':0,
        'model_sha256':MODEL_SHA,'codec_sha256':CODEC_SHA,'isolated_gpu_process':True}
    p['fingerprint']=digest(p);write_json(args.output/'producer.json',p)
    return p


@torch.inference_mode()
def execute(args):
    p=freeze(args)
    if args.command=='run' and read(args.pilot/'result.json')['status']!='passed':
        raise ValueError('nine-copy systems pilot must pass')
    gpt,llama=tokenizers()
    pool=ReplicaPool(args.copies)
    torch.cuda.reset_peak_memory_stats()
    data=read(FIXTURES)['examples']
    dev=[e for e in data if e['split']=='development']
    result={'schema':SCHEMA_VERSION,'hypothesis_id':HYPOTHESIS_ID,
        'copies':args.copies,'producer_fingerprint':p['fingerprint'],
        'unique_runtime_weight_buffer_bytes':pool.model_bytes,
        'graph_setup_seconds':sum(r.setup_seconds for r in pool.runtimes)}
    if args.command=='pilot':
        checks=[]
        for pattern in (dev[:args.copies],list(reversed(dev[:args.copies])),[dev[0]]*args.copies):
            expected=[pool.runtimes[i].generate(e['prefix_ids'],gpt) for i,e in enumerate(pattern)]
            actual,_=pool.serve(pattern,gpt)
            checks.extend({'id':e['id'],'copy':i,'parity':x['native_output_ids']==y['native_output_ids'] and x['text']==y['text']}
                for i,(e,x,y) in enumerate(zip(pattern,expected,actual)))
        write_json(args.output/'checks.json',{'rows':checks})
        if not all(r['parity'] for r in checks):raise RuntimeError('concurrent replica parity failed')
        result.update(status='passed',systems_label='systems_lifecycle_only_not_quality_evidence',
            parity_checks=len(checks),checks_sha256=sha256(args.output/'checks.json'))
    else:
        reference={(r['id'],r['repeat']):r for r in read(REFERENCE)['rows']}
        warm,seconds=pool.serve(dev[:args.copies],gpt)
        write_json(args.output/'excluded_warmup.json',{'ids':[e['id'] for e in dev[:args.copies]],'wall_seconds':seconds,'outputs':warm})
        examples=[e for e in data if e['split']=='heldout']
        rows=[];batches=[]
        for rep in range(3):
            for wave,part in enumerate(waves(examples,args.copies)):
                outputs,seconds=pool.serve(part,gpt)
                batches.append({'repeat':rep,'wave':wave,'ids':[e['id'] for e in part],'wall_seconds':seconds})
                for slot,(e,r) in enumerate(zip(part,outputs)):
                    old=reference[e['id'],rep]
                    r.update(id=e['id'],repeat=rep,wave=wave,copy=slot,
                        source_sha256=e['source_sha256'],common_output_tokens=len(llama.encode(r['text'],add_special_tokens=False)),
                        parity=r['native_output_ids']==old['native_output_ids'] and r['text']==old['text'] and e['source_sha256']==old['source_sha256'])
                    rows.append(r)
                if not all(r['parity'] for r in rows[-len(part):]):raise RuntimeError('parent output parity failed')
            write_json(args.output/'rows.json',{'rows':rows,'batches':batches})
            print(json.dumps({'copies':args.copies,'completed_repeat':rep+1,'timed_requests':len(rows)}),flush=True)
        result.update(status='completed',summary=summarize(rows,batches,args.copies),rows_sha256=sha256(args.output/'rows.json'))
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    for path,h in {**p['source_hashes'],**p['input_hashes']}.items():
        if sha256(path)!=h:raise ValueError('producer/reference changed')
    write_json(args.output/'result.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('pilot','run'))
    parser.add_argument('--copies',type=int,choices=ORDER,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,default=BASE/'20261021_pilot_v1')
    args=parser.parse_args();Config(args.copies)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('select physical GPU0 before Python starts')
    try:execute(args)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/'result.json').exists():
            write_json(args.output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':main()
