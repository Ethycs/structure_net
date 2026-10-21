"""Measure a shared-weight batched NCM with the existing inference computation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
from types import MethodType

import torch

from experiments.structure_net.ncm_llama import FIXTURES,VENDOR,digest,final_latent,load_ncm,sha256,storage_bytes,tokenizers,write_json
from experiments.structure_net.ncm_llama_graph import capture_sample,unpack_output,validate_prefix
from experiments.structure_net.ncm_capture_generate import generate as capture_generate
from experiments.structure_net.run_ncm_llama_graph import MODEL,CODEC,MODEL_SHA,CODEC_SHA
from experiments.structure_net.run_ncm_replicas import REFERENCE,summarize,waves

BASE=Path('data/experiments/ncm_batching')
PLAN=Path('docs/07 - Status Reports/2026-10-21_ncm-batching-plan.md')
HYPOTHESIS_ID='llama32-3b-ncm-batching-v1'


def static_inputs(examples,size):
    if not 0<len(examples)<=size or size not in (1,4,9):raise ValueError('supported nonempty static batch required')
    for e in examples:validate_prefix(e['prefix_ids'])
    return [e['prefix_ids'] for e in examples]+[examples[0]['prefix_ids']]*(size-len(examples))


class BatchRuntime:
    @torch.inference_mode()
    def __init__(self,vae,net,size):
        self.vae,self.net,self.size=vae,net,size
        net.generate=MethodType(capture_generate,net);net._sample_level_raw=MethodType(capture_sample,net)
        self.prefix=torch.full((size,512),50257,device='cuda',dtype=torch.long);self.prefix[:,0]=50256
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());start=time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(3):self.raw()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.output=self.raw()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-start

    def raw(self):
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            self.net._graph_sampling_finite=torch.ones((),device='cuda',dtype=torch.bool)
            codes=self.net.generate(None,self.prefix,temperature=1.,top_k=1,top_p=0.,cfg_scale=0.,use_kv_cache=False)
            logits=self.vae.decoder(final_latent(self.vae,codes))
            finite=torch.isfinite(logits).all() & self.net._graph_sampling_finite
            return torch.cat((logits.argmax(-1),finite.long().expand(self.size,1)),dim=1)

    @torch.inference_mode()
    def serve(self,examples,gpt):
        torch.cuda.synchronize();start=time.perf_counter()
        ids=static_inputs(examples,self.size)
        self.prefix.copy_(torch.tensor(ids,device='cuda',dtype=torch.long));self.graph.replay()
        # Decode every physical output, including static-tail padding work.
        outputs=[unpack_output(x,gpt) for x in self.output.tolist()]
        torch.cuda.synchronize()
        return outputs[:len(examples)],time.perf_counter()-start


def read(p):return json.loads(Path(p).read_text())


def freeze(args):
    if args.output.exists():raise ValueError('fresh batch evidence root required')
    if sha256(MODEL)!=MODEL_SHA or sha256(CODEC)!=CODEC_SHA:raise ValueError('frozen model changed')
    args.output.mkdir(parents=True);dst=args.output/'producer_source';dst.mkdir()
    sources=[Path(__file__),PLAN,Path('experiments/structure_net/ncm_llama.py'),
        Path('experiments/structure_net/ncm_llama_graph.py'),Path('experiments/structure_net/ncm_capture_generate.py'),
        Path('experiments/structure_net/run_ncm_llama_graph.py'),Path('experiments/structure_net/run_ncm_replicas.py'),
        Path('experiments/structure_net/prepare_llama_cotenancy.py')]+sorted(VENDOR.rglob('*.py'))
    for i,path in enumerate(sources):shutil.copyfile(path,dst/f'{i:03d}_{path.name}')
    inputs=[MODEL,CODEC,FIXTURES,REFERENCE]
    if args.command=='run':inputs.append(args.pilot/'result.json')
    p={'hypothesis_id':HYPOTHESIS_ID,'schema':'nal.ncm-batching.v1','command':sys.argv,'stage':args.command,
        'config':{'batch':args.batch,'repeats':3,'prefix':512,'canvas':48,'tail_padding_counted_as_outputs':False},
        'source_hashes':{str(x):sha256(x) for x in sources},'input_hashes':{str(x):sha256(x) for x in inputs},
        'created_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'physical_gpu':0}
    p['fingerprint']=digest(p);write_json(args.output/'producer.json',p);return p


@torch.inference_mode()
def execute(args):
    p=freeze(args);gpt,llama=tokenizers();vae,net,_=load_ncm(CODEC,MODEL,adaptation=True)
    dev=[e for e in read(FIXTURES)['examples'] if e['split']=='development']
    checks=[]
    if args.command=='pilot':
        from experiments.structure_net.ncm_llama import infer_ncm
        expected={e['id']:infer_ncm(vae,net,torch.tensor([e['prefix_ids']],device='cuda'),gpt)[0] for e in dev[:args.batch]}
    runtime=BatchRuntime(vae,net,args.batch);torch.cuda.reset_peak_memory_stats()
    if args.command=='pilot':
        for part in (dev[:args.batch],list(reversed(dev[:args.batch])),dev[:args.batch]):
            outputs,_=runtime.serve(part,gpt)
            checks.extend({'id':e['id'],'parity':o['native_output_ids']==expected[e['id']]['native_output_ids'] and o['text']==expected[e['id']]['text']} for e,o in zip(part,outputs))
        write_json(args.output/'checks.json',{'rows':checks})
        if not all(x['parity'] for x in checks):raise RuntimeError('batch/single output parity failed')
        result={'status':'passed','systems_label':'systems_lifecycle_only_not_quality_evidence','parity_checks':len(checks),
            'checks_sha256':sha256(args.output/'checks.json')}
    else:
        if read(args.pilot/'result.json')['status']!='passed':raise ValueError('batch9 systems pilot required')
        warm,seconds=runtime.serve(dev[:args.batch],gpt)
        write_json(args.output/'excluded_warmup.json',{'outputs':warm,'wall_seconds':seconds})
        reference={(x['id'],x['repeat']):x for x in read(REFERENCE)['rows']}
        examples=[e for e in read(FIXTURES)['examples'] if e['split']=='heldout'];rows=[];batches=[]
        for rep in range(3):
            for wave,part in enumerate(waves(examples,args.batch)):
                outputs,seconds=runtime.serve(part,gpt)
                batches.append({'repeat':rep,'wave':wave,'ids':[e['id'] for e in part],'wall_seconds':seconds,'physical_batch':args.batch})
                for e,o in zip(part,outputs):
                    old=reference[e['id'],rep]
                    o.update(id=e['id'],repeat=rep,wave=wave,source_sha256=e['source_sha256'],
                        common_output_tokens=len(llama.encode(o['text'],add_special_tokens=False)),
                        parity=o['native_output_ids']==old['native_output_ids'] and o['text']==old['text'])
                    rows.append(o)
                if not all(o['parity'] for o in rows[-len(part):]):raise RuntimeError('batch output differs from original')
            write_json(args.output/'rows.json',{'rows':rows,'batches':batches})
            print(json.dumps({'batch':args.batch,'completed_repeat':rep+1}),flush=True)
        result={'status':'completed','summary':summarize(rows,batches,args.batch),'rows_sha256':sha256(args.output/'rows.json'),
            'timed_dummy_outputs':sum(args.batch-len(x['ids']) for x in batches)}
    result.update(hypothesis_id=HYPOTHESIS_ID,producer_fingerprint=p['fingerprint'],batch=args.batch,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        model_bytes=storage_bytes(vae,net),graph_setup_seconds=runtime.setup_seconds)
    if result['peak_reserved_bytes']>6794772480:raise RuntimeError('original Llama reserved memory budget exceeded')
    for path,h in {**p['source_hashes'],**p['input_hashes']}.items():
        if sha256(path)!=h:raise ValueError('producer input changed')
    write_json(args.output/'result.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('pilot','run'))
    p.add_argument('--batch',type=int,choices=(1,4,9),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',type=Path,default=BASE/'20261021_pilot_v1');args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('select physical GPU0 before Python starts')
    try:execute(args)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/'result.json').exists():
            write_json(args.output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':main()
