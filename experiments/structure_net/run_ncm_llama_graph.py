"""Systems validation and isolated graph replay of the frozen distilled NCM."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from experiments.structure_net.ncm_llama import (
    BASE as PARENT,FIXTURES,VENDOR,digest,infer_ncm,load_ncm,sha256,storage_bytes,tokenizers,write_json)
from experiments.structure_net.ncm_llama_graph import GraphRuntime

BASE=Path('data/experiments/ncm_llama_graph')
HYPOTHESIS_ID='llama32-3b-ncm-graph-runtime-v1'
PLAN=Path('docs/07 - Status Reports/2026-10-21_ncm-graph-runtime-plan.md')
MODEL=PARENT/'20261021_generator_v1/model.pt'
CODEC=PARENT/'20261021_codec_v1/model.pt'
MODEL_SHA='ff7133b2de50bc1f66ef1e490fec4c723e0fc5a5c1957107209e923b198c3e3c'
CODEC_SHA='b87f7288edfc0eb1f5b9122d1139cc2e61aa2331be7d6fe53f0582993086ede2'


def read(p):return json.loads(Path(p).read_text())


def freeze(args):
    if args.output.exists():raise ValueError('fresh graph evidence root required')
    if read(PARENT/'20261021_evaluation_campaign_v1/execution.json')['status']!='completed_pending_source_review':
        raise ValueError('do not overlap the parent isolated timing blocks')
    if sha256(MODEL)!=MODEL_SHA or sha256(CODEC)!=CODEC_SHA:raise ValueError('frozen weights differ')
    args.output.mkdir(parents=True);copies=args.output/'producer_source';copies.mkdir()
    sources=[Path(__file__),Path('experiments/structure_net/ncm_llama_graph.py'),
        Path('experiments/structure_net/ncm_capture_generate.py'),
        Path('experiments/structure_net/ncm_capture_LICENSE'),
        Path('docs/07 - Status Reports/2026-10-21_ncm-graph-capture-amendment.md'),
        Path('docs/07 - Status Reports/2026-10-21_ncm-graph-sampler-amendment.md'),
        Path('experiments/structure_net/ncm_llama.py'),PLAN]+sorted(VENDOR.rglob('*.py'))
    for i,p in enumerate(sources):shutil.copyfile(p,copies/f'{i:03d}_{p.name}')
    inputs=[FIXTURES,MODEL,CODEC]
    for a in ('selected','teacher','fp16','released'):
        inputs += [PARENT/f'20261021_{a}_heldout_v1'/n for n in ('producer.json','result.json','rows.json')]
    if args.command=='run':inputs.append(args.pilot/'result.json')
    p={'hypothesis_id':HYPOTHESIS_ID,'schema':'nal.ncm-graph.v1','stage':args.command,
        'source_hashes':{str(x):sha256(x) for x in sources},'input_hashes':{str(x):sha256(x) for x in inputs},
        'config':{'repeats':3,'prefix':512,'canvas':48,'original_sampling_law':True,'training_updates':0,'implementation_version':3},
        'created_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,
        'gpu':torch.cuda.get_device_name(),'physical_gpu':0,'slots':1,'torch':torch.__version__}
    p['fingerprint']=digest(p);write_json(args.output/'producer.json',p)
    return p


def finish(args,p,result):
    for path,h in {**p['source_hashes'],**p['input_hashes']}.items():
        if sha256(path)!=h:raise ValueError('graph producer/reference changed')
    result.update(producer_fingerprint=p['fingerprint'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(args.output/'result.json',result);print(json.dumps(result),flush=True)


@torch.inference_mode()
def execute(args):
    p=freeze(args);gpt,llama=tokenizers();vae,net,_=load_ncm(CODEC,MODEL,adaptation=True)
    torch.cuda.reset_peak_memory_stats();runtime=GraphRuntime(vae,net)
    examples=read(FIXTURES)['examples']
    if args.command=='pilot':
        ex=[e for e in examples if e['split']=='development'][:8];checks=[]
        for e in ex+[ex[0],ex[1],ex[0]]:
            prefix=torch.tensor([e['prefix_ids']],device='cuda')
            shim,sampler=net.generate,net._sample_level_raw
            net.generate=runtime.original_generate;net._sample_level_raw=runtime.original_sample
            eager=infer_ncm(vae,net,prefix,gpt)[0]
            net.generate=shim;net._sample_level_raw=sampler
            graph=runtime.generate(e['prefix_ids'],gpt)
            checks.append({'id':e['id'],'native_ids_equal':eager['native_output_ids']==graph['native_output_ids'],
                'text_equal':eager['text']==graph['text']})
        write_json(args.output/'checks.json',{'rows':checks})
        if not all(r['native_ids_equal'] and r['text_equal'] for r in checks):raise RuntimeError('graph/eager pilot output parity failed')
        finish(args,p,{'status':'passed','systems_label':'systems_lifecycle_only_not_quality_evidence',
            'checks':len(checks),'checks_sha256':sha256(args.output/'checks.json'),
            'graph_setup_seconds':runtime.setup_seconds,'unique_runtime_weight_buffer_bytes':storage_bytes(vae,net)})
        return
    if read(args.pilot/'result.json')['status']!='passed':raise ValueError('graph lifecycle gate required')
    reference={(r['id'],r['repeat']):r for r in read(PARENT/'20261021_selected_heldout_v1/rows.json')['rows']}
    rows=[];warm=[]
    for e in [e for e in examples if e['split']=='heldout']:
        for rep in (-1,0,1,2):
            torch.cuda.synchronize();start=time.perf_counter();r=runtime.generate(e['prefix_ids'],gpt);torch.cuda.synchronize()
            r.update(id=e['id'],repeat=rep,condition='graph',source_sha256=e['source_sha256'],
                wall_seconds=time.perf_counter()-start,common_output_tokens=len(llama.encode(r['text'],add_special_tokens=False)))
            if rep>=0:
                old=reference[e['id'],rep]
                r['native_ids_equal_eager']=r['native_output_ids']==old['native_output_ids']
                r['text_equal_eager']=r['text']==old['text'];rows.append(r)
            else:warm.append(r)
        write_json(args.output/'rows.json',{'status':'running','rows':rows})
        write_json(args.output/'excluded_warmups.json',{'rows':warm})
        print(json.dumps({'graph_cases':len(warm),'total':100}),flush=True)
    write_json(args.output/'rows.json',{'status':'completed','rows':rows})
    finish(args,p,{'status':'completed_pending_audit','rows':len(rows),
        'parity_cells':sum(r['native_ids_equal_eager'] and r['text_equal_eager'] for r in rows),
        'rows_sha256':sha256(args.output/'rows.json'),'warmups_sha256':sha256(args.output/'excluded_warmups.json'),
        'graph_setup_seconds':runtime.setup_seconds,'unique_runtime_weight_buffer_bytes':storage_bytes(vae,net),
        'reused_control_timing_cells':900,'fresh_graph_timing_cells':len(rows)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('pilot','run'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',type=Path,default=BASE/'20261021_pilot_v3')
    args=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('select physicalGPU0 before Python starts')
    try:execute(args)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/'result.json').exists():
            write_json(args.output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':main()
