"""Bounded follow-up using free VRAM; same frozen runtime and task corpus."""
import argparse
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import requests
from experiments.word_games import three_gpu_experts as base
from experiments.word_games.common import PACKAGE, digest, read, sha, write
from experiments.word_games.prepare_qwen38_moe import WEIGHTS,SHA256,SIZE
from experiments.word_games.qwen_runtime import REQUESTS,REQUESTS_SHA,memory
from experiments.word_games.check_qwen_tokenizers import gguf_metadata
from experiments.word_games.qwen38_moe_runtime import validate_metadata

PLAN=Path('docs/07 - Status Reports/2026-10-31_word-games-packed-experts-amendment.md')
GROUPS={'CUDA0':list(range(18,23)), 'CUDA1':list(range(8))+[23],
        'CUDA2':list(range(8,18)), 'CPU':[24,25]}
ORIGINAL_COMMAND=base.command
ORIGINAL_VERIFY=base.verify


def packed_command(extra,port,name):
    if extra!=10:raise ValueError('packed layout requires ten regular-2060 banks')
    args=ORIGINAL_COMMAND(extra,port,name)
    specs=['^blk\\.('+'|'.join(map(str,layers))+')\\.ffn_.*_exps\\.weight$='+dev for dev,layers in GROUPS.items()]
    args[args.index('--override-tensor')+1]=','.join(specs)
    return args


def packed_verify(log,extra):
    layers={int(i):d for i,d in re.findall(r'layer\s+(\d+) assigned to device ([^,\s]+)',log)}
    if layers!={i:'CUDA0' for i in range(42)}:raise ValueError('attention moved')
    found={}
    for layer,part,dev in re.findall(r'tensor blk\.(\d+)\.ffn_(\w+)_exps\.weight \([^)]*\) buffer type overridden to (\S+)',log):
        i=int(layer);expect=next((d for d,ids in GROUPS.items() if i in ids),None)
        actual='CPU' if dev.startswith('CPU') or dev=='CUDA_Host' else dev
        if actual!=expect:raise ValueError('expert misplacement')
        found.setdefault(i,set()).add(part)
    if set(found)!=set(range(26)) or any(parts!={'gate','up','down'} for parts in found.values()):
        raise ValueError('incomplete expert overrides')
    return {'verified':True,'layer_devices':layers,'expert_overrides':GROUPS,'cpu_banks':2,'gpu_banks':38}


def cell(out,packed,phase,producer,metadata,port):
    base.command=packed_command if packed else ORIGINAL_COMMAND
    base.verify=packed_verify if packed else ORIGINAL_VERIFY
    try:return base.run_cell(out,10,phase,producer,metadata,port)
    finally:base.command=ORIGINAL_COMMAND;base.verify=ORIGINAL_VERIFY


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    reference=args.reference
    old=read(reference/'campaign_results.json')
    if not old['service_restored'] or len([r for r in old['results'] if r['phase']=='full' and r['completed']])!=2:
        raise ValueError('completed reference and service restoration required')
    print('Verifying checkpoint before packed follow-up',flush=True)
    if WEIGHTS.stat().st_size!=SIZE or sha(WEIGHTS)!=SHA256 or sha(REQUESTS)!=REQUESTS_SHA:raise ValueError('identity mismatch')
    metadata=gguf_metadata(WEIGHTS);validate_metadata(metadata)
    sources=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    sources += [Path(__file__),PLAN,base.PLAN,Path('pyproject.toml'),Path('pixi.lock')]
    bindings={str(p):sha(p) for p in sources}
    producer={'hypothesis_id':base.HYPOTHESIS_ID,'followup':'packed38','source_hashes':bindings,
              'weights_sha256':SHA256,'requests_sha256':REQUESTS_SHA,'groups':GROUPS,
              'reference':str(reference),'reference_sha256':sha(reference/'campaign_results.json'),
              'binary_hashes':{p.name:sha(p) for p in base.BINARY.iterdir() if p.is_file() and not p.is_symlink()}}
    producer['fingerprint']=digest(producer);write(out/'producer.json',producer)
    for i,p in enumerate(sources):
        dst=out/'producer_source'/f'{i}_{p.name}';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    url=base.TRITON;service=base.SERVICE
    before=requests.get(url+f'/v2/models/{service}/config',timeout=10);before.raise_for_status()
    write(out/'service_before.json',{'config':before.json(),'gpu_memory':memory()})
    results=[];attempted=False;restored=False
    try:
        attempted=True
        response=requests.post(url+f'/v2/repository/models/{service}/unload',json={},timeout=120)
        write(out/'unload.json',{'status':response.status_code,'body':response.text});response.raise_for_status()
        base.wait_for_release(out)
        b=cell(out/'pilot_reference18',False,'pilot',producer,metadata,args.port);results.append({'condition':'reference18',**b})
        if not b['completed']:return
        c=cell(out/'pilot_packed38',True,'pilot',producer,metadata,args.port);results.append({'condition':'packed38',**c})
        advance=c['completed'] and c['totals']['request_seconds']<=.95*b['totals']['request_seconds']
        write(out/'pilot_gate.json',{'advance':advance,'baseline':b['totals'],'candidate':c['totals']})
        if advance:
            full=cell(out/'full_packed38',True,'full',producer,metadata,args.port);results.append({'condition':'packed38',**full})
            end=cell(out/'pilot_ending_reference18',False,'pilot',producer,metadata,args.port);results.append({'condition':'ending_reference18',**end})
    finally:
        if attempted:
            response=requests.post(url+f'/v2/repository/models/{service}/load',json={},timeout=240)
            write(out/'reload.json',{'status':response.status_code,'body':response.text});response.raise_for_status()
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if requests.get(url+f'/v2/models/{service}/ready',timeout=5).status_code==200:
                    restored=True;break
                time.sleep(1)
            after=requests.get(url+f'/v2/models/{service}/config',timeout=10);after.raise_for_status()
            restored=restored and after.json()==before.json()
            write(out/'service_restored.json',{'ready_and_config_equal':restored,'gpu_memory':memory(),'config':after.json()})
            print('Qwen3-8B restored: '+str(restored),flush=True)
        write(out/'campaign_results.json',{'hypothesis_id':base.HYPOTHESIS_ID,'followup':'packed38',
              'fingerprint':producer['fingerprint'],'results':results,'service_restored':restored,'confirmed':False})
    if not restored:raise RuntimeError('service restoration failed')
    if any(sha(p)!=h for p,h in bindings.items()):raise ValueError('producer changed during campaign')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=18088);run(p.parse_args())
