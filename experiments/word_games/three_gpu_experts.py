"""Temporarily reclaim Qwen3-8B's GPU, measure expert banks, restore service."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import requests
from .common import PACKAGE, digest, read, sha, write
from .moe_placement import CONFIGS, SUPER, command as placement_command
from .native_speed import BINARY, GPU, IMAGE, PILOT, output_tokens
from .qwen38_moe_runtime import NativeQwen38, MODEL, host_memory, totals, validate_metadata
from .check_qwen_tokenizers import gguf_metadata
from .prepare_qwen38_moe import WEIGHTS, SHA256, SIZE
from .qwen_runtime import REQUESTS, REQUESTS_SHA, memory

SMALL='GPU-8cf440df-e8f4-2210-a2cd-29a9d00aca14'
TRITON='http://127.0.0.1:8000'
SERVICE='qwen3-8b-gguf'
HYPOTHESIS_ID='word-games-three-gpu-experts-v1'
PLAN=Path('docs/07 - Status Reports/word-games/2026-10-31_word-games-three-gpu-experts-plan.md')


def command(extra,port,name):
    if extra not in (0,8,10):raise ValueError('unsupported extra expert banks')
    args=placement_command(CONFIGS['experts8'],port,name)
    args[args.index('--gpus')+1]=f'"device={GPU},{SUPER},{SMALL}"'
    args[args.index(f'CUDA_VISIBLE_DEVICES={GPU},{SUPER}')]=f'CUDA_VISIBLE_DEVICES={GPU},{SUPER},{SMALL}'
    args[args.index('--device')+1]='CUDA0,CUDA1,CUDA2'
    args[args.index('--tensor-split')+1]='1,0,0'
    specs=[]
    for layers,dev in [(range(8),'CUDA1'),(range(8,8+extra),'CUDA2'),(range(8+extra,26),'CPU')]:
        if not layers:continue
        specs.append('^blk\\.('+'|'.join(map(str,layers))+')\\.ffn_.*_exps\\.weight$='+dev)
    args[args.index('--override-tensor')+1]=','.join(specs)
    return args


def verify(log,extra):
    layers={int(i):d for i,d in re.findall(r'layer\s+(\d+) assigned to device ([^,\s]+)',log)}
    if layers!={i:'CUDA0' for i in range(42)}:raise ValueError('attention/layer assignment changed')
    found={}
    for layer,part,dev in re.findall(r'tensor blk\.(\d+)\.ffn_(\w+)_exps\.weight \([^)]*\) buffer type overridden to (\S+)',log):
        i=int(layer);expected='CUDA1' if i<8 else ('CUDA2' if i<8+extra else 'CPU')
        if expected=='CPU':
            if not (dev.startswith('CPU') or dev=='CUDA_Host'):raise ValueError('CPU expert misplaced')
        elif dev!=expected:raise ValueError('GPU expert misplaced')
        found.setdefault(i,set()).add(part)
    if set(found)!=set(range(26)) or any(parts!={'gate','up','down'} for parts in found.values()):
        raise ValueError('incomplete expert placement')
    return {'verified':True,'layer_devices':layers,'super_layers':list(range(8)),
            'small_layers':list(range(8,8+extra)),'cpu_layers':list(range(8+extra,26))}


def wait_for_release(out):
    """Triton unload returns before the child CUDA context disappears."""
    samples=[];stable=0;deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free',
                                     '--format=csv,noheader,nounits'],text=True)
        free={parts[0]:int(parts[1]) for line in raw.strip().splitlines() for parts in [line.split(', ')]}
        samples.append({'monotonic':time.monotonic(),'free_mib':free})
        stable=stable+1 if free.get(SMALL,0)>=5200 else 0
        if stable>=2:
            write(out/'release_check.json',{'released':True,'samples':samples});return
        time.sleep(1)
    write(out/'release_check.json',{'released':False,'samples':samples})
    raise TimeoutError('Qwen3-8B CUDA allocation did not release')


class Monitor:
    def __init__(self,out,name):
        self.out=out;self.name=name;self.stop=threading.Event();self.error=None
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        bad=0;count=0
        with (self.out/'telemetry.jsonl').open('x') as f:
            while not self.stop.is_set():
                try:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits'],text=True)
                    values=[r.split(', ') for r in raw.strip().splitlines()]
                    f.write(json.dumps({'monotonic':time.monotonic(),'gpus':values})+'\n');f.flush()
                    if any(int(r[2])<384 for r in values if r[0] in (GPU,SUPER,SMALL)):self.error='GPU headroom below 384 MiB'
                    if count%10==0:
                        try:ok=requests.get(TRITON+'/v2/health/ready',timeout=3).status_code==200
                        except requests.RequestException:ok=False
                        bad=0 if ok else bad+1
                        if bad>=2:self.error='Triton server readiness failed twice'
                    count+=1
                    if self.error:
                        subprocess.run(['docker','stop','--time','5',self.name],capture_output=True);return
                except Exception as exc:self.error=repr(exc);return
                self.stop.wait(1)
    def start(self):self.thread.start()
    def close(self):self.stop.set();self.thread.join(timeout=10)


def run_cell(out,extra,phase,producer,metadata,port):
    out.mkdir(parents=True,exist_ok=False)
    name=f'word-games-three-gpu-{os.getpid()}';cmd=command(extra,port,name)
    write(out/'producer.json',{'extra_banks':extra,'phase':phase,'command':cmd,'parent_fingerprint':producer['fingerprint']})
    client=NativeQwen38(port);process=None;monitor=None;error=None;total=None
    try:
        with (out/'server.log').open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            monitor=Monitor(out,name);monitor.start();deadline=time.monotonic()+360
            while True:
                if process.poll() is not None:raise RuntimeError('runtime exited; inspect server.log')
                if monitor.error:raise RuntimeError(monitor.error)
                try:
                    if client.http.get(client.url+'/health',timeout=1).status_code==200:break
                except requests.RequestException:pass
                if time.monotonic()>deadline:raise TimeoutError('startup')
                time.sleep(.5)
            props=client.http.get(client.url+'/props',timeout=10).json()
            if props['model_alias']!=MODEL or props['chat_template'].rstrip('\n')!=metadata['tokenizer.chat_template'].rstrip('\n'):
                raise ValueError('model/template mismatch')
            write(out/'placement.json',verify((out/'server.log').read_text(),extra))
            write(out/'runtime.json',{'props':props,'memory':memory(),'host_memory':host_memory(name)})
            client.complete('Return exactly {"ready":true} as JSON.',out/'warmup',max_tokens=24)
            if client.rows[-1]['parsed']!={'ready':True}:raise ValueError('warm-up failed')
            client.rows.clear();inputs=read(REQUESTS)
            selected=[next(r for r in inputs if r['id']==i) for i in PILOT] if phase=='pilot' else inputs
            for request in selected:
                if monitor.error:raise RuntimeError(monitor.error)
                client.complete(request['prompt'],out/'runs'/request['id'],max_tokens=request['max_tokens'])
                row=client.rows[-1]
                if not isinstance(row['parsed'],dict) or row['response'].get('stop_type')=='limit' or row['response'].get('truncated',False):
                    raise ValueError('invalid or truncated JSON')
                print(json.dumps({'extra_banks':extra,'phase':phase,'id':request['id'],'seconds':row['elapsed_seconds'],
                                  'tokens':output_tokens(row['response'])}),flush=True)
            total=totals(client.rows)
            total.update(prefill_seconds=sum(r['response']['timings']['prompt_ms'] for r in client.rows)/1000,
                         decode_seconds=sum(r['response']['timings']['predicted_ms'] for r in client.rows)/1000)
            print(json.dumps({'extra_banks':extra,'phase':phase,'totals':total}),flush=True)
    except Exception as exc:error=repr(exc);print(json.dumps({'extra_banks':extra,'error':error}),flush=True)
    finally:
        if process:
            subprocess.run(['docker','stop','--time','10',name],capture_output=True);process.wait(timeout=30)
        if monitor:monitor.close()
    result={'hypothesis_id':HYPOTHESIS_ID,'extra_banks':extra,'phase':phase,'completed':error is None,'error':error,
            'totals':total,'response_hashes':{str(p):sha(p) for p in (out/'runs').rglob('response.json')},
            'confirmed':False,'classification':'systems_lifecycle_only_not_quality_evidence' if phase=='pilot' else 'completed_exploratory'}
    write(out/'result.json',result);return result


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    print('Verifying checkpoint before the service pause',flush=True)
    if WEIGHTS.stat().st_size!=SIZE or sha(WEIGHTS)!=SHA256 or sha(REQUESTS)!=REQUESTS_SHA:raise ValueError('identity mismatch')
    metadata=gguf_metadata(WEIGHTS);validate_metadata(metadata)
    sources=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    sources += [PLAN,Path('docs/07 - Status Reports/word-games/2026-10-31_word-games-three-gpu-release-amendment.md'),
                Path('pyproject.toml'),Path('pixi.lock')]
    bindings={str(p):sha(p) for p in sources}
    producer={'hypothesis_id':HYPOTHESIS_ID,'source_hashes':bindings,'weights_sha256':SHA256,'requests_sha256':REQUESTS_SHA,
              'binary_hashes':{p.name:sha(p) for p in BINARY.iterdir() if p.is_file() and not p.is_symlink()},
              'pilot_full_gate':.95,'full_passes_per_arm':1,'user_authorized_temporary_unload':SERVICE}
    producer['fingerprint']=digest(producer);write(out/'producer.json',producer)
    for i,p in enumerate(sources):
        dst=out/'producer_source'/f'{i}_{p.name}';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    before=requests.get(TRITON+f'/v2/models/{SERVICE}/config',timeout=10);before.raise_for_status()
    write(out/'service_before.json',{'config':before.json(),'gpu_memory':memory(),
          'repository':requests.post(TRITON+'/v2/repository/index',timeout=10).json()})
    results=[];unload_attempted=False;restored=False
    try:
        unload_attempted=True
        response=requests.post(TRITON+f'/v2/repository/models/{SERVICE}/unload',json={},timeout=120)
        write(out/'unload.json',{'status':response.status_code,'body':response.text});response.raise_for_status()
        print('Qwen3-8B unloaded temporarily',flush=True)
        wait_for_release(out)
        write(out/'memory_unloaded.json',memory())
        baseline=run_cell(out/'pilot_experts8',0,'pilot',producer,metadata,args.port);results.append(baseline)
        if not baseline['completed']:return
        candidate=run_cell(out/'pilot_experts18',10,'pilot',producer,metadata,args.port);results.append(candidate)
        if not candidate['completed'] and ('headroom' in str(candidate['error']) or 'runtime exited' in str(candidate['error'])):
            candidate=run_cell(out/'pilot_experts16',8,'pilot',producer,metadata,args.port);results.append(candidate)
        advance=candidate['completed'] and candidate['totals']['request_seconds']<=.95*baseline['totals']['request_seconds']
        write(out/'pilot_gate.json',{'advance':advance,'baseline':baseline['totals'],'candidate':candidate['totals']})
        if advance:
            full=run_cell(out/'full_experts8',0,'full',producer,metadata,args.port);results.append(full)
            if full['completed']:
                results.append(run_cell(out/f'full_experts{8+candidate["extra_banks"]}',candidate['extra_banks'],'full',producer,metadata,args.port))
    finally:
        if unload_attempted:
            response=requests.post(TRITON+f'/v2/repository/models/{SERVICE}/load',json={},timeout=240)
            write(out/'reload.json',{'status':response.status_code,'body':response.text});response.raise_for_status()
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if requests.get(TRITON+f'/v2/models/{SERVICE}/ready',timeout=5).status_code==200:
                    restored=True;break
                time.sleep(1)
            after=requests.get(TRITON+f'/v2/models/{SERVICE}/config',timeout=10);after.raise_for_status()
            restored=restored and after.json()==before.json()
            write(out/'service_restored.json',{'ready_and_config_equal':restored,'gpu_memory':memory(),'config':after.json()})
            print('Qwen3-8B restored: '+str(restored),flush=True)
        write(out/'campaign_results.json',{'hypothesis_id':HYPOTHESIS_ID,'fingerprint':producer['fingerprint'],
              'results':results,'service_restored':restored,'confirmed':False})
    if not restored:raise RuntimeError('Qwen3-8B restoration failed')
    if any(sha(p)!=h for p,h in bindings.items()):raise ValueError('producer changed during campaign')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18088)
    run(p.parse_args())
