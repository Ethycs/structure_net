"""Exploratory exact-weight MoE placement and CPU tuning, via Pixi at repo root."""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import requests

from .common import PACKAGE, digest, read, sha, write
from .native_speed import BINARY, GPU, IMAGE, PILOT, output_tokens
from .qwen38_moe_runtime import (MODEL, NativeQwen38, command as base_command,
                                host_memory, totals, validate_metadata)
from .check_qwen_tokenizers import gguf_metadata
from .prepare_qwen38_moe import WEIGHTS, SHA256, SIZE, REPO, REVISION
from .qwen_runtime import REQUESTS, REQUESTS_SHA, memory

HYPOTHESIS_ID = 'word-games-moe-placement-v1'
SCHEMA_VERSION = 'word-games.moe-placement.v1'
PLAN = Path('docs/07 - Status Reports/2026-10-31_word-games-moe-placement-plan.md')
EXPERT_PLAN = Path('docs/07 - Status Reports/2026-10-31_word-games-moe-experts-only-amendment.md')
SUPER = 'GPU-7b1db3cb-ec9c-beb3-e7db-826c2543edf9'
HEALTH = 'http://127.0.0.1:8000/v2/models/qwen3-8b-gguf/ready'


@dataclass(frozen=True)
class Config:
    name: str
    threads: int = 12
    pin: bool = False
    super_layers: int = 0
    cache_mib: int = 0
    experts_only: bool = False

    def __post_init__(self):
        if self.threads not in (6, 8, 12, 24): raise ValueError('unsupported CPU count')
        if self.super_layers not in (0, 6, 8): raise ValueError('unsupported layer count')
        if self.cache_mib not in (0, 2048): raise ValueError('unsupported cache size')
        if self.pin and self.threads > 12: raise ValueError('one thread per physical core')
        if self.experts_only and (not self.super_layers or self.cache_mib):
            raise ValueError('experts-only arm requires resident experts and no cache')


CONFIGS = {c.name:c for c in [Config('control'), Config('cpu6',6), Config('cpu8',8),
    Config('cpu24',24), Config('cpu12pin',12,True), Config('gpu8',super_layers=8),
    Config('gpu6',super_layers=6), Config('cache2048',cache_mib=2048),
    Config('gpu8pin',pin=True,super_layers=8),
    Config('gpu8pin_cache2048',pin=True,super_layers=8,cache_mib=2048),
    Config('experts8',super_layers=8,experts_only=True)]}


def command(config, port, name):
    args = base_command(port, name)
    def set_arg(flag, value): args[args.index(flag)+1] = str(value)
    set_arg('--threads', config.threads)
    args += ['--threads-batch',str(config.threads)]
    if config.pin: args += ['--cpu-range','0-11','--cpu-strict','1',
                           '--cpu-range-batch','0-11','--cpu-strict-batch','1']
    if config.super_layers:
        # Device list order defines split order. Fraction is deliberately between
        # (N-1)/42 and N/42 to avoid floating rounding assigning an extra block.
        set_arg('--gpus',f'"device={GPU},{SUPER}"')
        args[args.index(f'CUDA_VISIBLE_DEVICES={GPU}')] = f'CUDA_VISIBLE_DEVICES={GPU},{SUPER}'
        set_arg('--device','CUDA0,CUDA1' if config.experts_only else 'CUDA1,CUDA0')
        set_arg('--split-mode','layer')
        set_arg('--n-cpu-moe',0)
        split = config.super_layers - .5
        args += ['--tensor-split','1,0' if config.experts_only else f'{split},{42-split}']
        pattern = '^blk\\.('+'|'.join(str(i) for i in range(config.super_layers,26))+')\\.ffn_.*_exps\\.weight$=CPU'
        if config.experts_only:
            pattern += ',^blk\\.('+'|'.join(str(i) for i in range(config.super_layers))+')\\.ffn_.*_exps\\.weight$=CUDA1'
        args += ['--override-tensor',pattern]
    if config.cache_mib: args += ['--moe-cache-mib',str(config.cache_mib)]
    return args


def verify_placement(log, config):
    matches=re.findall(r'tensor blk\.(\d+)\.ffn_(\w+)_exps\.weight \([^)]*\) buffer type overridden to (\S+)',log)
    cpu={};gpu={}
    for layer,part,buffer in matches:
        if buffer.startswith('CPU') or buffer=='CUDA_Host':target=cpu
        elif config.experts_only and buffer=='CUDA1':target=gpu
        else:raise ValueError('unexpected expert buffer')
        target.setdefault(int(layer),set()).add(part)
    if set(cpu)!=set(range(config.super_layers,26)): raise ValueError('wrong CPU expert layers')
    if any(parts not in ({'gate','up','down'},{'gate_up','down'}) for parts in cpu.values()):
        raise ValueError('incomplete expert placement')
    if set(gpu)!=set(range(config.super_layers if config.experts_only else 0)):
        raise ValueError('wrong GPU expert overrides')
    if any(parts not in ({'gate','up','down'},{'gate_up','down'}) for parts in gpu.values()):
        raise ValueError('incomplete GPU expert placement')
    layers={int(i):dev for i,dev in re.findall(r'layer\s+(\d+) assigned to device ([^,\s]+)',log)}
    if set(layers)!=set(range(42)): raise ValueError('missing layer placement')
    if any(layers[i]!=('CUDA1' if i<config.super_layers and not config.experts_only else 'CUDA0') for i in range(42)):
        raise ValueError('wrong contiguous device split')
    if config.experts_only and not any(float(n)>0 for n in re.findall(r'CUDA1 model buffer size\s*=\s*([\d.]+) MiB',log)):
        raise ValueError('missing resident GPU expert allocation')
    cache_lines=[s for s in log.splitlines() if 'moe_cache' in s or 'moe-cache' in s
                 or 'MoE cache' in s or re.search(r'\d+ slots \(',s)]
    cache_sizes=[float(n) for n in re.findall(r'MoE cache size\s*=\s*([\d.]+) MiB',log)]
    if config.cache_mib and (not cache_sizes or sum(cache_sizes)<=0):
        raise ValueError('cache allocation not verified')
    return {'cpu_layers':sorted(cpu),'gpu_expert_overrides':sorted(gpu),'layer_devices':layers,'cache_lines':cache_lines,'verified':True}


def quality_compare(reference, candidate):
    if set(reference)!=set(candidate) or len(reference)!=36: raise ValueError('paired 36 requests required')
    changed=[k for k in reference if reference[k]['parsed']!=candidate[k]['parsed']]
    valid=all(isinstance(r['parsed'],dict) and r['response'].get('stop_type')!='limit'
              and not r['response'].get('truncated',False) for r in candidate.values())
    return {'parsed_equal':36-len(changed),'changed':changed,'valid':valid,
            'text_equal':sum(reference[k]['response']['content']==candidate[k]['response']['content'] for k in reference),
            'quality_parity':valid and not changed}


class Monitor:
    def __init__(self,out,config,container):
        self.out,self.config,self.container=out,config,container
        self.stop=threading.Event();self.error=None;self.samples=[]
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        bad_health=0
        with (self.out/'telemetry.jsonl').open('x') as handle:
            while not self.stop.is_set():
                try:
                    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits'],text=True)
                    values=[s.split(', ') for s in rows.strip().splitlines()]
                    sample={'monotonic':time.monotonic(),'gpus':values,'loadavg':Path('/proc/loadavg').read_text().strip()}
                    self.samples.append(sample);handle.write(json.dumps(sample)+'\n');handle.flush()
                    ids={GPU,SUPER} if self.config.super_layers else {GPU}
                    if any(int(r[2])<384 for r in values if r[0] in ids): self.error='GPU headroom below 384 MiB'
                    if len(self.samples)%10==1:
                        try: ok=requests.get(HEALTH,timeout=3).status_code==200
                        except requests.RequestException: ok=False
                        bad_health=0 if ok else bad_health+1
                        if bad_health>=2:self.error='resident Qwen3-8B health check failed twice'
                    if self.error:
                        subprocess.run(['docker','stop','--time','5',self.container],capture_output=True)
                        return
                except Exception as exc:
                    self.error='telemetry failure '+repr(exc);return
                self.stop.wait(1)
    def start(self):self.thread.start()
    def close(self):self.stop.set();self.thread.join(timeout=10)


def run_cell(out,config,phase,repeats,metadata,producer,port):
    out.mkdir(parents=True,exist_ok=False)
    name=f'word-games-moe-placement-{os.getpid()}'
    cmd=command(config,port,name)
    cell={'schema':SCHEMA_VERSION,'hypothesis_id':HYPOTHESIS_ID,'config':asdict(config),
          'phase':phase,'repeats':repeats,'command':cmd,'parent_fingerprint':producer['fingerprint'],
          'gpu_before':memory()}
    write(out/'producer.json',{**cell,'fingerprint':digest(cell)})
    client=NativeQwen38(port);process=None;monitor=None
    start=time.perf_counter();passes=[]
    try:
        if requests.get(HEALTH,timeout=5).status_code!=200: raise RuntimeError('resident service not ready before test')
        with (out/'server.log').open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+360
            while True:
                if process.poll() is not None: raise RuntimeError('server exited; inspect server.log')
                try:
                    if client.http.get(client.url+'/health',timeout=1).status_code==200:break
                except requests.RequestException:pass
                if time.monotonic()>deadline:raise TimeoutError('startup timeout')
                time.sleep(.5)
            props=client.http.get(client.url+'/props',timeout=10).json()
            if props['model_alias']!=MODEL or props['chat_template'].rstrip('\n')!=metadata['tokenizer.chat_template'].rstrip('\n'):
                raise ValueError('model/template identity mismatch')
            write(out/'placement.json',verify_placement((out/'server.log').read_text(),config))
            write(out/'runtime.json',{'properties':props,'load_seconds':time.perf_counter()-start,
                                     'gpu_loaded':memory(),'host_memory':host_memory(name)})
            monitor=Monitor(out,config,name);monitor.start()
            client.complete('Return exactly {"ready":true} as JSON.',out/'warmup',max_tokens=24)
            if client.rows[-1]['parsed']!={'ready':True}:raise ValueError('warm-up JSON failed')
            inputs=read(REQUESTS)
            selected=[next(r for r in inputs if r['id']==i) for i in PILOT] if phase=='pilot' else inputs
            for rep in range(repeats):
                client.rows.clear();repdir=out/f'pass_{rep:02d}'
                for request in selected:
                    if monitor.error:raise RuntimeError(monitor.error)
                    client.complete(request['prompt'],repdir/'runs'/request['id'],max_tokens=request['max_tokens'])
                    row=client.rows[-1]
                    if not isinstance(row['parsed'],dict) or row['response'].get('stop_type')=='limit':
                        raise ValueError('invalid or truncated JSON')
                    print(json.dumps({'config':config.name,'rep':rep,'id':request['id'],
                          'seconds':row['elapsed_seconds'],'tokens':output_tokens(row['response'])}),flush=True)
                t=totals(client.rows)
                t.update(prefill_seconds=sum(r['response']['timings']['prompt_ms'] for r in client.rows)/1000,
                         decode_seconds=sum(r['response']['timings']['predicted_ms'] for r in client.rows)/1000)
                result={'schema':SCHEMA_VERSION,'hypothesis_id':HYPOTHESIS_ID,'config':asdict(config),
                        'fingerprint':digest(cell),'totals':t,'confirmed':False,
                        'classification':'systems_lifecycle_only_not_quality_evidence' if phase=='pilot' else 'completed_exploratory',
                        'response_hashes':{str(p):sha(p) for p in repdir.rglob('response.json')}}
                write(repdir/'result.json',result);passes.append(result)
                print(json.dumps({'config':config.name,'rep':rep,'totals':t}),flush=True)
            if monitor.error:raise RuntimeError(monitor.error)
            write(out/'result.json',{'schema':SCHEMA_VERSION,'hypothesis_id':HYPOTHESIS_ID,'completed':True,
                  'config':asdict(config),'passes':passes,'host_memory_after':host_memory(name),'confirmed':False})
    except Exception as exc:
        write(out/'interruption.json',{'error':repr(exc),'passes_completed':len(passes),'calls_in_current_pass':len(client.rows)})
        print(json.dumps({'config':config.name,'error':repr(exc)}),flush=True)
    finally:
        if monitor:monitor.close()
        if process:
            subprocess.run(['docker','stop','--time','5',name],capture_output=True);process.wait(timeout=30)
        write(out/'gpu_after.json',{'memory':memory()})


def run(args):
    configs=list(args.configs)
    if args.ending_control:
        if args.phase!='full' or 'control' in configs:raise ValueError('ending control requires full intervention-only conditions')
        configs.append('control')
    repeats={c:(1 if args.ending_control and c=='control' else args.repeats) for c in configs}
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    print('Verifying pinned checkpoint once for campaign',flush=True)
    if WEIGHTS.stat().st_size!=SIZE or sha(WEIGHTS)!=SHA256:raise ValueError('checkpoint mismatch')
    if sha(REQUESTS)!=REQUESTS_SHA:raise ValueError('request mismatch')
    metadata=gguf_metadata(WEIGHTS);validate_metadata(metadata)
    sources=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    sources += [PLAN,EXPERT_PLAN,Path('pyproject.toml'),Path('pixi.lock')]
    bindings={str(p):sha(p) for p in sources}
    producer={'schema':SCHEMA_VERSION,'hypothesis_id':HYPOTHESIS_ID,'configs':configs,'phase':args.phase,'repeats':args.repeats,
              'repeats_by_condition':repeats,'ending_control':args.ending_control,
              'source_hashes':bindings,'weights_sha256':SHA256,'weights_path':str(WEIGHTS),'repository':REPO,'revision':REVISION,
              'requests_sha256':REQUESTS_SHA,'binary_hashes':{p.name:sha(p) for p in BINARY.glob('*') if p.is_file() and not p.is_symlink()},
              'temperature':0,'seed':73,'thinking':False,'training':False,'speculation':False,'weight_quant':'Q4_K_M',
              'flash_attention':True,'kv':'q8_0','git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'cpu':subprocess.check_output(['lscpu'],text=True)}
    producer['fingerprint']=digest(producer);write(out/'producer.json',producer)
    for i,p in enumerate(sources):
        dst=out/'producer_source'/f'{i}_{p.name}';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    for config in configs:run_cell(out/config,CONFIGS[config],args.phase,repeats[config],metadata,producer,args.port)
    if any(sha(p)!=h for p,h in bindings.items()):raise ValueError('producer changed during campaign')
    completed=[c for c in configs if (out/c/'result.json').exists()]
    write(out/'campaign_results.json',{'schema':SCHEMA_VERSION,'hypothesis_id':HYPOTHESIS_ID,'fingerprint':producer['fingerprint'],
          'requested':configs,'completed':completed,'failed':[c for c in configs if c not in completed],
          'repeats_by_condition':repeats,
          'phase':args.phase,'repeats':args.repeats,'confirmed':False,
          'artifacts':{str(p):sha(p) for p in out.rglob('result.json')}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','full'],default='pilot')
    p.add_argument('--configs',nargs='+',choices=list(CONFIGS),default=list(CONFIGS))
    p.add_argument('--repeats',type=int,choices=[1,2],default=1)
    p.add_argument('--ending-control',action='store_true',help='append one full control pass after the intervention cells')
    p.add_argument('--port',type=int,default=18088)
    run(p.parse_args())
