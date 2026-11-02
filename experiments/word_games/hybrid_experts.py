"""Actual within-layer CPU/SUPER partition, matched native controls and telemetry."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import time
import requests

from .common import PACKAGE, digest, read, sha, write
from .moe_placement import CONFIGS, Monitor, SUPER, HEALTH, command as placement_command, verify_placement
from .native_speed import BINARY, GPU, IMAGE, PILOT, output_tokens
from .qwen38_moe_runtime import NativeQwen38, MODEL, totals, host_memory, validate_metadata
from .check_qwen_tokenizers import gguf_metadata
from .prepare_qwen38_moe import WEIGHTS, SHA256, SIZE
from .qwen_runtime import REQUESTS, REQUESTS_SHA, memory
from .prepare_hybrid_runtime import ROOT, SOURCE

HYPOTHESIS_ID='word-games-hybrid-experts-v1'
SCHEMA='word-games.hybrid-experts.v1'
PLAN=Path('docs/07 - Status Reports/word-games/2026-10-31_word-games-hybrid-experts-plan.md')
RUNTIME=ROOT/'runtime'
ARMS=('control','experts8','pins0','pins64','pins80')


def command(arm, port, name, out):
    if arm not in ARMS: raise ValueError('unknown arm')
    args=placement_command(CONFIGS['experts8' if arm=='experts8' else 'control'],port,name)
    args[args.index(f'{BINARY.resolve()}:/llama:ro')]=f'{RUNTIME.resolve()}:/llama:ro'
    if arm.startswith('pins'):
        args[args.index('--gpus')+1]=f'"device={GPU},{SUPER}"'
        args[args.index(f'CUDA_VISIBLE_DEVICES={GPU}')]=f'CUDA_VISIBLE_DEVICES={GPU},{SUPER}'
        args[args.index('--device')+1]='CUDA0,CUDA1'
        args[args.index('--split-mode')+1]='layer'
        args+=['--tensor-split','1,0']
        i=args.index(IMAGE)
        args[i:i]=['-v',f'{out.resolve()}:/evidence:rw','-e',f'WORD_GAMES_HYBRID_PINS={int(arm[4:])}',
                   '-e','WORD_GAMES_HYBRID_STATS=/evidence/hybrid_stats.json']
    return args


def validate_stats(stats, arm):
    if not arm.startswith('pins'): return
    if stats['pins']!=int(arm[4:]) or stats['layers']!=26 or stats['validated_layers']!=26:
        raise ValueError('hybrid layer/pin/validation mismatch')
    if stats['calls']<=0 or stats['gpu_hits']+stats['cpu_misses']!=8*stats['calls']:
        raise ValueError('routed expert accounting failed')
    if stats['cpu_misses']<=0 or (stats['pins']>0 and stats['gpu_hits']<=0):
        raise ValueError('requested partition did not execute')
    if not 0<=stats['validation_max_relative_l2']<(0.03 if stats['pins'] else 1e-6):
        raise ValueError('expert numerical check failed')


def run_cell(out, arm, phase, repeats, producer, metadata, port):
    out.mkdir(parents=True,exist_ok=False)
    name=f'word-games-hybrid-{os.getpid()}'
    cmd=command(arm,port,name,out)
    write(out/'producer.json',{'arm':arm,'command':cmd,'parent_fingerprint':producer['fingerprint']})
    process=None; monitor=None; client=NativeQwen38(port); passes=[]; error=None
    try:
        if requests.get(HEALTH,timeout=5).status_code!=200: raise RuntimeError('protected service unavailable')
        with (out/'server.log').open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            monitor=Monitor(out,CONFIGS['gpu8'] if arm!='control' else CONFIGS['control'],name)
            monitor.start()
            deadline=time.monotonic()+360
            while True:
                if process.poll() is not None: raise RuntimeError('runtime exited; see server.log')
                if monitor.error: raise RuntimeError(monitor.error)
                try:
                    if client.http.get(client.url+'/health',timeout=1).status_code==200:break
                except requests.RequestException: pass
                if time.monotonic()>deadline: raise TimeoutError('runtime startup')
                time.sleep(.5)
            props=client.http.get(client.url+'/props',timeout=10).json()
            if props['model_alias']!=MODEL or props['chat_template'].rstrip('\n')!=metadata['tokenizer.chat_template'].rstrip('\n'):
                raise ValueError('model/template identity mismatch')
            text=(out/'server.log').read_text()
            verify_placement(text,CONFIGS['experts8' if arm=='experts8' else 'control'])
            if arm.startswith('pins') and 'byte_verification=passed' not in text: raise ValueError('GPU copies unverified')
            write(out/'runtime.json',{'props':props,'memory':memory(),'host_memory':host_memory(name)})
            client.complete('Return exactly {"ready":true} as JSON.',out/'warmup',max_tokens=24)
            if client.rows[-1]['parsed']!={'ready':True}: raise ValueError('warm-up failed')
            if arm.startswith('pins'):
                checks=re.findall(r'WORD_GAMES_HYBRID_VALIDATE: layer=(\d+)',(out/'server.log').read_text())
                if set(map(int,checks))!=set(range(26)):raise ValueError('all layers must validate before timing')
            inputs=read(REQUESTS)
            selected=[next(r for r in inputs if r['id']==i) for i in PILOT] if phase=='pilot' else inputs
            for rep in range(repeats):
                client.rows.clear(); repdir=out/f'pass_{rep:02d}'
                for request in selected:
                    if monitor.error:raise RuntimeError(monitor.error)
                    client.complete(request['prompt'],repdir/'runs'/request['id'],max_tokens=request['max_tokens'])
                    row=client.rows[-1]
                    if not isinstance(row['parsed'],dict) or row['response'].get('stop_type')=='limit' or row['response'].get('truncated',False):
                        raise ValueError('invalid or truncated JSON')
                    print(json.dumps({'arm':arm,'rep':rep,'id':request['id'],'seconds':row['elapsed_seconds'],
                                      'tokens':output_tokens(row['response'])}),flush=True)
                t=totals(client.rows)
                t.update(prefill_seconds=sum(r['response']['timings']['prompt_ms'] for r in client.rows)/1000,
                         decode_seconds=sum(r['response']['timings']['predicted_ms'] for r in client.rows)/1000)
                result={'schema':SCHEMA,'hypothesis_id':HYPOTHESIS_ID,'arm':arm,'totals':t,'confirmed':False,
                        'classification':'systems_lifecycle_only_not_quality_evidence' if phase=='pilot' else 'completed_exploratory',
                        'response_hashes':{str(p):sha(p) for p in repdir.rglob('response.json')}}
                write(repdir/'result.json',result);passes.append(result)
                print(json.dumps({'arm':arm,'rep':rep,'totals':t}),flush=True)
    except Exception as exc:
        error=repr(exc); print(json.dumps({'arm':arm,'error':error}),flush=True)
    finally:
        if process:
            subprocess.run(['docker','stop','--time','10',name],capture_output=True)
            process.wait(timeout=30)
        if monitor:monitor.close()
    stats=read(out/'hybrid_stats.json') if (out/'hybrid_stats.json').exists() else None
    if not error:
        try: validate_stats(stats,arm)
        except Exception as exc:error=repr(exc)
    result={'schema':SCHEMA,'hypothesis_id':HYPOTHESIS_ID,'arm':arm,'completed':error is None,
            'error':error,'passes':passes,'stats':stats,'gpu_after':memory(),'confirmed':False}
    write(out/'result.json',result)
    return result


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    print('Hashing pinned weights and runtime',flush=True)
    if WEIGHTS.stat().st_size!=SIZE or sha(WEIGHTS)!=SHA256 or sha(REQUESTS)!=REQUESTS_SHA:
        raise ValueError('model or request identity mismatch')
    metadata=gguf_metadata(WEIGHTS); validate_metadata(metadata)
    sources=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    sources += [PLAN,Path('pyproject.toml'),Path('pixi.lock')]
    sources += [SOURCE/'src'/s for s in ['llama-context.cpp','llama-graph.cpp','CMakeLists.txt']]
    bindings={str(p):sha(p) for p in sources}
    producer={'schema':SCHEMA,'hypothesis_id':HYPOTHESIS_ID,'arms':args.arms,'phase':args.phase,'repeats':args.repeats,
              'weights_sha256':SHA256,'requests_sha256':REQUESTS_SHA,'source_hashes':bindings,
              'upstream_archive_sha256':sha(ROOT/'b11960.tar.gz'),
              'binary_hashes':{p.name:sha(p) for p in RUNTIME.iterdir() if p.is_file() and not p.is_symlink()},
              'build_image':subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','word-games-hybrid-build:b11960'],text=True).strip(),
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    producer['fingerprint']=digest(producer);write(out/'producer.json',producer)
    for i,p in enumerate(sources):
        dst=out/'producer_source'/f'{i}_{p.name}';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    results=[]
    for arm in args.arms:
        results.append(run_cell(out/arm,arm,args.phase,args.repeats,producer,metadata,args.port))
    if any(sha(p)!=h for p,h in bindings.items()):raise ValueError('producer changed during campaign')
    write(out/'campaign_results.json',{'schema':SCHEMA,'hypothesis_id':HYPOTHESIS_ID,'fingerprint':producer['fingerprint'],
          'results':results,'completed':[r['arm'] for r in results if r['completed']],'confirmed':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',choices=ARMS,required=True)
    p.add_argument('--phase',choices=('pilot','full'),default='pilot');p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18088)
    run(p.parse_args())
