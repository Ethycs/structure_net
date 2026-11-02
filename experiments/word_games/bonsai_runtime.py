"""Standalone Bonsai 2 27B, native non-thinking WORD_GAMES evaluation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

import requests

from .check_qwen_tokenizers import gguf_metadata
from .common import PACKAGE, digest, read, sha, write
from .native_speed import BINARY, GPU, IMAGE, PILOT, NativePhi4, aggregate, output_tokens
from .prepare_bonsai import WEIGHTS, SHA256, SIZE, REPO, REVISION
from .qwen_runtime import REQUESTS, REQUESTS_SHA, memory

MODEL = 'word-games-bonsai2-27b'
HYPOTHESIS = 'word-games-bonsai27-naked-v1'
PLAN = Path('docs/07 - Status Reports/word-games/2026-10-31_word-games-bonsai-naked-plan.md')
PAIRED = Path('data/experiments/word_games/20261031_qwen_baseline_live_v1')
NO_THINK_SUFFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'


class NativeBonsai(NativePhi4):
    def format_prompt(self, prompt):
        r = self.http.post(self.url+'/apply-template', json={
            'messages':[{'role':'user','content':prompt}], 'add_generation_prompt':True,
            'chat_template_kwargs':{'enable_thinking':False}}, timeout=30)
        r.raise_for_status()
        result = r.json()['prompt']
        # The embedded Qwen3.8 template applies Jinja |trim to user content.
        # Require the exact resulting turn, allowing only that normalization.
        user_turn = '<|im_start|>user\n' + prompt.strip() + '<|im_end|>\n'
        if not result.endswith(NO_THINK_SUFFIX) or user_turn not in result:
            raise ValueError('native Bonsai template did not disable thinking or preserve prompt')
        return result


def command(port, name):
    cache = Path('.pixi/word-games-cuda-cache').resolve()
    cache.mkdir(exist_ok=True)
    return ['docker','run','--rm','--name',name,'--gpus',f'device={GPU}',
            '--user',f'{os.getuid()}:{os.getgid()}', '-p',f'127.0.0.1:{port}:8080',
            '-v',f'{BINARY.resolve()}:/llama:ro','-v','/usr/local/cuda/lib64:/cuda:ro',
            '-v',f'{WEIGHTS.resolve()}:/model.gguf:ro','-v',f'{cache}:/cuda-cache:rw',
            '-e',f'CUDA_VISIBLE_DEVICES={GPU}','-e','CUDA_CACHE_PATH=/cuda-cache',
            '-e','CUDA_CACHE_MAXSIZE=2147483648',IMAGE,'-m','/model.gguf','--alias',MODEL,
            '--host','0.0.0.0','--port','8080','--ctx-size','8192','--parallel','1','--fit','off',
            '--device','CUDA0','--split-mode','none','--main-gpu','0','--n-gpu-layers','99',
            '--flash-attn','on','--cache-type-k','q8_0','--cache-type-v','q8_0',
            '--batch-size','512','--ubatch-size','512','--threads','4','--cache-ram','0',
            '--no-cache-prompt','--jinja','--chat-template-kwargs','{"enable_thinking":false}',
            '--spec-type','none','--verbosity','4']


def totals(rows):
    result = aggregate(rows)
    result['output_bytes'] = sum(len(r['response']['content'].encode()) for r in rows)
    result['output_bytes_per_second'] = result['output_bytes']/result['request_seconds']
    return result


def run(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    if sha(REQUESTS)!=REQUESTS_SHA: raise ValueError('request corpus changed')
    print('Verifying Bonsai checkpoint',flush=True)
    if WEIGHTS.stat().st_size!=SIZE or sha(WEIGHTS)!=SHA256: raise ValueError('checkpoint differs')
    metadata=gguf_metadata(WEIGHTS)
    if metadata['general.architecture']!='qwen35' or metadata.get('prism.hadamard.version')!=1:
        raise ValueError('expected rotated Qwen35 Bonsai checkpoint')
    sources=[p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    sources += [PLAN,Path('pyproject.toml'),Path('pixi.lock')]
    bindings={str(p):sha(p) for p in sources}
    fixtures={str(p):sha(p) for p in sorted((PAIRED/'quality/cases').glob('*/context.json'))} if args.stage=='live' else {}
    if args.stage=='live' and len(fixtures)!=12: raise ValueError('twelve paired cases required')
    name=f'word-games-bonsai-{os.getpid()}'
    cmd=command(args.port,name)
    config={'hypothesis_id':HYPOTHESIS,'model':MODEL,'stage':args.stage,'command':cmd,
            'repository':REPO,'revision':REVISION,'weights_sha256':SHA256,'weights_bytes':SIZE,
            'source_hashes':bindings,'requests_sha256':REQUESTS_SHA,'fixture_hashes':fixtures,
            'binary_hashes':{p.name:sha(p) for p in BINARY.glob('*') if p.is_file() and not p.is_symlink()},
            'temperature':0,'seed':73,'thinking':False,'speculation':False,'cross_request_cache':False,
            'metadata_sha256':digest(metadata),'chat_template_sha256':digest(metadata['tokenizer.chat_template']),
            'hadamard_version':metadata['prism.hadamard.version'],'gpu_before':memory()}
    write(out/'producer.json',{**config,'fingerprint':digest(config)})
    for i,p in enumerate(sources):
        dest=out/'producer_source'/f'{i}_{p.name}';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    client,process=NativeBonsai(args.port),None
    started=time.perf_counter()
    try:
        with (out/'server.log').open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+240
            while True:
                if process.poll() is not None: raise RuntimeError('Bonsai server exited; see server.log')
                try:
                    if client.http.get(client.url+'/health',timeout=1).status_code==200: break
                except requests.RequestException: pass
                if time.monotonic()>deadline: raise TimeoutError('Bonsai startup')
                time.sleep(.5)
            props=client.http.get(client.url+'/props',timeout=10).json()
            write(out/'runtime.json',{'properties':props,'load_seconds':time.perf_counter()-started,'gpu_loaded':memory()})
            if props['model_alias']!=MODEL or props['model_path']!='/model.gguf': raise ValueError('wrong runtime identity')
            if props['chat_template'].rstrip('\n')!=metadata['tokenizer.chat_template'].rstrip('\n'):
                raise ValueError('chat template differs')
            log_text=(out/'server.log').read_text()
            offloads=re.findall(r'offloaded (\d+)/(\d+) layers',log_text)
            if not offloads or any(a!=b for a,b in offloads): raise ValueError('full GPU offload not verified')
            write(out/'offload_check.json',{'layers':offloads,'verified':True})
            client.complete('Return exactly {"ready":true} as JSON.',out/'warmup',max_tokens=24)
            if client.rows[-1]['parsed']!={'ready':True}: raise ValueError('JSON warm-up failed')
            client.rows.clear()
            identity={'model':MODEL,'ready':True,'weights_sha256':SHA256,'fingerprint':digest(config),
                      'backend':'llama.cpp b11960 native','thinking':False,'speculation':False}
            write(out/'identity.json',identity)
            if args.stage=='live':
                from .native_quality import run_quality
                q=run_quality(out/'quality',out/'identity.json',client,PAIRED,
                              expected_model=MODEL,hypothesis_id=HYPOTHESIS,plan=PLAN)
                rows=[read(p) for p in (out/'quality/cases').glob('*/result.json')]
                write(out/'quality_summary.json',{'metrics':q['metrics'],'cases':len(rows),
                      'joint_case_success':sum(all(r[k] for k in ('sender_operation_match','boundary_outcome_match','receiver_state_correct')) for r in rows),
                      'semantic_review_completed':False})
            else:
                inputs=read(REQUESTS)
                chosen=[next(r for r in inputs if r['id']==name) for name in PILOT] if args.stage=='pilot' else inputs
                for r in chosen:
                    client.complete(r['prompt'],out/'runs'/r['id'],max_tokens=r['max_tokens'])
                    row=client.rows[-1]
                    if args.stage=='pilot' and (not isinstance(row['parsed'],dict) or row['response'].get('stop_type')=='limit'):
                        raise ValueError('invalid or truncated pilot JSON')
                    print(json.dumps({'id':r['id'],'seconds':row['elapsed_seconds'],'tokens':output_tokens(row['response'])}),flush=True)
            if any(sha(p)!=h for p,h in {**bindings,**fixtures}.items()): raise ValueError('producer or fixtures changed')
            result={'hypothesis_id':HYPOTHESIS,'stage':args.stage,'model':MODEL,'fingerprint':digest(config),
                    'classification':'systems_lifecycle_only_not_quality_evidence' if args.stage=='pilot' else 'completed_exploratory',
                    'confirmed':False,'totals':totals(client.rows),
                    'artifacts':{str(p):sha(p) for p in out.rglob('response.json')}}
            write(out/'campaign_results.json',result)
            print(json.dumps(result['totals']),flush=True)
    except Exception as exc:
        write(out/'interruption.json',{'error':repr(exc),'completed_calls':len(client.rows)})
        raise
    finally:
        if process:
            subprocess.run(['docker','stop','--time','5',name],capture_output=True);process.wait(timeout=30)
        write(out/'gpu_after.json',{'memory':memory()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['pilot','full','live'],default='pilot')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--port',type=int,default=18086)
    run(p.parse_args())
