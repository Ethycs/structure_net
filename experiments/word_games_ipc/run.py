"""Measure two GPU workers over Unix sockets; serial target reference only."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from experiments.word_games.common import digest, read, sha, write
from experiments.word_games.check_qwen_tokenizers import gguf_metadata
from experiments.word_games.native_speed import BINARY, GPU, IMAGE, PILOT
from experiments.word_games.qwen_pair import PAIRS, directory, entrypoint
from experiments.word_games.qwen_runtime import CHECK, DRAFT_GPU, REQUESTS, REQUESTS_SHA, memory
from .transport import DraftSession, UnixHTTP, accepted_prefix

PLAN = Path('docs/07 - Status Reports/2026-10-31_word-games-qwen-ipc-plan.md')


def command(role, gpu, socket_dir, cache, name):
    return ['docker','run','--rm','--name',name,'--network','none','--gpus',f'device={gpu}',
            '--user',f'{os.getuid()}:{os.getgid()}', '-e',f'CUDA_VISIBLE_DEVICES={gpu}',
            '-e','CUDA_CACHE_PATH=/cuda-cache','-e','CUDA_CACHE_MAXSIZE=2147483648',
            '-v',f'{cache}:/cuda-cache:rw','-v',f'{socket_dir}:/ipc:rw',
            '-v',f'{BINARY.resolve()}:/llama:ro','-v','/usr/local/cuda/lib64:/cuda:ro',
            '-v',f'{directory(role).resolve()}:/weights:ro',IMAGE,
            '-m','/weights/'+PAIRS[role]['files'][0][0], '--alias','word-games-qwen-'+role,
            '--host',f'/ipc/{role}.sock','--ctx-size','8192','--parallel','1','--fit','off',
            '--device','CUDA0','--n-gpu-layers','99','--flash-attn','on',
            '--cache-type-k','q8_0','--cache-type-v','q8_0','--batch-size','512','--ubatch-size','512',
            '--threads','4','--cache-ram','0','--no-cache-prompt','--jinja','--spec-type','none']


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    check = read(CHECK)
    if not check['compatible'] or sha(REQUESTS) != REQUESTS_SHA:
        raise ValueError('verified tokenizer and frozen prompts required')
    for role,cfg in PAIRS.items():
        for filename,size,h in cfg['files']:
            p=directory(role)/filename
            if p.stat().st_size != size or sha(p) != h: raise ValueError('changed weights')
    socket_dir = (Path('.pixi/word-games-ipc') / f'run-{os.getpid()}').resolve()
    socket_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    cache = Path('.pixi/word-games-cuda-cache').resolve()
    cache.mkdir(exist_ok=True)
    sources = [Path(__file__),Path(__file__).with_name('transport.py'),PLAN,
               Path('experiments/word_games/qwen_pair.py'),Path('experiments/word_games/qwen_runtime.py'),
               Path('experiments/word_games/common.py'),Path('experiments/word_games/check_qwen_tokenizers.py'),
               Path('experiments/word_games/native_speed.py'),Path('pyproject.toml'),Path('pixi.lock')]
    bindings = {str(p):sha(p) for p in sources}
    names = {r:f'word-games-ipc-{r}-{os.getpid()}' for r in PAIRS}
    commands = {r:command(r,g,socket_dir,cache,names[r]) for r,g in [('target',GPU),('draft',DRAFT_GPU)]}
    config = {'hypothesis_id':'word-games-qwen-ipc-v1','commands':commands,'models':PAIRS,
              'source_hashes':bindings,'requests_sha256':REQUESTS_SHA,'tokenizer_check_sha256':sha(CHECK),
              'binary_hashes':{p.name:sha(p) for p in BINARY.glob('*') if p.is_file() and not p.is_symlink()},
              'gpu_before':memory(),'verification':'autoregressive reference, not batched speculation',
              'external_network':False,'socket_directory_mode':oct(socket_dir.stat().st_mode & 0o777)}
    write(out/'producer.json',{**config,'fingerprint':digest(config)})
    for i,p in enumerate(sources): (out/f'{i}_{p.name}').write_bytes(p.read_bytes())
    processes, logs, clients, rows = {}, {}, {}, []
    try:
        for r in PAIRS:
            logs[r] = (out/f'{r}.log').open('x')
            processes[r] = subprocess.Popen(commands[r],stdout=logs[r],stderr=subprocess.STDOUT)
            clients[r] = UnixHTTP(socket_dir/f'{r}.sock')
        deadline = time.monotonic()+240
        for r,c in clients.items():
            while True:
                if processes[r].poll() is not None: raise RuntimeError(r+' exited during load')
                try:
                    health,_=c.call('/health')
                    if health.get('status')=='ok': break
                except (OSError,RuntimeError): pass
                if time.monotonic()>deadline: raise TimeoutError('IPC worker startup')
                time.sleep(.5)
            props,_=c.call('/props')
            if props['model_alias'] != 'word-games-qwen-'+r or props['chat_template'].rstrip('\n') != check['chat_template'].rstrip('\n'):
                raise ValueError('worker identity/template differs')
            write(out/f'{r}_props.json',props)
            result,elapsed=c.call('/completion',{'prompt':'Hello','n_predict':8,'temperature':0,'cache_prompt':False,'return_tokens':True})
            write(out/f'{r}_warmup.json',{'response':result,'elapsed_seconds':elapsed})
        write(out/'gpu_loaded.json',{'memory':memory()})
        latency = {}
        for r,c in clients.items():
            for _ in range(10): c.call('/health')
            samples = [c.call('/health')[1]*1000 for _ in range(100)]
            latency[r] = {'samples_ms':samples,'median_ms':statistics.median(samples),'p95_ms':sorted(samples)[94]}
        metadata=gguf_metadata(entrypoint('draft'))
        ordinary=[i for i,t in enumerate(metadata['tokenizer.ggml.token_type'])
                  if t in (1,6) and metadata['tokenizer.ggml.tokens'][i] != '</s>']
        tokenizer_identity=digest({k:check[k] for k in ('target_tokenizer_sha256','draft_tokenizer_sha256','shared_ids_checked')})
        session=DraftSession(clients['draft'],ordinary,tokenizer_identity)
        inputs=read(REQUESTS)
        for name in PILOT:
            request=next(r for r in inputs if r['id']==name)
            formatted,_=clients['target'].call('/apply-template',{'messages':[{'role':'user','content':request['prompt']}],'add_generation_prompt':True})
            encoded,_=clients['target'].call('/tokenize',{'content':formatted['prompt'],'add_special':False,'parse_special':True})
            prefix=encoded['tokens']
            # Check the second worker interprets this exact prompt identically.
            draft_encoded,_=clients['draft'].call('/tokenize',{'content':formatted['prompt'],'add_special':False,'parse_special':True})
            if prefix != draft_encoded['tokens']: raise ValueError('live tokenizer mismatch')
            session.begin(prefix,tokenizer_fingerprint=tokenizer_identity)
            proposal=session.propose(prefix,3)
            tokens=session.validate(proposal,prefix)
            verified,seconds=clients['target'].call('/completion',{'prompt':prefix,'n_predict':3,
                'temperature':0,'seed':73,'repeat_penalty':1.0,'cache_prompt':False,'return_tokens':True,'stream':False})
            target_ids=verified['tokens']
            if not target_ids: raise ValueError('empty target verification')
            bad=[(target_ids[0]+1)%1000]+tokens[1:]
            if accepted_prefix(bad,target_ids)!=0: raise ValueError('wrong-draft rejection failed')
            row={'id':name,'prefix_tokens':len(prefix),'proposal':proposal,'target_tokens':target_ids,
                 'accepted_prefix':accepted_prefix(tokens,target_ids),'target_seconds':seconds,'wrong_draft_accepted':0}
            rows.append(row);write(out/'runs'/name/'result.json',row)
            print(json.dumps(row),flush=True)
        if any(sha(p)!=h for p,h in bindings.items()): raise ValueError('producer changed')
        result={'hypothesis_id':config['hypothesis_id'],'classification':'two_worker_ipc_reference_validated',
                'confirmed':False,'fingerprint':digest(config),'cases':rows,'latency':latency,
                'payload_bytes':{r:{'sent':c.sent_bytes,'received':c.received_bytes} for r,c in clients.items()},
                'external_draft_batched_verifier_implemented':False,'end_to_end_speedup_measured':False,
                'artifacts':{str(p):sha(p) for p in out.rglob('result.json')}}
        write(out/'campaign_results.json',result)
        print(json.dumps({r:{k:v for k,v in m.items() if k!='samples_ms'} for r,m in latency.items()}),flush=True)
    except Exception as exc:
        write(out/'interruption.json',{'error':repr(exc),'completed_cases':len(rows)})
        raise
    finally:
        for c in clients.values(): c.close()
        for r,p in processes.items():
            subprocess.run(['docker','stop','--time','5',names[r]],capture_output=True);p.wait(timeout=30)
        for log in logs.values(): log.close()
        write(out/'gpu_after.json',{'memory':memory()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output)
