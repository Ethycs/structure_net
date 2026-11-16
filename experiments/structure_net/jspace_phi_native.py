"""Reservation-backed, layer-streamed Phi-4 quantized-parent reference cache.

This is native Phi execution on decoded GPTQ weights in BF16, not a GPTQ-kernel
benchmark. One block is resident at a time. Gold response suffixes are causal;
prompt predictions never see their future letter or terminator.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_phi_data as data
from . import jspace_phi_weights as weights
from .jspace_recovery_data import read, write, sha, digest, ROOT

SCHEMA = 'jspace.phi-native.v1'
ALLOCATION = 11264
LIMIT = 11008


def full_items(items, responses):
    if len(items) != len(responses) or not items:
        raise ValueError('Matching nonempty prompt/response records required')
    result, offset, seen = [], 0, set()
    for c, r in zip(items, responses):
        if (c['identity'] != r['example_id'] or c['identity'] in seen or
                c['input_ids'] != r['prompt_ids'] or c['split'] != r['split'] or
                len(r['response_ids']) != 2 or r['response_ids'][-1] != r['eos_token_id'] or
                r['finish_reason'] != 'eos'):
            raise ValueError('Exact unique prompt plus observed letter/EOS response required')
        seen.add(c['identity'])
        ids = c['input_ids'] + r['response_ids']
        if not 3 <= len(c['input_ids']) < len(ids) <= 416:
            raise ValueError('Invalid complete sequence length')
        result.append(dict(c, offset=offset, stop=offset+len(c['input_ids']),
                           full_stop=offset+len(ids), full_input_ids=ids))
        offset += len(ids)
    prompt_offset=0
    for row in result:
        row.update(prompt_offset=prompt_offset,prompt_stop=prompt_offset+len(row['input_ids']))
        prompt_offset=row['prompt_stop']
    return result


def block_arguments(x, rotary):
    positions = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
    mask = torch.full((x.shape[0], 1, x.shape[1], x.shape[1]), torch.finfo(x.dtype).min,
                      device=x.device, dtype=x.dtype).triu(1)
    return dict(attention_mask=mask, position_ids=positions, cache_position=positions[0],
                position_embeddings=rotary(x, positions), use_cache=False)


def hidden(value):
    return value[0] if isinstance(value, tuple) else value


def put(bank, start, value):
    value = value.detach().squeeze(0).contiguous()
    if value.dtype != torch.bfloat16 or value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError('Finite BF16 token matrix required')
    bank[start:start+value.shape[0]] = value.cpu().view(torch.int16).numpy()


def get(bank, case, device, *, prompt_only=False):
    stop = case['stop'] if prompt_only else case['full_stop']
    return torch.from_numpy(np.array(bank[case['offset']:stop], copy=True)).view(torch.bfloat16)[None].to(device)


def event(root, kind, **values):
    row = dict(kind=kind, time=time.time(), **values)
    with (Path(root)/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)


def prepare(root, data_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Hide CUDA during CPU preparation')
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    if root.exists(): raise FileExistsError(root)
    d = data.verify(data_root, live=True)
    items = full_items(read(data_root/'pilot_items.json')['items'],
                       [json.loads(line) for line in (data_root/'pilot_responses.jsonl').read_text().splitlines()])
    # Explicit producer set: no mutable launch/training code is part of native execution.
    from . import jspace_full_lease, jspace_single_block_reserved
    import structure_net_resources.client as resource_client
    paths = [Path(__file__), Path(data.__file__), Path(weights.__file__),
             ROOT/'experiments/structure_net/jspace_recovery_data.py',
             Path(jspace_full_lease.__file__), Path(jspace_single_block_reserved.__file__),
             Path(resource_client.__file__), ROOT/'pixi.lock', ROOT/'pyproject.toml',
             ROOT/'tests/structure_net/test_jspace_phi_native.py',
             ROOT/'docs/07 - Status Reports/jspace/2026-11-15_phi4-single-block-pilot.md']
    p = dict(schema=SCHEMA, model=d['model'], data_root=str(data_root),
             data_fingerprint=d['fingerprint'], data_manifest_sha256=sha(data_root/'manifest.json'),
             items_digest=digest(items), layer=39, layers=40, width=5120,
             tokens=items[-1]['full_stop'], prompt_tokens=items[-1]['prompt_stop'],label_ids=d['label_ids'],
             dtype='bfloat16', source_kind=weights.SOURCE_KIND,
             source_sha256={str(f.resolve()):sha(f) for f in paths},
             allocation_mib=ALLOCATION, allocator_mib=LIMIT, confirmed=False,
             scope='Native reference for one final-block compression pilot; quantized-source BF16 execution; no general quality or speed claim')
    p['fingerprint'] = digest(p)
    root.mkdir(parents=True)
    for path, value in p['source_sha256'].items():
        dest = root/'producer'/Path(path).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        if sha(dest) != value: raise ValueError('Source archive changed')
    write(root/'plan.json', p)
    write(root/'items.json', items)
    return dict(status='prepared', fingerprint=p['fingerprint'], cases=len(items), tokens=p['tokens'])


def verify_plan(root, *, live=True):
    root = Path(root).resolve(); p = read(root/'plan.json')
    if (p.get('schema') != SCHEMA or p.get('fingerprint') != digest({k:v for k,v in p.items() if k!='fingerprint'}) or
            (p.get('layer'), p.get('layers'), p.get('width')) != (39,40,5120) or
            p.get('dtype') != 'bfloat16' or p.get('allocation_mib') != ALLOCATION or
            p.get('allocator_mib') != LIMIT or p.get('confirmed') is not False):
        raise ValueError('Phi native plan changed')
    d = data.verify(p['data_root'])
    if d['fingerprint'] != p['data_fingerprint'] or sha(Path(p['data_root'])/'manifest.json') != p['data_manifest_sha256']:
        raise ValueError('Data source changed')
    if p['model'] != d['model'] or digest(read(root/'items.json')) != p['items_digest']:
        raise ValueError('Model or native sequence inventory changed')
    for path, value in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT)) != value or live and sha(path) != value:
            raise ValueError('Native producer changed')
    return p


def reservation():
    from .jspace_full_lease import wait_ready
    from .jspace_single_block_reserved import allocator_settings
    lease = wait_ready([0])
    if len(lease['allocation']) != 1 or not all(k.startswith('GPU-0370fdb0') and v==ALLOCATION for k,v in lease['allocation'].items()):
        raise ValueError('Phi pilot requires the declared RTX3060 reservation')
    settings = allocator_settings(lease, dict(headroom_mib=256, allocator_backend='native'))
    torch.cuda.reset_peak_memory_stats()
    def guard():
        current = wait_ready([0])
        if current['id'] != lease['id'] or current['allocation'] != lease['allocation']:
            raise RuntimeError('Reservation changed')
        if torch.cuda.max_memory_reserved() > LIMIT*1024**2:
            raise MemoryError('Reserved allocator budget exceeded')
    return lease, settings, guard


def checked(root, *, hash_payload=True):
    root = Path(root).resolve(); p = verify_plan(root, live=False)
    m = read(root/'native/manifest.json')
    if (m.get('schema') != SCHEMA or m.get('status') != 'completed' or m.get('fingerprint') != p['fingerprint'] or
            set(m.get('artifacts',{})) != {'items.json','prefix.npy','teacher_hidden.npy',
                'prompt_prefix.npy','prompt_teacher_hidden.npy','baseline.json'}):
        raise ValueError('Completed native reference receipt required')
    from collections import Counter
    if (Counter(c['split'] for c in read(root/'native/items.json')) != {'train':128,'development':64} or
            len(m.get('allocation',{}))!=1 or
            not all(k.startswith('GPU-0370fdb0') and v==ALLOCATION for k,v in m['allocation'].items()) or
            m.get('allocator',{}).get('allocator_limit_bytes') != LIMIT*1024**2 or
            not 0 < m.get('peak_reserved_bytes',0) <= LIMIT*1024**2):
        raise ValueError('Native population/reservation evidence incomplete')
    if hash_payload:
        for name, value in m['artifacts'].items():
            if sha(root/'native'/name) != value: raise ValueError('Native artifact changed')
    return m


@torch.no_grad()
def stream_blocks(out, items, width, layers, bank, load_block, rotary, device, guard):
    """Single-example native causal execution with bounded ping-pong disk banks."""
    out=Path(out); shape=(items[-1]['full_stop'],width)
    scratch=[out/'scratch_a.npy',out/'scratch_b.npy']
    for layer in range(layers):
        guard(); tick=time.monotonic(); block=load_block(layer)
        destination=out/'teacher_hidden.npy' if layer==layers-1 else scratch[(layer+1)%2]
        dest=np.lib.format.open_memmap(destination,mode='w+',dtype=np.int16,shape=shape)
        for c in items:
            guard(); x=get(bank,c,device)
            y=hidden(block(x,**block_arguments(x,rotary)))
            put(dest,c['offset'],y)
            del x,y
        dest.flush(); bank._mmap.close(); del bank,block
        if layer==layers-2:
            dest._mmap.close(); del dest
            scratch[(layer+1)%2].rename(out/'prefix.npy')
            bank=np.load(out/'prefix.npy',mmap_mode='r',allow_pickle=False)
        else: bank=dest
        gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
        event(out,'layer_complete',layer=layer,seconds=time.monotonic()-tick)
    return bank


def separate_banks(out, full_tokens, prompt_tokens, width):
    """Separate the two execution contexts without holding a second RAM copy."""
    out=Path(out)
    for name in ('prefix.npy','teacher_hidden.npy'):
        source=out/('combined_'+name);(out/name).rename(source)
        bank=np.load(source,mmap_mode='r',allow_pickle=False)
        if bank.shape != (full_tokens+prompt_tokens,width):raise ValueError('Combined cache shape changed')
        for target,start,count in ((name,0,full_tokens),('prompt_'+name,full_tokens,prompt_tokens)):
            dest=np.lib.format.open_memmap(out/target,mode='w+',dtype=np.int16,shape=(count,width))
            for lo in range(0,count,1024):dest[lo:lo+1024]=bank[start+lo:start+min(lo+1024,count)]
            dest.flush();dest._mmap.close()
        bank._mmap.close();source.unlink()


@torch.no_grad()
def run(root):
    root = Path(root).resolve(); p = verify_plan(root)
    out = root/'native'
    if out.exists(): raise FileExistsError(out)
    out.mkdir(); items = read(root/'items.json'); write(out/'items.json', items)
    lease = settings = None; started = time.monotonic()
    try:
        lease, settings, guard = reservation()
        config = weights.load_config(p['model']['path'])
        device = 'cuda:0'; shape = (p['tokens']+p['prompt_tokens'],p['width'])
        prompt_items=[dict(c,offset=p['tokens']+c['prompt_offset'],
            stop=p['tokens']+c['prompt_stop'],full_stop=p['tokens']+c['prompt_stop'],
            full_input_ids=c['input_ids']) for c in items]
        stream_items=items+prompt_items
        rotary = weights.load_rotary(p['model']['path'],device,guard=guard)
        scratch = [out/'scratch_a.npy',out/'scratch_b.npy']
        bank = np.lib.format.open_memmap(scratch[0],mode='w+',dtype=np.int16,shape=shape)
        embedding = weights.load_embedding(p['model']['path'],device,guard=guard)
        for c in stream_items:
            guard(); put(bank,c['offset'],embedding[torch.tensor(c['full_input_ids'],device=device)][None])
        bank.flush(); del embedding; gc.collect(); torch.cuda.empty_cache()
        event(out,'embedding_complete',cases=len(items),tokens=shape[0],separate_prompt_execution=True)
        bank=stream_blocks(out,stream_items,p['width'],40,bank,
            lambda layer:weights.load_native_block(p['model']['path'],layer,device,guard=guard),
            rotary,device,guard)
        bank._mmap.close();del bank
        separate_banks(out,p['tokens'],p['prompt_tokens'],p['width'])
        bank=np.load(out/'prompt_teacher_hidden.npy',mmap_mode='r',allow_pickle=False)
        head,norm,_ = weights.load_readout_norm(p['model']['path'],device,guard=guard)
        rows=[]
        for c in items:
            guard(); pc=dict(c,offset=c['prompt_offset'],full_stop=c['prompt_stop'])
            h=get(bank,pc,device)[:,-1:]
            logits=torch.nn.functional.linear(norm(h),head).float()[0,0]
            allowed=logits[torch.tensor(p['label_ids'][:len(c['choices'])],device=device)]
            prediction=int(allowed.argmax())
            rows.append(dict(identity=c['identity'],family=c['family'],split=c['split'],answer=c['answer'],
                             prediction=prediction,correct=prediction==c['answer'],label_logits=allowed.cpu().tolist(),
                             full_vocabulary_argmax=int(logits.argmax())))
        bank._mmap.close(); del bank
        metrics={split:dict(correct=sum(r['correct'] for r in rows if r['split']==split),
                            count=sum(r['split']==split for r in rows)) for split in ('train','development')}
        write(out/'baseline.json',dict(rows=rows,metrics=metrics,confirmed=False,scope=p['scope']))
        for path in scratch:
            if path.exists(): path.unlink()
        guard()
        manifest=dict(schema=SCHEMA,status='completed',fingerprint=p['fingerprint'],
                      artifacts={name:sha(out/name) for name in ('items.json','prefix.npy','teacher_hidden.npy',
                          'prompt_prefix.npy','prompt_teacher_hidden.npy','baseline.json')},
                      lease_id=lease['id'],allocation=lease['allocation'],allocator=settings,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started,
                      metrics=metrics,confirmed=False)
        write(out/'manifest.json',manifest); event(out,'completed',metrics=metrics)
        return manifest
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',error=repr(error),lease=lease,allocator=settings,
                                     elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run','verify'))
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--data-root',type=Path)
    args=parser.parse_args()
    result=prepare(args.root,args.data_root) if args.phase=='prepare' else run(args.root) if args.phase=='run' else checked(args.root)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
