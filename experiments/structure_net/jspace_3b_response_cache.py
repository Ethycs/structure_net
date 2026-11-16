"""Authenticated, lossless native targets for complete-response Qwen recovery.

Only admitted TRAIN/DEV records enter this producer. CUDA is owned by an outer
inference-ctl lease. Hidden-state storage preserves BF16 bits; the loss uses the
original tied full-vocabulary readout, not a top-k approximation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.neural_architecture_lab.distillation_recovery import module
from . import jspace_3b_final_evaluate as execution
from . import jspace_3b_broad_recovery_cache as prior
from . import jspace_3b_response_eval as evaluation
from .jspace_full_lease import wait_ready

data = module('distillation_data')
SCHEMA = 'jspace.3b-complete-response-cache.v1'
ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    p = Path(path).resolve(); s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns,
                inode=s.st_ino, device=s.st_dev)


def records_at(root):
    root = Path(root)
    return [*data.stream_records(root/'train.jsonl', allowed_splits=('train',)),
            *data.stream_records(root/'development.jsonl', allowed_splits=('development',))]


def index_records(records):
    data.dataset_fingerprint(records)  # Full group/exact-token split admission.
    rows = []; offset = 0
    for r in records:
        if len(r.prompt_ids) > 288 or len(r.response_ids) > 128 or len(r.input_ids) > 416:
            raise ValueError('Untruncated 288 prompt / 128 response / 416 total profile required')
        if r.finish_reason != 'eos':
            raise ValueError('Teacher-forced reference must include its actual template EOS')
        rows.append(dict(example_id=r.example_id, split=r.split,
            record_sha256=data.record_fingerprint(r), offset=offset,
            stop=offset+len(r.input_ids), input_ids=list(r.input_ids)))
        offset = rows[-1]['stop']
    if {r.split for r in records} != {'train', 'development'}:
        raise ValueError('Both admitted TRAIN and DEV are required')
    return rows


def generation_indices(records, per_family=2):
    families={}
    for i,record in enumerate(records):
        if record.split=='development':
            family=record.provenance.get('source_family',record.task)
            families.setdefault(family,[]).append(i)
    return [i for family in sorted(families)
            for i in sorted(families[family],key=lambda j:records[j].example_id)[:per_family]]


class NativeModel:
    def __init__(self, model, guard):
        self.model,self.guard=model,guard

    @torch.no_grad()
    def next_logits(self, ids):
        self.guard(); ids=ids.to(self.model.lm_head.weight.device)
        hidden=self.model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1]
        return torch.nn.functional.linear(hidden,self.model.lm_head.weight)[0]


def prepare(root, corpus, native_plan, *, cut=20):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU preparation requires CUDA hidden')
    root, corpus, native_plan = map(lambda p: Path(p).resolve(), (root, corpus, native_plan))
    if root.exists(): raise FileExistsError(root)
    model = read(native_plan)['model']; records = records_at(corpus)
    rows = index_records(records)
    if not 0 < cut < 36: raise ValueError('Native prefix capture requires cut 1..35')
    inputs = {str(p):sha(p) for p in (corpus/'manifest.json', corpus/'train.jsonl',
        corpus/'development.jsonl', corpus/'text.jsonl', native_plan)}
    for name, digest in model['files'].items():
        path = Path(model['path'])/name
        if sha(path) != digest: raise ValueError('Pinned native file changed: '+str(path))
        inputs[str(path)] = digest
    paths = prior.source.inherited.local_import_closure([Path(__file__),
        ROOT/'src/neural_architecture_lab/distillation_data.py',
        ROOT/'src/neural_architecture_lab/distillation_training.py',
        ROOT/'experiments/neural_architecture_lab/distillation_recovery.py'])
    p = dict(schema=SCHEMA, corpus=str(corpus), model=model, cut=cut, width=2048,
        vocabulary=151936, max_length=416, records_sha256=data.dataset_fingerprint(records),
        rows=rows, total_tokens=rows[-1]['stop'], generation_indices=generation_indices(records),
        generation_max_tokens=128,
        generation_eos_token_ids=read(Path(model['path'])/'generation_config.json')['eos_token_id'],
        input_sha256=inputs,
        input_identity={path:identity(path) for path in inputs},
        source_sha256={str(f):sha(f) for f in sorted(paths)}, confirmed=False)
    p['fingerprint'] = data.digest(p)
    root.mkdir(parents=True)
    write(root/'plan.json', p)
    for f, digest in p['source_sha256'].items():
        target=root/'producer'/Path(f).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f,target)
        if sha(target) != digest: raise ValueError('Producer archive mismatch')
    return dict(status='prepared', fingerprint=p['fingerprint'], records=len(rows),
                total_tokens=p['total_tokens'], cache_bytes=4*p['total_tokens']*2048)


def verify(root, *, live=True):
    root=Path(root); p=read(root/'plan.json')
    body=dict(p); fp=body.pop('fingerprint')
    if p['schema'] != SCHEMA or data.digest(body) != fp:
        raise ValueError('Cache plan changed')
    for path, old in p['input_identity'].items():
        if identity(path) != old: raise ValueError('Cache input changed: '+path)
    for path, digest in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT)) != digest:
            raise ValueError('Archived producer changed')
        if live and sha(path) != digest: raise ValueError('Live producer changed: '+path)
    records=records_at(p['corpus'])
    if data.dataset_fingerprint(records) != p['records_sha256'] or index_records(records) != p['rows']:
        raise ValueError('Admitted corpus differs from cache plan')
    return p,records


def reservation(limit_mib):
    lease=wait_ready([0]); torch.cuda.set_device(0)
    if len(lease['allocation']) != 1: raise ValueError('Single reserved worker GPU required')
    reserved=next(iter(lease['allocation'].values()))
    if not 0 < limit_mib < reserved: raise ValueError('Allocator must fit reservation with headroom')
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(limit_mib*1024**2/total,0)
    torch.cuda.reset_peak_memory_stats()
    def guard():
        current=wait_ready([0])
        if current['id'] != lease['id'] or current['allocation'] != lease['allocation']:
            raise RuntimeError('Worker reservation changed')
        if torch.cuda.max_memory_reserved() > limit_mib*1024**2:
            raise MemoryError('Declared allocator cap exceeded')
    return lease,guard


@torch.inference_mode()
def native_parity(model, prompt_ids, guard):
    """Independent HF forward vs hidden/readout for prompt and generated prefixes."""
    ids=list(prompt_ids); records=[]
    for step in range(3):
        guard(); tensor=torch.tensor([ids],device=model.lm_head.weight.device)
        ref=model(input_ids=tensor,use_cache=False).logits[:,-1]
        hidden=model.model(input_ids=tensor,use_cache=False).last_hidden_state[:,-1]
        actual=torch.nn.functional.linear(hidden,model.lm_head.weight)
        if not torch.isfinite(actual).all() or not torch.equal(actual,ref):
            # GEMM dimensions can change BF16 rounding; compare each row with
            # a prespecified BF16 allowance and require equal greedy decisions.
            torch.testing.assert_close(actual,ref,rtol=1/128,atol=.03125)
        if not torch.equal(actual.argmax(-1),ref.argmax(-1)):
            raise ValueError('Native full-vocabulary greedy parity failed')
        records.append(dict(step=step,vocabulary=actual.shape[-1],
            maximum_absolute_difference=float((actual.float()-ref.float()).abs().max()),
            greedy_equal=True))
        ids.append(int(ref.argmax(-1)))
    return dict(passed=True, prefixes=records)


def capture(root, *, limit_mib=11008):
    root=Path(root).resolve(); p,records=verify(root); out=root/'native'
    if out.exists(): raise FileExistsError(out)
    out.mkdir(); start=time.monotonic(); lease=None; banks=[]
    try:
        lease,guard=reservation(limit_mib); torch.set_num_threads(2)
        with execution.arithmetic_policy():
            model=execution.load_native_model(p['model']['path'],device='cuda:0',attention='eager',guard=guard)
            frozen=[(id(v),v._version) for v in (*model.parameters(),*model.buffers())]
            parity=native_parity(model,records[0].prompt_ids,guard)
            arrays={name:np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=np.int16,
                shape=(p['total_tokens'],p['width'])) for name in ('teacher_hidden',f'boundary_{p["cut"]:02d}')}
            banks=list(arrays.values()); boundary={}
            def hook(_module,args): boundary['value']=args[0].detach()
            handle=model.model.layers[p['cut']].register_forward_pre_hook(hook)
            dev_rows=[]
            with torch.inference_mode():
                for i,(record,row) in enumerate(zip(records,p['rows'])):
                    guard(); ids=torch.tensor([record.input_ids],device='cuda:0')
                    hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state
                    values={'teacher_hidden':hidden, f'boundary_{p["cut"]:02d}':boundary.pop('value')}
                    for name,value in values.items():
                        if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                            raise ValueError('Native hidden states nonfinite or wrong dtype')
                        arrays[name][row['offset']:row['stop']]=value[0].contiguous().cpu().view(torch.int16).numpy()
                    if record.split=='development':
                        nll=0.
                        for start_pos in range(0,len(record.response_ids),8):
                            positions=record.prediction_positions[start_pos:start_pos+8]
                            logits=torch.nn.functional.linear(hidden[0,list(positions)],model.lm_head.weight).float()
                            targets=torch.tensor(record.response_ids[start_pos:start_pos+8],device=logits.device)
                            nll+=float(torch.nn.functional.cross_entropy(logits,targets,reduction='sum'))
                        dev_rows.append(dict(example_id=record.example_id,task=record.task,
                            family=record.provenance.get('source_family',record.task),
                            response_tokens=len(record.response_ids),nll_sum=nll,nll=nll/len(record.response_ids)))
                    if (i+1)%32==0 or i+1==len(records):
                        print(json.dumps(dict(phase='native_complete_response_cache',completed=i+1,
                            total=len(records),tokens=row['stop'],seconds=time.monotonic()-start)),flush=True)
            handle.remove()
            if frozen != [(id(v),v._version) for v in (*model.parameters(),*model.buffers())]:
                raise ValueError('Frozen original teacher changed')
            for bank in banks: bank.flush(); bank._mmap.close()
            banks=[]; guard()
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(p['model']['path'],local_files_only=True,trust_remote_code=False)
            native=NativeModel(model,guard)
            texts={r['example_id']:r for r in map(json.loads,(Path(p['corpus'])/'text.jsonl').read_text().splitlines())}
            canaries=evaluation.evaluate_canaries(native,tokenizer,device='cuda:0',
                eos_token_ids=p['generation_eos_token_ids'])
            generation=[]
            for i in p['generation_indices']:
                source=texts[records[i].example_id]
                generation.append(evaluation.generate_record(native,records[i],tokenizer,device='cuda:0',
                    source_text=source.get('context',source.get('prompt_text','')),
                    eos_token_ids=p['generation_eos_token_ids']))
                print(json.dumps(dict(phase='native_generation',completed=len(generation),
                    total=len(p['generation_indices']))),flush=True)
            write(out/'generation.json',dict(canaries=canaries,corpus=generation,
                teacher_revision=p['model']['revision'],max_new_tokens=128))
            write(out/'development.json',dict(records=dev_rows,
                response_tokens=sum(r['response_tokens'] for r in dev_rows),
                nll=sum(r['nll_sum'] for r in dev_rows)/sum(r['response_tokens'] for r in dev_rows)))
            result=dict(status='completed',fingerprint=p['fingerprint'],teacher_unchanged=True,
                full_vocabulary_parity=parity,records=len(records),tokens=p['total_tokens'],
                seconds=time.monotonic()-start,lease_id=lease['id'],allocation=lease['allocation'],
                allocator_mib=limit_mib,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                storage='Lossless BF16 bits in int16 arrays; normalized teacher hidden and native prefix residual')
            write(out/'result.json',result)
            write(out/'manifest.json',dict(status='completed',fingerprint=p['fingerprint'],
                artifacts={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
            return result
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',error=repr(error),lease=lease,seconds=time.monotonic()-start))
        raise
    finally:
        for bank in banks:
            if not bank._mmap.closed: bank.flush(); bank._mmap.close()


class ResponseDataset:
    def __init__(self, root, *, cut, pad_token_id=151643):
        self.root=Path(root); self.plan,self.records=verify(root,live=False)
        if cut not in (0,self.plan['cut']): raise ValueError('Unsupported native-prefix boundary')
        manifest=read(self.root/'native/manifest.json')
        if manifest['status']!='completed' or manifest['fingerprint']!=self.plan['fingerprint']:
            raise ValueError('Incomplete native response cache')
        for name,digest in manifest['artifacts'].items():
            path=self.root/'native'/name
            if path.parent.resolve()!=(self.root/'native').resolve() or sha(path)!=digest:
                raise ValueError('Native response target changed')
        self.rows=self.plan['rows']; self.pad_token_id=pad_token_id; self.cut=cut
        self.training=[i for i,r in enumerate(self.records) if r.split=='train']
        self.development=[i for i,r in enumerate(self.records) if r.split=='development']
        self.teacher=np.load(self.root/'native/teacher_hidden.npy',mmap_mode='r')
        self.boundary=None if cut==0 else np.load(self.root/'native'/f'boundary_{cut:02d}.npy',mmap_mode='r')
        for bank in (self.teacher,self.boundary):
            if bank is not None and (bank.dtype!=np.int16 or bank.shape!=(self.plan['total_tokens'],self.plan['width'])):
                raise ValueError('Native target bank shape/dtype changed')

    def batch(self,indices,*,device='cpu',dtype=torch.bfloat16):
        if dtype!=torch.bfloat16: raise ValueError('Lossless BF16 target profile required')
        records=[self.records[i] for i in indices]
        batch=data.response_batch(records,pad_token_id=self.pad_token_id,max_length=416,device=device)
        shape=(*batch['input_ids'].shape,self.plan['width'])
        for name,bank in (('teacher_hidden',self.teacher),('boundary_hidden',self.boundary)):
            if bank is None: continue
            target=torch.zeros(shape,dtype=dtype)
            for j,i in enumerate(indices):
                row=self.rows[i]
                if row['input_ids']!=list(records[j].input_ids): raise ValueError('Cached prefix mismatch')
                value=torch.from_numpy(np.array(bank[row['offset']:row['stop']],copy=True)).view(torch.bfloat16)
                target[j,:len(value)]=value
            batch[name]=target.to(device)
        batch.update(indices=list(indices),teacher_input_ids=batch['input_ids'].clone(),
                     teacher_record_ids=[r.example_id for r in records])
        return batch

    def close(self):
        for bank in (self.teacher,self.boundary):
            if bank is not None and not bank._mmap.closed: bank._mmap.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','capture','check'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--corpus',type=Path); parser.add_argument('--native-plan',type=Path)
    parser.add_argument('--cut',type=int,default=20); parser.add_argument('--limit-mib',type=int,default=11008)
    args=parser.parse_args()
    if args.phase=='prepare': result=prepare(args.root,args.corpus,args.native_plan,cut=args.cut)
    elif args.phase=='capture': result=capture(args.root,limit_mib=args.limit_mib)
    else:
        p,rs=verify(args.root,live=False); result=dict(fingerprint=p['fingerprint'],records=len(rs))
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__': main()
