"""Disjoint TRAIN selections and fixed native anchors for finite-removal search."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from . import jspace_3b_response_cache as cache
from . import jspace_3b_full_progressive_data as banks

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/07 - Status Reports/jspace/2026-11-16_qwen3b-ablation32-all-at-once.md'
SCHEMA='jspace.3b-ablation32-data.v1'
read,write,sha,identity=cache.read,cache.write,cache.sha,cache.identity


def choose(records,*,fit_count=2500,selection_count=500,seed=220301):
    """Distinct groups, family round robin, with no DEV selection or targets."""
    train=[i for i,r in enumerate(records) if r.split=='train']
    families=sorted({records[i].provenance['source_family'] for i in train})
    def take(count,role,excluded):
        pools={f:sorted((i for i in train if records[i].provenance['source_family']==f
                         and records[i].group_id not in excluded),
                        key=lambda i:cache.data.digest([seed,role,records[i].example_id])) for f in families}
        used=set(excluded);out=[]
        while len(out)<count:
            before=len(out)
            for family in families:
                while pools[family] and records[pools[family][0]].group_id in used:pools[family].pop(0)
                if pools[family]:
                    i=pools[family].pop(0);out.append(i);used.add(records[i].group_id)
                    if len(out)==count:break
            if len(out)==before:raise ValueError('Insufficient distinct TRAIN groups for declared selection')
        return out
    selection=take(selection_count,'selection',set())
    held_groups={records[i].group_id for i in selection}
    fitting=take(fit_count,'fit',held_groups)
    distillation=[i for i in train if records[i].group_id not in held_groups]
    counts=lambda indices:dict(Counter(records[i].provenance['source_family'] for i in indices))
    return dict(training_indices=fitting,selection_indices=selection,distillation_indices=distillation,
        family_counts=dict(fitting=counts(fitting),selection=counts(selection),distillation=counts(distillation)),
        fitting_response_tokens=sum(len(records[i].response_ids) for i in fitting),
        selection_response_tokens=sum(len(records[i].response_ids) for i in selection),seed=seed)


def epoch_order(plan,epoch):
    if type(epoch)is not int or not 0<=epoch<8:raise ValueError('Search epochs are0..7')
    order=list(plan['training_indices']);random.Random(plan['seed']+epoch).shuffle(order)
    return order


def prepare(root,response_cache_root,full_data_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation requires CUDA hidden')
    root,response_cache_root,full_data_root=map(lambda p:Path(p).resolve(),(root,response_cache_root,full_data_root))
    if root.exists():raise FileExistsError(root)
    p,records=cache.verify(response_cache_root,live=False)
    full,_=banks.verify(full_data_root,live=False)
    if full['model']!=p['model'] or full['records_sha256']!=p['records_sha256']:
        raise ValueError('Original teacher and native boundary plans differ')
    manifest=read(response_cache_root/'native/manifest.json')
    if manifest.get('status')!='completed' or manifest['fingerprint']!=p['fingerprint']:
        raise ValueError('Completed original teacher cache required')
    inputs={**p['input_sha256'],str(response_cache_root/'plan.json'):sha(response_cache_root/'plan.json'),
        str(full_data_root/'plan.json'):sha(full_data_root/'plan.json'),
        str(response_cache_root/'native/manifest.json'):sha(response_cache_root/'native/manifest.json')}
    teacher=response_cache_root/'native/teacher_hidden.npy'
    if sha(teacher)!=manifest['artifacts']['teacher_hidden.npy']:raise ValueError('Original teacher bank changed')
    inputs[str(teacher)]=manifest['artifacts']['teacher_hidden.npy']
    files=cache.prior.source.inherited.local_import_closure([Path(__file__),PROTOCOL,
        ROOT/'tests/structure_net/test_jspace_3b_ablation32_data.py'])
    plan=dict(schema=SCHEMA,response_cache_root=str(response_cache_root),response_cache_fingerprint=p['fingerprint'],
        full_data_root=str(full_data_root),full_data_fingerprint=full['fingerprint'],
        model=p['model'],corpus=p['corpus'],records_sha256=p['records_sha256'],rows=p['rows'],
        width=p['width'],vocabulary=p['vocabulary'],total_tokens=p['total_tokens'],max_length=416,
        generation_indices=p['generation_indices'],generation_eos_token_ids=p['generation_eos_token_ids'],
        **choose(records),input_sha256=inputs,input_identity={path:identity(path) for path in inputs},
        source_sha256={str(path):sha(path) for path in sorted(files)},
        supervision='Example-mean full response/EOS teacher KL; no label loss',
        anchor_policy='Native absolute-position means from2500basis-fitting TRAIN only; unseen positions use global TRAIN-token mean',
        confirmed=False)
    plan['fingerprint']=cache.data.digest(plan);root.mkdir(parents=True);write(root/'plan.json',plan)
    for path,wanted in plan['source_sha256'].items():
        target=root/'producer'/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha(target)!=wanted:raise ValueError('Data producer archive differs')
    return dict(status='prepared',fingerprint=plan['fingerprint'],fitting=len(plan['training_indices']),
        selection=len(plan['selection_indices']),distillation=len(plan['distillation_indices']),
        family_counts=plan['family_counts'])


def verify(root,*,live=True):
    root=Path(root);p=read(root/'plan.json');body=dict(p);fp=body.pop('fingerprint')
    if p.get('schema')!=SCHEMA or cache.data.digest(body)!=fp:raise ValueError('Ablation data plan changed')
    for path,old in p['input_identity'].items():
        if identity(path)!=old:raise ValueError('Ablation data input changed: '+path)
    for path,wanted in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT))!=wanted or live and sha(path)!=wanted:
            raise ValueError('Ablation data producer changed')
    original,records=cache.verify(p['response_cache_root'],live=False)
    if (original['fingerprint']!=p['response_cache_fingerprint'] or original['rows']!=p['rows']
            or original['records_sha256']!=p['records_sha256']):raise ValueError('Original teacher records differ')
    selected=choose(records)
    if any(p[k]!=v for k,v in selected.items()):raise ValueError('Ablation TRAIN selections changed')
    return p,records


def positional_means(bank,rows,indices,*,capacity,width):
    sums=torch.zeros(capacity,width,dtype=torch.float64);counts=torch.zeros(capacity,dtype=torch.long)
    for i in indices:
        row=rows[i];n=row['stop']-row['offset']
        if not 0<n<=capacity:raise ValueError('Complete sequence exceeds anchor capacity')
        values=torch.from_numpy(np.array(bank[row['offset']:row['stop']],copy=True)).view(torch.bfloat16).double()
        sums[:n]+=values;counts[:n]+=1
    if counts.sum()==0:raise ValueError('Empty TRAIN anchor inventory')
    global_mean=sums.sum(0)/counts.sum()
    means=sums/counts.clamp_min(1)[:,None];means[counts==0]=global_mean
    return means.float(),counts


def anchors(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Anchor preparation is CPU-only')
    root=Path(root).resolve();p,_=verify(root);out=root/'anchors'
    if out.exists():raise FileExistsError(out)
    native=Path(p['full_data_root'])/'native'
    inputs=banks._manifest_files(native,p['full_data_fingerprint'],hash_payloads=False)
    result=read(native/'result.json')
    if not all(result.get(k) for k in ('full_teacher_bit_exact','existing_boundary_bit_exact','teacher_unchanged')):
        raise ValueError('Completed authenticated original native banks required')
    if shutil.disk_usage(root).free<2*1024**3:raise OSError('Need2GiB scratch headroom for anchor export')
    out.mkdir();torch.set_num_threads(2);means=[];counts=[]
    for layer in range(36):
        bank=banks._bank(native/f'boundary_{layer:02d}.npy',p['total_tokens'],p['width'])
        try:m,c=positional_means(bank,p['rows'],p['training_indices'],capacity=p['max_length'],width=p['width'])
        finally:bank._mmap.close()
        means.append(m);counts.append(c)
        print(json.dumps(dict(phase='anchors',completed=layer+1,total=36)),flush=True)
    with (out/'anchors.pt').open('xb')as stream:torch.save(dict(anchors=torch.stack(means),counts=torch.stack(counts)),stream)
    receipt=dict(status='completed',fingerprint=p['fingerprint'],shape=[36,p['max_length'],p['width']],
        fitting_indices=p['training_indices'],source_native_manifest_sha256=sha(native/'manifest.json'),
        source_input_identity={path:identity(path) for path in inputs},
        anchors_sha256=sha(out/'anchors.pt'),anchor_identity=identity(out/'anchors.pt'),
        accumulation_dtype='float64',output_dtype='float32',heldout_groups_excluded=True)
    write(out/'result.json',receipt);return receipt


class SearchDataset:
    def __init__(self,root,*,require_anchors=True):
        self.root=Path(root).resolve();self.plan,self.records=verify(root,live=False)
        p=self.plan;self.rows=p['rows'];self.training_indices=p['training_indices'];self.selection_indices=p['selection_indices']
        self.training=self.training_indices;self.development=self.selection_indices;self.pad_token_id=151643
        self.teacher=banks._bank(Path(p['response_cache_root'])/'native/teacher_hidden.npy',p['total_tokens'],p['width'])
        self.anchors=None
        if require_anchors:
            receipt=read(self.root/'anchors/result.json')
            if receipt.get('status')!='completed' or receipt.get('fingerprint')!=p['fingerprint']:
                raise ValueError('Matching scientific anchors are required')
            if identity(self.root/'anchors/anchors.pt')!=receipt['anchor_identity']:
                raise ValueError('Scientific anchor payload changed')
            saved=torch.load(self.root/'anchors/anchors.pt',map_location='cpu',weights_only=True)
            self.anchors=saved['anchors']
            if self.anchors.shape!=(36,416,2048) or self.anchors.dtype!=torch.float32:
                raise ValueError('Scientific anchor shape/precision differs')

    def batch(self,indices,*,device='cpu'):
        if not indices or any(type(i)is not int or not 0<=i<len(self.records) for i in indices):
            raise ValueError('Admitted complete response indices required')
        records=[self.records[i]for i in indices]
        batch=cache.data.response_batch(records,pad_token_id=self.pad_token_id,max_length=416,device=device)
        hidden=torch.zeros(*batch['input_ids'].shape,self.plan['width'],dtype=torch.bfloat16)
        for j,i in enumerate(indices):
            row=self.rows[i];n=row['stop']-row['offset']
            hidden[j,:n]=torch.from_numpy(np.array(self.teacher[row['offset']:row['stop']],copy=True)).view(torch.bfloat16)
        batch.update(teacher_hidden=hidden.to(device),teacher_input_ids=batch['input_ids'].clone(),
            teacher_record_ids=[r.example_id for r in records],indices=list(indices))
        return batch

    def close(self):
        if not self.teacher._mmap.closed:self.teacher._mmap.close()


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=('prepare','anchors','check'))
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--response-cache-root',type=Path)
    parser.add_argument('--full-data-root',type=Path);args=parser.parse_args()
    if args.phase=='prepare':
        if args.response_cache_root is None or args.full_data_root is None:parser.error('Both native source roots are required')
        result=prepare(args.root,args.response_cache_root,args.full_data_root)
    elif args.phase=='anchors':result=anchors(args.root)
    else:
        p,_=verify(args.root,live=False);result=dict(status='prepared',fingerprint=p['fingerprint'])
        if (args.root/'anchors/result.json').exists():result=read(args.root/'anchors/result.json')
    print(json.dumps({k:result[k]for k in ('status','fingerprint','fitting','selection','distillation','shape','family_counts')if k in result}))


if __name__=='__main__':main()
