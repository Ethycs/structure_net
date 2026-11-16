"""Reserved block39 Phi-4 pilot capture -> basis -> interface -> relation.

Consumes the separately executed prompt-only decoded-GPTQ prefix bank; it does
not score heldout data, train a student, or claim full-model compression. The
distinct full-response bank remains pinned for training, never substituted for
the prompt-only BF16 execution used in calibration and task evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from . import jspace_phi_data as data
from . import jspace_phi_weights as weights
from . import jspace_phi_native as native_reference
from . import jspace_qwen_parallel_fit as fitting
from . import jspace_allblocks_rankfit as rankfit
from .jspace_bench_data import sha, write
from .jspace_phi_affine_block import AffinePhiBlock, validate_phi4_config
from .jspace_phi_capture import PhiGroupedTeacherCapture, capture_compiled_development
from .jspace_full_lease import wait_ready
from .jspace_3b_staged_campaign import local_import_closure

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'jspace.phi-fit.v1'
PHASES = ('capture', 'basis', 'interface', 'relation')
SETTINGS = dict(layer=39, rank=2560, head_dim=64, anchors=24, remove_count=1664,
    pad_to_multiple=128, probe_groups=2, probes_per_group=8, activation_positions=16,
    position_capacity=416, seed=20861, relation_seed=861, svd_rank=64,
    oversample=16, power_iterations=1, ridge=.01, eligibility_nrmse=.5, chunk_rows=256,
    allocation_mib=11264, allocator_mib=11008)
read = lambda path: json.loads(Path(path).read_text())


def digest(value):
    return data.digest(value)


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def validate_items(items, source_items, *, production=True):
    if not isinstance(items, list) or len(items) != len(source_items) or not items:
        raise ValueError('Native/Phi data item populations differ')
    seen, cursor, prompt_cursor = set(), 0, 0
    for row, source in zip(items, source_items):
        if row['identity'] in seen: raise ValueError('Duplicate native case identity')
        seen.add(row['identity'])
        for key in ('identity', 'family', 'split', 'input_ids', 'choices', 'answer', 'loss_positions', 'targets'):
            if row.get(key) != source.get(key): raise ValueError('Native case differs from Phi pilot data: ' + key)
        length = len(row['input_ids'])
        if (row['split'] not in ('train', 'development') or not 1 <= length <= 384 or
                row.get('offset') != cursor or row.get('stop') != cursor + length or
                row.get('prompt_offset') != prompt_cursor or row.get('prompt_stop') != prompt_cursor + length or
                type(row.get('full_stop')) is not int or not row['stop'] < row['full_stop'] <= cursor + 416 or
                len(row['loss_positions']) != 4 or any(type(p) is not int or not 0 <= p < length for p in row['loss_positions'])):
            raise ValueError('Invalid prompt/full-response offsets or positions')
        cursor = row['full_stop']
        prompt_cursor = row['prompt_stop']
    if production:
        expected = Counter({(split, family): count for split, count in (('train',32),('development',16))
                            for family in ('arc','boolq','hellaswag','winogrande')})
        if Counter((c['split'], c['family']) for c in items) != expected:
            raise ValueError('Phi pilot requires 128 TRAIN and 64 DEV, balanced across four families')
    return cursor


def _native(root):
    directory = Path(root) / 'native'
    manifest = native_reference.checked(root,hash_payload=False)
    if (manifest.get('schema') != 'jspace.phi-native.v1' or manifest.get('status') != 'completed'
            or not manifest.get('fingerprint') or not {'prefix.npy','prompt_prefix.npy','items.json'} <= set(manifest.get('artifacts',{}))):
        raise ValueError('Completed authenticated native Phi prefix capture required')
    for name, expected in manifest['artifacts'].items():
        target = (directory/name).resolve()
        if not target.is_relative_to(directory.resolve()) or sha(target) != expected:
            raise ValueError('Native Phi artifact changed: ' + name)
    return manifest, read(directory/'items.json')


def prepare(root, native_root, data_root, model=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU prepare requires CUDA hidden')
    root, native_root, data_root = map(lambda p: Path(p).resolve(), (root,native_root,data_root))
    if root.exists(): raise FileExistsError(root)
    dp = data.verify(data_root, live=True); native, items = _native(native_root)
    native_plan=read(native_root/'plan.json')
    if (native_plan['model']!=dp['model'] or native_plan['data_fingerprint']!=dp['fingerprint'] or
            Path(native_plan['data_root']).resolve()!=data_root or native['fingerprint']!=native_plan['fingerprint']):
        raise ValueError('Native source model/data identity differs from requested Phi pilot')
    model = Path(model or dp['model']['path']).resolve()
    if str(model) != dp['model']['path']: raise ValueError('Model path differs from frozen Phi data')
    config = weights.load_config(model); validate_phi4_config(config)
    tokens = validate_items(items, read(data_root/'pilot_items.json')['items'])
    prefix = native_root/'native/prefix.npy'
    bank = np.load(prefix, mmap_mode='r', allow_pickle=False)
    try:
        if bank.dtype != np.dtype('int16') or not bank.dtype.isnative or bank.shape != (tokens,5120):
            raise ValueError('Native prefix bank must be lossless BF16 bits [full sequence tokens,5120]')
    finally: bank._mmap.close()
    prompt_prefix=native_root/'native/prompt_prefix.npy'
    bank=np.load(prompt_prefix,mmap_mode='r',allow_pickle=False)
    try:
        if bank.dtype!=np.dtype('int16') or not bank.dtype.isnative or bank.shape!=(items[-1]['prompt_stop'],5120):
            raise ValueError('Native prompt prefix must be separately executed lossless BF16 [prompt tokens,5120]')
    finally:bank._mmap.close()
    pins = {str(native_root/'native/manifest.json'):sha(native_root/'native/manifest.json'),
            str(data_root/'manifest.json'):sha(data_root/'manifest.json'),
            str(data_root/'plan.json'):sha(data_root/'plan.json'),
            str(data_root/'pilot_items.json'):sha(data_root/'pilot_items.json')}
    for name, value in native['artifacts'].items(): pins[str((native_root/'native'/name).resolve())] = value
    for name, value in dp['model']['files'].items(): pins[str(model/name)] = value
    pins[str(native_root/'plan.json')] = sha(native_root/'plan.json')
    files = local_import_closure([Path(__file__),ROOT/'tests/structure_net/test_jspace_phi_fit.py'])
    p = dict(schema=SCHEMA, settings=SETTINGS, model=str(model), native_root=str(native_root), data_root=str(data_root),
        native_fingerprint=native['fingerprint'], data_fingerprint=dp['fingerprint'],
        items_sha256=sha(native_root/'native/items.json'), prefix_sha256=sha(prefix), total_full_tokens=tokens,
        prompt_prefix_sha256=sha(prompt_prefix),total_prompt_tokens=items[-1]['prompt_stop'],
        source_kind=weights.SOURCE_KIND, scope='one Phi-4 block39 pilot; original prompt-only task-Jacobian estimator',
        confirmed=False, input_sha256=pins, input_identity={name:identity(name) for name in pins},
        source_sha256={str(f):sha(f) for f in sorted(files)})
    p['fingerprint'] = digest(p); root.mkdir(parents=True)
    for name, expected in p['source_sha256'].items():
        target=root/'producer'/Path(name).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(name,target)
        if sha(target)!=expected: raise ValueError('Producer archive differs')
    write(root/'plan.json',p)
    return dict(status='prepared',fingerprint=p['fingerprint'],source_kind=p['source_kind'])


def verify(root):
    root=Path(root);p=read(root/'plan.json')
    if (p.get('schema')!=SCHEMA or p.get('settings')!=SETTINGS or p.get('confirmed')is not False or
            p.get('fingerprint')!=digest({k:v for k,v in p.items() if k!='fingerprint'})):
        raise ValueError('Phi fitting plan changed')
    # Large source payloads were SHA-admitted at prepare. Every phase checks
    # the frozen file identities; newly consumed phase artifacts are SHA-read.
    for name, old in p['input_identity'].items():
        if identity(name)!=old: raise ValueError('Authenticated Phi input identity changed: '+name)
    for name, expected in p['source_sha256'].items():
        if sha(name)!=expected or sha(root/'producer'/Path(name).relative_to(ROOT))!=expected:
            raise ValueError('Fitting source/archive changed')
    return p


def checked(root, phase, plan=None):
    root=Path(root);p=plan or verify(root)
    if phase not in PHASES: raise ValueError('Unknown Phi fit phase')
    receipt=read(root/(phase+'_receipt.json'))
    if (receipt.get('schema')!=SCHEMA or receipt.get('phase')!=phase or receipt.get('status')!='completed'
            or receipt.get('fingerprint')!=p['fingerprint'] or not receipt.get('artifacts')):
        raise ValueError('Incomplete or mismatched Phi fitting receipt')
    allocation=receipt.get('allocation',{});cards=list(allocation);allocator=receipt.get('allocator',{})
    if (len(cards)!=1 or not cards[0].startswith('GPU-0370fdb0') or allocation[cards[0]]!=11264 or
            receipt.get('peak_reserved_bytes',float('inf'))>SETTINGS['allocator_mib']*1024**2 or
            allocator.get('allocator_bytes')!=SETTINGS['allocator_mib']*1024**2 or
            allocator.get('lease_bytes')!=11264*1024**2 or allocator.get('physical_uuid')!=cards[0]):
        raise ValueError('Phi fitting resource receipt differs')
    previous={prior:sha(root/(prior+'_receipt.json')) for prior in PHASES[:PHASES.index(phase)]}
    if receipt.get('previous_receipts')!=previous:raise ValueError('Phi phase predecessor receipt changed')
    for name, expected in receipt['artifacts'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root.resolve()) or sha(target)!=expected:
            raise ValueError('Phi phase artifact changed: '+name)
    return receipt


def block_arguments(x, rotary):
    positions=torch.arange(x.shape[1],device=x.device)[None]
    mask=torch.full((1,1,x.shape[1],x.shape[1]),torch.finfo(x.dtype).min,device=x.device,dtype=x.dtype).triu(1)
    return dict(attention_mask=mask,position_ids=positions,cache_position=positions[0],
                position_embeddings=rotary(x,positions),use_cache=False)


class PrefixTeacher(nn.Module):
    """Read the separately executed prompt-only native bank at prompt offsets."""
    def __init__(self, items, prefix_path, block, norm, readout, rotary, *, guard):
        super().__init__(); guard()
        self.items,self.guard=list(items),guard
        layer=block.self_attn.layer_idx
        self.model=nn.Module();self.model.layers=nn.ModuleList([nn.Identity() for _ in range(layer)]+[block])
        self.model.norm=norm;self.rotary=rotary;self.layer=layer
        self.lm_head=nn.Module();self.lm_head.register_buffer('weight',readout)
        if any(p.requires_grad for p in self.parameters()) or any(m.training for m in (block,norm,rotary)):
            raise ValueError('Phi prefix teacher must already be frozen/eval')
        self.eval();self.bank=np.load(prefix_path,mmap_mode='r',allow_pickle=False)
        if self.bank.dtype!=np.dtype('int16') or self.bank.shape!=(items[-1]['prompt_stop'],readout.shape[1]):
            self.close();raise ValueError('Native Phi prefix shape/precision differs')

    def forward_case(self,index):
        self.guard();c=self.items[index]
        bits=np.array(self.bank[c['prompt_offset']:c['prompt_stop']],copy=True)
        x=torch.from_numpy(bits).view(torch.bfloat16).to(self.lm_head.weight.device)[None]
        if x.shape[1]!=len(c['input_ids']) or not torch.isfinite(x).all():raise ValueError('Invalid prompt boundary')
        value=self.model.layers[self.layer](x,**block_arguments(x,self.rotary))
        return self.model.norm(value[0] if isinstance(value,tuple) else value)

    def close(self):
        if getattr(self,'bank',None) is not None:self.bank._mmap.close();self.bank=None


def build_block(fit_root,native,trainable=True):
    """Canonical BF16 learned state and FP32 geometry for train/reload."""
    path=Path(fit_root)/'basis/block_39'
    fit=torch.load(path/'fit.pt',map_location='cpu',weights_only=True)
    relation=torch.load(path/'relation.pt',map_location='cpu',weights_only=True)
    device=native.self_attn.qkv_proj.weight.device
    if (native.self_attn.layer_idx!=39 or fit['rank']!=2560 or fit['head_dim']!=64 or
            len(relation['anchors'])!=24 or len(relation['removed'])!=1664):
        raise ValueError('Phi block39 fitting geometry/budget differs')
    block=AffinePhiBlock(native,fit['basis'][:,:fit['rank']].to(device),fit['means'].to(device),
        trainable=trainable,parameter_dtype=torch.bfloat16,buffer_dtype=torch.float32,
        head_projection={k:fit[k].to(device) for k in ('rope_pairs','value_basis')})
    block.compile_polynomial(relation,relation['removed'])
    for parameter in block.parameters():parameter.data=parameter.data.to(torch.bfloat16)
    return block


def allocator_settings(lease, *, cuda=None, visible=None):
    cuda=cuda or torch.cuda;visible=os.environ.get('CUDA_VISIBLE_DEVICES','') if visible is None else visible
    allocation=lease.get('allocation',{});cards=list(allocation)
    if (len(cards)!=1 or not cards[0].startswith('GPU-0370fdb0') or allocation[cards[0]]!=11264 or
            not visible.startswith('GPU-0370fdb0') or ',' in visible):
        raise ValueError('Phi fit requires exactly the leased 3060 at11264MiB')
    properties=cuda.get_device_properties(0);limit=11008*1024**2
    if properties.total_memory<=limit:raise ValueError('Device memory below Phi allocator limit')
    cuda.set_per_process_memory_fraction(limit/properties.total_memory,0)
    cuda.set_device(0);cuda.reset_peak_memory_stats(0)
    return dict(physical_uuid=cards[0],allocator_bytes=limit,lease_bytes=11264*1024**2)


def _execute(root,p,phase,device,guard,progress):
    root=Path(root);s=p['settings'];path=root/'basis/block_39'
    if phase=='capture':
        items=read(Path(p['native_root'])/'native/items.json')
        block=weights.load_native_block(p['model'],39,device,guard=guard)
        head,norm,rotary=weights.load_readout_norm(p['model'],device,guard=guard)
        teacher=PrefixTeacher(items,Path(p['native_root'])/'native/prompt_prefix.npy',block,norm,head,rotary,guard=guard)
        calibration=[i for i,c in enumerate(items) if c['split']=='train']
        try:
            with PhiGroupedTeacherCapture(root/'capture',teacher,[39],items,calibration,
                    teacher_fingerprint=p['native_fingerprint'],probe_groups=2,probes_per_group=8,
                    activation_positions=16,position_capacity=416,seed=20861,guard=guard) as recorder:
                for i in range(len(items)):
                    recorder.capture_case(i,lambda index=i:teacher.forward_case(index))
                    progress(dict(phase='capture_case',completed=i+1,total=len(items)))
                return recorder.finish()
        finally:teacher.close()
    if phase=='basis':
        return fitting.finalize_basis(root/'capture',root/'basis',layer_ids=[39],rank=2560,head_dim=64,
                                      device=device,guard=guard,progress=progress)
    if phase=='interface':
        block=weights.load_native_block(p['model'],39,device,guard=guard)
        rotary=weights.load_rotary(p['model'],device,guard=guard)
        return capture_compiled_development(path,root/'capture',39,block,
            lambda compiled,x,case:compiled(x,**block_arguments(x,rotary)),device=device,guard=guard,progress=progress)
    return rankfit.fit_relation(path,device=device,guard=guard,progress=progress,
        seed=s['relation_seed'],anchors=s['anchors'],remove_count=s['remove_count'],pad_to_multiple=s['pad_to_multiple'],
        svd_rank=s['svd_rank'],oversample=s['oversample'],power_iterations=s['power_iterations'],ridge=s['ridge'],
        eligibility_nrmse=s['eligibility_nrmse'],chunk_rows=s['chunk_rows'])


def run(root,phase):
    root=Path(root);p=verify(root)
    if phase not in PHASES:raise ValueError('Unknown Phi fitting phase')
    receipt=root/(phase+'_receipt.json')
    if receipt.exists():return checked(root,phase,p)
    if (root/(phase+'_failure.json')).exists():raise ValueError('Failed phase preserved; use a new root')
    previous={}
    for prior in PHASES[:PHASES.index(phase)]:
        checked(root,prior,p);previous[prior]=sha(root/(prior+'_receipt.json'))
    expected={'capture':[root/'capture'],'basis':[root/'basis'],
        'interface':[root/'basis/block_39'/n for n in ('interface.json','compiled_dev.npy','output_weight.pt')],
        'relation':[root/'basis/block_39'/n for n in ('relation.pt','relations.json','rank_spectrum.pt','rank_diagnostic.json')]}[phase]
    if any(path.exists() for path in expected):raise ValueError('Incomplete phase output exists; no overwrite')
    lease=settings=None;started=time.monotonic()
    try:
        lease=wait_ready([0]);settings=allocator_settings(lease)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        def guard():
            current=wait_ready([0])
            if current['id']!=lease['id'] or current['allocation']!=lease['allocation']:
                raise RuntimeError('Phi fitting reservation changed')
        def progress(value):
            print(json.dumps(dict(kind='phi_fit_progress',stage=phase,**value),allow_nan=False),flush=True)
        before={str(f.relative_to(root)) for d in ('capture','basis') for f in (root/d).rglob('*') if f.is_file()}
        with torch.cuda.device(0):result=_execute(root,p,phase,'cuda:0',guard,progress)
        guard();torch.cuda.synchronize(0)
        peak=torch.cuda.max_memory_reserved(0)
        if peak>settings['allocator_bytes']:raise RuntimeError('Phi fitting exceeded allocator budget')
        artifacts={str(f.relative_to(root)):sha(f) for d in ('capture','basis') for f in (root/d).rglob('*')
                   if f.is_file() and str(f.relative_to(root)) not in before}
        value=dict(schema=SCHEMA,status='completed',phase=phase,fingerprint=p['fingerprint'],
            previous_receipts=previous,artifacts=artifacts,lease_id=lease['id'],allocation=lease['allocation'],
            allocator=settings,peak_reserved_bytes=peak,elapsed_seconds=time.monotonic()-started,
            source_kind=p['source_kind'],confirmed=False)
        write(receipt,value)
        return value
    except BaseException as error:
        write(root/(phase+'_failure.json'),dict(status='failed',phase=phase,fingerprint=p['fingerprint'],
            error=f'{type(error).__name__}: {error}',lease=lease,allocator=settings,elapsed_seconds=time.monotonic()-started))
        raise
    finally:gc.collect()


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--root',required=True);prep.add_argument('--native-root',required=True)
    prep.add_argument('--data-root',required=True);prep.add_argument('--model')
    launch=commands.add_parser('run');launch.add_argument('--root',required=True);launch.add_argument('--phase',choices=PHASES,required=True)
    args=parser.parse_args();result=prepare(args.root,args.native_root,args.data_root,args.model) if args.command=='prepare' else run(args.root,args.phase)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
