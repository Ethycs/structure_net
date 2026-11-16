"""Broad complete-response native boundaries for all36 progressive stages.

Preparation is CPU-only. Capture requires an outer inference-ctl reservation.
The admitted response corpus is unchanged; old multiple-choice DEV is a separate
retention control and cannot enter calibration, feature fitting or recovery.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import mmap
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_3b_response_cache as cache
from . import jspace_3b_rebuild_data as seed_source
from .jspace_3b_response_campaign import training_windows, checkpoint_boundaries

SCHEMA = 'jspace.3b-full-progressive-response-data.v1'
FAMILIES = ('arc', 'boolq', 'dolly', 'gsm8k', 'hellaswag', 'mbpp', 'winogrande', 'xsum')
SEED = 20261116
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/07 - Status Reports/jspace/2026-11-16_qwen3b-full-progressive-compression.md'
SYSTEMS_AMENDMENT = ROOT/'docs/07 - Status Reports/jspace/2026-11-16_qwen3b-progressive-response-rebuild-systems-amendment.md'
read, write, sha, identity, data = cache.read, cache.write, cache.sha, cache.identity, cache.data


def selections(records, *, budget=25000, accumulation=16, seed=SEED):
    """Distinct TRAIN calibration/diagnostics; all response positions, real EOS."""
    cache.index_records(records)
    pools = {family: [] for family in FAMILIES}
    for i, record in enumerate(records):
        family = record.provenance.get('source_family')
        if family not in pools:
            raise ValueError('Only the admitted eight response families are supported')
        if record.split == 'train': pools[family].append(i)
    calibration, diagnostics, alignment, used_groups = [], [], [], set()
    for family in FAMILIES:
        order = sorted(pools[family], key=lambda i: (data.digest([seed, 'geometry', records[i].example_id]),
                                                    records[i].example_id))
        distinct=[]
        for i in order:
            if records[i].group_id not in used_groups:
                distinct.append(i); used_groups.add(records[i].group_id)
            if len(distinct)==52: break
        if len(distinct)<52:
            raise ValueError('Each family needs 16 calibration, 4 diagnostic and 32 local-alignment distinct TRAIN groups')
        calibration.extend(distinct[:16]); diagnostics.extend(distinct[16:20]); alignment.extend(distinct[20:52])
    # Round-robin these independent groups so every accumulation spans families.
    alignment=[alignment[f*32+j] for j in range(32) for f in range(len(FAMILIES))]
    training = training_windows(records, budget=budget, accumulation=accumulation, seed=seed)
    return dict(calibration_indices=calibration, diagnostic_indices=diagnostics,
        local_alignment_indices=alignment,
        local_alignment_windows=[alignment[i:i+accumulation] for i in range(0,len(alignment),accumulation)],
        calibration_response_tokens=sum(len(records[i].response_ids) for i in calibration),
        diagnostic_response_tokens=sum(len(records[i].response_ids) for i in diagnostics),
        training=training, recovery_order=training['order'],
        recovery_response_tokens=training['actual_supervised_tokens'],
        recovery_checkpoints=checkpoint_boundaries(training['windows'], records,
            thresholds=tuple(t for t in (5000, 10000, budget) if t <= budget)),
        supervision='Every response predicting-logit position, including actual terminal EOS; no prompt targets',
        diagnostics_scope='Disjoint TRAIN geometry diagnostics, not untouched DEV or final confirmation')


def capture_items(records, rows):
    """Variable-response capture contract, retaining complete native offsets."""
    if len(records) != len(rows): raise ValueError('Record/offset count differs')
    items = []
    for i, (record, row) in enumerate(zip(records, rows)):
        if row['input_ids'] != list(record.input_ids) or row['record_sha256'] != data.record_fingerprint(record):
            raise ValueError('Response identity differs from the native input bank')
        items.append(dict(row, source_index=i, identity=record.example_id, group_id=record.group_id,
            family=record.provenance['source_family'], task=record.task,
            loss_positions=list(record.prediction_positions), targets=list(record.response_ids),
            response_tokens=len(record.response_ids), eos_token_id=record.eos_token_id))
    return items


def _manifest_files(directory, fingerprint, *, hash_payloads=False):
    directory = Path(directory).resolve(); manifest = read(directory/'manifest.json')
    if manifest.get('status') != 'completed' or manifest.get('fingerprint') != fingerprint:
        raise ValueError('Incomplete or mismatched native cache receipt')
    paths = {str(directory/'manifest.json'): sha(directory/'manifest.json')}
    for name, digest in manifest['artifacts'].items():
        path = directory/name
        if path.parent.resolve() != directory or not path.is_file():
            raise ValueError('Unadmitted cache artifact path')
        if hash_payloads:
            if sha(path) != digest: raise ValueError('Native cache payload changed')
        elif identity(path)!=manifest.get('artifact_identity',{}).get(name):
            raise ValueError('Captured immutable cache file identity changed or is missing')
        paths[str(path)] = digest
    return paths


def _mc_inventory(root):
    """Read only the explicitly exposed old MC inventory; retain DEV only."""
    root = Path(root).resolve(); p = read(root/'plan.json')
    cache.prior.source.check_plan(p)
    if p.get('schema') != cache.prior.SCHEMA: raise ValueError('Expected exposed broad MC cache')
    rows = read(root/'items.json')['items']
    if cache.prior.base.digest(rows) != p['cases_digest']: raise ValueError('Exposed MC inventory changed')
    dev = [row for row in rows if row['split'] == 'development']
    if len(dev) != 256 or Counter(r['family'] for r in dev) != dict.fromkeys(('arc','boolq','hellaswag','winogrande'),64):
        raise ValueError('Expected the exposed 256-case MC development inventory')
    m = read(root/'native/manifest.json'); result = read(root/'native/result.json')
    if (m.get('status') != 'completed' or m.get('fingerprint') != p['fingerprint']
            or m['artifacts'].get('result.json') != sha(root/'native/result.json')
            or result.get('status') != 'completed' or result.get('correct') != 195):
        raise ValueError('Original exposed MC native control is unauthenticated')
    cache.prior.sanity.validate_records(result['records'], dev)
    items, offset = [], 0
    for row in dev:
        items.append(dict(row, old_offset=row['offset'], old_stop=row['stop'],
                          offset=offset, stop=offset+len(row['input_ids'])))
        offset = items[-1]['stop']
    return dict(root=str(root), fingerprint=p['fingerprint'], items=items, total_tokens=offset,
        label_ids=p['label_ids'], native_records=result['records'], native_correct=195,
        scope='Exposed MC DEV retention only; excluded from geometry fitting and recovery'), {
            str(root/name):sha(root/name) for name in ('plan.json','items.json','native/manifest.json','native/result.json')}


def boundary_inventory(cuts):
    cuts=list(cuts)
    if not cuts or any(type(c)is not int or not 0<=c<=35 for c in cuts) or len(set(cuts))!=len(cuts):
        raise ValueError('Distinct original layer cuts0..35 are required')
    cuts=sorted(cuts,reverse=True)
    return cuts,sorted(set(cuts)|{c+1 for c in cuts})


def disk_admission(root, *, total_tokens, mc_tokens, cuts, boundary_cuts, width=2048,
                   reserve_bytes=16*1024**3):
    """Check space before sparse mmap allocation; reserve room for phase receipts."""
    if reserve_bytes<0: raise ValueError('Nonnegative disk reserve required')
    payload=2*width*(total_tokens*len(boundary_cuts)+mc_tokens*len(cuts))
    required=payload+4096*(len(boundary_cuts)+len(cuts))+reserve_bytes
    path=Path(root).resolve()
    while not path.exists(): path=path.parent
    available=shutil.disk_usage(path).free
    if available<required:
        raise OSError(f'Insufficient disk space: need {required} bytes including reserve, available {available}')
    return dict(payload_bytes=payload,reserve_bytes=reserve_bytes,required_bytes=required,
                available_bytes=available,filesystem_path=str(path),passed=True)


def seed_inventory(root, source, records, mc):
    root=Path(root).resolve(); p,old_records=seed_source.verify(root,live=False)
    if (p['model']!=source['model'] or p['response_cache_fingerprint']!=source['fingerprint']
            or p['records_sha256']!=data.dataset_fingerprint(records)
            or p['rows']!=source['rows'] or old_records!=records
            or p['mc'] is None or mc is None or p['mc']!=mc):
        raise ValueError('Existing one-block seed must use exactly the same native model, corpus and MC controls')
    selected=selections(records)
    if any(p[k]!=v for k,v in selected.items()): raise ValueError('Seed and full progression scientific selections differ')
    m=read(root/'native/manifest.json'); r=read(root/'native/result.json')
    if (m.get('status')!='completed' or m.get('fingerprint')!=p['fingerprint']
            or r.get('status')!='completed' or r.get('fingerprint')!=p['fingerprint']
            or not all(r.get(k) for k in ('full_teacher_bit_exact','existing_boundary_bit_exact',
                                          'final_normalization_bit_exact','teacher_unchanged'))):
        raise ValueError('Existing native seed cache is incomplete or failed')
    inputs={str(root/'plan.json'):sha(root/'plan.json'),str(root/'native/manifest.json'):sha(root/'native/manifest.json')}
    for name in ('result.json','boundary_35.npy'):
        actual=sha(root/'native'/name)
        if actual!=m['artifacts'].get(name): raise ValueError('Existing native seed cache payload changed')
        inputs[str(root/'native'/name)]=actual
    return dict(root=str(root),fingerprint=p['fingerprint'],model=p['model'],
        records_sha256=p['records_sha256'],response_cache_fingerprint=p['response_cache_fingerprint'],
        boundary35_sha256=inputs[str(root/'native/boundary_35.npy')],
        mc_inventory_digest=data.digest(mc['items']),quality_admission='Not applied; data identity only'),inputs


def prepare(root, response_cache_root, *, cuts=tuple(range(36)), mc_cache_root=None, seed_data_root=None,
            target_response_tokens=25000, accumulation=16, seed=SEED):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU preparation requires CUDA hidden')
    root, response_cache_root = Path(root).resolve(), Path(response_cache_root).resolve()
    if root.exists(): raise FileExistsError(root)
    if (target_response_tokens,accumulation,seed)!=(25000,16,SEED):
        raise ValueError('Full progression retains the frozen25k/16-example recovery profile and seed')
    cuts,boundary_cuts=boundary_inventory(cuts)
    if seed_data_root is None: raise ValueError('Authenticated v2 native seed data root required')
    source, records = cache.verify(response_cache_root, live=False)
    if source['width'] != 2048 or source['model']['revision'] != 'aa8e72537993ba99e69dfaafa59ed015b17504d1':
        raise ValueError('Pinned original Qwen2.5-3B native cache required')
    inputs = _manifest_files(response_cache_root/'native', source['fingerprint'],hash_payloads=True)
    original_result = read(response_cache_root/'native/result.json')
    if (not original_result.get('teacher_unchanged') or original_result.get('status') != 'completed'
            or not original_result.get('full_vocabulary_parity',{}).get('passed')):
        raise ValueError('Original response teacher controls failed')
    # The old producer hashed model shards and pinned their identity. verify()
    # above authenticated those unchanged files; do not reread 6GB needlessly.
    inputs.update(source['input_sha256'])
    inputs[str(response_cache_root/'plan.json')] = sha(response_cache_root/'plan.json')
    mc = None
    if mc_cache_root is not None:
        mc, mc_inputs = _mc_inventory(mc_cache_root); inputs.update(mc_inputs)
    if mc is None: raise ValueError('Separate exposed MC DEV controls required')
    seed_data,seed_inputs=seed_inventory(seed_data_root,source,records,mc);inputs.update(seed_inputs)
    space=disk_admission(root,total_tokens=source['total_tokens'],mc_tokens=mc['total_tokens'],
        cuts=cuts,boundary_cuts=boundary_cuts,width=source['width'])
    rows = cache.index_records(records)
    p = dict(schema=SCHEMA, response_cache_root=str(response_cache_root),
        response_cache_fingerprint=source['fingerprint'], corpus=source['corpus'], model=source['model'],
        records_sha256=source['records_sha256'], cuts=cuts,
        boundary_cuts=boundary_cuts, existing_boundary_cut=source['cut'],seed_data=seed_data,disk_admission=space,
        rows=rows, width=source['width'], vocabulary=source['vocabulary'], max_length=416,
        total_tokens=source['total_tokens'], items=capture_items(records,rows),
        generation_indices=source['generation_indices'], generation_max_tokens=128,
        generation_eos_token_ids=source['generation_eos_token_ids'], mc=mc,
        input_sha256=inputs, input_identity={path:identity(path) for path in inputs},
        **selections(records,budget=target_response_tokens,accumulation=accumulation,seed=seed),
        confirmed=False, promoted=False)
    files = cache.prior.source.inherited.local_import_closure([Path(__file__), PROTOCOL, SYSTEMS_AMENDMENT,
        ROOT/'tests/structure_net/test_jspace_3b_full_progressive_data.py'])
    p['source_sha256'] = {str(f):sha(f) for f in sorted(files)}
    p['fingerprint'] = data.digest(p)
    root.mkdir(parents=True); write(root/'plan.json',p)
    for path,digest in p['source_sha256'].items():
        target=root/'producer'/Path(path).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
        if sha(target) != digest: raise ValueError('Archived rebuild producer changed')
    return dict(status='prepared',fingerprint=p['fingerprint'],records=len(records),cuts=cuts,
        calibration_count=len(p['calibration_indices']),diagnostic_count=len(p['diagnostic_indices']),
        recovery_examples=len(p['recovery_order']),recovery_response_tokens=p['recovery_response_tokens'],
        new_boundary_bytes=space['payload_bytes'],disk_available_bytes=space['available_bytes'],
        seed_data_fingerprint=seed_data['fingerprint'])


def verify(root, *, live=True):
    root=Path(root).resolve(); p=read(root/'plan.json'); body=dict(p); fp=body.pop('fingerprint')
    if p.get('schema')!=SCHEMA or data.digest(body)!=fp: raise ValueError('Rebuild plan changed')
    if boundary_inventory(p['cuts'])!=(p['cuts'],p['boundary_cuts']): raise ValueError('Native boundary inventory changed')
    for path,old in p['input_identity'].items():
        if identity(path)!=old: raise ValueError('Pinned rebuild input changed: '+path)
    for path,digest in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT))!=digest or live and sha(path)!=digest:
            raise ValueError('Rebuild producer changed')
    source,records=cache.verify(p['response_cache_root'],live=False)
    if (source['fingerprint']!=p['response_cache_fingerprint'] or source['rows']!=p['rows']
            or data.dataset_fingerprint(records)!=p['records_sha256']
            or capture_items(records,p['rows'])!=p['items']):
        raise ValueError('Rebuild corpus or native record offsets changed')
    selected=selections(records,budget=p['training']['nominal_supervised_tokens'],
        accumulation=p['training']['accumulation_examples'],seed=p['training']['seed'])
    if any(p[k]!=v for k,v in selected.items()): raise ValueError('Rebuild TRAIN selection changed')
    return p,records


@contextmanager
def boundary_hooks(model, cuts):
    """Capture exact layer inputs; index36 is the actual pre-final-norm output."""
    captured, handles = {}, []
    def hook(index):
        def save(_module,args):
            if index in captured: raise ValueError('Native boundary executed more than once')
            captured[index]=args[0].detach()
        return save
    try:
        for cut in cuts:
            if not 0 <= cut <= len(model.model.layers): raise ValueError('Invalid native boundary')
            target=model.model.norm if cut==len(model.model.layers) else model.model.layers[cut]
            handles.append(target.register_forward_pre_hook(hook(cut)))
        yield captured
    finally:
        for handle in handles: handle.remove()


def _bits(value, length, width):
    if value.dtype!=torch.bfloat16 or value.shape!=(1,length,width) or not torch.isfinite(value).all():
        raise ValueError('Complete finite BF16 native sequence required')
    return value[0].contiguous().cpu().view(torch.int16).numpy()


def _bank(path, total, width):
    bank=np.load(path,mmap_mode='r')
    if bank.dtype!=np.int16 or bank.shape!=(total,width):
        bank._mmap.close(); raise ValueError('Lossless native bank shape/dtype changed')
    return bank


def boundary_bits(captured,cuts,hidden,length,width):
    """One device-to-host transfer per complete record, including final teacher."""
    values=[captured[cut] for cut in cuts]+[hidden]
    if any(v.dtype!=torch.bfloat16 or v.shape!=(1,length,width) for v in values):
        raise ValueError('Complete BF16 native boundary shapes required')
    joined=torch.stack([v[0] for v in values])
    if not torch.isfinite(joined).all(): raise ValueError('Nonfinite native boundary')
    return joined.contiguous().cpu().view(torch.int16).numpy()


def flush_banks(banks):
    for bank in banks:
        bank.flush()
        if hasattr(bank._mmap,'madvise'): bank._mmap.madvise(mmap.MADV_DONTNEED)


def capture(root, *, limit_mib=11008):
    root=Path(root).resolve(); p,records=verify(root); out=root/'native'
    if out.exists(): raise FileExistsError(out)
    space=disk_admission(root,total_tokens=p['total_tokens'],mc_tokens=p['mc']['total_tokens'],
        cuts=p['cuts'],boundary_cuts=p['boundary_cuts'],width=p['width'],
        reserve_bytes=p['disk_admission']['reserve_bytes'])
    out.mkdir(); start=time.monotonic(); lease=None; banks=[]
    try:
        lease,guard=cache.reservation(limit_mib); torch.set_num_threads(2)
        old=Path(p['response_cache_root'])/'native'
        teacher=_bank(old/'teacher_hidden.npy',p['total_tokens'],p['width'])
        seam=_bank(old/f'boundary_{p["existing_boundary_cut"]:02d}.npy',p['total_tokens'],p['width'])
        seed35=_bank(Path(p['seed_data']['root'])/'native/boundary_35.npy',p['total_tokens'],p['width'])
        banks.extend((teacher,seam,seed35))
        arrays={cut:np.lib.format.open_memmap(out/f'boundary_{cut:02d}.npy',mode='w+',
            dtype=np.int16,shape=(p['total_tokens'],p['width'])) for cut in p['boundary_cuts']}
        banks.extend(arrays.values())
        # Construct ordinary tensors so real parameter/buffer version counters
        # remain available for the frozen-teacher proof. no_grad avoids graphs
        # without inference_mode's removal of those counters (even if a caller
        # entered inference mode before invoking this producer).
        with cache.execution.arithmetic_policy(),torch.inference_mode(False),torch.no_grad():
            model=cache.execution.load_native_model(p['model']['path'],device='cuda:0',attention='eager',guard=guard)
            if model.lm_head.weight is not model.model.embed_tokens.weight: raise ValueError('Original tied readout required')
            frozen=[(id(v),v._version) for v in (*model.parameters(),*model.buffers())]
            requested=sorted(set(p['boundary_cuts'])|{p['existing_boundary_cut'],35})
            cut_positions={cut:i for i,cut in enumerate(requested)}
            with boundary_hooks(model,requested) as captured:
                for i,(record,row) in enumerate(zip(records,p['rows'])):
                    guard(); captured.clear()
                    hidden=model.model(input_ids=torch.tensor([record.input_ids],device='cuda:0'),use_cache=False).last_hidden_state
                    if set(captured)!=set(requested): raise ValueError('Native capture missed a boundary')
                    start_row,stop=row['offset'],row['stop']; length=len(record.input_ids)
                    host=boundary_bits(captured,requested,hidden,length,p['width'])
                    if not np.array_equal(host[-1],teacher[start_row:stop]):
                        raise ValueError('Original full-response teacher replay is not bit exact')
                    if 36 in captured and not torch.equal(model.model.norm.forward(captured[36]),hidden):
                        raise ValueError('Pre-final-norm boundary does not reconstruct original normalized teacher')
                    if not np.array_equal(host[cut_positions[p['existing_boundary_cut']]],seam[start_row:stop]):
                        raise ValueError('Original native-prefix seam replay is not bit exact')
                    if not np.array_equal(host[cut_positions[35]],seed35[start_row:stop]):
                        raise ValueError('Existing v2 cut35 native boundary replay is not bit exact')
                    for cut,bank in arrays.items(): bank[start_row:stop]=host[cut_positions[cut]]
                    if (i+1)%32==0 or i+1==len(records):
                        flush_banks(arrays.values())
                        print(json.dumps(dict(phase='broad_rebuild_native_boundaries',completed=i+1,total=len(records),
                            cuts=p['boundary_cuts'],seconds=time.monotonic()-start)),flush=True)
                mc_records=[]
                if p['mc'] is not None:
                    mc=p['mc']; mc_arrays={cut:np.lib.format.open_memmap(out/f'mc_boundary_{cut:02d}.npy',mode='w+',
                        dtype=np.int16,shape=(mc['total_tokens'],p['width'])) for cut in p['cuts']}
                    banks.extend(mc_arrays.values())
                    for i,row in enumerate(mc['items']):
                        guard(); captured.clear()
                        hidden=model.model(input_ids=torch.tensor([row['input_ids']],device='cuda:0'),use_cache=False).last_hidden_state
                        result=cache.prior.sanity._record(row,hidden[:,-1],model.lm_head.weight,mc['label_ids'])
                        if result!=mc['native_records'][i]: raise ValueError('Exposed MC original native control replay differs')
                        mc_records.append(result)
                        host=boundary_bits(captured,p['cuts'],hidden,len(row['input_ids']),p['width'])
                        for position,(cut,bank) in enumerate(mc_arrays.items()):
                            bank[row['offset']:row['stop']]=host[position]
                        if (i+1)%32==0: flush_banks(mc_arrays.values())
                    write(out/'mc_development.json',dict(records=mc_records,count=len(mc_records),
                        correct=sum(r['prediction']==r['answer'] for r in mc_records),native_control_bit_exact=True,
                        scope=mc['scope']))
            if frozen!=[(id(v),v._version) for v in (*model.parameters(),*model.buffers())]:
                raise ValueError('Frozen original teacher changed')
            guard()
        for bank in banks: bank.flush(); bank._mmap.close()
        banks=[]
        result=dict(status='completed',fingerprint=p['fingerprint'],records=len(records),
            full_teacher_bit_exact=True,existing_boundary_bit_exact=True,teacher_unchanged=True,
            seed_boundary35_bit_exact=True,seed_data_fingerprint=p['seed_data']['fingerprint'],
            final_normalization_bit_exact=(36 in p['boundary_cuts']),
            mc_count=len(mc_records),cuts=p['boundary_cuts'],lease_id=lease['id'],allocation=lease['allocation'],
            allocator_mib=limit_mib,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            seconds=time.monotonic()-start,storage='Exact BF16 bits, full input positions; boundary36 is pre-final-norm',
            disk_admission=space,transfer='One stacked boundary transfer per record; flush and evict every32records',
            confirmed=False)
        write(out/'result.json',result)
        artifacts={f.name:sha(f) for f in out.iterdir() if f.is_file()}
        write(out/'manifest.json',dict(status='completed',fingerprint=p['fingerprint'],artifacts=artifacts,
            artifact_identity={name:identity(out/name) for name in artifacts},
            admission='Content hashed once at capture completion; subsequent readers pin manifest and unchanged file identities'))
        return result
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',error=repr(error),lease=lease,seconds=time.monotonic()-start))
        raise
    finally:
        for bank in banks:
            if not bank._mmap.closed: bank.flush(); bank._mmap.close()


class RebuildDataset:
    """Disk-backed native geometry/teacher targets; no CPU corpus-sized tensors."""
    def __init__(self,root,*,cut,pad_token_id=151643):
        self.root=Path(root).resolve(); self.plan,self.records=verify(root,live=False)
        if cut not in self.plan['cuts']: raise ValueError('Boundary cut was not admitted')
        _manifest_files(self.root/'native',self.plan['fingerprint'])
        result=read(self.root/'native/result.json')
        if not all(result.get(k) for k in ('full_teacher_bit_exact','existing_boundary_bit_exact',
                                         'teacher_unchanged','seed_boundary35_bit_exact')):
            raise ValueError('Native teacher/boundary seam controls failed')
        self.cut,self.pad_token_id=cut,pad_token_id; self.rows=self.plan['rows']
        self.items=self.plan['items']; self.width=self.plan['width']
        self.training=[i for i,r in enumerate(self.records) if r.split=='train']
        self.development=[i for i,r in enumerate(self.records) if r.split=='development']
        self.teacher=_bank(Path(self.plan['response_cache_root'])/'native/teacher_hidden.npy',self.plan['total_tokens'],self.width)
        self.boundary=_bank(self.root/'native'/f'boundary_{cut:02d}.npy',self.plan['total_tokens'],self.width)
        self.next_boundary=_bank(self.root/'native'/f'boundary_{cut+1:02d}.npy',self.plan['total_tokens'],self.width)

    def batch(self,indices,*,device='cpu',dtype=torch.bfloat16):
        if dtype!=torch.bfloat16: raise ValueError('Exact BF16 target profile required')
        if not indices or any(type(i)is not int or not 0<=i<len(self.records) for i in indices):
            raise ValueError('Nonempty admitted record indices required')
        records=[self.records[i] for i in indices]
        batch=data.response_batch(records,pad_token_id=self.pad_token_id,max_length=416,device=device)
        for name,bank in (('teacher_hidden',self.teacher),('boundary_hidden',self.boundary),
                          ('next_boundary_hidden',self.next_boundary)):
            target=torch.zeros((*batch['input_ids'].shape,self.width),dtype=dtype)
            for j,i in enumerate(indices):
                row=self.rows[i]
                value=torch.from_numpy(np.array(bank[row['offset']:row['stop']],copy=True)).view(torch.bfloat16)
                target[j,:len(value)]=value
            batch[name]=target.to(device)
        batch.update(indices=list(indices),teacher_input_ids=batch['input_ids'].clone(),
            teacher_record_ids=[r.example_id for r in records])
        return batch

    def close(self):
        for bank in (self.teacher,self.boundary,self.next_boundary):
            if not bank._mmap.closed: bank._mmap.close()


class MCDataset:
    """Separate old exposed DEV retention controls; deliberately no TRAIN rows."""
    def __init__(self,root,*,cut,pad_token_id=151643):
        self.root=Path(root).resolve(); self.plan,_=verify(root,live=False)
        if self.plan['mc'] is None or cut not in self.plan['cuts']: raise ValueError('MC boundary not admitted')
        _manifest_files(self.root/'native',self.plan['fingerprint'])
        mc=self.plan['mc']; self.items=mc['items']; self.label_ids=mc['label_ids']
        self.native_records=mc['native_records']; self.native_correct=mc['native_correct']
        self.training=[]; self.development=list(range(len(self.items))); self.width=self.plan['width']
        self.cut,self.pad_token_id=cut,pad_token_id
        self.boundary=_bank(self.root/'native'/f'mc_boundary_{cut:02d}.npy',mc['total_tokens'],self.width)

    def batch(self,indices,*,device='cpu',dtype=torch.bfloat16):
        if dtype!=torch.bfloat16 or not indices or any(i not in self.development for i in indices):
            raise ValueError('Only admitted MC DEV BF16 batches are supported')
        rows=[self.items[i] for i in indices]; length=max(len(r['input_ids']) for r in rows)
        ids=torch.full((len(rows),length),self.pad_token_id,dtype=torch.long)
        valid=torch.zeros_like(ids,dtype=torch.bool); hidden=torch.zeros((len(rows),length,self.width),dtype=dtype)
        for i,row in enumerate(rows):
            n=len(row['input_ids']); ids[i,:n]=torch.tensor(row['input_ids']); valid[i,:n]=True
            hidden[i,:n]=torch.from_numpy(np.array(self.boundary[row['offset']:row['stop']],copy=True)).view(torch.bfloat16)
        return dict(input_ids=ids.to(device),attention_mask=valid.to(device),boundary_hidden=hidden.to(device))

    def close(self):
        if not self.boundary._mmap.closed: self.boundary._mmap.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','capture','check')); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--response-cache-root',type=Path); parser.add_argument('--mc-cache-root',type=Path)
    parser.add_argument('--seed-data-root',type=Path,required=False)
    parser.add_argument('--cuts',nargs='+',type=int,default=list(range(36))); parser.add_argument('--limit-mib',type=int,default=11008)
    parser.add_argument('--target-response-tokens',type=int,default=25000)
    args=parser.parse_args()
    if args.phase=='prepare': result=prepare(args.root,args.response_cache_root,cuts=args.cuts,
        mc_cache_root=args.mc_cache_root,seed_data_root=args.seed_data_root,target_response_tokens=args.target_response_tokens)
    elif args.phase=='capture': result=capture(args.root,limit_mib=args.limit_mib)
    else:
        p,records=verify(args.root,live=False); result=dict(fingerprint=p['fingerprint'],records=len(records),cuts=p['cuts'])
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__': main()
