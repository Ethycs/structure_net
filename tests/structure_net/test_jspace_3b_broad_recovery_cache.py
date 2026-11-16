"""Lossless teacher-slot alignment and bounded paired-cache controls."""
import json

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_3b_broad_recovery_cache as study
from test_jspace_3b_sanity_evaluate import tiny, arguments


def items():
    rows=[];offset=0
    for i,ids in enumerate(([1,4,3],[2,3,7,4,5])):
        n=len(ids);gold=i%2
        rows.append(dict(identity=f'case{i}',family='boolq',split='train' if i==0 else 'development',
            input_ids=ids,choices=['yes','no'],answer=gold,offset=offset,stop=offset+n,
            legacy_loss_positions=[0,0,n-2,n-1],content_loss_positions=[1,n-2,1,n-1],
            legacy_targets=[ids[1],ids[1],ids[-1],[3,8][gold]],
            content_targets=[ids[2],ids[-1],ids[2],[3,8][gold]]))
        offset+=n
    return rows


def cache_files(tmp_path):
    rows=items();h=torch.arange(8*8).reshape(8,8).bfloat16()/8
    logits=(torch.arange(2*8*16).reshape(2,8,16)%29).bfloat16()
    for name,value in [('boundary',h),('native',logits),('parent',logits[:,:4]+100)]:
        np.save(tmp_path/f'{name}.npy',value.contiguous().view(torch.int16).numpy())
    return rows,h,logits


def test_raw_data_rows_canonicalize_for_frozen_parent_feature_metadata():
    raw=items()
    for c in raw:
        for key in ('offset','stop'):c.pop(key)
    rows=study.canonical_items(raw)
    assert all('offset' not in c and 'loss_positions' not in c for c in raw)
    assert study.validate_items(rows,[3,8])==8
    expected=study.feature.target_rows(rows)
    assert len(expected)==1 and expected[0]['identity']=='case0'
    for actual,source in zip(rows,raw):
        assert actual['loss_positions']==source['legacy_loss_positions']
        assert actual['targets']==source['legacy_targets']
        assert actual['content_loss_positions']==source['content_loss_positions']


@pytest.mark.parametrize('arm,start', [('parent_legacy',0),('native_legacy',0),('native_content',4)])
def test_actual_slot_views_preserve_bf16_rows_targets_and_causal_padding(tmp_path,arm,start):
    rows,hidden,logits=cache_files(tmp_path)
    name='parent' if arm=='parent_legacy' else 'native'
    dataset=study.RecoveryDataset(rows,tmp_path/'boundary.npy',tmp_path/f'{name}.npy',arm=arm,label_ids=[3,8])
    try:
        batch=dataset.batch([1,0],device='cpu')
        wanted=logits[[1,0],start:start+4]+(100 if arm=='parent_legacy' else 0)
        assert torch.equal(batch['teacher'],wanted)
        assert batch['teacher'].dtype==torch.bfloat16 and batch['lengths']==[5,3]
        assert torch.equal(batch['hidden'][0],hidden[3:])
        assert torch.equal(batch['hidden'][1,:3],hidden[:3]) and torch.count_nonzero(batch['hidden'][1,3:])==0
        kind='content' if arm=='native_content' else 'legacy'
        assert batch['loss_positions'].tolist()==[rows[i][kind+'_loss_positions'] for i in (1,0)]
        assert batch['targets'].tolist()==[rows[i][kind+'_targets'] for i in (1,0)]
        for b,n in enumerate([5,3]):
            for q in range(5):
                for k in range(5):
                    assert (batch['attention_mask'][b,0,q,k]==0).item()==(k<=q and k<n)
        assert dataset.training==[0] and dataset.development==[1]
        assert 'original' not in dataset.batch([0],include_original=True)
    finally:dataset.close()
    assert dataset.boundary._mmap.closed and dataset._logits._mmap.closed


@pytest.mark.parametrize('defect', ['heldout','offset','duplicate','position','gold'])
def test_invalid_rows_cannot_enter_teacher_cache(defect):
    rows=items()
    if defect=='heldout':rows[0]['split']='evaluation'
    elif defect=='offset':rows[1]['offset']-=1
    elif defect=='duplicate':rows[1]['identity']=rows[0]['identity']
    elif defect=='position':rows[0]['content_loss_positions'][-1]=0
    else:rows[0]['content_targets'][-1]=8
    with pytest.raises(ValueError):study.validate_items(rows,[3,8])


def test_cache_rejects_wrong_slot_inventory_and_nonfinite_values(tmp_path):
    rows,_,logits=cache_files(tmp_path)
    with pytest.raises(ValueError,match='shape'):
        study.RecoveryDataset(rows,tmp_path/'boundary.npy',tmp_path/'parent.npy',arm='native_content',label_ids=[3,8])
    logits[0,4,0]=float('nan');np.save(tmp_path/'bad.npy',logits.view(torch.int16).numpy())
    dataset=study.RecoveryDataset(rows,tmp_path/'boundary.npy',tmp_path/'bad.npy',arm='native_content',label_ids=[3,8])
    try:
        with pytest.raises(FloatingPointError):dataset.batch([0])
    finally:dataset.close()


@torch.inference_mode()
def test_actual_native_capture_matches_full_vocab_and_before_block_boundaries(tiny):
    native,_,_,_,labels=tiny
    for case in items():
        boundaries,logits,row=study.native_outputs(native,case,labels,cuts=(1,2))
        ids=torch.tensor([case['input_ids']]);hidden=native.model.embed_tokens(ids)
        kwargs=arguments(hidden,native.model.rotary_emb)
        expected={}
        for layer,block in enumerate(native.model.layers):
            if layer in (1,2):expected[layer]=hidden.clone()
            hidden=block(hidden,**kwargs)[0]
        normalized=native.model.norm(hidden)
        positions=case['legacy_loss_positions']+case['content_loss_positions']
        manual=torch.nn.functional.linear(normalized[:,positions],native.lm_head.weight)[0]
        assert logits.shape==(8,16) and logits.dtype==torch.bfloat16 and logits.device.type=='cpu'
        assert torch.equal(logits,manual) and torch.equal(logits[3],logits[7])
        assert all(torch.equal(boundaries[k],expected[k]) for k in expected)
        assert row==study.sanity._record(case,normalized[:,-1],native.lm_head.weight,labels)
        assert not any(block._forward_pre_hooks for block in native.model.layers)


def test_native_capture_denied_guard_precedes_forward(tiny):
    native,*_=tiny
    def denied():raise RuntimeError('no lease')
    with pytest.raises(RuntimeError,match='lease'):
        study.native_outputs(native,items()[0],[3,8],cuts=(1,2),guard=denied)
    assert not any(block._forward_pre_hooks for block in native.model.layers)


def test_parent_phase_requires_completed_native_receipt_before_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'verify',lambda *_:{'fingerprint':'fixture'})
    def denied(*_):raise ValueError('native incomplete')
    monkeypatch.setattr(study,'checked',denied)
    monkeypatch.setattr(study,'wait_ready',lambda *_:pytest.fail('lease requested before native admission'))
    with pytest.raises(ValueError,match='native incomplete'):study.run(tmp_path,'parent')


def test_native_phase_denied_reservation_never_initializes_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'verify',lambda *_:{'fingerprint':'fixture'})
    def denied(*_):raise RuntimeError('lease denied')
    monkeypatch.setattr(study,'wait_ready',denied)
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('CUDA initialized'))
    with pytest.raises(RuntimeError,match='lease denied'):study.run(tmp_path,'native')


def test_completed_native_receipt_binds_bank_hash_control_counts_and_resource(tmp_path):
    rows=items();directory=tmp_path/'native';directory.mkdir()
    study.base.write(tmp_path/'items.json',{'items':rows})
    record=dict(identity=rows[1]['identity'],family='boolq',answer=1,prediction=1,logits=[0.,1.])
    result=dict(status='completed',records=[record],count=1,correct=1,fixed_state_unchanged=True,
        boundary_controls={str(i):dict(passed=True,bit_exact=True) for i in (22,23)})
    (directory/'boundary_22.npy').write_bytes(b'immutable bank')
    uuid='GPU-0370fdb0-fixture';plan={'fingerprint':'fixture'}
    def write(value):
        (directory/'result.json').write_text(json.dumps(value))
        manifest=dict(status='completed',phase='native',fingerprint='fixture',allocation={uuid:11264},
            allocator=dict(physical_uuid=uuid,allocator_limit_bytes=11008*1024**2),peak_reserved_bytes=1024,
            artifacts={p.name:study.base.sha(p) for p in directory.iterdir() if p.name!='manifest.json'})
        (directory/'manifest.json').write_text(json.dumps(manifest))
    write(result);assert study.checked(tmp_path,'native',plan)==result
    for bad in (dict(result,correct=0),dict(result,fixed_state_unchanged=False)):
        write(bad)
        with pytest.raises(ValueError):study.checked(tmp_path,'native',plan)
    write(result);(directory/'boundary_22.npy').write_bytes(b'changed bank')
    with pytest.raises(ValueError,match='artifact changed'):study.checked(tmp_path,'native',plan)
