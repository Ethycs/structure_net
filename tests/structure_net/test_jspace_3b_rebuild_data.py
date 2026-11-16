from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_3b_rebuild_data as rebuild


def record(family, index, split='train'):
    return rebuild.data.ResponseRecord(example_id=f'{family}:{split}:{index}',
        group_id=f'{family}:{split}:{index}',split=split,task=family,
        prompt_ids=(10+index,11+index),response_ids=(30+index,31+index,999),
        eos_token_id=999,finish_reason='eos',provenance=dict(source_uri='fixture://source',
            source_revision='pinned',source_family=family,tokenizer_sha256='a'*64,
            chat_template_sha256='b'*64,prompt_provenance='fixture',response_provenance='fixture',
            assistant_boundary_verified=True))


def corpus():
    # Distinct prompts across families/splits; synthetic fixtures are tests only.
    return [record(family,i+1000*f) for f,family in enumerate(rebuild.FAMILIES) for i in range(60)]+[
        record(family,9000+f,'development') for f,family in enumerate(rebuild.FAMILIES)]


def test_balanced_train_only_independent_geometry_and_local_alignment():
    records=corpus(); p=rebuild.selections(records,budget=500,accumulation=16)
    assert p==rebuild.selections(records,budget=500,accumulation=16)
    selected=[]
    for key,n in [('calibration_indices',16),('diagnostic_indices',4),('local_alignment_indices',32)]:
        indices=p[key]; assert Counter(records[i].task for i in indices)==dict.fromkeys(rebuild.FAMILIES,n)
        assert all(records[i].split=='train' for i in indices); selected.extend(indices)
    assert len(set(selected))==len(selected)==416
    assert len({records[i].group_id for i in selected})==416
    assert len(p['local_alignment_windows'])==16
    order=p['recovery_order']
    assert len(set(order))==len(order)
    assert all(records[i].split=='train' for i in order)
    assert p['recovery_response_tokens']==sum(len(records[i].response_ids) for i in order)>=500
    assert p['recovery_response_tokens']-500 < 16*3


def test_selection_rejects_short_or_leaking_corpus():
    records=corpus()
    with pytest.raises(ValueError,match='distinct TRAIN groups'):
        rebuild.selections([r for r in records if not (r.task=='arc' and r.split=='train' and int(r.example_id.split(':')[-1])>=40)],budget=100)
    records[-1]=replace(records[-1],group_id=records[0].group_id)
    with pytest.raises(ValueError,match='Cross-split source group'):
        rebuild.selections(records,budget=100)


def test_capture_items_keep_all_causal_response_positions_and_eos():
    records=corpus(); rows=rebuild.cache.index_records(records)
    items=rebuild.capture_items(records,rows)
    for record,row,item in zip(records,rows,items):
        assert item['loss_positions']==list(range(len(record.prompt_ids)-1,len(record.input_ids)-1))
        assert item['targets']==list(record.response_ids)
        assert item['targets'][-1]==record.eos_token_id
        assert (item['offset'],item['stop'])==(row['offset'],row['stop'])
    rows[0]['input_ids'][0]+=1
    with pytest.raises(ValueError,match='identity differs'):
        rebuild.capture_items(records,rows)


class Add(nn.Module):
    def __init__(self,value): super().__init__(); self.value=value
    def forward(self,value): return value+self.value


def test_actual_boundary_hooks_include_pre_final_norm_and_remove_on_failure():
    layers=nn.ModuleList([Add(1),Add(2),Add(4)])
    norm=Add(100); model=SimpleNamespace(model=SimpleNamespace(layers=layers,norm=norm))
    x=torch.zeros((1,3,2),dtype=torch.bfloat16)
    with rebuild.boundary_hooks(model,[1,2,3]) as captured:
        for layer in layers: x=layer(x)
        y=norm(x)
        assert torch.equal(captured[1],torch.ones_like(x))
        assert torch.equal(captured[2],torch.full_like(x,3))
        assert torch.equal(captured[3],torch.full_like(x,7))
        assert torch.equal(y,torch.full_like(x,107))
        with pytest.raises(ValueError,match='more than once'): norm(x)
    assert not norm._forward_pre_hooks and all(not layer._forward_pre_hooks for layer in layers)
    with pytest.raises(RuntimeError):
        with rebuild.boundary_hooks(model,[3]): raise RuntimeError('fixture')
    assert not norm._forward_pre_hooks


def test_lossless_bf16_bits_and_wrong_dtype_rejection():
    x=torch.tensor([[[1.001,-0.0],[1e-8,-1000.3]]],dtype=torch.bfloat16)
    bits=rebuild._bits(x,2,2)
    assert bits.dtype==np.int16
    assert torch.equal(torch.from_numpy(bits).view(torch.int16),x[0].view(torch.int16))
    with pytest.raises(ValueError,match='finite BF16'): rebuild._bits(x.float(),2,2)
    x[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='finite BF16'): rebuild._bits(x,2,2)


def dataset_fixture(tmp_path,monkeypatch):
    records=[record('arc',1),record('arc',10,'development')]
    rows=rebuild.cache.index_records(records); root=tmp_path/'rebuild'; (root/'native').mkdir(parents=True)
    old=tmp_path/'old'; (old/'native').mkdir(parents=True)
    values=torch.arange(30,dtype=torch.float32).reshape(10,3).to(torch.bfloat16)
    for name,offset in [('boundary_34',10),('boundary_35',20),('boundary_36',30)]:
        np.save(root/'native'/f'{name}.npy',(values+offset).view(torch.int16).numpy())
    np.save(old/'native/teacher_hidden.npy',values.view(torch.int16).numpy())
    mc=dict(items=[dict(identity='old-dev',split='development',input_ids=[2,3],offset=0,stop=2)],
        total_tokens=2,label_ids=[4,5],native_records=[{'fixture':True}],native_correct=195)
    np.save(root/'native/mc_boundary_35.npy',(values[:2]+50).view(torch.int16).numpy())
    p=dict(fingerprint='pinned',cuts=[35,34],rows=rows,items=rebuild.capture_items(records,rows),
        width=3,total_tokens=10,response_cache_root=str(old),mc=mc)
    rebuild.write(root/'native/result.json',dict(full_teacher_bit_exact=True,existing_boundary_bit_exact=True,teacher_unchanged=True))
    rebuild.write(root/'native/manifest.json',dict(status='completed',fingerprint='pinned',
        artifacts={f.name:rebuild.sha(f) for f in (root/'native').iterdir()}))
    monkeypatch.setattr(rebuild,'verify',lambda *a,**k:(p,records))
    return root,records,values


def test_streamed_batch_binds_teacher_and_adjacent_targets_to_exact_response(tmp_path,monkeypatch):
    root,records,values=dataset_fixture(tmp_path,monkeypatch)
    ds=rebuild.RebuildDataset(root,cut=35,pad_token_id=0)
    try:
        batch=ds.batch([0])
        assert torch.equal(batch['teacher_hidden'][0],values[:5])
        assert torch.equal(batch['boundary_hidden'][0],values[:5]+20)
        assert torch.equal(batch['next_boundary_hidden'][0],values[:5]+30)
        assert batch['targets'][batch['loss_mask']].tolist()==list(records[0].response_ids)
        assert batch['teacher_record_ids']==[records[0].example_id]
        assert torch.equal(batch['teacher_input_ids'],batch['input_ids'])
        assert ds.training==[0] and ds.development==[1]
        with pytest.raises(ValueError,match='single-split'): ds.batch([0,1])
    finally: ds.close()
    assert ds.teacher._mmap.closed and ds.next_boundary._mmap.closed


def test_old_mc_control_has_no_training_rows_and_cannot_supply_response_targets(tmp_path,monkeypatch):
    root,_,values=dataset_fixture(tmp_path,monkeypatch)
    ds=rebuild.MCDataset(root,cut=35,pad_token_id=0)
    try:
        batch=ds.batch([0])
        assert ds.training==[] and ds.development==[0]
        assert 'targets' not in batch and 'loss_mask' not in batch and 'teacher_hidden' not in batch
        assert torch.equal(batch['boundary_hidden'][0],values[:2]+50)
        assert batch['attention_mask'].all()
    finally: ds.close()


def test_changed_boundary_bank_is_rejected_before_dataset_read(tmp_path,monkeypatch):
    root,_,_=dataset_fixture(tmp_path,monkeypatch)
    with (root/'native/boundary_35.npy').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='payload changed'): rebuild.RebuildDataset(root,cut=35)


def test_prepare_requires_cpu_and_does_not_touch_existing_root(tmp_path,monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    with pytest.raises(ValueError,match='CUDA hidden'): rebuild.prepare(tmp_path/'new',tmp_path/'missing')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(FileExistsError): rebuild.prepare(tmp_path,tmp_path/'missing')
