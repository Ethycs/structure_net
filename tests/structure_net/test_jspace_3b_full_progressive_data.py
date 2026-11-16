from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_3b_full_progressive_data as study


def record(name,split,start,family='arc'):
    return study.data.ResponseRecord(example_id=name,group_id=name,split=split,task=family,
        prompt_ids=(start,start+1),response_ids=(start+2,99),eos_token_id=99,finish_reason='eos',
        provenance=dict(source_uri='fixture',source_revision='v1',source_family=family,
            tokenizer_sha256='t',chat_template_sha256='c',prompt_provenance='fixture',
            response_provenance='fixture',assistant_boundary_verified=True))


def test_full_inventory_includes_embedding_and_prenorm_without_duplicate_cuts():
    cuts,boundaries=study.boundary_inventory(range(36))
    assert cuts==list(range(35,-1,-1)) and boundaries==list(range(37))
    for bad in ([],[-1],[36],[0,0],[False]):
        with pytest.raises(ValueError):study.boundary_inventory(bad)


def test_disk_space_checked_for_all_banks_and_reserve(tmp_path,monkeypatch):
    monkeypatch.setattr(study.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**13))
    p=study.disk_admission(tmp_path,total_tokens=710537,mc_tokens=34828,
        cuts=range(36),boundary_cuts=range(37))
    assert p['payload_bytes']==2*2048*(710537*37+34828*36)
    assert p['required_bytes']>p['payload_bytes']+16*1024**3
    monkeypatch.setattr(study.shutil,'disk_usage',lambda _:SimpleNamespace(free=p['required_bytes']-1))
    with pytest.raises(OSError,match='Insufficient disk'):
        study.disk_admission(tmp_path,total_tokens=710537,mc_tokens=34828,
            cuts=range(36),boundary_cuts=range(37))


def test_frozen_scientific_selection_is_identical_to_v2():
    records=[record(f'{family}:{i}','train',1000*f+10+i,family)
             for f,family in enumerate(study.FAMILIES) for i in range(60)]
    records += [record('dev','development',90000)]
    assert study.selections(records,budget=500)==study.seed_source.selections(records,budget=500)
    p=study.selections(records,budget=500)
    assert len(p['calibration_indices'])==128 and len(p['diagnostic_indices'])==32
    assert len(p['local_alignment_indices'])==256
    items=study.capture_items(records,study.cache.index_records(records))
    assert all(c['targets'][-1]==99 and len(c['loss_positions'])==len(c['targets']) for c in items)


def test_seed_admission_binds_original_data_without_a_quality_gate(tmp_path,monkeypatch):
    records=[record(f'{family}:{i}','train',1000*f+10+i,family)
             for f,family in enumerate(study.FAMILIES) for i in range(60)]
    # Enough actual complete tokens for the unchanged25k profile.
    from dataclasses import replace
    records=[replace(r,response_ids=tuple([123]*60+[99])) for r in records]
    records+=[record('dev','development',90000)]
    source=dict(model={'revision':'native'},fingerprint='original',rows=study.cache.index_records(records))
    mc={'items':[{'identity':'exposed'}]}
    p=dict(model=source['model'],response_cache_fingerprint='original',rows=source['rows'],mc=mc,
        records_sha256=study.data.dataset_fingerprint(records),fingerprint='seed',
        quality_status='retention_not_established',**study.selections(records))
    (tmp_path/'native').mkdir();study.write(tmp_path/'plan.json',p)
    np.save(tmp_path/'native/boundary_35.npy',np.zeros((2,3),dtype=np.int16))
    study.write(tmp_path/'native/result.json',dict(status='completed',fingerprint='seed',
        full_teacher_bit_exact=True,existing_boundary_bit_exact=True,final_normalization_bit_exact=True,teacher_unchanged=True))
    study.write(tmp_path/'native/manifest.json',dict(status='completed',fingerprint='seed',
        artifacts={f.name:study.sha(f) for f in (tmp_path/'native').iterdir()}))
    monkeypatch.setattr(study.seed_source,'verify',lambda *a,**k:(p,records))
    receipt,_=study.seed_inventory(tmp_path,source,records,mc)
    assert receipt['fingerprint']=='seed' and receipt['quality_admission'].startswith('Not applied')
    with pytest.raises(ValueError,match='same native model'):
        study.seed_inventory(tmp_path,{**source,'model':{'revision':'wrong'}},records,mc)
    with (tmp_path/'native/boundary_35.npy').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='payload changed'):study.seed_inventory(tmp_path,source,records,mc)


class Add(nn.Module):
    def forward(self,x):return x+1


class Body(nn.Module):
    def __init__(self):
        super().__init__();self.embed_tokens=nn.Embedding(100,3)
        self.layers=nn.ModuleList([Add() for _ in range(36)]);self.norm=nn.Identity()

    def forward(self,input_ids,use_cache=False):
        x=self.embed_tokens(input_ids)
        for layer in self.layers:x=layer(x)
        return SimpleNamespace(last_hidden_state=self.norm(x))


class Native(nn.Module):
    def __init__(self):
        super().__init__();self.model=Body();self.lm_head=nn.Linear(3,100,bias=False)
        self.lm_head.weight=self.model.embed_tokens.weight;self.register_buffer('tracked',torch.zeros(()))


def capture_fixture(tmp_path,monkeypatch,*,mutate_teacher=False,mutate_seed=False):
    torch.manual_seed(42);original=Native().bfloat16().eval().requires_grad_(False)
    state={k:v.clone() for k,v in original.state_dict().items()}
    records=[record('train','train',1),record('dev','development',10)];rows=study.cache.index_records(records)
    expected={k:[] for k in ('teacher',0,1,20,35,36)}
    with torch.no_grad(),study.boundary_hooks(original,[0,1,20,35,36]) as captured:
        for r in records:
            captured.clear();hidden=original.model(input_ids=torch.tensor([r.input_ids])).last_hidden_state
            expected['teacher'].append(study._bits(hidden,4,3).copy())
            for cut in (0,1,20,35,36):expected[cut].append(study._bits(captured[cut],4,3).copy())
    expected={k:np.concatenate(v) for k,v in expected.items()}
    old=tmp_path/'old';(old/'native').mkdir(parents=True)
    seed=tmp_path/'seed';(seed/'native').mkdir(parents=True)
    np.save(old/'native/teacher_hidden.npy',expected['teacher']);np.save(old/'native/boundary_20.npy',expected[20])
    seed_bits=expected[35].copy()
    if mutate_seed:seed_bits[0,0]+=1
    np.save(seed/'native/boundary_35.npy',seed_bits)
    case=dict(identity='mc',split='development',family='arc',answer=0,choices=['a','b'],
        input_ids=list(records[0].input_ids),offset=0,stop=4)
    with torch.no_grad():
        h=original.model(input_ids=torch.tensor([case['input_ids']])).last_hidden_state
        native_mc=study.cache.prior.sanity._record(case,h[:,-1],original.lm_head.weight,[1,2])
    mc=dict(items=[case],label_ids=[1,2],native_records=[native_mc],native_correct=195,total_tokens=4,scope='fixture')
    root=tmp_path/'capture';root.mkdir()
    plan=dict(response_cache_root=str(old),total_tokens=8,width=3,existing_boundary_cut=20,
        boundary_cuts=list(range(37)),cuts=list(range(35,-1,-1)),model={'path':'fixture'},rows=rows,
        items=study.capture_items(records,rows),mc=mc,fingerprint='fixture',
        seed_data={'root':str(seed),'fingerprint':'seed-fp'},disk_admission={'reserve_bytes':0})
    monkeypatch.setattr(study,'verify',lambda *a,**k:(plan,records))
    def loader(*a,**k):
        assert not torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        model=Native().bfloat16().eval().requires_grad_(False);model.load_state_dict(state)
        assert all(isinstance(v._version,int) for v in (*model.parameters(),*model.buffers()))
        if mutate_teacher:
            def mutate(*_):model.tracked.add_(0)
            model.model.register_forward_hook(mutate)
        return model
    monkeypatch.setattr(study.cache.execution,'load_native_model',loader)
    monkeypatch.setattr(study.cache,'reservation',lambda *_:({'id':'cpu','allocation':{'fixture':11009}},lambda:None))
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *a:0)
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CPU regression initialized CUDA'))
    tensor=torch.tensor
    def cpu_tensor(*a,**k):
        if k.get('device')=='cuda:0':k['device']='cpu'
        return tensor(*a,**k)
    monkeypatch.setattr(torch,'tensor',cpu_tensor)
    return root,plan,records,expected


def test_actual_capture_all_boundaries_and_cut0_dataset_preserve_bits(tmp_path,monkeypatch):
    root,plan,records,expected=capture_fixture(tmp_path,monkeypatch)
    calls=[];function=study.boundary_bits
    def counted(*a,**k):calls.append(True);return function(*a,**k)
    monkeypatch.setattr(study,'boundary_bits',counted)
    with torch.inference_mode():result=study.capture(root)
    assert result['seed_boundary35_bit_exact'] and result['teacher_unchanged'] and result['final_normalization_bit_exact']
    assert len(calls)==3 # two responses and one separate MC row; one batched transfer each.
    for cut in range(37):
        a=np.load(root/'native'/f'boundary_{cut:02d}.npy');assert a.shape==(8,3) and a.dtype==np.int16
    for cut in (0,1,20,35,36):assert np.array_equal(np.load(root/'native'/f'boundary_{cut:02d}.npy'),expected[cut])
    for cut in range(36):assert np.load(root/'native'/f'mc_boundary_{cut:02d}.npy').shape==(4,3)
    dataset=study.RebuildDataset(root,cut=0)
    try:
        b=dataset.batch([0]);assert torch.equal(b['boundary_hidden'][0].view(torch.int16),torch.from_numpy(expected[0][:4]))
        assert torch.equal(b['next_boundary_hidden'][0].view(torch.int16),torch.from_numpy(expected[1][:4]))
        assert b['targets'][b['loss_mask']].tolist()==list(records[0].response_ids)
    finally:dataset.close()
    mc=study.MCDataset(root,cut=0)
    try:assert mc.training==[] and torch.equal(mc.batch([0])['boundary_hidden'][0].view(torch.int16),torch.from_numpy(expected[0][:4]))
    finally:mc.close()
    original_sha=study.sha
    def only_metadata(path):
        assert not str(path).endswith('.npy'),'Dataset admission reread a whole bank'
        return original_sha(path)
    monkeypatch.setattr(study,'sha',only_metadata)
    study._manifest_files(root/'native','fixture')
    with (root/'native/boundary_00.npy').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='file identity changed'):study._manifest_files(root/'native','fixture')


@pytest.mark.parametrize('mutation,error',[(dict(mutate_teacher=True),'Frozen original teacher changed'),
                                         (dict(mutate_seed=True),'v2 cut35 native boundary replay')])
def test_actual_capture_rejects_teacher_or_seed_mutation(tmp_path,monkeypatch,mutation,error):
    root,*_=capture_fixture(tmp_path,monkeypatch,**mutation)
    with pytest.raises(ValueError,match=error):study.capture(root)
    assert (root/'native/failure.json').is_file() and not (root/'native/manifest.json').exists()
