from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from transformers import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3RMSNorm, Phi3RotaryEmbedding

from experiments.structure_net import jspace_phi_fit as fit


def tiny_items():
    items=[];cursor=0;prompt_cursor=0
    for i,length in enumerate((5,3,4)):
        items.append(dict(identity=f'x{i}',split='train' if i<2 else 'development',family='arc',
            input_ids=list(range(1,length+1)),choices=['a','b'],answer=i%2,
            loss_positions=[0,1,2,length-1],targets=[1,2,3,4],
            offset=cursor,stop=cursor+length,full_stop=cursor+length+2,
            prompt_offset=prompt_cursor,prompt_stop=prompt_cursor+length))
        cursor+=length+2;prompt_cursor+=length
    return items


def test_distinct_prompt_offsets_and_stream_preserve_input_autograd(tmp_path):
    torch.manual_seed(4021)
    config=Phi3Config(hidden_size=16,intermediate_size=24,num_hidden_layers=40,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=32,max_position_embeddings=32,
        original_max_position_embeddings=32,rope_theta=250000,tie_word_embeddings=False)
    config._attn_implementation='eager'
    native=Phi3DecoderLayer(config,39).bfloat16().eval().requires_grad_(False)
    norm=Phi3RMSNorm(16,eps=config.rms_norm_eps).bfloat16().eval().requires_grad_(False)
    rotary=Phi3RotaryEmbedding(config).eval();head=torch.randn(32,16).bfloat16()
    items=tiny_items();bank=torch.randn(items[-1]['prompt_stop'],16).bfloat16()
    path=tmp_path/'prompt_prefix.npy';np.save(path,bank.view(torch.int16).numpy())
    # The separate full-response stream is deliberately unrelated.
    np.save(tmp_path/'prefix.npy',torch.full((items[-1]['full_stop'],16),999,dtype=torch.bfloat16).view(torch.int16).numpy())
    calls=[]
    teacher=fit.PrefixTeacher(items,path,native,norm,head,rotary,guard=lambda:calls.append(True))
    try:
        for i,row in enumerate(items):
            x=bank[row['prompt_offset']:row['prompt_stop']][None]
            expected=norm(native(x,**fit.block_arguments(x,rotary))[0])
            assert torch.equal(teacher.forward_case(i),expected)
        # A normal input-gradient hook must remain connected through norm.
        def gradient_input(module,args):return (args[0].detach().requires_grad_(True),)
        hook=native.register_forward_pre_hook(gradient_input)
        result=teacher.forward_case(1);hook.remove()
        assert result.requires_grad
        result.float().square().sum().backward()
        assert all(p.grad is None for p in teacher.parameters())
        assert len(calls)==5
    finally:teacher.close()


def test_validates_production_128_64_without_hardcoded_prompt_contiguity():
    items=[];cursor=0;prompt_cursor=0
    for split,count in [('train',32),('development',16)]:
        for family in ('arc','boolq','hellaswag','winogrande'):
            for i in range(count):
                items.append(dict(identity=f'{split}:{family}:{i}',split=split,family=family,input_ids=[1,2,3],
                    loss_positions=[0,1,1,2],targets=[1,2,2,3],choices=['a','b'],answer=0,
                    offset=cursor,stop=cursor+3,full_stop=cursor+5,
                    prompt_offset=prompt_cursor,prompt_stop=prompt_cursor+3));cursor+=5;prompt_cursor+=3
    sources=[{**c,'offset':i*3,'stop':i*3+3} for i,c in enumerate(items)]
    assert fit.validate_items(items,sources)==960
    corrupt=deepcopy(items);corrupt[1]['offset']=3
    with pytest.raises(ValueError,match='offsets'):fit.validate_items(corrupt,sources)
    corrupt=deepcopy(items);corrupt[0]['input_ids'][0]=8
    with pytest.raises(ValueError,match='differs'):fit.validate_items(corrupt,sources)
    with pytest.raises(ValueError,match='128 TRAIN'):fit.validate_items(items[:-1],sources[:-1])


def test_denied_lease_before_cuda_or_native_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(fit,'verify',lambda _:dict(fingerprint='study'))
    def deny(_):raise PermissionError('no lease')
    monkeypatch.setattr(fit,'wait_ready',deny)
    monkeypatch.setattr(fit,'allocator_settings',lambda *a,**k:pytest.fail('CUDA before lease'))
    monkeypatch.setattr(fit,'_execute',lambda *a,**k:pytest.fail('native load before lease'))
    with pytest.raises(PermissionError):fit.run(tmp_path,'capture')
    failure=fit.read(tmp_path/'capture_failure.json')
    assert failure['status']=='failed' and failure['lease']is None
    with pytest.raises(ValueError,match='preserved'):fit.run(tmp_path,'capture')


def test_allocator_validates_physical_card_before_cuda():
    calls=[]
    cuda=SimpleNamespace(get_device_properties=lambda _:calls.append('query') or SimpleNamespace(total_memory=12*1024**3),
        set_per_process_memory_fraction=lambda f,d:calls.append((f,d)),set_device=lambda _:calls.append('device'),
        reset_peak_memory_stats=lambda _:calls.append('reset'))
    with pytest.raises(ValueError):fit.allocator_settings({'allocation':{'GPU-wrong':11264}},cuda=cuda,visible='GPU-wrong')
    assert calls==[]
    value=fit.allocator_settings({'allocation':{'GPU-0370fdb0-test':11264}},cuda=cuda,visible='GPU-0370fdb0-test')
    assert value['allocator_bytes']==11008*1024**2 and calls[0]=='query' and calls[-1]=='reset'


def receipt(root,phase,*,previous=None):
    directory=root/phase;directory.mkdir(exist_ok=True)
    path=directory/'output';path.write_text(phase)
    value=dict(schema=fit.SCHEMA,status='completed',phase=phase,fingerprint='study',
        previous_receipts=previous or {},artifacts={str(path.relative_to(root)):fit.sha(path)},
        allocation={'GPU-0370fdb0-test':11264},allocator=dict(physical_uuid='GPU-0370fdb0-test',
            allocator_bytes=11008*1024**2,lease_bytes=11264*1024**2),peak_reserved_bytes=1024)
    fit.write(root/(phase+'_receipt.json'),value)
    return value


@pytest.mark.parametrize('mutation',['payload','card','peak','status','predecessor'])
def test_completed_phase_admission_rejects_artifact_resource_and_chain_damage(tmp_path,mutation):
    receipt(tmp_path,'capture')
    value=receipt(tmp_path,'basis',previous={'capture':fit.sha(tmp_path/'capture_receipt.json')})
    assert fit.checked(tmp_path,'basis',{'fingerprint':'study'})['status']=='completed'
    if mutation=='payload':(tmp_path/'basis/output').write_text('different')
    elif mutation=='card':value['allocation']={'GPU-other':11264}
    elif mutation=='peak':value['peak_reserved_bytes']=11008*1024**2+1
    elif mutation=='status':value['status']='started'
    else:value['previous_receipts']['capture']='0'*64
    if mutation!='payload':(tmp_path/'basis_receipt.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):fit.checked(tmp_path,'basis',{'fingerprint':'study'})


def test_resume_completed_phase_does_not_acquire_gpu(tmp_path,monkeypatch):
    receipt(tmp_path,'capture')
    monkeypatch.setattr(fit,'verify',lambda _:dict(fingerprint='study'))
    monkeypatch.setattr(fit,'wait_ready',lambda _:pytest.fail('completed phase acquired GPU'))
    assert fit.run(tmp_path,'capture')['status']=='completed'


def test_phase_dispatch_fixed_geometry_order_and_native_phi_rotary(tmp_path,monkeypatch):
    calls=[];native=object();rotary=object()
    monkeypatch.setattr(fit.fitting,'finalize_basis',lambda *a,**k:calls.append(('basis',a,k)))
    monkeypatch.setattr(fit.weights,'load_native_block',lambda *a,**k:calls.append(('native',a,k)) or native)
    monkeypatch.setattr(fit.weights,'load_rotary',lambda *a,**k:calls.append(('rotary',a,k)) or rotary)
    monkeypatch.setattr(fit,'capture_compiled_development',lambda *a,**k:calls.append(('interface',a,k)))
    monkeypatch.setattr(fit.rankfit,'fit_relation',lambda *a,**k:calls.append(('relation',a,k)))
    p={'model':'pinned-gptq','settings':fit.SETTINGS};guard=lambda:None;progress=lambda _:None
    for phase in ('basis','interface','relation'):fit._execute(tmp_path,p,phase,'cpu',guard,progress)
    assert [c[0] for c in calls]==['basis','native','rotary','interface','relation']
    assert calls[0][2]['rank']==2560 and calls[0][2]['head_dim']==64
    assert calls[3][1][2:4]==(39,native)
    last=calls[-1][2]
    assert (last['anchors'],last['remove_count'],last['pad_to_multiple'],last['seed'])==(24,1664,128,861)
    assert last['chunk_rows']==256 and last['ridge']==.01


def test_native_manifest_tamper_rejected_before_fitting(tmp_path,monkeypatch):
    native=tmp_path/'native';native.mkdir()
    np.save(native/'prefix.npy',np.zeros((4,16),dtype=np.int16))
    np.save(native/'prompt_prefix.npy',np.zeros((3,16),dtype=np.int16))
    fit.write(native/'items.json',tiny_items())
    value=dict(schema='jspace.phi-native.v1',status='completed',fingerprint='native',
        artifacts={name:fit.sha(native/name) for name in ('prefix.npy','prompt_prefix.npy','items.json')})
    fit.write(native/'manifest.json',value)
    monkeypatch.setattr(fit.native_reference,'checked',lambda *a,**k:value)
    assert fit._native(tmp_path)[0]['fingerprint']=='native'
    (native/'items.json').write_text('[]')
    with pytest.raises(ValueError,match='artifact changed'):fit._native(tmp_path)


def test_cpu_prepare_hidden_gpu_guard(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-something')
    monkeypatch.setattr(fit.data,'verify',lambda *a,**k:pytest.fail('inputs opened before CPU guard'))
    with pytest.raises(ValueError,match='CUDA hidden'):fit.prepare(tmp_path/'fit','native','data')


@pytest.mark.parametrize('field',['model','data_fingerprint','data_root','fingerprint'])
def test_prepare_binds_actual_native_model_and_data_before_fitting(tmp_path,monkeypatch,field):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    native_root=tmp_path/'native_source';native_root.mkdir();data_root=tmp_path/'data';data_root.mkdir()
    dp=dict(model={'path':str(tmp_path/'model')},fingerprint='data')
    native=dict(model=dp['model'],data_fingerprint='data',data_root=str(data_root),fingerprint='native')
    native[field]={'path':'another-model'} if field=='model' else 'another-source'
    fit.write(native_root/'plan.json',native)
    monkeypatch.setattr(fit.data,'verify',lambda *a,**k:dp)
    monkeypatch.setattr(fit,'_native',lambda _:({'fingerprint':'native'},[]))
    monkeypatch.setattr(fit.weights,'load_config',lambda *a,**k:pytest.fail('Loaded weights before source identity admission'))
    with pytest.raises(ValueError,match='model/data identity'):fit.prepare(tmp_path/'fit',native_root,data_root)


def test_build_uses_canonical_bf16_parameters_without_quantizing_geometry(tmp_path,monkeypatch):
    path=tmp_path/'basis/block_39';path.mkdir(parents=True)
    torch.save(dict(rank=2560,head_dim=64,basis=torch.zeros(3,2560),means=torch.zeros(1,3),
        rope_pairs=torch.zeros(1,32,dtype=torch.long),value_basis=torch.zeros(1,128,64)),path/'fit.pt')
    torch.save(dict(anchors=list(range(24)),removed=list(range(24,1688))),path/'relation.pt')
    class Fake(torch.nn.Module):
        def __init__(self,native,basis,means,**kwargs):
            super().__init__();self.p=torch.nn.Parameter(torch.ones(2));self.register_buffer('basis',basis)
            self.kwargs=kwargs
        def compile_polynomial(self,relation,removed):
            self.poly=torch.nn.Parameter(torch.ones(3));self.removed=removed
    monkeypatch.setattr(fit,'AffinePhiBlock',Fake)
    native=SimpleNamespace(self_attn=SimpleNamespace(layer_idx=39,qkv_proj=SimpleNamespace(weight=torch.ones(1))))
    block=fit.build_block(tmp_path,native)
    assert all(p.dtype==torch.bfloat16 for p in block.parameters()) and block.basis.dtype==torch.float32
    assert block.kwargs['buffer_dtype']==torch.float32 and len(block.removed)==1664
