from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_progressive_capture as study


def tiny_rows():
    rows=[];offset=0
    for i,ids in enumerate(([1,2,3,4],[3,1,2],[2,5,1])):
        n=len(ids);legacy=[0,0,n-2,n-1];content=[1,n-2,1,n-1]
        rows.append(dict(identity=f'case{i}',family='arc',split='development' if i==2 else 'train',
            input_ids=list(ids),choices=['one','two'],answer=i%2,offset=offset,stop=offset+n,
            legacy_loss_positions=legacy,content_loss_positions=content,
            legacy_targets=[ids[p+1] for p in legacy[:-1]]+[6+i%2],
            content_targets=[ids[p+1] for p in content[:-1]]+[6+i%2],
            loss_positions=legacy,targets=[ids[p+1] for p in legacy[:-1]]+[6+i%2]))
        offset+=n
    return rows


def native():
    torch.manual_seed(832)
    config=Qwen2Config(hidden_size=8,intermediate_size=12,num_attention_heads=2,
        num_key_value_heads=1,num_hidden_layers=4,vocab_size=16,attention_dropout=0.,
        max_position_embeddings=32,tie_word_embeddings=True)
    config._attn_implementation='eager'
    return Qwen2ForCausalLM(config).bfloat16().eval().requires_grad_(False)


def production(tmp_path,monkeypatch,cut):
    rows=[];indices=[]
    for split,count in [('train',1152),('development',64)]:
        for family in study.FAMILIES:
            for i in range(count):
                row=deepcopy(tiny_rows()[0]);row.update(identity=f'{split}-{family}-{i}',family=family,
                    split=split,offset=4*len(rows),stop=4*(len(rows)+1))
                if split=='train' and i<32:indices.append(len(rows))
                rows.append(row)
    cache=tmp_path/'cache';cache.mkdir();study.base.write(cache/'items.json',dict(items=rows))
    parent=tmp_path/'previous';parent.mkdir();checkpoint=parent/'selected.pt';checkpoint.write_bytes(b'parent weights')
    source=parent/f'boundary_{cut+1:02d}.npy';source.write_bytes(b'native boundary content')
    receipt=parent/'step_receipt.json'
    schema='jspace.3b-broad-step-capture.v1' if cut==20 else study.SCHEMA
    control='native_pre22_control' if cut==20 else 'native_parent_boundary_control'
    study.base.write(receipt,dict(schema=schema,status='completed',phase='boundary',cut=cut+1,
        items_sha256=study.base.digest(study.content_items(rows)),shape=[rows[-1]['stop'],2048],
        fixed_state_unchanged=True,artifacts={source.name:study.base.sha(source)},
        **{control:dict(passed=True,bit_exact=True,count=4864,max_abs=0.)}))
    plan=dict(cut=cut,parent_cut=cut+1,allocation_mib=11264,allocator_mib=11008,
        cache_root=str(cache),data_root=str(tmp_path/'data'),model={'path':'native'},label_ids=[6,7,8,9],
        training_order=list(reversed(range(4608))),calibration_indices=indices,
        parent_fits={str(i):str(tmp_path/f'fit{i}') for i in range(cut+1,36)},
        parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=study.base.sha(checkpoint),
        expected_parent_fixed_state=dict(sha256='a'*64,tensor_bytes=10),
        parent_boundary_path=str(source),parent_boundary_sha256=study.base.sha(source),
        parent_boundary_receipt=str(receipt),parent_boundary_receipt_sha256=study.base.sha(receipt),
        boundary_root=str(tmp_path/'boundary'),capture_root=str(tmp_path/'capture'),fit_root=str(tmp_path/'fit'))
    cp={k:deepcopy(plan[k]) for k in ('model','label_ids','data_root','training_order')}
    cp['fits']={k:v for k,v in plan['parent_fits'].items() if int(k)>=22}
    monkeypatch.setattr(study.cache,'verify',lambda root:cp)
    return plan,rows


@pytest.mark.parametrize('cut',[0,7,20])
def test_arbitrary_cut_keeps_all_data_and_balanced_training_only_calibration(tmp_path,monkeypatch,cut):
    plan,rows=production(tmp_path,monkeypatch,cut)
    actual=study.validate_plan_items(plan)
    assert len(actual)==4864 and all(actual[i]['split']=='train' for i in plan['calibration_indices'])
    assert actual[0]['loss_positions']==rows[0]['content_loss_positions']
    assert study.parent_boundary(plan,actual,hash_payload=True)==Path(plan['parent_boundary_path']).resolve()
    different={**plan,'cut':cut+1}
    assert study.teacher_fingerprint(plan,actual)!=study.teacher_fingerprint(different,actual)


@pytest.mark.parametrize('defect',['development_j','duplicate_j','unbalanced_j','parent_weights','parent_geometry','parent_cut','parent_bank','parent_receipt','nested'])
def test_rejects_foreign_parent_data_and_unsafe_output_roots(tmp_path,monkeypatch,defect):
    plan,_=production(tmp_path,monkeypatch,7)
    if defect=='development_j':plan['calibration_indices'][0]=4608
    elif defect=='duplicate_j':plan['calibration_indices'][0]=plan['calibration_indices'][1]
    elif defect=='unbalanced_j':plan['calibration_indices'][0]=1152+32
    elif defect=='parent_weights':Path(plan['parent_checkpoint']).write_bytes(b'changed')
    elif defect=='parent_geometry':plan['parent_fits']['35']='wrong35'
    elif defect=='parent_cut':plan['parent_cut']=9
    elif defect=='parent_bank':plan['parent_boundary_sha256']='b'*64
    elif defect=='parent_receipt':Path(plan['parent_boundary_receipt']).write_text('{}')
    else:plan['boundary_root']=str(Path(plan['parent_boundary_path']).parent/'new')
    with pytest.raises(ValueError):study.validate_plan_items(plan)


@pytest.mark.parametrize('cut',[0,1,2])
def test_native_prefix_earlystop_is_bit_exact_including_raw_embedding_and_unused_tail(cut):
    model=native();row=tiny_rows()[0];ids=torch.tensor([row['input_ids']])
    with torch.no_grad():expected=model.model(input_ids=ids,use_cache=False,output_hidden_states=True).hidden_states
    calls=[];handles=[]
    for i,block in enumerate(model.model.layers):
        handles.append(block.register_forward_hook(lambda _m,_a,_o,layer=i:calls.append(layer)))
    try:actual=study.native_boundaries(model,row,cut,lambda:None)
    finally:
        for h in handles:h.remove()
    assert calls==list(range(cut+1))
    assert torch.equal(actual[cut],expected[cut]) and torch.equal(actual[cut+1],expected[cut+1])
    if cut==0:assert torch.equal(actual[0],model.model.embed_tokens(ids))
    assert not any(block._forward_pre_hooks for block in model.model.layers)


def test_real_native_error_propagates_and_capture_hooks_are_removed(monkeypatch):
    model=native()
    def fail(*a,**k):raise RuntimeError('real decoder failure')
    monkeypatch.setattr(model.model.layers[0],'forward',fail)
    with pytest.raises(RuntimeError,match='real decoder failure'):
        study.native_boundaries(model,tiny_rows()[0],0,lambda:None)
    assert not any(block._forward_pre_hooks for block in model.model.layers)


@pytest.mark.parametrize('phase',['boundary','capture','fit'])
def test_lease_guard_runs_before_model_validation_and_cuda(tmp_path,monkeypatch,phase):
    monkeypatch.setattr(study,'validate_plan_items',lambda _:pytest.fail('validation after denied reservation'))
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CUDA initialized'))
    def deny():raise RuntimeError('denied lease')
    with pytest.raises(RuntimeError,match='denied lease'):getattr(study,'run_'+phase)(tmp_path/phase,{},deny)


@pytest.mark.parametrize('cut',[0,8,20])
def test_fit_retains_exact_rank_head_and_relation_order_at_dynamic_cut(tmp_path,monkeypatch,cut):
    items=study.content_items(tiny_rows());capture=tmp_path/'capture';capture.mkdir()
    plan=dict(cut=cut,parent_cut=cut+1,capture_root=str(capture),fit_root=str(tmp_path/'fit'),
        parent_checkpoint_sha256='b'*64,calibration_indices=[0,1],model={'path':'native'})
    metadata=dict(layers=[cut],calibration_indices=[0,1],capture_order=list(range(3)),probe_groups=2,
        probes_per_group=8,activation_positions=16,seed=study.calibration.calibration_seed(cut),
        teacher_fingerprint=study.teacher_fingerprint(plan,items))
    study.base.write(capture/'metadata.json',metadata)
    study._save_receipt(capture,plan,'capture',items,artifacts={'metadata.json':study.base.sha(capture/'metadata.json')})
    monkeypatch.setattr(study,'validate_plan_items',lambda _:items)
    monkeypatch.setattr(study,'DEVICE','cpu')
    monkeypatch.setattr(study.trial,'_model_device',lambda _:nullcontext())
    monkeypatch.setattr(study.cache.sanity.execution,'arithmetic_policy',lambda:nullcontext())
    order=[]
    def basis(root,out,**kw):
        assert kw['layer_ids']==[cut] and kw['rank']==1024 and kw['head_dim']==64
        order.append('basis');(out/f'block_{cut:02d}').mkdir();(out/f'block_{cut:02d}/fit.pt').write_bytes(b'fit')
    def interface(directory,root,layer,native,callback,**kw):
        assert layer==cut and order==['basis'];order.append('compiled')
        (directory/'interface.json').write_text('{}')
    def relation(directory,**kw):
        assert order==['basis','compiled'];order.append('relation')
        expected=dict(seed=861,anchors=24,remove_count=1024,pad_to_multiple=128,svd_rank=64,
            oversample=16,power_iterations=1,ridge=.01,eligibility_nrmse=.5,chunk_rows=256)
        assert {k:kw[k] for k in expected}==expected
        (directory/'relation.pt').write_bytes(b'relation')
    monkeypatch.setattr(study.fitting,'finalize_basis',basis)
    monkeypatch.setattr(study.fitting,'capture_compiled_development',interface)
    monkeypatch.setattr(study.rankfit,'fit_relation',relation)
    monkeypatch.setattr(study.builder.native,'load_native_block',lambda *a,**k:object())
    monkeypatch.setattr(study.repair,'restore_rotary',lambda *a,**k:object())
    receipt=study.run_fit(plan['fit_root'],plan,lambda:None)
    assert order==['basis','compiled','relation']
    assert receipt['cut']==cut and receipt['fit_directory']==str((tmp_path/f'fit/block_{cut:02d}').resolve())
    assert receipt['rank']==1024 and receipt['head_dim']==64 and receipt['anchors']==24 and receipt['removed']==1024


@pytest.mark.parametrize('cut',[0,11,20])
def test_recovery_uses_distinct_dynamic_boundaries_and_unchanged_original_content_teacher(tmp_path,monkeypatch,cut):
    rows=study.content_items(tiny_rows());cache=tmp_path/'cache/native';cache.mkdir(parents=True)
    entrance=tmp_path/'boundary';entrance.mkdir();previous=tmp_path/'previous';previous.mkdir()
    hidden=torch.arange(rows[-1]['stop']*8).reshape(-1,8).bfloat16()
    logits=(torch.arange(3*8*16).reshape(3,8,16)%31).bfloat16()
    np.save(entrance/f'boundary_{cut:02d}.npy',hidden.view(torch.int16).numpy())
    np.save(previous/f'boundary_{cut+1:02d}.npy',(hidden+100).view(torch.int16).numpy())
    np.save(cache/'teacher_logits.npy',logits.view(torch.int16).numpy())
    plan=dict(cut=cut,parent_cut=cut+1,cache_root=str(cache.parent),boundary_root=str(entrance),
        parent_boundary_path=str(previous/f'boundary_{cut+1:02d}.npy'),label_ids=[6,7,8,9])
    monkeypatch.setattr(study,'validate_plan_items',lambda _:rows)
    for parent in (False,True):
        dataset=study.recovery_dataset(plan,parent_input=parent)
        try:
            batch=dataset.batch([2,0],device='cpu')
            assert torch.equal(batch['teacher'],logits[[2,0],4:8])
            for b,i in enumerate([2,0]):
                c=rows[i];expected=hidden[c['offset']:c['stop']]+(100 if parent else 0)
                assert torch.equal(batch['hidden'][b,:len(expected)],expected)
        finally:dataset.close()


def test_cut_zero_vjp_and_native_seam_preserve_frozen_parent_and_full_input_gradients(tmp_path):
    model=native();rows=study.content_items(tiny_rows())
    # A real native suffix accepts the same HF arguments as the packed suffix.
    class Tail(torch.nn.Module):
        def __init__(self,blocks):super().__init__();self.blocks=torch.nn.ModuleList(blocks)
        def forward(self,x,**kw):
            for block in self.blocks:x=study.calibration._hidden(block(x,**kw))
            return (x,)
    suffix=Tail(list(model.model.layers[1:])).eval().requires_grad_(False)
    embeddings=torch.cat([model.model.embed_tokens(torch.tensor(c['input_ids'])) for c in rows])
    source=tmp_path/'boundary_00.npy';np.save(source,embeddings.view(torch.int16).numpy())
    with study.SplitRotaryTeacher(rows,source,0,model.model.layers[0],suffix,model.model.norm,
            model.lm_head.weight,model.model.rotary_emb,native_rotary=deepcopy(model.model.rotary_emb)) as teacher:
        targets=torch.cat([teacher.native_forward(i)[0] for i in range(3)]).detach()
        nextbank=tmp_path/'boundary_01.npy';np.save(nextbank,targets.view(torch.int16).numpy())
        assert study.check_native_seam(teacher,nextbank)['bit_exact']
        inputs={}
        def leaf(_module,args):
            x=args[0].detach().requires_grad_();inputs['x']=x;return (x,*args[1:])
        handle=teacher.model.layers[0].register_forward_pre_hook(leaf)
        try:gradient=torch.autograd.grad(teacher.forward_case(0)[0,-1].float().square().sum(),inputs['x'])[0]
        finally:handle.remove()
        assert torch.isfinite(gradient).all() and gradient[0,0].abs().sum()>0
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        capture=tmp_path/'capture'
        with study.GroupedTeacherCapture(capture,teacher,[0],rows,[0,1],teacher_fingerprint='actual-cut0',
                probe_groups=2,probes_per_group=8,activation_positions=16,position_capacity=416,
                seed=study.calibration.calibration_seed(0)) as recorder:
            for index in range(3):recorder.capture_case(index,lambda i=index:teacher.forward_case(i))
            metadata=recorder.finish()
        assert metadata['blocks']['0']['shape']==dict(width=8,intermediate=12,heads=2,kv_heads=1,head_dim=4)
        assert metadata['calibration_indices']==[0,1] and metadata['development_indices']==[2]
        derivatives=np.load(capture/'block_00/mlp_jacobian_bf16.npy')
        assert derivatives.dtype==np.int16 and derivatives.shape==(7*2*8,12)
        statistics=torch.load(capture/'block_00/statistics.pt',weights_only=True)
        assert statistics['pair_scores'].shape==(2,1,2) and statistics['input_grams'].shape==(2,8,8)
        logits=torch.from_numpy(np.load(capture/'accepted_teacher_logits.npy').copy()).view(torch.bfloat16)
        expected=torch.nn.functional.linear(teacher.forward_case(0)[0,rows[0]['content_loss_positions']],model.lm_head.weight)
        assert torch.equal(logits[0],expected)
