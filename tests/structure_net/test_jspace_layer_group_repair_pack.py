"""Packing the selected delta into compact parents, with native endpoints fixed."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RMSNorm,Qwen2RotaryEmbedding

from experiments.structure_net import jspace_layer_group_repair_pack as packed


def fixture(tmp_path):
    torch.manual_seed(823)
    config=Qwen2Config(hidden_size=16,intermediate_size=24,num_attention_heads=4,
        num_key_value_heads=2,num_hidden_layers=4,vocab_size=23,max_position_embeddings=32,attention_dropout=0.)
    config._attn_implementation="eager"
    selected=tmp_path/"selected";selected.mkdir();native=[];recovered=[];blocks={};hashes={}
    selected_layers=[0,2,3]
    for layer in range(4):
        original=Qwen2DecoderLayer(config,layer_idx=layer).bfloat16().float()
        fit=dict(basis=torch.linalg.qr(torch.randn(16,8)).Q,means=torch.randn(1,16)/5)
        block=packed.repair.semantic.build_enriched_block(deepcopy(original),fit,parameter_dtype=torch.float32)
        # Distinguish recovered parents from recompiling native weights.
        with torch.no_grad():
            for index,parameter in enumerate(block.parameters()):parameter.add_((layer+1)*(index+1)*.001)
        initial=tmp_path/f"original_selected_{layer:02d}.pt"
        packed.recipe.save(initial,dict(block=packed.repair.joint._cpu(block.state_dict()),epoch=7,layer=layer))
        fit_path=tmp_path/f"fit_{layer:02d}.pt";packed.recipe.save(fit_path,fit)
        blocks[str(layer)]=dict(fit_path=str(fit_path),checkpoint=str(initial),checkpoint_sha256=packed.recipe.sha(initial),
            selected_epoch=7,rank=block.rank,head_dim=block.head_dim,intermediate_size=block.intermediate_size)
        if layer in selected_layers:
            with torch.no_grad():
                for index,parameter in enumerate(block.parameters()):parameter.add_((layer+1)*(index+1)*.003)
            path=selected/f"block_{layer:02d}.pt";packed.recipe.save(path,packed.repair.joint._cpu(block.state_dict()))
            hashes[path.name]=packed.recipe.sha(path)
        native.append(original);recovered.append(block)
    plan=dict(model=dict(path="tiny-native"),blocks=blocks,selected_directory=str(selected),
        checkpoint_artifacts=hashes,selected_layers=selected_layers)
    embedding=torch.randn(23,16).bfloat16().float();norm=Qwen2RMSNorm(16)
    with torch.no_grad():norm.weight.copy_(torch.linspace(.7,1.3,16).bfloat16().float())
    return plan,native,recovered,embedding,norm,config


def test_actual_packed_four_block_stream_keeps_unselected_compact_and_native_norm(tmp_path):
    plan,native,recovered,embedding,norm,config=fixture(tmp_path);calls=[]
    def native_loader(path,layer,**_):
        assert path=="tiny-native";calls.append(layer);return deepcopy(native[layer])
    blocks=[packed.load_packed_block(plan,i,loader=native_loader) for i in range(4)]
    assert calls==list(range(4))
    assert not (tmp_path/"selected/block_01.pt").exists() and not (tmp_path/"selected/norm_gain.pt").exists()
    original_norm=packed.norm_fingerprint(norm)
    for layer,block in enumerate(blocks):
        assert not any(p.requires_grad for p in block.parameters())
        torch.testing.assert_close(block.basis,recovered[layer].basis,atol=0,rtol=0)
    cases=[dict(identity="binary",family="tiny",split="development",input_ids=[1,4,2],choices=["A","B"],answer=1),
        dict(identity="four",family="tiny",split="development",input_ids=[7,3,1,5,2],choices=list("ABCD"),answer=2)]
    labels=[19,20,21,22];rotary=Qwen2RotaryEmbedding(config=config)
    result=packed.assembly.single.stream_score(cases,input_embedding=embedding,block_loader=lambda i:blocks[i],
        final_readout=(norm,embedding),label_ids=labels,scratch_dir=tmp_path/"scratch",rotary=rotary,layer_count=4,device="cpu",microbatch=2)
    with torch.no_grad():
        for case,record in zip(cases,result["records"]):
            hidden=embedding[case["input_ids"]][None];positions=torch.arange(hidden.shape[1])[None]
            mask=torch.ones(1,1,hidden.shape[1],hidden.shape[1],dtype=torch.bool).tril()
            for block in recovered:
                hidden=block(hidden,attention_mask=mask,position_ids=positions,position_embeddings=rotary(hidden,positions))
            expected=F.linear(norm(hidden[:,-1]),embedding[labels[:len(case["choices"])]])[0]
            torch.testing.assert_close(torch.tensor(record["logits"]),expected,atol=2e-5,rtol=2e-5)
    assert packed.norm_fingerprint(norm)==original_norm
    state=packed.assembly.tensor_storage(embedding,norm,rotary)
    assert state["storage_bytes"]==packed.assembly.tensor_storage(embedding,embedding,norm,rotary)["storage_bytes"]
    assert all(packed.assembly.tensor_storage(block)["storage_bytes"]>0 for block in blocks)


def test_selected_delta_fixed_geometry_cannot_change_even_with_updated_file_digest(tmp_path):
    plan,native,_,_,_,_=fixture(tmp_path);path=tmp_path/"selected/block_02.pt"
    # Detach the test edit from mmap storage before overwriting its backing file.
    state=deepcopy(packed.recipe.load(path));key=next(k for k in state if k.endswith(".basis"))
    state[key][0,0]+=.1;torch.save(state,path);plan["checkpoint_artifacts"][path.name]=packed.recipe.sha(path)
    with pytest.raises(ValueError,match="fixed geometry"):
        packed.load_packed_block(plan,2,loader=lambda _path,layer,**_:deepcopy(native[layer]))


def test_changed_unselected_parent_bytes_are_rejected(tmp_path):
    plan,native,_,_,_,_=fixture(tmp_path);path=tmp_path/"original_selected_01.pt"
    with path.open("ab") as stream:stream.write(b"tamper")
    with pytest.raises(ValueError,match="checkpoint changed"):
        packed.load_packed_block(plan,1,loader=lambda _path,layer,**_:deepcopy(native[layer]))


def authenticated_fixture(tmp_path,monkeypatch):
    selected=tmp_path/"train/selected";selected.mkdir(parents=True);verification=tmp_path/"verification";verification.mkdir()
    plan=dict(fingerprint="repair",selected_layers=[0,2,3],assembly_fingerprint="assembly",assembly_root="parent",
        selection=dict(search_fingerprint="search"))
    metrics=dict(correct=46,count=64);delta=dict(schema=packed.repair.SCHEMA,selected_layers=[0,2,3],
        parent_assembly_fingerprint="assembly",search_fingerprint="search",
        artifacts={f"block_{i:02d}.pt":"unused" for i in (0,2,3)},metadata=dict(epoch=4,development=metrics))
    packed.recipe.write(selected/"manifest.json",delta)
    result=dict(selected_epoch=4,selected=metrics,quality_pass=False,selected_manifest_sha256=packed.recipe.sha(selected/"manifest.json"))
    fresh=dict(status="completed",passed=True,fingerprint="repair",selected_layers=[0,2,3],selected_epoch=4,
        selected_manifest_sha256=result["selected_manifest_sha256"],training_quality_pass=False,strict_state_load=True,
        frozen_state_unchanged=True,predictions_exact=True,logits_pass=True,metrics_pass=True)
    packed.recipe.write(verification/"manifest.json",dict(passed=True))
    monkeypatch.setattr(packed.repair,"verify",lambda _:plan)
    monkeypatch.setattr(packed.repair,"checked",lambda _root,stage,_plan:result if stage=="train" else fresh if stage=="verification" else {})
    monkeypatch.setattr(packed.assembly,"verify",lambda _:dict(fingerprint="assembly",arms=dict(all_compact=list(range(36)))))
    return plan,result,fresh,delta


def test_packing_accepts_scientific_quality_miss_but_requires_matching_fresh_receipt(tmp_path,monkeypatch):
    _,result,fresh,_=authenticated_fixture(tmp_path,monkeypatch)
    assert packed.authenticated(tmp_path)[1]==result and result["quality_pass"] is False
    fresh["selected_epoch"]=3
    with pytest.raises(ValueError,match="fresh subgroup"):packed.authenticated(tmp_path)


def test_delta_manifest_must_not_include_a_trained_final_norm(tmp_path,monkeypatch):
    _,_,_,delta=authenticated_fixture(tmp_path,monkeypatch)
    delta["artifacts"]["norm_gain.pt"]="wrong"
    (tmp_path/"train/selected/manifest.json").write_text(__import__("json").dumps(delta))
    with pytest.raises(ValueError,match="Delta3"):packed.authenticated(tmp_path)


def test_both_reference_logits_and_predictions_are_required():
    cases=[dict(identity="x",family="tiny",answer=0,choices=["A","B"])]
    records=[dict(identity="x",family="tiny",answer=0,prediction=0,logits=[1.,0.])]
    selected,fresh=dict(records=deepcopy(records)),dict(records=deepcopy(records))
    assert packed.compare(records,selected,fresh,cases)["passed"]
    fresh["records"][0]["logits"][0]=2.
    assert not packed.compare(records,selected,fresh,cases)["passed"]


def test_packing_requires_exact_3060_7168_reservation():
    with pytest.raises(ValueError,match="leased3060"):
        packed.cap_allocator(dict(allocation={"GPU-0370fdb0-test":5120}),visible="GPU-0370fdb0-test")
    with pytest.raises(ValueError,match="leased3060"):
        packed.cap_allocator(dict(allocation={"GPU-7b1db3cb-test":7168}),visible="GPU-7b1db3cb-test")


def test_denied_reservation_never_initializes_cuda_or_loads_weights(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease,jspace_qwen_parallel_model
    monkeypatch.setattr(packed,"verify",lambda _:dict(fingerprint="packing"))
    def denied(_):raise RuntimeError("no reservation")
    monkeypatch.setattr(jspace_full_lease,"wait_ready",denied)
    monkeypatch.setattr(packed,"cap_allocator",lambda *_a,**_kw:pytest.fail("Allocator before lease"))
    monkeypatch.setattr(torch.cuda,"init",lambda:pytest.fail("CUDA before lease"))
    monkeypatch.setattr(jspace_qwen_parallel_model,"load_readout_norm",lambda *_a,**_kw:pytest.fail("Weights before lease"))
    with pytest.raises(RuntimeError,match="no reservation"):packed.run(tmp_path)
    failure=packed.recipe.read(tmp_path/"verification/failure.json")
    assert failure["lease_id"] is None and failure["peak_reserved_bytes"] is None
