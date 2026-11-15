"""CPU contracts for one-step suffix expansion, recovery and BF16 controls."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RMSNorm,Qwen2RotaryEmbedding

from experiments.structure_net import jspace_incremental_suffix as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def components():
    torch.manual_seed(713)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=12,num_attention_heads=2,num_key_value_heads=1,
        num_hidden_layers=36,vocab_size=16,max_position_embeddings=32,attention_dropout=0.)
    cfg._attn_implementation="eager";blocks=[];bases=[]
    for layer in range(29,36):
        native=Qwen2DecoderLayer(cfg,layer_idx=layer).bfloat16()
        u=torch.linalg.qr(torch.randn(8,8)).Q[:,:4];bases.append(u.clone())
        block=AffineQwenBlock(native,u,torch.randn(1,8)/10,parameter_dtype=torch.bfloat16,
            head_projection=dict(rope_pairs=torch.tensor([[0]]),value_basis=torch.eye(4)[None,:,:2]))
        relation=dict(anchors=[0,1],mean=torch.zeros(12),scale=torch.ones(12),coefficients={"2":torch.randn(12,6)/100})
        blocks.append(block.compile_polynomial(relation,[10,11]))
    old=CompactQwenStack(deepcopy(blocks[1:]),geometry_mode="bf16",pad_to_multiple=16)
    norm=Qwen2RMSNorm(8).bfloat16()
    with torch.no_grad():
        norm.weight.fill_(3.)
        for p in old.parameters():p.add_(.015625)
    previous=dict(stack=study.core.recursive_cpu(old.state_dict()),norm=study.core.recursive_cpu(norm.state_dict()))
    for index,block in enumerate(blocks[1:]):
        prefix=f"blocks.{index}.";block.load_state_dict({k[len(prefix):]:v for k,v in previous["stack"].items() if k.startswith(prefix)},strict=True)
        block.basis=bases[index+1]
    stack=CompactQwenStack(blocks,geometry_mode="bf16",pad_to_multiple=16,checkpoint_blocks=True)
    fresh_norm=Qwen2RMSNorm(8).bfloat16();proof=study.preserve_inheritance(stack,fresh_norm,previous)
    return stack,fresh_norm,torch.randn(16,8).bfloat16(),Qwen2RotaryEmbedding(config=cfg),previous,proof


def data(tmp_path):
    torch.manual_seed(614);items=[];offset=0
    for index,length in enumerate((4,6,5,4,5,4,6,5)):
        answer=index%2
        items.append(dict(identity=str(index),family="tiny",split="train" if index<4 else "development",
            input_ids=list(range(length)),choices=["A","B"],answer=answer,targets=[4,5,6,2+answer],
            loss_positions=[0,1,length-2,length-1],offset=offset,stop=offset+length))
        offset+=length
    values=dict(boundary=torch.randn(offset,8).bfloat16(),accepted=torch.randn(8,4,16).bfloat16(),original=torch.randn(8,4,16).bfloat16())
    for name,value in values.items():np.save(tmp_path/(name+".npy"),value.view(torch.int16).numpy())
    return study.core.BoundaryDataset(items,tmp_path/"boundary.npy",tmp_path/"accepted.npy",
        original_logits_path=tmp_path/"original.npy",original_indices=list(range(8)))


def trainer(tmp_path,epochs=2):
    stack,norm,readout,rotary,*_=components();dataset=data(tmp_path)
    config=study.core.RecoveryConfig(epochs=epochs,microbatch=2,accumulation=2,seed=20861,optimizer_chunk_bytes=128)
    return study.Trainer(stack,norm,readout,rotary,dataset,[2,3],config=config)


def test_fixed_protocol_and_accuracy_first_ties_and_two_gates():
    cfg=study.CONFIG
    assert (cfg.epochs,cfg.microbatch,cfg.accumulation,cfg.learning_rate,cfg.seed)==(12,4,4,1.2e-4,20861)
    assert 512//(cfg.microbatch*cfg.accumulation)*cfg.epochs==384
    assert (cfg.kl_weight,cfg.ce_weight,cfg.temperature)==(.7,.3,2.)
    assert study.selection_key(dict(correct=48,ce=9.))>study.selection_key(dict(correct=47,ce=.01))
    assert study.selection_key(dict(correct=48,ce=1.))>study.selection_key(dict(correct=48,ce=2.))
    assert study.gates(48)["passed"] and not study.gates(47)["passed"]
    assert study.gates(48)["native"]["minimum_correct"]==45 and study.gates(48)["parent"]["minimum_correct"]==48


def test_seven_initializer_preserves_exact_old_six_and_interior_bridges():
    stack,norm,_,_,previous,proof=components()
    assert proof==dict(passed=True,unchanged_layers=list(range(30,36)),interior_bridges_preserved=5,
        new_layer=29,new_bridge=[29,30],norm_exact=True)
    for i,block in enumerate(stack.blocks[1:]):
        prefix=f"blocks.{i}."
        assert study.state_equal(block.state_dict(),{k[len(prefix):]:v for k,v in previous["stack"].items() if k.startswith(prefix)})
    for i,bridge in enumerate(stack.bridges[1:]):
        prefix=f"bridges.{i}."
        assert study.state_equal(bridge.state_dict(),{k[len(prefix):]:v for k,v in previous["stack"].items() if k.startswith(prefix)})
    with torch.no_grad():next(stack.blocks[3].parameters()).add_(.25)
    with pytest.raises(ValueError,match="Inherited block"):study.preserve_inheritance(stack,norm,previous)


def test_actual_seven_block_update_moves_all_fp32_masters_even_when_bf16_norm_is_unchanged(tmp_path):
    value=trainer(tmp_path);before=study.fixed_state(value.stack,value.readout,value.rotary)
    masters=[study.master_fingerprint(b,value.optimizer) for b in value.stack.blocks]
    norm_master=study.master_fingerprint(value.norm,value.optimizer);norm_bf16=value.norm.weight.detach().clone()
    result=value.train_window(value.dataset.training)
    assert result["gradients_present_by_block"]==[True]*7
    assert all(study.master_fingerprint(b,value.optimizer)!=old for b,old in zip(value.stack.blocks,masters))
    assert study.master_fingerprint(value.norm,value.optimizer)!=norm_master
    assert torch.equal(value.norm.weight,norm_bf16)
    assert result["optimizer"]["unclipped_gradient_norm"]>0
    assert study.fixed_state(value.stack,value.readout,value.rotary)==before
    assert value.readout.grad is None and all(not b.requires_grad for b in value.stack.buffers())
    value.dataset.close()


def test_initializer_mismatch_stops_before_any_scientific_update(tmp_path):
    value=trainer(tmp_path);metrics,rows=value.packed_development();wrong=deepcopy(rows)
    wrong[0]["logits"]=list(reversed(wrong[0]["logits"]));wrong[0]["prediction"]=1-wrong[0]["prediction"]
    with pytest.raises(RuntimeError,match="before training"):
        value.fit(tmp_path/"fit",initial_reference=dict(initial=metrics,initial_records=wrong))
    assert value.updates==0
    value.dataset.close()


def test_small_real_fit_records_initializer_every_update_and_selected_reload(tmp_path):
    value=trainer(tmp_path);metrics,rows=value.packed_development();events=[]
    result=value.fit(tmp_path/"fit",initial_reference=dict(initial=metrics,initial_records=rows),
        event=lambda kind,**kw:events.append((kind,kw)))
    assert result["updates"]==2 and result["completed_epochs"]==2
    assert [r["epoch"] for r in result["history"]]==[0,1,2]
    assert sum(len(r["training"]) for r in result["history"])==2
    assert result["strict_reload_exact"] and result["fixed_state_unchanged"]
    selected=max(result["history"],key=lambda r:study.selection_key(r["metrics"]))
    assert selected["epoch"]==result["selected_epoch"]
    rates=[v["learning_rate"] for kind,v in events if kind=="update"]
    assert rates==pytest.approx([1.2e-4,6e-5])
    assert study.read(tmp_path/"fit/initializer_replay.json")["passed"]
    assert len(list((tmp_path/"fit").glob("epoch_*_development.json")))==3
    value.dataset.close()


def test_cache_rows_use_explicit_labels_and_original_identity_mapping(tmp_path):
    dataset=data(tmp_path)
    rows=study.rows_from_cache(dataset,[2,3],"accepted")
    for row,index in zip(rows,dataset.development):
        expected=dataset._bits(dataset.accepted[index,-1],"cpu").float()[[2,3]]
        assert row["prediction"]==int(expected.argmax()) and row["logits"]==expected.tolist()
    dataset.close()


def test_bf16_comparison_distinguishes_exact_decisions_and_logit_tolerance():
    cases=[dict(identity="x",family="tiny",answer=0,choices=["A","B"])]
    reference=[dict(identity="x",family="tiny",answer=0,prediction=0,logits=[40.,20.])]
    observed=deepcopy(reference);observed[0]["logits"][0]+=.25
    result=study.compare_rows(observed,reference,cases,expected_correct=1)
    assert result["passed"] and result["max_abs_logit_delta"]==.25
    observed[0]["logits"][0]+=1.
    assert not study.compare_rows(observed,reference,cases)["passed"]


def test_metadata_inventory_counts_original_prefix_tiedE_and_compact_suffix_once(tmp_path):
    stack,norm,_,rotary,*_=components()
    header={"model.embed_tokens.weight":dict(dtype="BF16",shape=[16,8]),
        "model.norm.weight":dict(dtype="BF16",shape=[8]),
        **{f"model.layers.{i}.weight":dict(dtype="BF16",shape=[8,8]) for i in range(36)}}
    raw=json.dumps(header).encode()
    with (tmp_path/"model.safetensors").open("wb") as stream:stream.write(len(raw).to_bytes(8,"little"));stream.write(raw)
    result=study.inventory(dict(model=dict(path=str(tmp_path))),stack,norm,rotary)
    assert result["frozen_native_prefix_bytes"]==29*8*8*2
    assert result["tied_embedding_readout_bytes"]==16*8*2
    assert result["full_model_tensor_bytes"]==sum(result[k] for k in ("frozen_native_prefix_bytes","tied_embedding_readout_bytes","packed_suffix_and_norm_bytes","rotary_bytes"))


def test_consumed_model_or_cache_identity_changes_are_detectable(tmp_path):
    path=tmp_path/"weights.bin";path.write_bytes(b"before");first=study.file_identity(path)
    path.write_bytes(b"after and different size");assert study.file_identity(path)!=first


def test_denied_reservation_precedes_cuda_and_model(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(study,"verify",lambda _:dict(fingerprint="test"))
    def deny(_):raise RuntimeError("no lease")
    monkeypatch.setattr(jspace_full_lease,"wait_ready",deny)
    monkeypatch.setattr(study,"cap_allocator",lambda *_a,**_kw:pytest.fail("allocator before lease"))
    monkeypatch.setattr(study,"build",lambda *_a,**_kw:pytest.fail("model before lease"))
    with pytest.raises(RuntimeError,match="no lease"):study.run(tmp_path,"pilot")
    assert study.read(tmp_path/"pilot/failure.json")["lease_id"] is None
