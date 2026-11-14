from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding,Qwen2RMSNorm

from experiments.structure_net import jspace_joint_evaluate as evaluate
from experiments.structure_net import jspace_joint_recovery as joint
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_quadratic_qwen import QuadraticQwenBlock
from experiments.structure_net.jspace_residual_carry import ResidualCarryBlock
from experiments.structure_net.jspace_quadratic_evaluate import stream_score


def fixture():
    torch.manual_seed(77)
    config=Qwen2Config(hidden_size=16,intermediate_size=24,num_attention_heads=4,num_key_value_heads=2,
        num_hidden_layers=2,vocab_size=23,max_position_embeddings=32,attention_dropout=0.)
    config._attn_implementation="eager"
    blocks=[]
    for layer in range(2):
        native=Qwen2DecoderLayer(config,layer_idx=layer).eval()
        affine=AffineQwenBlock(native,torch.linalg.qr(torch.randn(16,8)).Q,torch.randn(1,16)/4,
            trainable=True,parameter_dtype=torch.float32)
        wrapped=QuadraticQwenBlock(affine,mode="softmax",learn_metric=True)
        for p in wrapped.parameters():p.data=p.data.bfloat16()
        blocks.append(ResidualCarryBlock(wrapped))
    return joint.JointCarryModel(torch.randn(23,16).bfloat16(),blocks,torch.rand(16)+.5,1e-6,
        Qwen2RotaryEmbedding(config=config)),config


def test_fresh_shard_reload_matches_direct_fp32_full_model_and_streamed_scores(tmp_path):
    original,config=fixture()
    selected=tmp_path/"selected"
    joint.save_selected(original,selected,metadata=dict(epoch=1,development={}))
    control=deepcopy(original).float().eval()
    def factory(layer,device):
        blank=deepcopy(original.blocks[layer]).to(device)
        with torch.no_grad():
            for p in blank.parameters():p.zero_()
        return blank
    restored=[]
    for layer in range(2):
        block,sizes=evaluate.load_candidate_block({},selected,layer,"cpu",block_factory=factory)
        restored.append(block)
        assert sizes["checkpoint_tensor_bytes"]==evaluate.tensor_storage_bytes(original.blocks[layer])
        assert sizes["fp32_evaluation_tensor_bytes"]==evaluate.tensor_storage_bytes(control.blocks[layer])
        assert sizes["checkpoint_tensor_bytes"]<sizes["fp32_evaluation_tensor_bytes"]
    cases=[dict(identity="a",family="tiny",answer=0,choices=["A","B"],input_ids=[1,2,3,4]),
           dict(identity="b",family="tiny",answer=1,choices=["A","B"],input_ids=[1,5,6,3,7])]
    norm=Qwen2RMSNorm(16,eps=1e-6);norm.weight=torch.nn.Parameter(control.norm_gain.detach().clone(),requires_grad=False)
    result=stream_score(cases,input_embedding=control.embedding,block_loader=lambda layer:restored[layer],
        final_readout=(norm,control.embedding),label_ids=[20,21],scratch_dir=tmp_path/"scratch",
        rotary=control.rotary,layer_count=2,microbatch=2,device="cpu")
    with torch.no_grad():
        for row,case in enumerate(cases):
            ids=torch.tensor([case["input_ids"]]);length=ids.shape[1]
            positions=torch.arange(length)[None]
            mask=(positions[:,:,None]>=positions[:,None,:])[:,None]
            chosen=torch.tensor([[0,1,length-2,length-1]])
            logits=control(ids,positions,mask,chosen)[0,-1,[20,21]]
            torch.testing.assert_close(torch.tensor(result["records"][row]["logits"]),logits,atol=1e-5,rtol=1e-5)
    assert not (tmp_path/"scratch").exists()


def test_storage_counts_tied_embedding_once_and_includes_nonfloating_buffers():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(7,5,dtype=torch.bfloat16))
            self.alias=self.weight;self.register_buffer("indices",torch.arange(3))
    model=Tied()
    assert evaluate.tensor_storage_bytes(model,model.weight,model.weight[:2])==7*5*2+3*8


def test_contaminated_duplicate_or_wrong_size_confirmation_is_rejected():
    cases=[dict(identity="fresh1"),dict(identity="fresh2")]
    assert evaluate.validate_confirmation(cases,{"old","training"},expected_count=2)==cases
    with pytest.raises(ValueError,match="overlaps"):
        evaluate.validate_confirmation(cases,{"fresh2"},expected_count=2)
    with pytest.raises(ValueError,match="uniqueness"):
        evaluate.validate_confirmation([cases[0],cases[0]],set(),expected_count=2)
    with pytest.raises(ValueError,match="population"):
        evaluate.validate_confirmation(cases,set())


def test_confirmation_is_not_opened_before_completed_training_gate(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    calls=[]
    def incomplete(_):calls.append("training_gate");raise ValueError("training incomplete")
    monkeypatch.setattr(evaluate,"completed_training",incomplete)
    monkeypatch.setattr(evaluate.recipe,"read",lambda *_:pytest.fail("Read confirmation before completion"))
    with pytest.raises(ValueError,match="training incomplete"):
        evaluate.prepare(tmp_path/"eval",tmp_path/"train",split="confirmation",confirmation_root=tmp_path/"unopened")
    assert calls==["training_gate"] and not (tmp_path/"eval").exists()


def test_completed_training_checks_strict_reload_before_selected_state(tmp_path,monkeypatch):
    fingerprint="fixed"
    monkeypatch.setattr(joint,"verify",lambda _:dict(fingerprint=fingerprint))
    monkeypatch.setattr(joint,"verify_pilot",lambda *_:None)
    directory=tmp_path/"train";directory.mkdir()
    joint.recipe.write(directory/"result.json",dict(status="completed",systems_only=False,strict_reload_exact=False))
    joint.recipe.write(directory/"manifest.json",dict(status="completed",systems_only=False,fingerprint=fingerprint,
        result_sha256=joint.recipe.sha(directory/"result.json")))
    with pytest.raises(ValueError,match="exact selected reload"):
        evaluate.completed_training(tmp_path)


def test_five_percentage_point_gate_uses_paired_counts():
    native=[dict(identity=str(i),family="tiny",answer=0,prediction=0) for i in range(20)]
    candidate=deepcopy(native);candidate[0]["prediction"]=1
    quality=evaluate.paired_summary(native,candidate)
    assert quality["aggregate_quality_pass"]
    assert quality["quality"]["overall"]["losses"]==1
    candidate[1]["prediction"]=1
    assert not evaluate.paired_summary(native,candidate)["aggregate_quality_pass"]
