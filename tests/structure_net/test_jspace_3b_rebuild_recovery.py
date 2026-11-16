"""Actual tiny local/joint training, packed scoring, and selection boundaries."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_rebuild_recovery as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack

data = study.cache.data


class TinyDataset:
    def __init__(self, native, cut):
        provenance = dict(source_uri="public-fixture", source_revision="v1", tokenizer_sha256="tokenizer",
            chat_template_sha256="template", prompt_provenance="fixture", response_provenance="reference",
            assistant_boundary_verified=True, source_family="fixture")
        self.records = [data.ResponseRecord(example_id=f"case{i}",group_id=f"group{i}",
            split="train" if i < 2 else "development", task="fixture",prompt_ids=(1,3+i),
            response_ids=(7+i,31),eos_token_id=31,finish_reason="eos",provenance=provenance) for i in range(3)]
        self.training, self.development, self.pad_token_id = [0,1],[2],0
        self.saved = []
        for record in self.records:
            captured = {}
            def save(which):
                def hook(_module,args): captured[which]=args[0].detach().clone()
                return hook
            h1 = native.model.layers[cut].register_forward_pre_hook(save("boundary_hidden"))
            target = native.model.norm if cut+1 == len(native.model.layers) else native.model.layers[cut+1]
            h2 = target.register_forward_pre_hook(save("next_boundary_hidden"))
            with torch.no_grad():
                out = native.model(torch.tensor([record.input_ids]),use_cache=False).last_hidden_state
            h1.remove(); h2.remove(); captured["teacher_hidden"]=out.detach()
            self.saved.append(captured)

    def batch(self,indices):
        assert len(indices)==1
        i=indices[0]; record=self.records[i]
        b=data.response_batch([record],pad_token_id=0,max_length=8)
        b.update({k:v.clone() for k,v in self.saved[i].items()})
        b.update(teacher_input_ids=b["input_ids"].clone(),teacher_record_ids=[record.example_id])
        return b


def fixture(cut):
    torch.manual_seed(20261116)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,vocab_size=32,max_position_embeddings=32,
        tie_word_embeddings=True,attention_dropout=0.,pad_token_id=0,eos_token_id=31)
    cfg._attn_implementation="eager"
    native=Qwen2ForCausalLM(cfg).bfloat16().eval().requires_grad_(False)
    blocks=[]
    for layer in range(cut,2):
        U=torch.linalg.qr(torch.randn(8,8)).Q[:,:4]
        b=AffineQwenBlock(native.model.layers[layer],U,torch.randn(8,8)/100,
            trainable=True,parameter_dtype=torch.bfloat16)
        b.compile_polynomial(dict(anchors=[0,1],mean=torch.zeros(12),scale=torch.ones(12),
            coefficients={"2":torch.randn(12,6)/100}),[10,11])
        for parameter in b.parameters(): parameter.data=parameter.data.bfloat16()
        blocks.append(b)
    stack=CompactQwenStack(blocks,geometry_mode="bf16",pad_to_multiple=16)
    norm=deepcopy(native.model.norm)
    E=native.lm_head.weight.detach()
    rotary=study.model_source.partial.training_rotary(cfg,"cpu")
    return cfg,stack,norm,E,rotary,TinyDataset(native,cut)


@pytest.mark.parametrize("cut",[0,1])
def test_local_one_pass_updates_only_new_block_then_joint_updates_entire_suffix(cut,monkeypatch):
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:pytest.fail("CPU fixture initialized CUDA"))
    cfg,stack,norm,E,rotary,dataset=fixture(cut)
    local=study.LocalTrainer(stack,norm,E,rotary,dataset,[0,1],
        config=study.LocalConfig(learning_rate=.02,optimizer_chunk_bytes=64))
    assert local.normalization["response_tokens"]==4 # EOS is an eligible target.
    fixed=local.fixed_signature()
    receipt=local.train_window([0,1])
    assert receipt["only_new_block_gradients"]
    assert all(receipt["optimizer"]["master_changed_by_parameter"])
    assert local.fixed_signature()==fixed
    assert local.finish()["actual_bf16_new_block_changed"]
    with pytest.raises(ValueError,match="One-pass"):local.train_window([0])
    local.close()
    trainer=study.training.ResponseTrainer(stack,norm,E,rotary,dataset,teacher_id="native-fixture",
        config=replace(study.training.ResponseConfig(),learning_rate=.02,max_length=8,
                       optimizer_chunk_bytes=64,max_supervised_tokens=4,token_chunk_size=1),
        expected_layers=2,host_reserve_bytes=0)
    try:
        before=trainer.parameter_fingerprints()
        row=trainer.train_window([0,1])
        assert row["gradients_present_by_block"]==[True]*(2-cut)
        assert trainer.change_report(before)["bf16_changed_by_block"]==[True]*(2-cut)
    finally:trainer.close()


def test_local_normalizer_uses_only_train_response_prediction_positions():
    _,stack,norm,E,rotary,dataset=fixture(1)
    # Prompt position0 and target tokens outside the selected predicting rows
    # must not affect the frozen scalar. DEV targets also never enter it.
    selected=dataset.batch([0])["loss_mask"].bool()
    expected=sum(float(dataset.saved[i]["next_boundary_hidden"][selected].float().square().mean(-1).sum())
                 for i in (0,1))/4
    for i in (0,1):dataset.saved[i]["next_boundary_hidden"][~selected]=200
    dataset.saved[2]["next_boundary_hidden"].fill_(300)
    local=study.LocalTrainer(stack,norm,E,rotary,dataset,[0,1],
        config=study.LocalConfig(optimizer_chunk_bytes=64))
    try:
        assert local.normalization["energy"]==expected
        with pytest.raises(ValueError,match="One-pass"):local.train_window([2])
    finally:local.close()
    with pytest.raises(ValueError,match="TRAIN"):
        study.LocalTrainer(stack,norm,E,rotary,dataset,[0,2])


def test_local_forward_is_same_first_block_reconstructed_geometry():
    _,stack,norm,E,rotary,dataset=fixture(1)
    batch=dataset.batch([0]); h=batch["boundary_hidden"]
    actual=study._first_output(stack,rotary,batch,torch.device("cpu"))
    positions=torch.arange(h.shape[1])[None]
    mask=torch.full((1,1,h.shape[1],h.shape[1]),torch.finfo(h.dtype).min,dtype=h.dtype).triu(1)
    expected=stack(h,attention_mask=mask,position_ids=positions,
                   position_embeddings=rotary(h,positions),use_cache=False)[0]
    assert torch.equal(actual,expected)


def test_packed_mc_evaluates_fresh_reload_and_not_unpacked_scorer(tmp_path,monkeypatch):
    cfg,stack,norm,E,rotary,dataset=fixture(1)
    packed=study.packed_source.PackedResponseModel(study.builder.packed_clone(stack),norm,E,rotary,
                                                  config=cfg,max_length=8)
    receipt=study.packed_source.export_packed_snapshot(packed,tmp_path/"export",identities={"fixture":"native"})
    model,reload=study.packed_source.load_packed_snapshot(tmp_path/"export")
    assert receipt["packed_state_sha256"]==reload["packed_state_sha256"]
    cases=[dict(identity=f"c{i}",family=f"f{i//64}",answer=0,choices=["a","b"],split="development")
           for i in range(256)]
    native=[dict(identity=c["identity"],family=c["family"],answer=0,prediction=0,logits=[1.,0.]) for c in cases]
    batch=dataset.batch([2]); calls=[]
    original=model.hidden_forward
    def hidden(b):calls.append(True);return original(b)
    monkeypatch.setattr(model,"hidden_forward",hidden)
    monkeypatch.setattr(study.model_source.joint.training,"score_unpacked",
                        lambda *a,**k:pytest.fail("Packed evaluation used unpacked scorer"))
    mc=SimpleNamespace(items=cases,training=[],development=list(range(256)),native_correct=256,
        native_records=native,label_ids=[7,8],batch=lambda indices:batch)
    result=study.packed_mc(model,mc)
    assert result["packed"] and result["metrics"]["count"]==256 and len(calls)==256


def candidate(label,*,passed=False,nll=1.,tokens=0,update=0,semantic=None,grading="completed"):
    return dict(label=label,eligibility={"passed":passed},nll=nll,response_tokens=tokens,update=update,
                semantic_task_success=semantic,grading_status=grading)


def test_selection_never_promotes_missing_judge_or_cheap_failure_and_prefers_early_equivalent():
    raw=candidate("prefit",nll=1.1,grading="not_run_failed_cheap_retention")
    early=candidate("early",passed=True,nll=1.,tokens=5000,semantic=.75)
    later=candidate("later",passed=True,nll=1.,tokens=10000,semantic=.75)
    missing=candidate("missing",nll=.5,grading="judge_unavailable")
    result=study.select([raw,early,later,missing])
    assert result["selected"] is early and result["diagnostic_best"] is missing
    assert result["relative_retention_pass"] and not result["confirmed"]
    failed=study.select([raw,missing])
    assert not failed["relative_retention_pass"] and failed["selected"] is None
    with pytest.raises(ValueError,match="Missing"):
        study.select([raw,candidate("forged",passed=True,grading="judge_unavailable")])


def test_cheap_retention_does_not_allow_mc_to_rescue_nll_or_nonfinite():
    mc={"eligibility":{"passed":True}}
    assert study.cheap_gate({"nll":1.09},{"nll":1.},mc)["passed"]
    assert not study.cheap_gate({"nll":1.11},{"nll":1.},mc)["passed"]
    assert not study.cheap_gate({"nll":float("nan")},{"nll":1.},mc)["passed"]
    assert not study.cheap_gate({"nll":1.},{"nll":1.},{"eligibility":{"passed":False}})["passed"]


def test_new_relative_canary_gate_can_retain_a_teacher_below_absolute_fifty_percent():
    rows=[dict(id=str(i),input_ids=[1,3],max_new_tokens=128,eos_token_ids=[151645,151643],
               generation_success=i<2) for i in range(6)]
    native={"records":rows}; student={"records":deepcopy(rows)}
    assert not study.evaluation.generation_gate(native,student)["passed"]
    assert study.evaluation.generation_gate(native,student,minimum_student_rate=0.)["passed"]
    student["records"][1]["generation_success"]=False
    assert not study.evaluation.generation_gate(native,student,minimum_student_rate=0.)["passed"]


def test_discarded_systems_pilot_saves_real_updates_and_exact_packed_reload(tmp_path):
    cfg,stack,norm,E,rotary,dataset=fixture(0)
    before=study.initializer_fingerprint(stack,norm,E,rotary)
    result=study.systems_pilot(stack,norm,E,rotary,dataset,local_indices=[0,1],joint_indices=[0,1],
        local_config=study.LocalConfig(learning_rate=.02,optimizer_chunk_bytes=64),
        response_config=replace(study.training.ResponseConfig(),learning_rate=.02,max_length=8,
            optimizer_chunk_bytes=64,max_supervised_tokens=4,token_chunk_size=1),teacher_id="native-fixture",
        model_config=cfg,directory=tmp_path/"pilot",expected_layers=2,host_reserve_bytes=0)
    assert result["passed"] and result["discarded"] and result["systems_only"]
    assert result["initializer"]==before
    assert study.initializer_fingerprint(stack,norm,E,rotary)!=before
    assert result["strict_trainable_reload"] and result["full_vocabulary_packed_replay_exact"]
    assert result["joint_update"]["cumulative_supervised_tokens"]==4
    assert result["local_fixed_parent"] and result["joint_change"]["bf16_changed_by_block"]==[True,True]
    # The actual scientific constructor recreates the exact *pre-pilot* state.
    _,fresh,fresh_norm,fresh_E,fresh_rope,_=fixture(0)
    assert study.initializer_fingerprint(fresh,fresh_norm,fresh_E,fresh_rope)==before
