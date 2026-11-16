from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
from transformers import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3RMSNorm, Phi3RotaryEmbedding

from experiments.structure_net import jspace_phi_train as trial


@pytest.fixture
def system(tmp_path):
    torch.manual_seed(731)
    config=Phi3Config(hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=4,
        num_key_value_heads=2,vocab_size=32,max_position_embeddings=16,original_max_position_embeddings=16,
        bos_token_id=1,eos_token_id=2,pad_token_id=2,tie_word_embeddings=False,
        resid_pdrop=0.,embd_pdrop=0.,attention_dropout=0.,rope_theta=250000.)
    config._attn_implementation="eager"
    native=Phi3DecoderLayer(config,0).bfloat16().eval().requires_grad_(False)
    rotary=Phi3RotaryEmbedding(config)
    norm=Phi3RMSNorm(16,eps=config.rms_norm_eps).bfloat16()
    head=torch.randn(32,16).bfloat16()/8
    basis=torch.linalg.qr(torch.randn(16,16)).Q[:,:8]
    block=trial.AffinePhiBlock(native,basis,torch.randn(8,16)/10,parameter_dtype=torch.bfloat16,
        buffer_dtype=torch.float32,trainable=True)
    relation=dict(anchors=[0,2],mean=torch.zeros(24),scale=torch.ones(24),coefficients={"2":torch.randn(24,6)/100})
    block.compile_polynomial(relation,[3,4])
    root=tmp_path/"native_source"; (root/"native").mkdir(parents=True)
    response_dir=tmp_path/"data";response_dir.mkdir()
    labels=[9,10,11,12];(response_dir/"plan.json").write_text(json.dumps(dict(label_ids=labels)))
    records=[];items=[];inputs=[];teachers=[];pin=[];pout=[];offset=poffset=0
    for index in range(4):
        prompt=[3+index,15+index,20+index];response=[labels[index%2],2]
        provenance=dict(source_uri="fixture://"+str(index),source_revision="one",tokenizer_sha256="tok",
            chat_template_sha256="template",prompt_provenance="fixture",response_provenance="gold",
            assistant_boundary_verified=True)
        record=trial.response_data.ResponseRecord(str(index),str(index),"train" if index<2 else "development",
            "arc",prompt,response,provenance,"eos",2)
        records.append(asdict(record));x=torch.randn(1,5,16).bfloat16()
        pos=torch.arange(5).unsqueeze(0);mask=torch.full((1,1,5,5),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16).triu(1)
        with torch.no_grad(): y=native(x,position_embeddings=rotary(x,pos),attention_mask=mask)[0]
        px=x[:,:3].clone();ppos=torch.arange(3).unsqueeze(0)
        with torch.no_grad(): py=native(px,position_embeddings=rotary(px,ppos),attention_mask=mask[:,:,:3,:3])[0]
        inputs.append(x[0]);teachers.append(y[0]);pin.append(px[0]);pout.append(py[0])
        items.append(dict(input_ids=prompt,identity=str(index),split=record.split,family="arc",answer=index%2,
            choices=["one","two"],offset=offset,stop=offset+3,full_stop=offset+5,prompt_offset=poffset,prompt_stop=poffset+3))
        offset+=5;poffset+=3
    for name,rows in (("prefix",inputs),("teacher_hidden",teachers),("prompt_prefix",pin),("prompt_teacher_hidden",pout)):
        np.save(root/"native"/(name+".npy"),torch.cat(rows).view(torch.int16).numpy())
    (root/"native/items.json").write_text(json.dumps(items))
    (root/"plan.json").write_text(json.dumps(dict(fingerprint="native-fixture")))
    trial.write(root/"native/manifest.json",dict(schema="jspace.phi-native.v1",status="completed",fingerprint="native-fixture",
        artifacts={p.name:trial.sha(p) for p in (root/"native").iterdir()}))
    response_path=response_dir/"pilot_responses.jsonl"
    response_path.write_text("".join(json.dumps(r)+"\n" for r in records))
    data=trial.PhiTrainingData(root,response_path,require_population=False,layer=0)
    cfg=trial.PhiTrainConfig(accumulation=2,optimizer_chunk_bytes=256,max_cpu_state_bytes=1024**2)
    def make():return trial.PhiTrainer(deepcopy(block),deepcopy(norm),head.clone(),deepcopy(rotary),data,
        config=cfg,identities=dict(model="fixture",fit="fixture"),native_block_bytes=sum(p.numel()*p.element_size() for p in native.parameters()))
    yield data,make,root,response_path
    data.close()


def test_full_response_alignment_and_prompt_bank_contract(system):
    data,make,_,_=system
    case=data.case(0,"cpu")
    assert case["loss_mask"].nonzero().tolist()==[[0,2],[0,3]]
    assert case["targets"][case["loss_mask"]].tolist()==[9,2]
    assert case["hidden"].shape==(1,5,16)
    assert data.prompt_case(0,"cpu")["hidden"].shape==(1,3,16)
    assert len(data.training)==2 and len(data.development)==2


def test_chunked_combined_gradient_matches_dense_objective_and_keeps_teacher_frozen(system):
    data,make,_,_=system;trainer=make();case=data.case(0,"cpu")
    values,_,_=trainer.losses(case,backward=True)
    chunked={n:p.grad.clone() for n,p in trainer.block.named_parameters()}
    trainer.optimizer.zero_grad(set_to_none=True)
    z,location=trial.forward_coordinates(trainer.block,case["hidden"],case["position_ids"],trainer.rotary)
    target=trainer.block.encode(case["teacher_hidden"]).detach()
    feature=(z.float()-target.float()).square().mean()/trainer.normalization["denominator"]
    positions=case["loss_mask"].nonzero();selected=trial.decoded_rows(trainer.block,z,location,positions)
    # Same per-row readout execution, but one dense backbone autograd objective.
    logits=torch.cat([trainer._readout(row[None]) for row in selected])
    with torch.no_grad():
        teacher=torch.cat([trainer._readout(case["teacher_hidden"][b,p][None]) for b,p in positions])
    losses=trial.response_training.loss_vectors(logits,teacher,case["targets"][case["loss_mask"]],trainer.loss_config)
    dense=losses["loss"].mean()+feature
    dense.backward()
    assert abs(float(dense)-values["loss"])<1e-5
    for name,p in trainer.block.named_parameters():
        torch.testing.assert_close(p.grad.float(),chunked[name].float(),atol=2e-3,rtol=.02)
    assert trainer.head.grad is None and all(p.grad is None for p in trainer.norm.parameters())
    assert trainer.unchanged()


def test_training_normalization_uses_no_development_targets(system,monkeypatch):
    data,make,_,_=system;first=make();original=data.case
    def altered(index,device):
        result=original(index,device)
        if index in data.development: result["teacher_hidden"]*=100
        return result
    monkeypatch.setattr(data,"case",altered)
    second=make()
    assert first.normalization==second.normalization


def test_mc_scoring_uses_prompt_only_separately_from_full_response_metrics(system,monkeypatch):
    data,make,_,_=system;trainer=make();before=trainer.development();original=data.case
    def altered(index,device):
        result=original(index,device);result["hidden"]*=3;result["teacher_hidden"]*=-2;return result
    monkeypatch.setattr(data,"case",altered)
    after=trainer.development()
    assert before["ce"]!=after["ce"]
    for a,b in zip(before["records"],after["records"]):
        assert a["choice_logits"]==b["choice_logits"] and a["native_choice_logits"]==b["native_choice_logits"]


def test_pilot_discard_and_complete_three_epoch_lifecycle_fresh_reload(system,tmp_path):
    data,make,_,_=system
    pilot=make();receipt=pilot.pilot(data.training)
    assert receipt["passed"] and receipt["fixed_state_unchanged"] and receipt["discarded"]
    changed=receipt["effective_parameter_change"]
    assert receipt["bf16_parameters_changed"] and changed["changed_tensor_count"]>0
    assert changed["changed_tensor_count"]==sum(changed["changed_by_name"].values())
    assert all(changed["changed_by_name"][k]==(v!=changed["initial_sha256_by_name"][k])
               for k,v in changed["current_sha256_by_name"].items())
    with pytest.raises(ValueError):pilot.fit(tmp_path/"forbidden")
    trainer=make();result=trainer.fit(tmp_path/"training")
    assert result["epochs"]==3 and result["updates"]==3 and result["supervised_tokens"]==12
    assert [r["epoch"] for r in result["history"]]==[0,1,2,3]
    assert result["selected"]==min(result["history"],key=trial.selection_key)
    assert result["history"][0]["effective_parameter_change"]["changed"] is False
    assert result["selected_effective_parameter_change"]==result["selected"]["effective_parameter_change"]
    assert result["fixed_state_unchanged"] and result["internal_reload"]["passed"]
    verification=trial.verify_checkpoint(make(),tmp_path/"training")
    assert verification["passed"] and verification["records_and_metrics_exact"]
    assert verification["metrics"]==result["metrics"]
    manifest=trial.read(tmp_path/"training/manifest.json")
    del manifest["artifacts"]["epoch_03.pt"]
    (tmp_path/"training/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="Complete initializer"):
        trial.verify_checkpoint(make(),tmp_path/"training")


def test_selection_uses_earliest_accuracy_tie_not_ce():
    rows=[dict(epoch=0,correct=45,ce=9),dict(epoch=1,correct=45,ce=1),dict(epoch=2,correct=44,ce=.01)]
    assert min(rows,key=trial.selection_key)==rows[0]
    rows.append(dict(epoch=3,correct=46,ce=99))
    assert min(rows,key=trial.selection_key)==rows[3]


def test_pilot_rejects_master_only_update_if_all_stored_bf16_weights_round_back(system,monkeypatch):
    data,make,_,_=system;trainer=make()
    original=trainer.optimizer.step
    before={name:p.detach().clone() for name,p in trainer.block.named_parameters()}
    def master_only(*args,**kwargs):
        result=original(*args,**kwargs)
        with torch.no_grad():
            for name,p in trainer.block.named_parameters():p.copy_(before[name])
        return result
    monkeypatch.setattr(trainer.optimizer,"step",master_only)
    result=trainer.pilot(data.training)
    assert result["master_before"]!=result["master_after"]
    assert not result["bf16_parameters_changed"] and not result["passed"]


def test_data_rejects_eos_offset_or_hash_mismatch(system):
    data,_,root,responses=system
    with (root/"native/prefix.npy").open("ab") as stream:stream.write(b"changed")
    with pytest.raises(ValueError,match="hash"):
        trial.PhiTrainingData(root,responses,require_population=False,layer=0)


def test_development_cannot_enter_optimizer(system):
    data,make,_,_=system
    with pytest.raises(ValueError,match="TRAIN"):
        make().train_window(data.development)
