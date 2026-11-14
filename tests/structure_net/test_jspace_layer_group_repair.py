"""Selected-block gradients, matched teacher targets and authenticated lifecycle."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RMSNorm,Qwen2RotaryEmbedding

from experiments.structure_net import jspace_layer_group_repair as repair


def config():
    value=Qwen2Config(hidden_size=16,intermediate_size=24,num_attention_heads=4,
        num_key_value_heads=2,num_hidden_layers=4,vocab_size=23,max_position_embeddings=32,attention_dropout=0.)
    value._attn_implementation="eager"
    return value


def tiny_model(layers=(0,2,3),checkpoint=True):
    torch.manual_seed(615);cfg=config();blocks=[]
    for layer in range(4):
        native=Qwen2DecoderLayer(cfg,layer_idx=layer).bfloat16().float()
        fit=dict(basis=torch.linalg.qr(torch.randn(16,8)).Q,means=torch.randn(1,16)/7)
        blocks.append(repair.semantic.build_enriched_block(native,fit,parameter_dtype=torch.float32))
    model=repair.GroupModel(torch.randn(23,16)/3,blocks,torch.ones(16),1e-6,
        Qwen2RotaryEmbedding(config=cfg),split=2,checkpoint_blocks=checkpoint)
    return repair.configure_group(model,list(layers))


def cases():
    result=[];offset=0
    for i,length in enumerate((4,6,5,4)):
        result.append(dict(identity=str(i),family="tiny",split="train" if i<2 else "development",
            input_ids=[(i+j)%19 for j in range(length)],choices=list("ABCD"),answer=i%4,
            loss_positions=[0,1,length-2,length-1],offset=offset,stop=offset+length))
        offset+=length
    return result


class TinyData:
    def __init__(self):
        self.items=cases();self.training=[0,1];self.development=[2,3]
        self.label_ids=[19,20,21,22];self.vocabulary=23
        self.logits=torch.randn(4,2,23)
        self.targets={l:torch.randn(4,6,16) for l in (0,2,3)}

    def batch(self,indices,device):
        result=repair.teacher.token_batch(self.items,indices,device);length=result["input_ids"].shape[1]
        lengths=torch.tensor([len(self.items[i]["input_ids"]) for i in indices],device=device)
        return dict(**result,teacher=self.logits[indices].to(device),
            feature_targets={l:x[indices,:length].clone() for l,x in self.targets.items()},
            valid_tokens=result["position_ids"]<lengths[:,None])


def plan():
    return dict(selected_layers=[0,2,3],assembly_fingerprint="assembly",selection=dict(search_fingerprint="search"))


def test_fixed_schedule_and_exact_triple():
    cfg=repair.Config()
    assert (cfg.epochs,cfg.seed,cfg.microbatch,cfg.accumulation,cfg.learning_rate)==(6,27866,1,16,1e-5)
    assert 512//(cfg.microbatch*cfg.accumulation)*cfg.epochs==192
    assert cfg.feature_weight==1 and cfg.gradient_clip==1
    assert asdict(cfg.objective)==dict(temperature=2.,full_kl_weight=.2,choice_kl_weight=.5,choice_ce_weight=.3)
    for value in ([0,1],[2,1,3],[0,0,1],[0,1,36],[False,1,2]):
        with pytest.raises(ValueError,match="Exactly three"):repair.selected_layers(value)


def test_projected_next_state_loss_full_derivative_fixed_teacher_normalization():
    torch.manual_seed(817)
    upstream=torch.randn(2,4,5,dtype=torch.float64,requires_grad=True)
    update=torch.randn_like(upstream,requires_grad=True)
    target=torch.randn_like(upstream,requires_grad=True)
    basis=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([[True,True,False,False],[True,True,True,True]])
    y=upstream+update
    actual=repair.feature_objective({3:y},{3:target},{3:basis},valid,"cpu")["feature_loss"]
    manual=[]
    for i in range(2):
        numerator=((y[i,valid[i]]-target[i,valid[i]].detach())@basis.detach()).square().sum(-1).mean()
        denominator=(target[i,valid[i]].detach()@basis.detach()).square().sum(-1).mean().clamp_min(1e-8)
        manual.append(numerator/denominator)
    expected=torch.stack(manual).mean()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    grad=torch.autograd.grad(actual,(upstream,update,target,basis),allow_unused=True,retain_graph=True)
    expected_grad=torch.autograd.grad(expected,(upstream,update))
    torch.testing.assert_close(grad[0],expected_grad[0]);torch.testing.assert_close(grad[1],expected_grad[1])
    assert grad[0].abs().sum()>0 and grad[2] is None and grad[3] is None
    assert not grad[0][0,2:].count_nonzero()
    # Stop-gradient correction matching is exactly the same value and keeps upstream gradient.
    predicted=(y-upstream.detach())@basis.detach()
    wanted=(target.detach()-upstream.detach())@basis.detach()
    torch.testing.assert_close(predicted-wanted,(y-target.detach())@basis.detach())


def test_feature_loss_averages_cases_and_layers_and_clamps_teacher_only_floor():
    y=torch.ones(2,3,2,requires_grad=True);zero=torch.zeros_like(y)
    valid=torch.tensor([[True,False,False],[True,True,True]])
    result=repair.feature_objective({0:y,2:2*y},{0:zero,2:zero},{0:torch.eye(2),2:torch.eye(2)},valid,"cpu")
    torch.testing.assert_close(result["feature_loss"],torch.tensor(5e8))
    assert set(result["per_layer_feature_loss"])=={"0","2"}
    with pytest.raises(ValueError,match="valid tokens"):
        repair.feature_objective({0:y},{0:zero},{0:torch.eye(2)},torch.zeros_like(valid),"cpu")


def test_actual_four_block_path_matches_parent_and_propagates_through_frozen_middle():
    model=tiny_model();data=TinyData();batch=data.batch([0,1],"cpu")
    reference=repair.base.PartitionedCarryModel(model.embedding.clone(),deepcopy(list(model.blocks)),
        model.norm_gain.detach(),model.epsilon,deepcopy(model.rotary),split=2,checkpoint_blocks=False)
    reference.requires_grad_(True)
    args=[batch[k] for k in ("input_ids","position_ids","attention_mask")];positions=batch["loss_positions"][:,[1,3]]
    actual,features=model.forward_with_features(*args,positions)
    expected=reference(*args,positions)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert set(features)=={0,2,3}
    weights=torch.randn_like(actual);(actual*weights).sum().backward();(expected*weights).sum().backward()
    native_parameters=dict(reference.named_parameters())
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad,native_parameters[name].grad,atol=0,rtol=0)
        else:assert p.grad is None
    assert sum(float(p.grad.abs().sum()) for p in model.blocks[0].parameters())>0
    assert all(p.grad is None for p in model.blocks[1].parameters())
    assert model.norm_gain.grad is None and model.embedding.grad is None


def test_frozen_prefix_before_first_selected_has_no_gradient_but_selected_branches_do():
    model=tiny_model(layers=(1,2,3));batch=repair.teacher.token_batch(cases(),[0],"cpu")
    logits,features=model.forward_with_features(batch["input_ids"],batch["position_ids"],batch["attention_mask"],batch["loss_positions"][:,[1,3]])
    logits.square().mean().backward()
    assert set(features)=={1,2,3}
    assert all(p.grad is None for p in model.blocks[0].parameters())
    assert all(p.grad is not None for i in (1,2,3) for p in model.blocks[i].parameters())


def test_later_feature_loss_reaches_earlier_selected_layer_through_frozen_middle():
    model=tiny_model(checkpoint=False);batch=repair.teacher.token_batch(cases(),[0],"cpu")
    args=[batch[k] for k in ("input_ids","position_ids","attention_mask")]
    def loss():
        _,features=model.forward_with_features(*args,batch["loss_positions"][:,[1,3]])
        return repair.feature_objective({3:features[3]},{3:torch.ones_like(features[3])},
            {3:model.blocks[3].basis},torch.ones(1,4,dtype=torch.bool),"cpu")["feature_loss"]
    earliest=next(model.blocks[0].parameters())
    gradient=torch.autograd.grad(loss(),earliest)[0]
    assert gradient.abs().sum()>0
    # A detached middle is a negative control: it severs exactly this upstream path.
    handle=model.blocks[1].register_forward_hook(lambda _module,_args,value:value.detach())
    try:assert torch.autograd.grad(loss(),earliest,allow_unused=True)[0] is None
    finally:handle.remove()
    assert all(p.grad is None for p in model.blocks[1].parameters()) and model.norm_gain.grad is None


def test_one_update_changes_exactly_selected_blocks_and_preserves_all_fixed_state():
    model=tiny_model();data=TinyData();trainer=repair.Trainer(model,data,plan())
    fixed=repair.frozen_fingerprint(model);before=repair.parameter_fingerprints(model,[0,2,3])
    with pytest.raises(ValueError,match="Training identities"):trainer.train_window([2])
    result=trainer.train_window([0,1]);after=repair.parameter_fingerprints(model,[0,2,3])
    assert result["update"]==1 and result["gradient_parameter_count"]==21
    assert all(before[str(i)]!=after[str(i)] for i in (0,2,3))
    assert repair.frozen_fingerprint(model)==fixed
    assert result["feature_loss"]>0 and result["loss"]==pytest.approx(result["endpoint_loss"]+result["feature_loss"])
    metrics,records=trainer.development()
    assert set(records[0]["feature_loss_by_layer"])=={"0","2","3"}
    assert metrics["feature_loss"]==pytest.approx(sum(r["feature_loss"] for r in records)/len(records))
    with torch.no_grad():model.norm_gain[0].add_(.1)
    assert repair.frozen_fingerprint(model)!=fixed


def test_delta_only_three_blocks_reload_and_provenance(tmp_path):
    model=tiny_model();before=repair.frozen_fingerprint(model)
    directory=tmp_path/"selected";metadata=dict(epoch=0,development=dict(correct=27))
    repair.save_delta(model,directory,plan(),metadata)
    saved=deepcopy(model.blocks[2].state_dict())
    assert {p.name for p in directory.iterdir()}=={"manifest.json","block_00.pt","block_02.pt","block_03.pt"}
    with torch.no_grad():next(model.blocks[2].parameters()).add_(.5)
    assert repair.load_delta(model,directory,plan())["metadata"]==metadata
    for name,value in model.blocks[2].state_dict().items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(value,saved[name],atol=0,rtol=0)
        else:assert value==saved[name]
    assert repair.frozen_fingerprint(model)==before
    wrong={**plan(),"selected_layers":[0,1,3]}
    with pytest.raises(ValueError,match="provenance"):repair.load_delta(model,directory,wrong)
    with (directory/"block_02.pt").open("ab") as stream:stream.write(b"changed")
    with pytest.raises(ValueError,match="bytes changed"):repair.load_delta(model,directory,plan())


def test_six_epoch_fit_writes_frozen_proof_before_result_and_reloads_selected_delta(tmp_path,monkeypatch):
    model=tiny_model();data=TinyData();trainer=repair.Trainer(model,data,plan())
    # The tiny fixture tests the lifecycle; production requires the real27/64 control.
    monkeypatch.setattr(repair.base,"baseline_control",lambda *a,**kw:dict(passed=True))
    before=repair.frozen_fingerprint(model)
    result=trainer.fit(tmp_path/"train",reference={},frozen_before=before)
    assert result["frozen_state_unchanged"] is True and result["frozen_state"]==before
    assert result["strict_reload_exact"] is True and result["updates"]==6
    assert [r["epoch"] for r in result["history"]]==list(range(7))
    winner=max(result["history"],key=lambda r:repair.base.selection_key(r["metrics"]))
    assert result["selected_epoch"]==winner["epoch"] and result["selected"]==winner["metrics"]
    assert repair.recipe.read(tmp_path/"train/result.json")==result


def test_group_dataset_keeps_original_positions_and_fp32_targets(tmp_path,monkeypatch):
    items=cases();teacher_root=tmp_path/"teacher";teacher_root.mkdir();(teacher_root/"cache").mkdir()
    repair.recipe.write(teacher_root/"items.json",dict(items=items))
    np.save(teacher_root/"cache/teacher_logits.npy",np.zeros((4,2,23),np.float32))
    monkeypatch.setattr(repair.teacher,"verify",lambda *a,**kw:dict(label_ids=[19,20,21,22]))
    root=tmp_path/"repair";root.mkdir();(root/"capture").mkdir();repair.recipe.write(root/"items.json",dict(items=items))
    for layer in (0,2,3):
        bank=np.repeat(np.arange(items[-1]["stop"],dtype=np.float32)[:,None],2048,axis=1)+layer
        np.save(root/"capture"/f"boundary_{layer+1:02d}.npy",bank)
    descriptor=dict(teacher_root=str(teacher_root),items_digest=repair.recipe.digest(dict(items=items)),selected_layers=[0,2,3])
    data=repair.GroupDataset(root,descriptor)
    try:
        batch=data.batch([1,0],"cpu")
        assert batch["teacher"].dtype==torch.float32 and batch["teacher"].shape==(2,2,23)
        assert batch["loss_positions"].tolist()==[items[1]["loss_positions"],items[0]["loss_positions"]]
        assert batch["valid_tokens"].sum(1).tolist()==[6,4]
        assert batch["feature_targets"][2][0,0,0].item()==items[1]["offset"]+2
        assert not batch["feature_targets"][2][1,4:].count_nonzero()
    finally:data.close()


def test_actual_native_features_and_logits_match_unpartitioned_path():
    torch.manual_seed(115);cfg=config();native=[Qwen2DecoderLayer(cfg,layer_idx=i).bfloat16().float() for i in range(4)]
    model=repair.FeatureTeacher(torch.randn(23,16),native,Qwen2RMSNorm(16),
        Qwen2RotaryEmbedding(config=cfg),Qwen2RotaryEmbedding(config=cfg),cut=2)
    batch=repair.teacher.token_batch(cases(),[0,1],"cpu")
    logits,features=model.forward_features(batch,[0,2,3])
    expected=model(batch["input_ids"],batch["position_ids"],batch["attention_mask"],batch["loss_positions"])
    torch.testing.assert_close(logits,expected,atol=0,rtol=0)
    hidden=torch.nn.functional.embedding(batch["input_ids"],model.embedding)
    rope=model.rotary_front(hidden,batch["position_ids"])
    mask=torch.zeros(batch["attention_mask"].shape).masked_fill_(~batch["attention_mask"],torch.finfo(torch.float32).min)
    with torch.no_grad():
        for index,block in enumerate(model.blocks):
            value=block(hidden,attention_mask=mask,position_ids=batch["position_ids"],position_embeddings=rope,
                cache_position=batch["position_ids"][0],use_cache=False)
            hidden=value[0] if isinstance(value,tuple) else value
            if index in features:torch.testing.assert_close(features[index],hidden,atol=0,rtol=0)
    assert all(not x.requires_grad and x.dtype==torch.float32 for x in features.values())


def test_native_target_pinning_hashes_only_consumed_banks_and_rejects_changed_bytes(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_native_dev_cache
    native_plan=dict(fingerprint="native",source_fingerprint="original")
    monkeypatch.setattr(jspace_native_dev_cache,"verify",lambda *a,**kw:native_plan)
    root=tmp_path;(root/"cache").mkdir()
    repair.recipe.write(root/"plan.json",native_plan);repair.recipe.write(root/"items.json",dict(items=[]))
    repair.recipe.write(root/"cache/result.json",dict(status="completed",fingerprint="native",correct=50,count=64,
        native_baseline_predictions_exact=True))
    for index in (1,3,4):np.save(root/"cache"/f"boundary_{index:02d}.npy",np.ones((2,16),np.float32)*index)
    names=["items.json","cache/result.json",*[f"cache/boundary_{i:02d}.npy" for i in (1,3,4)]]
    manifest=dict(status="completed",fingerprint="native",artifact_sha256={name:repair.recipe.sha(root/name) for name in names})
    repair.recipe.write(root/"cache/manifest.json",manifest)
    pinned=repair.native_target_inputs(root,[0,2,3],"original")
    assert len(pinned)==7 and str(root/"cache/boundary_03.npy") in pinned
    with (root/"cache/boundary_03.npy").open("ab") as stream:stream.write(b"tamper")
    with pytest.raises(ValueError,match="Changed consumed"):repair.native_target_inputs(root,[0,2,3],"original")


def test_training_receipt_requires_fixed_state_proof_and_all_seven_authenticated_epochs(tmp_path):
    root=tmp_path;directory=root/"train";directory.mkdir();(directory/"selected").mkdir()
    repair.recipe.write(directory/"selected/manifest.json",dict(selected=True))
    selected_sha=repair.recipe.sha(directory/"selected/manifest.json");history=[]
    for epoch in range(7):
        metrics=dict(correct=27+epoch,choice_ce=2.)
        path=directory/f"epoch_{epoch:02d}_development.json";repair.recipe.write(path,dict(metrics=metrics,records=[]))
        history.append(dict(epoch=epoch,metrics=metrics,development_sha256=repair.recipe.sha(path),checkpoint_manifest_sha256=selected_sha))
    frozen=dict(sha256="fixed",tensor_count=1,bytes=4)
    result=dict(status="completed",systems_only=False,updates=192,strict_reload_exact=True,
        frozen_state_unchanged=True,frozen_state=frozen,selected_layers=[0,2,3],history=history,
        selected_epoch=6,selected=history[-1]["metrics"],selected_manifest_sha256=selected_sha)
    repair.recipe.write(directory/"result.json",result)
    receipt=directory/"frozen_state.json";repair.recipe.write(receipt,dict(passed=True,before=frozen,after=frozen))
    manifest=dict(status="completed",fingerprint="plan",result_sha256=repair.recipe.sha(directory/"result.json"),
        artifacts={"frozen_state.json":repair.recipe.sha(receipt)})
    repair.recipe.write(directory/"manifest.json",manifest)
    assert repair.checked(root,"train",dict(fingerprint="plan",selected_layers=[0,2,3]))==result
    receipt.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):repair.checked(root,"train",dict(fingerprint="plan",selected_layers=[0,2,3]))


def test_gpu_stage_checks_lease_before_cuda_or_model(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(repair,"verify",lambda root:dict(fingerprint="plan",selected_layers=[0,2,3]))
    def deny(*a,**kw):raise RuntimeError("no reservation")
    monkeypatch.setattr(jspace_full_lease,"wait_ready",deny)
    monkeypatch.setattr(repair.base,"allocator_settings",lambda *a,**kw:pytest.fail("CUDA reached before lease"))
    with pytest.raises(RuntimeError,match="no reservation"):repair.run(tmp_path,"capture")
    assert repair.recipe.read(tmp_path/"capture/failure.json")["lease_id"] is None
