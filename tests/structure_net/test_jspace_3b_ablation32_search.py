"""Real tiny frozen Qwen: complete-response search, QR VJPs and durable geometry."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_ablation32_search as study

data=study.module("distillation_data")


class Dataset:
    def __init__(self,native):
        self.training_indices=[0,1];self.selection_indices=[2,3];self.records=[];self.hidden=[]
        self.anchors=torch.randn(2,8,8,dtype=torch.float32)/20
        provenance=dict(source_uri="fixture",source_revision="v1",tokenizer_sha256="fixture",
            chat_template_sha256="fixture",prompt_provenance="fixture",response_provenance="reference",
            assistant_boundary_verified=True)
        for i in range(4):
            record=data.ResponseRecord(example_id=f"case{i}",group_id=f"group{i}",split="train",task="fixture",
                prompt_ids=(1,2+i),response_ids=((7,31) if i%2==0 else (9,11,13,31)),
                provenance=provenance,finish_reason="eos",eos_token_id=31)
            self.records.append(record)
            with torch.no_grad():
                self.hidden.append(native.model(input_ids=torch.tensor([record.input_ids]),use_cache=False).last_hidden_state.clone())

    def batch(self,indices):
        i,=indices
        batch=data.response_batch([self.records[i]],pad_token_id=0,max_length=8)
        return dict(batch,teacher_hidden=self.hidden[i],teacher_input_ids=batch["input_ids"].clone(),
                    teacher_record_ids=[self.records[i].example_id])


def fixture():
    torch.manual_seed(46365)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=1,vocab_size=32,max_position_embeddings=32,tie_word_embeddings=True,
        attention_dropout=0.,eos_token_id=31,pad_token_id=0)
    cfg._attn_implementation="eager"
    native=Qwen2ForCausalLM(cfg).bfloat16().eval().requires_grad_(False)
    dataset=Dataset(native)
    config=study.SearchConfig(retained=4,epochs=2,accumulation_examples=2,max_length=8,cpu_threads=1,token_chunk_size=1)
    return native,dataset,config


def trainer(native,dataset,config):
    return study.SearchTrainer(native,dataset,teacher_id="native",identities={"data":"fixture","teacher":"native"},
        config=config,systems_only=True)


def test_functional_qr_backend_matches_frozen_geometry_values_and_raw_gradient():
    module=study.geometry.DiscardedSubspace(12,6,seed=220101)
    raw=module.raw.detach().clone().requires_grad_(True)
    actual=study.backend_discarded_basis(raw);expected=module.discarded_basis()
    assert torch.equal(actual,expected)
    probe=torch.randn_like(actual)
    (actual*probe).sum().backward();(expected*probe).sum().backward()
    torch.testing.assert_close(raw.grad,module.raw.grad,rtol=0,atol=0)
    with pytest.raises(ValueError,match="FP64"):study.backend_discarded_basis(raw.float())


def test_real_pilot_updates_every_subspace_keeps_native_fixed_and_strictly_reloads(tmp_path,monkeypatch):
    native,dataset,config=fixture();before={n:t.clone() for n,t in native.state_dict().items()}
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:pytest.fail("CPU fixture initialized CUDA"))
    instance=trainer(native,dataset,config)
    try:
        result=instance.pilot([0,1],directory=tmp_path/"pilot")
        assert result["status"]=="systems_passed" and result["all_interfaces_updated"]
        assert result["update"]["supervised_tokens"]==6  # Both EOS, unequal response lengths.
        assert result["update"]["gradient_calls"]==[2,2]
        assert result["optimizer_state_dtypes"]==["torch.float64"]
        assert result["fresh_raw_reload_exact"] and result["fresh_projector_replay_exact"]
        assert result["unprojected_native_control"]["all_hidden_bit_exact"]
        assert all(v>0 for v in result["projector_probe_max_changes"])
        assert all(torch.equal(before[n],t) for n,t in native.state_dict().items())
        assert all(p.dtype==torch.float64 and p.device.type=="cpu" for p in instance.parameters)
        assert not instance.model.discarded and not instance.graphs and not instance.handles
        with pytest.raises(ValueError,match="Fresh scientific"):instance.fit(tmp_path/"scientific")
    finally:instance.close()


def test_one_qr_per_window_and_per_example_loss_matches_direct_autograd(monkeypatch):
    native,dataset,config=fixture();a=trainer(native,dataset,config);b=trainer(deepcopy(native),dataset,config)
    counts=[];original=study.geometry.DiscardedSubspace.discarded_basis
    monkeypatch.setattr(study.geometry.DiscardedSubspace,"discarded_basis",lambda self:counts.append(1) or original(self))
    monkeypatch.setattr(a.optimizer,"step",lambda:None)
    try:
        receipt=a.train_window([0,1])
        assert len(counts)==2  # Two interfaces, not two interfaces times two examples.
        b.model.checkpoint_blocks=False
        b.model.discarded=[original(c).float() for c in b.coordinates]
        objective=0.
        for i in (0,1):
            batch=dataset.batch([i]);b.teacher.prepare(batch);hidden=b.model.hidden_forward(batch)
            pos=batch["loss_mask"].nonzero();rows=hidden[pos[:,0],pos[:,1]]
            values=study.nal.loss_vectors(b.model.readout(rows),b.teacher(batch,pos),batch["targets"][pos[:,0],pos[:,1]],b.loss_config)
            loss=values["forward_kl"].mean()/2
            objective+=float(loss.detach());loss.backward(retain_graph=i==0)
        torch.nn.utils.clip_grad_norm_(b.parameters,config.gradient_clip)
        assert receipt["per_example_kl"]==pytest.approx(objective,rel=1e-5,abs=1e-6)
        for pa,pb in zip(a.parameters,b.parameters):
            torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-2,atol=2e-5)
    finally:a.close();b.close()


def test_selection_is_disjoint_train_only_and_counts_all_tokens():
    native,dataset,config=fixture();instance=trainer(native,dataset,config)
    try:
        result=instance.evaluate()
        assert result["examples"]==2 and result["supervised_tokens"]==6 and result["temperature"]==2.
        with pytest.raises(ValueError,match="selection"):instance.evaluate([0])
        dataset.records[2]=replace(dataset.records[2],group_id=dataset.records[0].group_id)
        with pytest.raises(ValueError,match="source groups"):study.validate_selections(dataset)
    finally:instance.close()


def test_geometry_export_is_matched_to_saved_raw_state_and_anchors(tmp_path):
    native,dataset,config=fixture();instance=trainer(native,dataset,config)
    try:
        score=instance.evaluate();saved=instance.save_checkpoint(tmp_path/"initial.pt",epoch=0,selection=score)
        receipt=instance.export(tmp_path/"random",epoch=0,checkpoint_receipt=saved,arm="matched_initial_random")
        manifest=json.loads((tmp_path/"random/manifest.json").read_text())
        assert manifest["schema"]==study.GEOMETRY_SCHEMA and manifest["projection_sites"]==study.PROJECTION_SITES
        assert manifest["checkpoint"]["sha256"]==saved["sha256"]
        reference=torch.load(manifest["projection_reference"]["path"],weights_only=True)
        assert reference["checkpoint"]["sha256"]==saved["sha256"]
        assert [r["example_id"]for r in reference["records"]]==["case2","case3"]
        instance.prepare_window(differentiable=False)
        with torch.no_grad():
            for item in reference["records"]:
                batch=dataset.batch([item["index"]]);hidden=instance.model.hidden_forward(batch)
                assert torch.equal(item["input_ids"],batch["input_ids"])
                assert torch.equal(item["normalized_hidden"],hidden)
                logits=instance.model.readout(hidden[:,item["last_prediction_position"]])
                assert torch.equal(item["last_prediction_logits"],logits)
        instance._clear_window()
        layer=torch.load(manifest["layers"]["0"]["path"],weights_only=True)
        assert torch.equal(layer["anchor"],dataset.anchors[0])
        assert torch.equal(layer["raw"],instance.parameters[0])
        assert layer["retained"].dtype==torch.float64 and layer["retained"].shape==(8,4)
        with torch.no_grad():instance.parameters[0].add_(.1)
        with pytest.raises(ValueError,match="exact current"):
            instance.export(tmp_path/"mismatched",epoch=0,checkpoint_receipt=saved,arm="mismatched")
        instance.load_checkpoint(saved["path"],expected_sha256=saved["sha256"])
        assert torch.equal(layer["raw"],instance.parameters[0])
    finally:instance.close()


def test_cosine_schedule_and_epochs_keep_explicit_seeds_and_all_examples():
    config=study.SearchConfig()
    assert config.basis_seed==220101 and config.shuffle_seed==220301
    assert study.learning_rate_at(0,632,config)==.01
    assert study.learning_rate_at(631,632,config)==pytest.approx(.0015)
    indices=list(range(2500));first=study.epoch_windows(indices,1,config);second=study.epoch_windows(indices,2,config)
    assert len(first)==79 and sorted(i for w in first for i in w)==indices
    assert first!=second and first==study.epoch_windows(indices,1,config)
