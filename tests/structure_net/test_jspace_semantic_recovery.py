from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding

from experiments.structure_net import jspace_joint_recovery as joint
from experiments.structure_net import jspace_semantic_recovery as semantic


def fixture(tmp_path):
    torch.manual_seed(181)
    config=Qwen2Config(hidden_size=16,intermediate_size=24,num_attention_heads=4,num_key_value_heads=2,
        num_hidden_layers=3,vocab_size=23,max_position_embeddings=32,attention_dropout=0.)
    config._attn_implementation="eager"
    blocks=[]
    for layer in range(3):
        native=Qwen2DecoderLayer(config,layer_idx=layer).bfloat16()
        fit=dict(basis=torch.linalg.qr(torch.randn(16,8)).Q,means=torch.randn(1,16)/5,
            head_dim=2,rope_pairs=torch.zeros(2,1,dtype=torch.long),value_basis=torch.randn(2,4,2))
        block=semantic.build_enriched_block(native,fit)
        assert block.head_dim==4 and not block.has_head_projection and block.wrapped.metric is None
        blocks.append(block)
    model=joint.JointCarryModel(torch.randn(23,16).bfloat16()/3,blocks,torch.ones(16),1e-6,Qwen2RotaryEmbedding(config=config))
    rows=[]
    for i,length in enumerate((4,6,5,4)):
        count=2 if i%2==0 else 4
        rows.append(dict(identity=str(i),family="binary" if count==2 else "four",split="train" if i<2 else "development",
            input_ids=[(i+j)%19 for j in range(length)],loss_positions=[0,1,length-2,length-1],
            choices=[chr(65+j) for j in range(count)],answer=(i+1)%count))
    path=tmp_path/"teacher.npy"
    np.save(path,torch.randn(4,4,23).bfloat16().view(torch.int16).numpy())
    data=joint.JointDataset(rows,path,[19,20,21,22])
    return model,data


def test_full_heads_ignore_stored_half_head_metadata_all_layers_receive_gradients(tmp_path):
    model,data=fixture(tmp_path)
    config=semantic.SemanticRecoveryConfig(optimizer_chunk_bytes=128)
    trainer=semantic.SemanticTrainer(model,data,config=config)
    batch=data.batch([0,1],"cpu");logits,values=trainer.losses(batch)
    assert logits.shape==(2,2,23)  # Only midpoint/answer; still all vocabulary.
    values["loss"].backward()
    for block in model.blocks:
        assert block.basis.dtype==torch.float32 and not block.basis.requires_grad
        assert all(p.dtype==torch.bfloat16 and p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        assert sum(float(p.grad.float().abs().sum()) for p in block.parameters())>0
    assert model.embedding.grad is None and not model.embedding.requires_grad
    assert model.norm_gain.dtype==torch.float32 and model.norm_gain.grad is not None
    data.close()


def test_semantic_fit_logs_loss_components_baseline_and_strict_reload(tmp_path):
    model,data=fixture(tmp_path)
    embedding=model.embedding.clone();bases=[b.basis.clone() for b in model.blocks]
    config=semantic.SemanticRecoveryConfig(epochs=1,microbatch=1,accumulation=2,optimizer_chunk_bytes=128)
    trainer=semantic.SemanticTrainer(model,data,config=config)
    with pytest.raises(ValueError,match="training identities"):
        trainer.train_window([data.development[0]])
    result=trainer.fit(tmp_path/"fit")
    assert result["strict_reload_exact"] and result["updates"]==1
    for key in ("full_kl","choice_kl","choice_ce","valid_label_probability","prediction_counts",
                "best_constant_label_correct","gained_correct_over_best_constant_label"):
        assert key in result["selected"]
    metrics,records=trainer.development()
    assert metrics["kl"]==metrics["full_kl"]
    assert metrics["ce"]==metrics["answer_full_vocab_ce"]
    assert metrics["loss"]==pytest.approx(metrics["weighted_full_kl"]+metrics["weighted_choice_kl"]+metrics["weighted_choice_ce"],rel=2e-6)
    assert all("choice_ce" in row and "answer_full_vocab_ce" in row and "ce" in row for row in records)
    torch.testing.assert_close(model.embedding,embedding,atol=0,rtol=0)
    for block,basis in zip(model.blocks,bases):torch.testing.assert_close(block.basis,basis,atol=0,rtol=0)
    data.close()


def test_configuration_roundtrip_and_constant_predictor_report():
    config=semantic.SemanticRecoveryConfig()
    assert semantic.config_from_dict(asdict(config))==config
    rows=[dict(answer=1,prediction=1,family="x"),dict(answer=1,prediction=1,family="x"),dict(answer=0,prediction=1,family="x")]
    summary=semantic.ranking_summary(rows)
    assert summary["correct"]==summary["best_constant_label_correct"]==2
    assert summary["gained_correct_over_best_constant_label"]==0
    assert summary["prediction_counts"]=={"B":3}


def basis_files(root,source,*,fit_ids=("train-fit",),check_ids=("train-check",)):
    (root/"basis").mkdir(parents=True)
    plan=dict(schema=semantic.BASIS_SCHEMA,source_root=str(source),source_fingerprint="source-fingerprint",
        layers=list(range(36)),rank=1024,fit_training_identities=list(fit_ids),internal_training_check_identities=list(check_ids),source_sha256={})
    plan["fingerprint"]=joint.recipe.digest(plan)
    joint.recipe.write(root/"plan.json",plan)
    manifest=dict(status="completed",fingerprint=plan["fingerprint"],source_fingerprint="source-fingerprint",
        artifacts={f"basis/block_{i:02d}/{name}":"unused" for i in range(36) for name in ("fit.pt","fit.json")})
    joint.recipe.write(root/"basis/manifest.json",manifest)
    return plan,manifest


def test_enrichment_provenance_rejects_development_fitting_and_incomplete_layers(tmp_path):
    source=(tmp_path/"source").resolve();parent=dict(fingerprint="source-fingerprint")
    items=[dict(identity="train-fit",split="train"),dict(identity="train-check",split="train"),dict(identity="dev",split="development")]
    good=tmp_path/"good";basis_files(good,source)
    semantic.validate_basis_root(good,source,parent,items)
    bad=tmp_path/"bad";basis_files(bad,source,fit_ids=("dev",))
    with pytest.raises(ValueError,match="training-only"):
        semantic.validate_basis_root(bad,source,parent,items)
    missing=tmp_path/"missing";_,manifest=basis_files(missing,source)
    manifest["artifacts"].pop("basis/block_35/fit.pt")
    (missing/"basis/manifest.json").write_text(__import__('json').dumps(manifest))
    with pytest.raises(ValueError,match="all36"):
        semantic.validate_basis_root(missing,source,parent,items)


def test_prepare_and_train_guards_run_before_gpu_or_model(tmp_path,monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    monkeypatch.setattr(semantic.recipe,"verify",lambda *_:pytest.fail("Read source before CUDA hidden guard"))
    with pytest.raises(ValueError,match="Hide CUDA"):
        semantic.prepare(tmp_path/"new",tmp_path/"source",tmp_path/"basis")
    monkeypatch.setattr(semantic,"verify",lambda _:dict(fingerprint="test"))
    monkeypatch.setattr(semantic,"build_model",lambda *_a,**_k:pytest.fail("Constructed model before successful pilot"))
    with pytest.raises(FileNotFoundError):semantic.run(tmp_path,"train")
