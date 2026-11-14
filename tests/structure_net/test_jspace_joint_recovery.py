from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_quadratic_qwen import QuadraticQwenBlock
from experiments.structure_net.jspace_residual_carry import ResidualCarryBlock
from experiments.structure_net import jspace_joint_recovery as joint


def model(dtype=torch.float32):
    torch.manual_seed(991)
    config=Qwen2Config(hidden_size=16,intermediate_size=28,num_attention_heads=4,num_key_value_heads=2,
        num_hidden_layers=4,vocab_size=23,max_position_embeddings=32,attention_dropout=0.)
    config._attn_implementation="eager"
    blocks=[]
    for layer in range(4):
        native=Qwen2DecoderLayer(config,layer_idx=layer).eval()
        u=torch.linalg.qr(torch.randn(16,8)).Q
        affine=AffineQwenBlock(native,u,torch.randn(1,16)/3,trainable=True,parameter_dtype=torch.float32)
        wrapped=QuadraticQwenBlock(affine,mode="softmax",learn_metric=True)
        for p in wrapped.parameters():p.data=p.data.to(dtype)
        blocks.append(ResidualCarryBlock(wrapped))
    return joint.JointCarryModel(torch.randn(23,16,dtype=dtype)/3,blocks,torch.ones(16),1e-6,
        Qwen2RotaryEmbedding(config=config))


def dataset(tmp_path):
    rows=[]
    for i,length in enumerate((4,6,5,4)):
        rows.append(dict(identity=str(i),family="one" if i%2 else "two",split="train" if i<2 else "development",
            input_ids=[(j+i)%19 for j in range(length)],loss_positions=[0,1,length-2,length-1],
            choices=["yes","no"],answer=i%2))
    teacher=torch.randn(4,4,23).bfloat16().view(torch.int16).numpy()
    path=tmp_path/"teacher.npy";np.save(path,teacher)
    return joint.JointDataset(rows,path,[20,21])


@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_nonreentrant_checkpoint_gradients_all_four_blocks_with_fixed_input(tmp_path,dtype):
    data=dataset(tmp_path);net=model(dtype);net.train()
    control=deepcopy(net);control.checkpoint_blocks=False
    batch=data.batch([0,1],"cpu")
    args=[batch[k] for k in ("input_ids","position_ids","attention_mask","loss_positions")]
    assert not F.embedding(batch["input_ids"],net.embedding).requires_grad
    actual=net(*args);expected=control(*args)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    joint.joint_losses(actual,batch["teacher"],batch["target_tokens"])["loss"].backward()
    joint.joint_losses(expected,batch["teacher"],batch["target_tokens"])["loss"].backward()
    for (name,p),(other,q) in zip(net.named_parameters(),control.named_parameters()):
        assert name==other
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    for block in net.blocks:
        assert block.basis.dtype==torch.float32 and not block.basis.requires_grad
        assert sum(float(p.grad.float().abs().sum()) for p in block.parameters())>0
    assert net.embedding.grad is None and not net.embedding.requires_grad
    assert net.norm_gain.grad.abs().sum()>0
    assert actual.shape==(2,4,23)
    data.close()


def test_full_vocabulary_kl_ce_teacher_detached_and_nonchoice_logits_matter():
    torch.manual_seed(1)
    student=torch.randn(2,4,7,requires_grad=True)
    teacher=torch.randn(2,4,7,requires_grad=True)
    targets=torch.tensor([1,2]);config=joint.JointConfig()
    values=joint.joint_losses(student,teacher,targets,config)
    p=(teacher.detach()/2).softmax(-1)
    reference=.7*4*(p*((teacher.detach()/2).log_softmax(-1)-(student/2).log_softmax(-1))).sum(-1).mean()
    reference=reference+.3*F.cross_entropy(student[:,-1],targets)
    torch.testing.assert_close(values["loss"],reference)
    values["loss"].backward()
    assert teacher.grad is None
    assert student.grad[:,:,6].abs().sum()>0  # Not a four-choice classifier.


def test_padding_is_excluded_and_padded_batch_matches_individuals(tmp_path):
    data=dataset(tmp_path);net=model().eval()
    batch=data.batch([0,1],"cpu")
    assert not batch["attention_mask"][0,0,4:].any()
    assert not batch["attention_mask"][0,0,:,4:].any()
    with torch.no_grad():
        actual=net(*[batch[k] for k in ("input_ids","position_ids","attention_mask","loss_positions")])
        for row,index in enumerate((0,1)):
            single=data.batch([index],"cpu")
            expected=net(*[single[k] for k in ("input_ids","position_ids","attention_mask","loss_positions")])
            torch.testing.assert_close(actual[row],expected[0],atol=3e-6,rtol=3e-6)
    data.close()


def test_training_scope_frozen_geometry_optimizer_and_strict_selection_reload(tmp_path):
    data=dataset(tmp_path);net=model()
    embedding=net.embedding.clone();bases=[b.basis.clone() for b in net.blocks]
    config=joint.JointConfig(epochs=1,microbatch=1,accumulation=2,learning_rate=1e-3,optimizer_chunk_bytes=128)
    trainer=joint.JointTrainer(net,data,config=config)
    with pytest.raises(ValueError,match="training identities"):
        trainer.train_window([data.development[0]])
    result=trainer.fit(tmp_path/"fit")
    assert result["strict_reload_exact"] and result["updates"]==1
    torch.testing.assert_close(net.embedding,embedding,atol=0,rtol=0)
    for b,basis in zip(net.blocks,bases):torch.testing.assert_close(b.basis,basis,atol=0,rtol=0)
    fresh=model();joint.load_selected(fresh,result["selected_directory"])
    fresh_trainer=joint.JointTrainer(fresh,data,config=config)
    assert fresh_trainer.development()==trainer.development()
    with pytest.raises(FileExistsError):joint.save_selected(net,result["selected_directory"],metadata={})
    data.close()


def test_selection_prioritizes_accuracy_before_kl_and_dataset_rejects_heldout(tmp_path):
    assert joint.selection_key(dict(correct=31,kl=9))>joint.selection_key(dict(correct=30,kl=.01))
    assert joint.selection_key(dict(correct=31,kl=.1))>joint.selection_key(dict(correct=31,kl=.2))
    data=dataset(tmp_path);items=deepcopy(data.items);data.close();items[0]["split"]="evaluation"
    with pytest.raises(ValueError,match="Only training/development"):
        joint.JointDataset(items,tmp_path/"teacher.npy",[20,21])
    with pytest.raises(ValueError):joint.JointConfig(learning_rate=0)
    with pytest.raises(ValueError):joint.run(tmp_path,"evaluate")


def test_prepare_requires_explicit_hidden_cuda_before_source_reads(tmp_path,monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    monkeypatch.setattr(joint.recipe,"verify",lambda *_:pytest.fail("Source read before CUDA visibility guard"))
    with pytest.raises(ValueError,match="Hide CUDA"):
        joint.prepare(tmp_path/"new",tmp_path/"source")


def pilot_receipt(root,fingerprint="fixed"):
    directory=root/"pilot";directory.mkdir()
    joint.recipe.write(directory/"result.json",dict(status="completed",systems_only=True,updates=1))
    manifest=dict(status="completed",systems_only=True,fingerprint=fingerprint,
        result_sha256=joint.recipe.sha(directory/"result.json"),allocation={"GPU-tiny":2},peak_reserved_bytes=1024**2)
    joint.recipe.write(directory/"manifest.json",manifest)
    return manifest


def test_pilot_gate_checks_fingerprint_result_hash_and_reserved_peak(tmp_path):
    manifest=pilot_receipt(tmp_path)
    assert joint.verify_pilot(tmp_path,"fixed")==manifest
    with pytest.raises(ValueError,match="successful matching"):
        joint.verify_pilot(tmp_path,"different")
    with pytest.raises(MemoryError):joint.check_reserved_peak({"GPU-tiny":1},1024**2+1)
    assert joint.check_reserved_peak({"GPU-tiny":1},1024**2)==1024**2
    (tmp_path/"pilot/result.json").write_text('{"status":"completed","systems_only":true,"updates":2}')
    with pytest.raises(ValueError,match="successful matching"):
        joint.verify_pilot(tmp_path,"fixed")


def test_train_routes_through_pilot_guard_before_reservation_or_model(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sys
    calls=[]
    def wait(_):calls.append("lease");raise RuntimeError("reservation denied")
    monkeypatch.setitem(sys.modules,"experiments.structure_net.jspace_full_lease",SimpleNamespace(wait_ready=wait))
    monkeypatch.setattr(joint,"verify",lambda _:dict(fingerprint="fixed"))
    monkeypatch.setattr(joint,"build_model",lambda *_a,**_k:pytest.fail("Model allocated before prerequisites"))
    with pytest.raises(FileNotFoundError):joint.run(tmp_path,"train")
    assert calls==[]
    pilot_receipt(tmp_path)
    with pytest.raises(RuntimeError,match="reservation denied"):joint.run(tmp_path,"train")
    assert calls==["lease"]
    failure=joint.recipe.read(tmp_path/"train/failure.json")
    assert failure["status"]=="failed" and failure["allocation"] is None


def test_reserved_peak_failure_has_allocation_receipt_without_real_cuda(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sys
    lease=dict(id="test-lease",allocation={"GPU-tiny":1})
    monkeypatch.setitem(sys.modules,"experiments.structure_net.jspace_full_lease",SimpleNamespace(wait_ready=lambda _:lease))
    monkeypatch.setattr(joint,"verify",lambda _:dict(fingerprint="fixed",source="unused",arm="softmax",
        initialization="selected",config={"seed":1}))
    monkeypatch.setattr(joint,"build_model",lambda *_a,**_k:None)
    monkeypatch.setattr(torch,"manual_seed",lambda _:None)
    monkeypatch.setattr(torch.cuda,"init",lambda:None)
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda:None)
    monkeypatch.setattr(torch.cuda,"max_memory_reserved",lambda:2*1024**2)
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda:1024**2)
    with pytest.raises(MemoryError,match="exceeds lease"):joint.run(tmp_path,"pilot")
    failure=joint.recipe.read(tmp_path/"pilot/failure.json")
    assert failure["allocation"]==lease["allocation"] and failure["lease_id"]==lease["id"]
    assert failure["peak_reserved_bytes"]==2*1024**2
    assert not (tmp_path/"pilot/manifest.json").exists()
