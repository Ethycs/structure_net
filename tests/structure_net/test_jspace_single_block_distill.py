from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding

from experiments.structure_net import jspace_single_block_distill as single


def fixture(tmp_path):
    torch.manual_seed(187)
    config=Qwen2Config(hidden_size=16,intermediate_size=24,num_attention_heads=4,num_key_value_heads=2,
        num_hidden_layers=2,vocab_size=23,max_position_embeddings=32,attention_dropout=0.)
    config._attn_implementation="eager"
    native=Qwen2DecoderLayer(config,layer_idx=0).bfloat16().float().eval().requires_grad_(False)
    suffix=Qwen2DecoderLayer(config,layer_idx=1).bfloat16().float().eval().requires_grad_(False)
    rotary=Qwen2RotaryEmbedding(config=config)
    embedding=torch.randn(23,16).bfloat16().float()/3
    items=[];inputs=[];targets=[];cursor=0
    for i,length in enumerate((4,6,5,4)):
        ids=[(i+j)%19 for j in range(length)]
        x=embedding[ids][None];pos=torch.arange(length)[None]
        mask=torch.zeros(length,length).masked_fill(torch.ones(length,length,dtype=torch.bool).triu(1),torch.finfo(torch.float32).min)[None,None]
        with torch.no_grad():y=native(x,attention_mask=mask,position_embeddings=rotary(x,pos))[0]
        inputs.append(x[0]);targets.append(y[0])
        items.append(dict(identity=str(i),family="tiny",split="train" if i<2 else "development",input_ids=ids,
            choices=["A","B"],answer=i%2,offset=cursor,stop=cursor+length));cursor+=length
    for name,values in (("input",inputs),("target",targets)):
        np.save(tmp_path/f"{name}.npy",torch.cat(values).bfloat16().view(torch.int16).numpy())
    data=single.NativePairs(items,tmp_path/"input.npy",tmp_path/"target.npy")
    fit=dict(basis=torch.linalg.qr(torch.randn(16,8)).Q,means=torch.cat(inputs[:2]).mean(0)[None])
    block=single.build_enriched_block(native,fit,parameter_dtype=torch.float32)
    def quality(live):
        frozen=deepcopy(live).pack(dtype=torch.float32,buffer_dtype=torch.float32);records=[]
        with torch.no_grad():
            for c in items[2:]:
                x=embedding[c["input_ids"]][None];length=x.shape[1];pos=torch.arange(length)[None]
                allowed=(pos[:,:,None]>=pos[:,None,:])[:,None]
                rope=rotary(x,pos)
                h=frozen(x,attention_mask=allowed,position_ids=pos,position_embeddings=rope)
                mask=torch.zeros_like(allowed,dtype=torch.float32).masked_fill(~allowed,torch.finfo(torch.float32).min)
                h=suffix(h,attention_mask=mask,position_embeddings=rope)[0]
                norm=h[:,-1]/(h[:,-1].square().mean(-1,keepdim=True)+1e-6).sqrt()
                logits=F.linear(norm,embedding[[20,21]])[0]
                records.append(dict(identity=c["identity"],family=c["family"],answer=c["answer"],
                    prediction=int(logits.argmax()),logits=logits.tolist()))
        return dict(records=records)
    return block,rotary,data,quality,native,suffix


def test_relative_update_loss_is_equal_token_not_ratio_of_global_energies():
    hidden=torch.zeros(1,2,2,requires_grad=True)
    target=torch.tensor([[[100.,100.],[1.,1.]]],requires_grad=True)
    student=torch.tensor([[[120.,120.],[1.1,1.1]]],requires_grad=True)
    values=single.feature_losses(student,hidden,target,torch.ones(1,2,dtype=torch.bool),floor=1e-6)
    assert float(values["loss"])==pytest.approx((.04+.01)/2,rel=2e-6)
    assert float(values["mse"])==pytest.approx((400+.01)/2,rel=2e-6)
    values["loss"].backward()
    assert target.grad is None and hidden.grad is None and student.grad is not None


def test_train_only_floor_gradients_and_frozen_teacher_suffix(tmp_path):
    block,rotary,data,quality,native,suffix=fixture(tmp_path)
    before=[deepcopy(m.state_dict()) for m in (native,suffix)]
    config=single.Config(epochs=1,microbatch=1,accumulation=2)
    trainer=single.Trainer(block,rotary,data,quality,config=config)
    assert data.normalization["train_case_count"]==2 and data.normalization["train_token_count"]==10
    expected=[]
    for i in data.training:
        batch=data.batch([i],"cpu");expected.extend((batch["target"]-batch["hidden"]).square().mean(-1)[batch["valid"]].tolist())
    assert data.normalization["floor"]==pytest.approx(torch.tensor(expected).median().item()*1e-4)
    batch=data.batch([0,1],"cpu");values=trainer.losses(batch);values["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert not batch["hidden"].requires_grad and not batch["target"].requires_grad
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.development()
    assert all(p.requires_grad for p in block.parameters())
    trainer.train_window([0,1])
    for module,state in zip((native,suffix),before):
        assert all(p.grad is None and not p.requires_grad for p in module.parameters())
        for key,value in module.state_dict().items():torch.testing.assert_close(value,state[key],atol=0,rtol=0)
    data.close()


def test_all_epoch_checkpoints_accuracy_selection_and_reload(tmp_path):
    block,rotary,data,quality,_,_=fixture(tmp_path)
    trainer=single.Trainer(block,rotary,data,quality,config=single.Config(epochs=1,microbatch=1,accumulation=2))
    with pytest.raises(ValueError,match="Training cases only"):trainer.train_window(data.development)
    result=trainer.fit(tmp_path/"fit",native_correct=2,minimum_correct=0)
    assert result["strict_reload_exact"] and result["updates"]==1
    assert (tmp_path/"fit/epoch_00.pt").exists() and (tmp_path/"fit/epoch_01.pt").exists()
    assert result["selected"]["correct"]==max(r["metrics"]["correct"] for r in result["history"])
    for group in ("first","content","answer"):
        assert group+"_relative_mse" in result["selected"]
    assert "irreducible_outside_relative_mse" in result["selected"]
    data.close()


def test_irreducible_floor_and_padding_do_not_change_gradient():
    h0=torch.zeros(1,3,2);target=torch.tensor([[[1.,2.],[2.,3.],[0.,0.]]])
    student=torch.tensor([[[.5,0.],[1.,0.],[1000.,1000.]]],requires_grad=True)
    valid=torch.tensor([[True,True,False]]);basis=torch.tensor([[1.],[0.]])
    a=single.feature_losses(student,h0,target,valid,floor=.01,basis=basis)
    b=single.feature_losses(student,h0,target,valid,floor=.01)
    torch.testing.assert_close(a["loss"],b["loss"])
    assert a["irreducible_outside_relative_mse"]>0
    a["loss"].backward();assert torch.equal(student.grad[:,2],torch.zeros_like(student.grad[:,2]))


def test_preparation_and_pilot_gates_precede_gpu(tmp_path,monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    monkeypatch.setattr(single.recipe,"verify",lambda *_:pytest.fail("Read before CPU guard"))
    with pytest.raises(ValueError,match="Hide CUDA"):single.prepare(tmp_path/"new",tmp_path,tmp_path,tmp_path)
    monkeypatch.setattr(single,"verify",lambda _:dict(fingerprint="test"))
    with pytest.raises(FileNotFoundError):single.run(tmp_path,"train")
