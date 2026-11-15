from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_teacher_targets as targets
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack
from experiments.structure_net.jspace_qwen_parallel_train import BoundaryDataset
from experiments.structure_net.jspace_incremental_suffix import score


def fixture(tmp_path):
    torch.manual_seed(9021)
    config=Qwen2Config(hidden_size=8,intermediate_size=12,num_attention_heads=2,num_key_value_heads=1,
        num_hidden_layers=3,vocab_size=16,max_position_embeddings=32,attention_dropout=0.,tie_word_embeddings=True)
    config._attn_implementation="eager"
    native=Qwen2ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    blocks=[]
    for layer in (1,2):
        basis=torch.linalg.qr(torch.randn(8,8)).Q[:,:4]
        blocks.append(AffineQwenBlock(native.model.layers[layer],basis,torch.randn(1,8)/10,trainable=False))
    stack=CompactQwenStack(blocks,geometry_mode="bf16",pad_to_multiple=128).eval()
    stack.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True).requires_grad_(False)
    norm=native.model.norm
    with torch.no_grad():norm.weight.fill_(1.25)
    items=[];offset=0;banks=[]
    for i in range(11):
        length=3+i%4;ids=torch.randint(0,16,(length,)).tolist()
        case=dict(identity=f"case{i}",family="a",answer=i%2,choices=["a","b"],
            split="train" if i%2==0 else "development",input_ids=ids,offset=offset,stop=offset+length,
            loss_positions=[0,(length-2)//2,length-2,length-1],targets=[0,1,2,3])
        items.append(case);offset+=length;banks.append(torch.randn(length,8).bfloat16())
    boundary=tmp_path/"boundary.npy";np.save(boundary,torch.cat(banks).view(torch.int16).numpy())
    placeholder=tmp_path/"original_logits.npy";np.save(placeholder,np.zeros((len(items),4,16),dtype=np.int16))
    data=BoundaryDataset(items,boundary,placeholder)
    return stack,norm,native.lm_head.weight,native.model.rotary_emb,data


def decoded(path):return torch.from_numpy(np.load(path).copy()).view(torch.bfloat16)


@torch.inference_mode()
def direct_logits(stack,norm,E,rotary,data,indices):
    batch=data.batch(indices)
    h=stack(batch["hidden"],attention_mask=batch["attention_mask"],position_ids=batch["position_ids"],
        cache_position=batch["position_ids"][0],position_embeddings=rotary(batch["hidden"],batch["position_ids"]),use_cache=False)[0]
    chosen=h[torch.arange(len(indices))[:,None],batch["loss_positions"]]
    return F.linear(norm(chosen),E)


def test_actual_tiny_packed_parent_full_vocab_and_old_score_match_exactly(tmp_path):
    stack,norm,E,rotary,data=fixture(tmp_path);path=tmp_path/"packed.npy";events=[]
    before={k:v.clone() for k,v in stack.state_dict().items() if isinstance(v,torch.Tensor)}
    norm_before=norm.weight.clone();E_before=E.clone()
    receipt=targets.capture_packed_teacher(path,stack,norm,E,rotary,data,progress=events.append,label_ids=[0,1,2,3])
    actual=decoded(path)
    for indices in (data.training[:4],data.training[4:],data.development[:4],data.development[4:]):
        assert torch.equal(actual[indices],direct_logits(stack,norm,E,rotary,data,indices))
    reference=score(stack,norm,E,rotary,data,[0,1,2,3],lambda:None,batch_size=4)
    assert receipt["records"]==reference
    assert receipt["shape"]==[11,4,16] and receipt["fixed_state_unchanged"] is True
    assert all(torch.equal(v,stack.state_dict()[k]) for k,v in before.items())
    assert torch.equal(norm.weight,norm_before) and torch.equal(E,E_before)
    assert all(not p.requires_grad and p.grad is None for p in stack.parameters())
    assert events[-1]["completed"]==11 and np.load(path).dtype==np.int16
    assert receipt["calibration_jacobians_changed"] is False
    data.close()


def test_split_grouping_and_source_case_row_order_are_preserved(tmp_path):
    stack,norm,E,rotary,data=fixture(tmp_path)
    receipt=targets.capture_packed_teacher(tmp_path/"packed.npy",stack,norm,E,rotary,data)
    assert receipt["batches"]==[
        dict(split="train",indices=[0,2,4,6]),dict(split="train",indices=[8,10]),
        dict(split="development",indices=[1,3,5,7]),dict(split="development",indices=[9])]
    assert receipt["records"] is None and receipt["training_count"]==6 and receipt["development_count"]==5
    with pytest.raises(FileExistsError):targets.capture_packed_teacher(tmp_path/"packed.npy",stack,norm,E,rotary,data)
    data.close()


def test_no_dtype_conversion_of_teacher_logits_and_no_training_allowed(tmp_path):
    stack,norm,E,rotary,data=fixture(tmp_path)
    with pytest.raises(ValueError,match="BF16 readout"):
        targets.capture_packed_teacher(tmp_path/"fp32.npy",stack,norm,E.float(),rotary,data)
    norm.weight.requires_grad_(True)
    with pytest.raises(ValueError,match="frozen and eval"):
        targets.capture_packed_teacher(tmp_path/"trainable.npy",stack,norm,E,rotary,data)
    assert not (tmp_path/"trainable.npy").exists()
    data.close()


def test_wrong_split_or_batch_policy_is_rejected_before_file_creation(tmp_path):
    stack,norm,E,rotary,data=fixture(tmp_path)
    with pytest.raises(ValueError,match="microbatch4"):
        targets.capture_packed_teacher(tmp_path/"batch.npy",stack,norm,E,rotary,data,microbatch=1)
    data.development.append(data.training[0])
    with pytest.raises(ValueError,match="Disjoint complete"):
        targets.capture_packed_teacher(tmp_path/"split.npy",stack,norm,E,rotary,data)
    assert not (tmp_path/"split.npy").exists();data.close()


def test_teacher_mutation_aborts_and_preserves_partial_cache(tmp_path):
    stack,norm,E,rotary,data=fixture(tmp_path);path=tmp_path/"partial.npy"
    def mutate(_m,_args,_out):norm.weight.add_(.125)
    handle=stack.register_forward_hook(mutate)
    try:
        with pytest.raises(ValueError,match="changed during target capture"):
            targets.capture_packed_teacher(path,stack,norm,E,rotary,data)
    finally:handle.remove();data.close()
    assert path.exists()


def test_cuda_requires_guard_before_any_initialization(tmp_path,monkeypatch):
    touched=[];monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:touched.append(True))
    with pytest.raises(ValueError,match="reservation guard"):
        targets.capture_packed_teacher(tmp_path/"never.npy",None,None,
            SimpleNamespace(device=torch.device("cuda")),None,None)
    assert touched==[] and not (tmp_path/"never.npy").exists()
