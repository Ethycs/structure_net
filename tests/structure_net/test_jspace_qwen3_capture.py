"""Qwen3 capture must preserve tasks, causal execution, precision and ownership."""
from copy import deepcopy
import json
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.structure_net.jspace_qwen3_capture import (
    FP32BoundaryDataset, TokenizationLimitError, load_raw_cases, prepare_items,
    stream_native_capture, systems_subset,
)


class Tokenizer:
    def __init__(self, extra=0):self.calls=[];self.extra=extra
    def encode(self,text,*,add_special_tokens):
        assert add_special_tokens is False
        return [4+ord(text)-65]
    def apply_chat_template(self,messages,**kwargs):
        self.calls.append((deepcopy(messages),dict(kwargs)))
        assert kwargs == dict(tokenize=True,add_generation_prompt=True,enable_thinking=False)
        return [1,2,3]+[8+i%7 for i in range(int(messages[1]["content"]))]+[9]*self.extra


def cases():
    return [dict(identity=f"case{i}",family=f,split=s,prompt=str(n),choices=["x","y"] if f in ("boolq","winogrande") else ["w","x","y","z"],
                 answer=i%2,input_ids=[99999],loss_positions=[999],targets=[999],offset=-1,stop=-1)
            for i,(f,s,n) in enumerate((("arc","train",4),("boolq","development",8),("hellaswag","train",5),("winogrande","development",3)))]


def prepared():return prepare_items(cases(),Tokenizer(),systems_only=True)


def tiny(dtype):
    torch.manual_seed(51)
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,head_dim=4,max_position_embeddings=64,
        attention_dropout=0.,tie_word_embeddings=False)
    config._attn_implementation="eager"
    return Qwen3ForCausalLM(config).to(dtype=dtype).eval()


@torch.inference_mode()
def direct(model,items,label_ids):
    """Full HF model uses its own causal/padding-mask implementation."""
    length=max(len(c["input_ids"]) for c in items)
    ids=torch.zeros(len(items),length,dtype=torch.long);mask=torch.zeros_like(ids)
    for row,c in enumerate(items):
        ids[row,:len(c["input_ids"])]=torch.tensor(c["input_ids"]);mask[row,:len(c["input_ids"])]=1
    boundaries={0:model.model.embed_tokens(ids).detach().clone()};hooks=[]
    for i,block in enumerate(model.model.layers):
        def capture(module,args,value,index=i):
            boundaries[index+1]=(value[0] if isinstance(value,tuple) else value).detach().clone()
        hooks.append(block.register_forward_hook(capture))
    try:output=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
    finally:
        for h in hooks:h.remove()
    targets=torch.stack([output[i,c["loss_positions"]] for i,c in enumerate(items)]).float()
    banks={cut:torch.cat([bank[i,:len(c["input_ids"])] for i,c in enumerate(items)]).float().numpy()
           for cut,bank in boundaries.items()}
    return banks,targets.numpy()


def capture(tmp_path,*,dtype=torch.float32,cuts=(0,1,3),microbatch=4,guard=None):
    model=tiny(dtype);data=prepared();items=data["items"];refs=[];events=[]
    expected_banks,expected_logits=direct(model,items,data["label_ids"])
    def load(index):
        assert not refs or refs[-1]() is None
        block=deepcopy(model.model.layers[index]);refs.append(weakref.ref(block));events.append(("block",index))
        return block
    def readout():
        assert refs[-1]() is None
        events.append(("readout",None))
        return model.lm_head.weight,deepcopy(model.model.norm)
    result=stream_native_capture(items,input_embedding=model.model.embed_tokens.weight.detach(),block_loader=load,
        readout_loader=readout,rotary=model.model.rotary_emb,label_ids=data["label_ids"],directory=tmp_path/"capture",
        capture_cuts=cuts,layer_count=3,dtype=dtype,microbatch=microbatch,systems_only=True,guard=guard)
    return result,expected_banks,expected_logits,events


def test_retokenization_rebuilds_targets_and_excludes_old_ids_and_heldout(tmp_path):
    raw=cases();path=tmp_path/"tasks.json"
    path.write_text(json.dumps(dict(splits={"train":[c for c in raw if c["split"]=="train"],
        "development":[c for c in raw if c["split"]=="development"],"evaluation":[dict(raw[0],identity="DO_NOT_USE")]})))
    loaded=load_raw_cases(path,systems_only=True);tokenizer=Tokenizer();value=prepare_items(loaded,tokenizer,systems_only=True)
    assert {c["identity"] for c in value["items"]} == {c["identity"] for c in raw}
    assert all("input_ids" not in c for c in loaded)
    for c in value["items"]:
        assert 99999 not in c["input_ids"] and c["targets"][-1] == value["label_ids"][c["answer"]]
        assert c["targets"][:-1] == [c["input_ids"][p+1] for p in c["loss_positions"][:-1]]
    assert all(call[1]["enable_thinking"] is False for call in tokenizer.calls)
    assert raw[0]["input_ids"] == [99999]  # No source mutation.
    with pytest.raises(ValueError,match="512"):prepare_items(raw,Tokenizer())


def test_length_limit_reports_all_overflows_without_truncation_or_dropping():
    with pytest.raises(TokenizationLimitError) as error:prepare_items(cases(),Tokenizer(extra=420),systems_only=True)
    report=error.value.report
    assert report["over_limit_count"] == 4 and report["max_observed_length"] == 431
    assert report["truncated_count"] == report["dropped_count"] == 0
    assert len(report["over_limit"]) == 4


@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_streamed_native_banks_and_teacher_match_full_hf_model(tmp_path,dtype):
    result,banks,teacher,events=capture(tmp_path,dtype=dtype)
    assert result["capture_cuts"] == [0,1,3]
    for cut in result["capture_cuts"]:
        actual=np.load(result["boundary_paths"][str(cut)])
        np.testing.assert_array_equal(actual,banks[cut])
        assert actual.dtype == np.float32
    np.testing.assert_array_equal(np.load(result["teacher_logits_path"]),teacher)
    assert events == [("block",0),("block",1),("block",2),("readout",None)]
    assert result["owned_native_blocks_max"] == 1 and result["readout_loaded_after_blocks"]
    assert result["execution_dtype"] == str(dtype).split(".")[-1]
    assert not (tmp_path/"capture/_scratch").exists()
    assert not (tmp_path/"capture/boundary_02.npy").exists()
    assert result["teacher_inventory"]["tensor_bytes"] == 4*4*32*4
    assert result["by_split"]["train"]["count"] == result["by_split"]["development"]["count"] == 2


def test_bf16_disk_values_round_trip_and_float32_reader_has_explicit_dtype(tmp_path):
    result,_,teacher,_=capture(tmp_path,dtype=torch.bfloat16,cuts=(0,))
    payload=json.loads(Path(result["items_path"]).read_text());items=payload["items"]
    data=FP32BoundaryDataset(items,result["boundary_paths"]["0"],result["teacher_logits_path"],label_ids=payload["label_ids"],
                            original_logits_path=result["teacher_logits_path"],original_indices=[0,1,2,3])
    try:
        batch=data.batch([1,3],dtype=torch.bfloat16,include_original=True)
        assert batch["hidden"].dtype==torch.bfloat16 and batch["teacher"].dtype==torch.float32
        torch.testing.assert_close(batch["teacher"],torch.from_numpy(teacher[[1,3]]),rtol=0,atol=0)
        assert torch.equal(batch["teacher"],batch["original"])
        bank=np.load(result["boundary_paths"]["0"])
        for row,index in enumerate((1,3)):
            c=items[index]
            np.testing.assert_array_equal(batch["hidden"][row,:len(c["input_ids"])] .float().numpy(),bank[c["offset"]:c["stop"]])
        assert data.training == [0,2] and data.development == [1,3]
        with pytest.raises(ValueError):data.batch([0],dtype=torch.float16)
    finally:data.close()
    assert data.boundary._mmap.closed


def test_reader_rejects_bf16_bit_storage_and_wrong_final_gold_target(tmp_path):
    p=tmp_path/"bf16.npy";np.save(p,np.zeros((4,16),dtype=np.int16))
    with pytest.raises(ValueError,match="float32"):FP32BoundaryDataset._open(p,2)
    result,_,_,_=capture(tmp_path,cuts=(0,));payload=json.loads(Path(result["items_path"]).read_text())
    payload["items"][0]["targets"][-1]=0
    with pytest.raises(ValueError,match="gold"):
        FP32BoundaryDataset(payload["items"],result["boundary_paths"]["0"],result["teacher_logits_path"],label_ids=payload["label_ids"])


def test_guard_denial_precedes_model_load_or_cuda_and_output_creation(tmp_path,monkeypatch):
    data=prepared();calls=[]
    def denied():calls.append("guard");raise RuntimeError("no lease")
    def loader(*a):raise AssertionError("model loader must not run")
    monkeypatch.setattr(torch.cuda,"init",lambda:(_ for _ in ()).throw(AssertionError("CUDA must not initialize")))
    with pytest.raises(RuntimeError,match="no lease"):
        stream_native_capture(data["items"],input_embedding=torch.zeros(32,16),block_loader=loader,
            readout_loader=loader,rotary=None,label_ids=data["label_ids"],directory=tmp_path/"blocked",
            capture_cuts=(0,),layer_count=3,device="cuda:0",systems_only=True,guard=denied)
    assert calls==["guard"] and not (tmp_path/"blocked").exists()


def test_failure_cleans_only_owned_scratch_and_keeps_completed_requested_bank(tmp_path):
    data=prepared();model=tiny(torch.float32);foreign=tmp_path/"unrelated.npy";foreign.write_bytes(b"keep")
    def broken(index):
        if index==1:raise RuntimeError("test failure")
        return deepcopy(model.model.layers[index])
    with pytest.raises(RuntimeError,match="test failure"):
        stream_native_capture(data["items"],input_embedding=model.model.embed_tokens.weight.detach(),block_loader=broken,
            readout_loader=lambda:None,rotary=model.model.rotary_emb,label_ids=data["label_ids"],directory=tmp_path/"failed",
            capture_cuts=(1,),layer_count=3,dtype=torch.float32,systems_only=True)
    assert foreign.read_bytes()==b"keep" and (tmp_path/"failed/boundary_01.npy").exists()
    assert not (tmp_path/"failed/_scratch").exists()
    assert json.loads((tmp_path/"failed/failure.json").read_text())["completed_cuts"] == [1]


def test_systems_subset_rebases_offsets_without_mutation_and_science_requires_full_set():
    original=prepared()["items"];subset=systems_subset(original,[3,1])
    assert subset[0]["offset"]==0 and subset[1]["offset"]==len(subset[0]["input_ids"])
    assert original[3]["offset"] > 0 and [c["identity"] for c in subset]==[original[3]["identity"],original[1]["identity"]]
    with pytest.raises(ValueError):systems_subset(original,[1,1])


def test_execution_disables_autocast_tf32_and_restores_flags(tmp_path):
    before=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    observed=[]
    def guard():observed.append((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))
    try:
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        capture(tmp_path,dtype=torch.float32,cuts=(),guard=guard)
        assert observed[0] == (True,True) and all(x==(False,False) for x in observed[1:])
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=before
