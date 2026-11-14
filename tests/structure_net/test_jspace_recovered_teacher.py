from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_recovered_teacher as teacher


def tiny():
    torch.manual_seed(442)
    config = Qwen2Config(hidden_size=16,intermediate_size=24,num_attention_heads=4,num_key_value_heads=2,
        num_hidden_layers=2,vocab_size=31,max_position_embeddings=32,tie_word_embeddings=True,attention_dropout=0.)
    config._attn_implementation = "eager"
    native = Qwen2ForCausalLM(config).bfloat16().float().eval().requires_grad_(False)
    model = teacher.PartitionedNativeTeacher(native.model.embed_tokens.weight.detach().clone(),
        [deepcopy(block) for block in native.model.layers],deepcopy(native.model.norm),
        deepcopy(native.model.rotary_emb),deepcopy(native.model.rotary_emb),cut=1)
    tokens = [[1,2,3],[1,2,7,8,9,10],[3,4,5,6],[3,8,9,6,5]]
    items = [dict(identity=str(i),split="train" if i<2 else "development",family="tiny",input_ids=ids,
        loss_positions=[0,len(ids)//2,len(ids)-2,len(ids)-1],choices=["A","B"],answer=i%2) for i,ids in enumerate(tokens)]
    return native,model,items,[29,30]


def test_partitioned_native_matches_real_unpartitioned_qwen_and_causality():
    native,model,items,labels = tiny()
    batch = teacher.token_batch(items,[0,1],"cpu")
    args = {k:v for k,v in batch.items() if k != "indices"}
    logits = model(**args)
    lengths = torch.tensor([len(items[i]["input_ids"]) for i in [0,1]])
    valid = batch["position_ids"] < lengths[:,None]
    with torch.inference_mode():
        full = native(input_ids=batch["input_ids"],position_ids=batch["position_ids"],attention_mask=valid,use_cache=False).logits
        wanted = full[torch.arange(2)[:,None],batch["loss_positions"][:,[1,3]]]
    torch.testing.assert_close(logits,wanted,atol=2e-6,rtol=2e-6)
    assert logits.shape == (2,2,31) and logits.dtype == torch.float32 and not logits.requires_grad
    assert all(not p.requires_grad for p in model.parameters())
    # Changing suffixes cannot alter the first case's midpoint at position1.
    changed = {k:v.clone() for k,v in args.items()};changed["input_ids"][0,-1] = 15
    torch.testing.assert_close(model(**changed)[0,0],logits[0,0],atol=0,rtol=0)

    class TensorResult(torch.nn.Module):
        def __init__(self,block):super().__init__();self.block=block
        def forward(self,*args,**kwargs):
            value=self.block(*args,**kwargs)
            return value[0] if isinstance(value,tuple) else value
    model.blocks=torch.nn.ModuleList([TensorResult(block) for block in model.blocks])
    torch.testing.assert_close(model(**args),logits,atol=0,rtol=0)


def test_fp32_capture_dataset_roundtrip_keeps_original_four_positions(tmp_path):
    _,model,items,labels = tiny()
    value = teacher.capture(model,items,labels,tmp_path/"cache",microbatch=2)
    dataset = teacher.FP32JointDataset(items,tmp_path/"cache/teacher_logits.npy",labels)
    try:
        batch = dataset.batch([1,2],"cpu")
        assert dataset.training == [0,1] and dataset.development == [2,3]
        assert batch["teacher"].shape == (2,2,31) and batch["teacher"].dtype == torch.float32
        assert batch["loss_positions"].tolist() == [items[i]["loss_positions"] for i in [1,2]]
        assert batch["target_tokens"].tolist() == [30,29]
        direct = model(**{k:batch[k] for k in ("input_ids","position_ids","attention_mask","loss_positions")})
        torch.testing.assert_close(batch["teacher"],direct,atol=2e-6,rtol=2e-6)
        assert value["shape"] == [4,2,31] and value["full_boundary_banks_written"] == 0
        assert len(list((tmp_path/"cache").glob("*.npy"))) == 1
        assert [r["identity"] for r in value["records"]] == ["2","3"]
    finally: dataset.close()


@pytest.mark.parametrize("bad",["dtype","duplicate","split","position","nonfinite"])
def test_dataset_rejects_malformed_or_nonfinite_targets(tmp_path,bad):
    _,_,items,labels = tiny();path = tmp_path/"logits.npy"
    values = np.zeros((4,2,31),dtype=np.float32)
    if bad == "dtype": values = values.astype(np.float16)
    elif bad == "duplicate":items[2]["identity"] = items[0]["identity"]
    elif bad == "split":items[2]["split"] = "confirmation"
    elif bad == "position":items[2]["loss_positions"][-1] = 99
    elif bad == "nonfinite":values[2,0,0] = np.nan
    np.save(path,values)
    if bad == "nonfinite":
        dataset = teacher.FP32JointDataset(items,path,labels)
        try:
            with pytest.raises(FloatingPointError):dataset.batch([2],"cpu")
        finally:dataset.close()
    else:
        with pytest.raises(ValueError):teacher.FP32JointDataset(items,path,labels)


def test_native_control_rejects_prediction_logit_and_identity_changes(tmp_path):
    _,model,items,labels = tiny();rows = teacher.capture(model,items,labels,tmp_path/"cache")["records"]
    cases = items[2:];correct = sum(r["prediction"] == r["answer"] for r in rows)
    assert teacher.check_control(rows,rows,cases,expected_correct=correct)["passed"]
    changed = deepcopy(rows);changed[0]["logits"] = [v+1 for v in changed[0]["logits"]]
    assert not teacher.check_control(changed,rows,cases,expected_correct=correct)["passed"]
    changed = deepcopy(rows);changed[0]["logits"] = [-100.,100.] if changed[0]["prediction"] == 0 else [100.,-100.]
    changed[0]["prediction"] = 1-changed[0]["prediction"]
    assert not teacher.check_control(changed,rows,cases,expected_correct=correct)["predictions_exact"]
    changed = deepcopy(rows);changed[0]["identity"] = "wrong"
    with pytest.raises(ValueError,match="identities"):teacher.check_control(changed,rows,cases,expected_correct=correct)


class FakeCUDA:
    def __init__(self):self.caps=[];self.memory=SimpleNamespace(get_allocator_backend=lambda:"native")
    def is_initialized(self):return False
    def get_device_properties(self,index):return SimpleNamespace(total_memory=(8 if index==0 else 12)*1024**3)
    def set_per_process_memory_fraction(self,fraction,device):self.caps.append((device,fraction))


def test_allocator_maps_reversed_physical_uuids_without_assuming_cuda_order():
    ids = ["GPU-7b1db3cb-SUPER","GPU-0370fdb0-3060"]
    lease = dict(id="lease",allocation={ids[0]:7168,ids[1]:11264});cuda = FakeCUDA()
    settings = teacher.allocator_settings(lease,cuda=cuda,visible=",".join(ids))
    assert settings["front"]["device"] == 1 and settings["tail"]["device"] == 0
    assert settings["front"]["allocator_limit_bytes"] == 11008*1024**2
    assert settings["tail"]["allocator_limit_bytes"] == 6912*1024**2
    assert len(cuda.caps) == 2
    lease["allocation"][ids[1]] = 7168
    with pytest.raises(ValueError,match="11264"):teacher.allocator_settings(lease,cuda=cuda,visible=",".join(ids))


def test_reservation_is_required_before_cuda_or_model_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(teacher,"verify",lambda _root:dict(fingerprint="fixture"))
    def reject():raise RuntimeError("No live lease")
    monkeypatch.setattr(teacher,"reservation",reject)
    monkeypatch.setattr(teacher,"allocator_settings",lambda *_a,**_k:pytest.fail("CUDA before lease"))
    with pytest.raises(RuntimeError,match="No live lease"):teacher.run(tmp_path)
    assert teacher.recipe.read(tmp_path/"failure.json")["error"] == "No live lease"


def test_cuda_capture_guard_and_cpu_prepare_guard(tmp_path,monkeypatch):
    fake = SimpleNamespace(front=torch.device("cuda:0"))
    with pytest.raises(ValueError,match="reservation guard"):teacher.capture(fake,[],[],tmp_path/"cache")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    monkeypatch.setattr(teacher.recipe,"verify",lambda *_:pytest.fail("Inputs read before CPU guard"))
    with pytest.raises(ValueError,match="Hide CUDA"):teacher.prepare(tmp_path,tmp_path,tmp_path)
