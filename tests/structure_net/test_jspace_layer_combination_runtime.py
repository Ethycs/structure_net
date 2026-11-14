"""Shared-prefix scoring must retain actual whole-model FP32 decisions."""
from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_layer_combination_runtime as runtime
from experiments.structure_net.jspace_semantic_recovery import build_enriched_block


def fixture():
    torch.manual_seed(553)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=3, vocab_size=11,
        max_position_embeddings=32, attention_dropout=0.)
    config._attn_implementation = "eager"
    native, compact = [], []
    for layer in range(3):
        original = Qwen2DecoderLayer(config, layer_idx=layer).bfloat16().float().eval().requires_grad_(False)
        fit = dict(basis=torch.linalg.qr(torch.randn(8,4)).Q, means=torch.randn(1,8)/5)
        recovered = build_enriched_block(deepcopy(original),fit,parameter_dtype=torch.float32)
        with torch.no_grad():
            for i,p in enumerate(recovered.parameters()):p.add_(.002*(i+1)*(layer+1))
        native.append(original);compact.append(recovered.pack(dtype=torch.float32,buffer_dtype=torch.float32))
    embedding = torch.randn(11,8).bfloat16().float()
    norm = Qwen2RMSNorm(8).eval().requires_grad_(False)
    rotary = Qwen2RotaryEmbedding(config=config)
    bank = runtime.MixedBlockBank(embedding,native,compact,norm,{"cpu":rotary})
    cases = [dict(identity="a",family="two",split="development",input_ids=[1,2,3],choices=["A","B"],answer=0),
             dict(identity="b",family="four",split="development",input_ids=[2,4,1,3,5],choices=list("ABCD"),answer=3),
             dict(identity="c",family="two",split="development",input_ids=[6,1,5,2],choices=["A","B"],answer=1)]
    labels = [7,8,9,10]
    with torch.no_grad():
        teacher = []
        for case in cases:
            hidden = embedding[case["input_ids"]][None]
            positions = torch.arange(hidden.shape[1])[None]
            allowed = torch.ones(1,1,hidden.shape[1],hidden.shape[1],dtype=torch.bool).tril()
            mask = torch.zeros_like(allowed,dtype=torch.float32).masked_fill(~allowed,torch.finfo(torch.float32).min)
            for block in native:
                hidden = block(hidden,attention_mask=mask,position_ids=positions,
                    position_embeddings=rotary(hidden,positions))[0]
            logits = F.linear(norm(hidden[:,-1]),embedding[labels[:len(case["choices"])]]).squeeze(0)
            teacher.append(dict(identity=case["identity"],family=case["family"],answer=case["answer"],
                prediction=int(logits.argmax()),logits=logits.tolist()))
    return bank,cases,labels,teacher


def masks():
    return {"".join("1" if x else "0" for x in bits):bits for bits in product((False,True),repeat=3)}


def test_all_masks_shared_trie_match_independent_actual_Qwen_with_fewer_decoder_calls():
    bank,cases,labels,teacher = fixture()
    fixed = {name:value.clone() for name,value in (*bank.named_parameters(),*bank.named_buffers())}
    before = runtime.state_digest(bank,chunk_bytes=64)
    common = dict(teacher_records=teacher,microbatch=2)
    shared = runtime.score_masks(bank,cases,masks(),labels,share_prefixes=True,**common)
    separate = runtime.score_masks(bank,cases,masks(),labels,share_prefixes=False,**common)
    assert shared["results"] == separate["results"]
    assert runtime.state_digest(bank,chunk_bytes=1024) == before
    for name,value in (*bank.named_parameters(),*bank.named_buffers()):
        torch.testing.assert_close(value,fixed[name],atol=0,rtol=0)
        assert not value.requires_grad and value.grad is None
    # Eight depth-three paths have14 distinct nonroot prefixes per minibatch.
    assert shared["execution"]["decoder_calls"] == 14*2
    assert separate["execution"]["decoder_calls"] == 24*2
    assert shared["execution"]["shared_prefix_saved_calls"] == (24-14)*2
    assert shared["execution"]["native_calls"]+shared["execution"]["compact_calls"] == 28
    native = shared["results"]["000"]
    assert native["teacher_agreement"] == 1.
    for actual,expected in zip(native["records"],teacher):
        assert actual["prediction"] == expected["prediction"]
        torch.testing.assert_close(torch.tensor(actual["logits"]),torch.tensor(expected["logits"]),atol=3e-6,rtol=3e-6)
    # An independent explicit alternating-layer calculation also establishes
    # which physical version each Boolean selects; trie equality alone cannot.
    with torch.no_grad():
        for key in ("101","010"):
            for case,record in zip(cases,shared["results"][key]["records"]):
                hidden = bank.embedding[case["input_ids"]][None]
                positions = torch.arange(hidden.shape[1])[None]
                allowed = torch.ones(1,1,hidden.shape[1],hidden.shape[1],dtype=torch.bool).tril()
                mask = torch.zeros_like(allowed,dtype=torch.float32).masked_fill(~allowed,torch.finfo(torch.float32).min)
                for layer,bit in enumerate(masks()[key]):
                    block = bank.compact[layer] if bit else bank.native[layer]
                    value = block(hidden,attention_mask=mask,position_ids=positions,
                        position_embeddings=bank.rotary_for("cpu")(hidden,positions))
                    hidden = value[0] if isinstance(value,tuple) else value
                expected = F.linear(bank.norm(hidden[:,-1]),bank.embedding[labels[:len(case["choices"])]]).squeeze(0)
                torch.testing.assert_close(torch.tensor(record["logits"]),expected,atol=4e-6,rtol=4e-6)


def test_mask_order_and_padding_do_not_change_per_case_decisions_or_conditional_metrics():
    bank,cases,labels,teacher = fixture()
    chosen = {"native":(False,False,False),"mixed":(True,False,True)}
    batched = runtime.score_masks(bank,cases,chosen,labels,teacher_records=teacher,microbatch=3)
    individual = runtime.score_masks(bank,cases,dict(reversed(list(chosen.items()))),labels,
        teacher_records=teacher,microbatch=1)
    for key in chosen:
        value = batched["results"][key];other = individual["results"][key]
        assert [r["identity"] for r in value["records"]] == [c["identity"] for c in cases]
        assert [r["prediction"] for r in value["records"]] == [r["prediction"] for r in other["records"]]
        ce,margins = [],[]
        for row,case in zip(value["records"],cases):
            assert len(row["logits"]) == len(case["choices"])
            logits = torch.tensor(row["logits"])
            ce.append(F.cross_entropy(logits[None],torch.tensor([case["answer"]])).item())
            remaining = logits.clone();remaining[case["answer"]] = -torch.inf
            margins.append(float(logits[case["answer"]]-remaining.max()))
        assert value["choice_ce"] == pytest.approx(sum(ce)/3,rel=2e-6,abs=2e-6)
        assert value["choice_margin"] == pytest.approx(sum(margins)/3,rel=2e-6,abs=2e-6)
        assert value["teacher_agreement"] == sum(a["prediction"]==b["prediction"] for a,b in zip(value["records"],teacher))/3
        assert value["choice_ce"] == pytest.approx(other["choice_ce"],rel=3e-6,abs=3e-6)


@pytest.mark.parametrize("bad", [
    {"short":(True,False)},
    {"integer":(1,0,1)},
    {"one":(True,False,True),"duplicate":(True,False,True)},
])
def test_invalid_or_duplicate_masks_fail_before_forward(bad,monkeypatch):
    bank,cases,labels,teacher = fixture()
    for block in (*bank.native,*bank.compact):
        monkeypatch.setattr(block,"forward",lambda *_a,**_k:pytest.fail("Forward before mask validation"))
    with pytest.raises(ValueError):runtime.score_masks(bank,cases,bad,labels,teacher_records=teacher)


def test_declined_guard_prevents_first_decoder_execution(monkeypatch):
    bank,cases,labels,teacher = fixture()
    def denied():raise RuntimeError("reservation revoked")
    for block in (*bank.native,*bank.compact):
        monkeypatch.setattr(block,"forward",lambda *_a,**_k:pytest.fail("Decoder before reservation guard"))
    monkeypatch.setattr(F,"embedding",lambda *_a,**_k:pytest.fail("Embedding before reservation guard"))
    with pytest.raises(RuntimeError,match="reservation revoked"):
        runtime.score_masks(bank,cases,masks(),labels,teacher_records=teacher,guard=denied)


def test_allocator_resolves_three_physical_cards_in_reversed_visible_order():
    uuids = ["GPU-8cf440df-test","GPU-7b1db3cb-test","GPU-0370fdb0-test"]
    allocation = dict(zip(uuids,[5120,7168,11264]));events = []
    fake = SimpleNamespace(is_initialized=lambda:False,device_count=lambda:3,
        get_device_properties=lambda i:SimpleNamespace(total_memory=[6,8,12][i]*1024**3),
        memory=SimpleNamespace(get_allocator_backend=lambda:"native"),
        set_per_process_memory_fraction=lambda fraction,device:events.append((fraction,device)))
    result = runtime.allocator_settings(dict(id="lease",allocation=allocation),cuda=fake,visible=",".join(uuids))
    assert result["devices"] == {"2060":"cuda:0","2060_SUPER":"cuda:1","3060":"cuda:2"}
    settings = {v["physical_uuid"]:v for v in result["settings"]}
    for uuid,mib in allocation.items():assert settings[uuid]["allocator_limit_bytes"] == (mib-256)*1024**2
    assert len(events) == 3
    events.clear()
    with pytest.raises(ValueError):
        runtime.allocator_settings(dict(id="missing",allocation={uuids[2]:11264}),cuda=fake,visible=",".join(uuids))
    assert not events


def test_state_digest_covers_actual_weights_and_fixed_geometry_independent_of_chunks():
    bank,*_ = fixture()
    original = runtime.state_digest(bank,chunk_bytes=64)
    assert runtime.state_digest(bank,chunk_bytes=1024) == original
    with torch.no_grad():next(bank.native[0].parameters()).reshape(-1)[0].add_(.01)
    weight_changed = runtime.state_digest(bank,chunk_bytes=64)
    assert weight_changed["sha256"] != original["sha256"]
    with torch.no_grad():bank.compact[0].basis[0,0].add_(.002)
    geometry_changed = runtime.state_digest(bank,chunk_bytes=64)
    assert geometry_changed["sha256"] != weight_changed["sha256"]
    assert geometry_changed["bytes"] == weight_changed["bytes"] == original["bytes"]
