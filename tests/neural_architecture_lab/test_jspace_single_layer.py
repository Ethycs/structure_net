from copy import deepcopy
import json
from pathlib import Path
import struct

import pytest
import torch

from neural_architecture_lab.jspace_single_layer_record import JSpaceSingleLayerRecord, STUDY_ID
from experiments.neural_architecture_lab import finalize_jspace_single_layer as finalizer


def evidence():
    families=("arc","boolq","hellaswag","winogrande")
    def records(correct):
        return [dict(identity=str(i),family=families[i//16],answer=0,prediction=int(i>=correct),
            logits=[1.,0.] if i<correct else [0.,1.]) for i in range(64)]
    features=dict(relative_mse=.1,mse=.01,first_relative_mse=.05,content_relative_mse=.1,
        answer_relative_mse=.2,irreducible_outside_relative_mse=.03,retained_relative_mse=.07)
    history=[]
    for epoch in range(13):
        metrics=dict(**features,correct=46 if epoch<5 else 48,count=64)
        metrics["relative_mse"] = .3-epoch*.01
        history.append(dict(epoch=epoch,metrics=metrics))
    selected=history[-1]["metrics"]
    normalization=dict(train_case_count=512,median_teacher_update_energy=.2,floor_fraction=1e-4,floor=.00002)
    value=dict(status="completed",confirmed=False,plan=dict(fingerprint="parent",trained_layers=[0],
        native_suffix=list(range(1,36)),rank=1024,head_dim=128,confirmation_used=False,
        config=dict(epochs=12,microbatch=4,accumulation=4)),
        training=dict(status="completed",systems_only=False,strict_reload_exact=True,selected_epoch=12,history=history,
            selected=selected,updates=384,native_correct=50,minimum_correct=47,quality_pass=True,normalization=normalization),
        verification=dict(status="completed",passed=True,strict_state_load=True,selected_epoch=12,
            campaign_fingerprint="parent",runtime_policy_fingerprint="runtime",training_quality_pass=True,records=records(48)),
        runtime_policy=dict(fingerprint="runtime"),runtime_train=dict(status="completed",stage="train"),
        native_records=records(50),initial_records=records(46),selected_records=records(48),
        oracle={"native_boundary":dict(records=records(50)),"projected_update":dict(records=records(48))},
        storage=dict(tied_embedding_readout_counted_once=True,changed_layers=[0],
            selected_unpacked=dict(tensor_bytes=124),packed_block=dict(tensor_bytes=120),
            native_matched_fp32_bytes=1000,native_block_fp32_bytes=200,whole_matched_fp32=dict(selected_bytes=920)),
        v1_failure=dict(status="failed",error_type="MemoryError",peak_reserved_bytes=4351590400,allocation={"gpu":4096}))
    return value


def test_record_uses_final_selection_keeps_oracle_separate_and_never_confirms():
    record=JSpaceSingleLayerRecord.from_evidence(evidence())
    assert record.study_id==STUDY_ID and record.selected_epoch==12 and record.updates==384
    assert record.primary_metric==48/64 and record.quality_pass and not record.confirmed
    assert record.selected["paired_to_native"]==dict(gains=0,losses=2,changed_predictions=2,net_correct=-2)
    assert record.projection_oracle["correct"]==48 and "not a task-accuracy upper bound" in record.projection_oracle["scope"]
    assert record.resource_failure_lineage["error_type"]=="MemoryError"


@pytest.mark.parametrize("problem",["epochs","updates","selection","reload","gate","identity","runtime","storage","failure","confirmation"])
def test_incomplete_or_inflated_closeout_is_rejected(problem):
    value=evidence()
    if problem=="epochs":value["plan"]["config"]["epochs"]=4
    elif problem=="updates":value["training"]["updates"]=383
    elif problem=="selection":value["training"]["selected_epoch"]=5
    elif problem=="reload":value["verification"]["passed"]=False
    elif problem=="gate":value["training"]["minimum_correct"]=46
    elif problem=="identity":value["selected_records"][0]["identity"]="different"
    elif problem=="runtime":value["runtime_train"]["status"]="failed"
    elif problem=="storage":value["storage"]["whole_matched_fp32"]["selected_bytes"]-=8
    elif problem=="failure":value["v1_failure"]["error_type"]="QualityError"
    elif problem=="confirmation":value["confirmed"]=True
    with pytest.raises(ValueError):JSpaceSingleLayerRecord.from_evidence(value)


def test_completed_scientific_failure_is_reported_without_changing_gate():
    value=evidence()
    value["selected_records"]=deepcopy(value["initial_records"])
    value["verification"]["records"]=deepcopy(value["initial_records"])
    for row in value["training"]["history"]:row["metrics"]["correct"]=46
    value["training"]["quality_pass"]=value["verification"]["training_quality_pass"]=False
    record=JSpaceSingleLayerRecord.from_evidence(value)
    assert record.classification=="single_layer_development_gate_failed" and not record.quality_pass


def fake_native_files(path, native_state):
    path.mkdir(); mapping={}; header={}; cursor=0
    tensors={f"model.layers.{i}.{key}":value for i in range(36) for key,value in native_state.items()}
    tensors.update({"model.embed_tokens.weight":torch.zeros(23,16),"lm_head.weight":torch.zeros(23,16),
                    "model.norm.weight":torch.zeros(16)})
    for key,value in tensors.items():
        count=value.numel()*2;header[key]=dict(dtype="BF16",shape=list(value.shape),data_offsets=[cursor,cursor+count])
        cursor+=count;mapping[key]="model.safetensors"
    raw=json.dumps(header).encode();raw+=b" "*((-len(raw))%8)
    (path/"model.safetensors").write_bytes(struct.pack("<Q",len(raw))+raw+bytes(cursor))
    (path/"model.safetensors.index.json").write_text(json.dumps(dict(weight_map=mapping)))
    (path/"config.json").write_text(json.dumps(dict(tie_word_embeddings=True,num_hidden_layers=36)))


def test_actual_packing_storage_matches_shape_accounting_and_tied_e_once(tmp_path):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from experiments.structure_net.jspace_semantic_recovery import build_enriched_block
    config=Qwen2Config(hidden_size=16,intermediate_size=24,num_attention_heads=4,num_key_value_heads=2,
        num_hidden_layers=2,vocab_size=23,max_position_embeddings=32)
    config._attn_implementation="eager"
    native=Qwen2DecoderLayer(config,layer_idx=0).bfloat16().float().eval().requires_grad_(False)
    fit=dict(basis=torch.eye(16)[:,:8],means=torch.zeros(1,16))
    block=build_enriched_block(deepcopy(native),fit,parameter_dtype=torch.float32)
    checkpoint=tmp_path/"selected.pt";torch.save(dict(block=block.state_dict()),checkpoint)
    native_path=tmp_path/"native";fake_native_files(native_path,native.state_dict())
    measured=finalizer.storage_accounting(checkpoint,native_path)
    packed=deepcopy(block).pack(dtype=torch.float32,buffer_dtype=torch.float32)
    assert measured["packed_block"]["tensor_bytes"]==finalizer.tensor_inventory(packed.state_dict())["tensor_bytes"]
    assert measured["packed_block"]["parameter_count"]==sum(p.numel() for p in packed.parameters())
    assert measured["packed_block"]["fixed_geometry_and_buffer_bytes"]==sum(b.numel()*b.element_size() for b in packed.buffers())
    native_parameters=36*sum(p.numel() for p in native.parameters())+23*16+16
    assert measured["native_parameters"]==native_parameters
    assert measured["native_source_tensor_bytes"]==native_parameters*2
    assert measured["native_matched_fp32_bytes"]==native_parameters*4
    assert measured["whole_matched_fp32"]["selected_bytes"]==native_parameters*4-measured["native_block_fp32_bytes"]+measured["packed_block"]["tensor_bytes"]


def test_tensor_inventory_reports_aliases_without_double_counting_serialized_storage():
    tensor=torch.ones(2,3)
    value=finalizer.tensor_inventory(dict(a=tensor,b=tensor,metadata={}))
    assert value["tensor_bytes"]==48 and value["serialized_storage_bytes"]==24


def test_audit_never_counts_weights_before_completed_fresh_verification(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_single_block_verify as fresh
    def incomplete(_root):raise ValueError("Verification incomplete")
    monkeypatch.setattr(fresh,"verify",incomplete)
    monkeypatch.setattr(finalizer,"storage_accounting",lambda *_:pytest.fail("Counted before qualification"))
    with pytest.raises(ValueError,match="incomplete"):
        finalizer.audit(tmp_path,tmp_path,tmp_path,tmp_path)


def test_uncommitted_report_cannot_be_registered(tmp_path):
    report=tmp_path/"new.md";report.write_text("new report")
    # A path outside the repo also cannot satisfy the exact committed-file gate.
    with pytest.raises(ValueError):finalizer.require_committed([report])
