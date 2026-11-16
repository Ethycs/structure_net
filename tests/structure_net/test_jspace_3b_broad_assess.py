"""Postselection admission, immutable reuse and matched assessment contracts."""
import copy
from pathlib import Path

import pytest
import numpy as np
import torch

from experiments.structure_net import jspace_3b_broad_assess as assess


def test_selection_is_authenticated_before_any_audit_data_access(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    calls=[]
    def deny(root):
        calls.append(("selection",root));raise ValueError("Both completed arms required")
    monkeypatch.setattr(assess.trial,"checked_selection",deny)
    monkeypatch.setattr(assess,"read",lambda path:pytest.fail("Data read before global selection"))
    with pytest.raises(ValueError,match="Both completed"):
        assess.prepare(tmp_path/"assessment",tmp_path/"recovery")
    assert len(calls)==1 and not (tmp_path/"assessment").exists()


def test_canaries_are_user_only_and_capacity_is_checked():
    config=assess.read(assess.GENERATION_CONFIG)
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert len(messages)==1 and messages[0]["role"]=="user"
            assert kwargs==dict(tokenize=True,add_generation_prompt=True)
            return [1,2,3]
    rows=assess.canary_inputs(config,Tokenizer())
    assert len(rows)==6 and all(r["input_ids"]==[1,2,3] for r in rows)
    class TooLong(Tokenizer):
        def apply_chat_template(self,*a,**kw):return [1]*353
    with pytest.raises(ValueError,match="capacity"):
        assess.canary_inputs(config,TooLong())


def test_identical_checkpoint_generation_reuse_is_exact_and_nonmutating():
    seed=dict(status="completed",phase="generate_seed",checkpoint_sha256="a"*64,
        fingerprint="fingerprint",records=[dict(id="example",response="text",output_ids=[3])])
    before=copy.deepcopy(seed)
    result=assess.reused_generation(seed,seed_sha="a"*64,selected_sha="a"*64,seed_manifest_sha="m"*64)
    assert result["records"]==seed["records"] and result["phase"]=="generate_selected"
    assert result["reused_manifest_sha256"]=="m"*64 and seed==before
    result["records"][0]["output_ids"].append(4)
    assert seed==before
    with pytest.raises(ValueError,match="identical"):
        assess.reused_generation(seed,seed_sha="a"*64,selected_sha="b"*64,seed_manifest_sha="m"*64)


def cases_and_rows():
    cases=[dict(identity=f"{i:064x}",family=family,answer=0,choices=["a","b"])
           for i,family in enumerate(("arc","boolq","hellaswag","winogrande"))]
    rows=[dict(identity=c["identity"],family=c["family"],answer=0,prediction=0,logits=[2.,1.]) for c in cases]
    return cases,rows


def test_audit_cannot_rescue_failed_development_or_choose_a_different_checkpoint():
    cases,rows=cases_and_rows()
    result=assess.audit_summary(rows,rows,cases,development_passed=False)
    assert result["descriptive_audit_passed"] is True
    assert result["development_passed"] is False and result["audit_can_rescue_development"] is False
    assert result["confirmed"] is False


def test_family_regression_is_reported_even_when_pooled_accuracy_is_unchanged():
    cases,rows=cases_and_rows();native=copy.deepcopy(rows);compact=copy.deepcopy(rows)
    native[0].update(prediction=1,logits=[1.,2.]);compact[1].update(prediction=1,logits=[1.,2.])
    value=assess.audit_summary(native,compact,cases,development_passed=True)
    assert value["overall_passed"] is True and value["family_passed"]["boolq"] is False
    assert value["descriptive_audit_passed"] is False


def test_missing_prior_phase_prevents_gpu_admission(tmp_path,monkeypatch):
    monkeypatch.setattr(assess,"verify",lambda root:{})
    def missing(*a,**kw):raise ValueError("Native audit incomplete")
    monkeypatch.setattr(assess,"checked",missing)
    monkeypatch.setattr(assess,"wait_ready",lambda *_:pytest.fail("GPU reached before prior phase"))
    with pytest.raises(ValueError,match="Native audit incomplete"):
        assess.run(tmp_path,"audit_compact")


def test_generation_validation_preserves_raw_empty_output_for_semantic_review():
    case=dict(id="canary",input_ids=[1,2])
    row=dict(id="canary",input_ids=[1,2],output_ids=[],response="",stop_reason="token_limit",use_cache=False,
        generation_config=dict(use_cache=False,max_new_tokens=64))
    assess.validate_generation([row],[case])
    row["use_cache"]=True
    with pytest.raises(ValueError,match="cache"):
        assess.validate_generation([row],[case])


def test_cuda_requires_reservation_before_initialization(tmp_path,monkeypatch):
    monkeypatch.setattr(assess,"verify",lambda root:{})
    def denied(*_):raise RuntimeError("No experiment lease")
    monkeypatch.setattr(assess,"wait_ready",denied)
    monkeypatch.setattr(assess.torch.cuda,"init",lambda:pytest.fail("CUDA initialized before reservation"))
    with pytest.raises(RuntimeError,match="No experiment lease"):
        assess.run(tmp_path,"audit_native")
    failure=assess.read(tmp_path/"audit_native/failure.json")
    assert failure["cuda_started"] is False


def test_cpu_prepare_freezes_actual_config_bytes_and_only_postselection_audit(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    recovery,cache_root,data_root=[tmp_path/name for name in ("recovery","cache","data")]
    for directory in (recovery,cache_root,data_root):directory.mkdir()
    checkpoint=tmp_path/"chosen.pt";checkpoint.write_bytes(b"test-checkpoint")
    report=tmp_path/"development.json";report.write_text("{}")
    checkpoint_sha=assess.base.sha(checkpoint)
    selection=dict(status="completed",fingerprint="selected",cache_root=str(cache_root),development_passed=False,
        candidate=dict(arm="parent_legacy",checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,report=str(report)))
    for name in ("plan.json","selection.json","manifest.json"):
        assess.base.write(recovery/name,selection if name=="selection.json" else {})
    for arm in assess.cache.ARMS:
        for phase in ("pilot","train","verify"):
            directory=recovery/arm/phase;directory.mkdir(parents=True)
            assess.base.write(directory/"result.json",{})
            assess.base.write(directory/"manifest.json",dict(artifacts={"result.json":assess.base.sha(directory/"result.json")}))
    cases=[dict(identity=f"{i:064x}",family=("arc","boolq","hellaswag","winogrande")[i//32],
        answer=0,choices=["a","b"],input_ids=[1,2,3],split="audit") for i in range(128)]
    assess.base.write(data_root/"audit_tasks.json",dict(splits=dict(audit=cases)))
    assess.base.write(data_root/"audit_identities.json",dict(identities=[c["identity"] for c in cases]))
    for name in ("plan.json","manifest.json"):assess.base.write(data_root/name,{})
    assess.base.write(cache_root/"plan.json",{})
    assess.base.write(cache_root/"items.json",dict(items=[]))
    (cache_root/"native").mkdir()
    assess.base.write(cache_root/"native/manifest.json",{})
    (cache_root/"native/boundary_22.npy").write_bytes(b"pinned-bank")
    controls=dict(seed=dict(checkpoint_sha256=checkpoint_sha),selected=dict(checkpoint_sha256=checkpoint_sha),
        boundary_path=str(cache_root/"native/boundary_22.npy"))
    monkeypatch.setattr(assess,"development_controls",lambda *args:controls)
    cp=dict(data_root=str(data_root),input_sha256={},model={"path":str(tmp_path/"native")},fits={},label_ids=[0,1,2,3],
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,development_control={},native_cache=str(tmp_path))
    events=[]
    monkeypatch.setattr(assess.trial,"checked_selection",lambda root:events.append("selection") or selection)
    monkeypatch.setattr(assess.cache,"verify",lambda root:cp)
    monkeypatch.setattr(assess.data_source,"verify",lambda root:events.append("data_verified"))
    monkeypatch.setattr(assess.cache,"SEED_SHA",checkpoint_sha)
    monkeypatch.setattr(assess.cache.source.inherited,"local_import_closure",lambda paths:[assess.GENERATION_CONFIG])
    class Tokenizer:
        def apply_chat_template(self,*a,**kw):return [1,2,3]
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,"from_pretrained",lambda *a,**kw:Tokenizer())
    out=tmp_path/"assessment"
    result=assess.prepare(out,recovery)
    assert events==["selection","data_verified"] and result["audit_count"]==128
    assert (out/"generation_config.json").read_bytes()==assess.GENERATION_CONFIG.read_bytes()
    assert assess.verify(out)["selection"]["development_passed"] is False


def test_existing_development_control_requires_exact_record_and_prefix(tmp_path,monkeypatch):
    case=dict(identity="dev",input_ids=[1,2,3],offset=1,stop=4)
    hidden=torch.arange(12).reshape(1,3,4).bfloat16()
    bank=np.zeros((5,4),dtype=np.int16);bank[1:4]=hidden[0].view(torch.int16).numpy()
    path=tmp_path/"boundary.npy";np.save(path,bank)
    row=dict(identity="dev",prediction=1,logits=[1.,2.])
    controls=dict(case=case,boundary_path=str(path),boundary_shape=[5,4],selected=dict(record=row))
    monkeypatch.setattr(assess.sanity,"native_case",lambda *args:(copy.deepcopy(row),hidden.clone()))
    actual=assess.partial_development_control(None,controls,"selected",[0,1],lambda:None)
    assert actual["records_exact"] and actual["boundary"]["bit_exact"]
    wrong=copy.deepcopy(row);wrong["logits"][0]+=.03125
    monkeypatch.setattr(assess.sanity,"native_case",lambda *args:(wrong,hidden.clone()))
    with pytest.raises(ValueError,match="choice record"):
        assess.partial_development_control(None,controls,"selected",[0,1],lambda:None)
    monkeypatch.setattr(assess.sanity,"native_case",lambda *args:(row,hidden+1))
    with pytest.raises(ValueError,match="boundary differs"):
        assess.partial_development_control(None,controls,"selected",[0,1],lambda:None)


def test_phase_receipts_require_whole_fixed_state_and_saved_development_logits():
    state=dict(sha256="exact",tensor_bytes=123)
    row=dict(identity="dev",prediction=1,logits=[1.,2.])
    plan=dict(compact_development_controls=dict(selected=dict(fixed_state=state,record=row)))
    result=dict(fixed_state_control=dict(passed=True,actual=state,expected=state),
        development_control=dict(records_exact=True,actual=row,expected=row,boundary=dict(bit_exact=True)))
    assess.check_compact_controls(result,plan,"selected",full_partial=True)
    bad=copy.deepcopy(result);bad["fixed_state_control"]["actual"]["sha256"]="other"
    with pytest.raises(ValueError,match="fixed-state"):
        assess.check_compact_controls(bad,plan,"selected",full_partial=True)
    bad=copy.deepcopy(result);bad["development_control"]["boundary"]["bit_exact"]=False
    with pytest.raises(ValueError,match="fixed-state"):
        assess.check_compact_controls(bad,plan,"selected",full_partial=True)


def test_control_references_use_selected_report_seed_pilot_and_original_dev_offsets(tmp_path):
    root=tmp_path/"recovery";cache_root=tmp_path/"cache";arm="native_content"
    items=[dict(identity=f"{i:064x}",family="arc",answer=i%2,choices=["a","b"],
        input_ids=[2,3,4],split="development",offset=30+3*i,stop=33+3*i) for i in range(256)]
    seed=[dict(identity=c["identity"],family="arc",answer=c["answer"],prediction=0,logits=[2.,1.]) for c in items]
    selected=copy.deepcopy(seed);selected[0].update(prediction=1,logits=[1.,3.])
    state=dict(sha256="fixed",tensor_bytes=99)
    for directory in (root/arm/"pilot",root/arm/"train",cache_root):directory.mkdir(parents=True)
    assess.base.write(root/arm/"pilot/result.json",dict(initial_records=seed,probe=dict(fixed_state=state),checkpoint_sha256="seed"))
    assess.base.write(root/arm/"train/result.json",dict(fixed_state=state))
    assess.base.write(cache_root/"items.json",dict(items=items))
    report=root/arm/"train/development.json";assess.base.write(report,dict(records=selected))
    value=assess.development_controls(root,cache_root,dict(arm=arm,report=str(report),checkpoint_sha256="selected"))
    assert value["case"]["offset"]==30 and value["case"]["identity"]==items[0]["identity"]
    assert value["seed"]["record"]==seed[0] and value["selected"]["record"]==selected[0]
    assert value["seed"]["fixed_state"]==value["selected"]["fixed_state"]==state
