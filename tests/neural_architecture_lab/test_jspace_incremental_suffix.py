"""Incremental suffix evidence must retain both matched controls and true task selection."""
from copy import deepcopy
import json

import pytest

from neural_architecture_lab.jspace_incremental_suffix_record import (
    ATOL, CONFIG, LAYERS, RTOL, JSpaceIncrementalSuffixRecord, parity, quality_gates,
)


def rows(correct):
    result = []
    for i in range(64):
        family = ("arc", "boolq", "hellaswag", "winogrande")[i//16]
        logits = [0.] * (2 if family in ("boolq", "winogrande") else 4)
        prediction = int(i >= correct); logits[prediction] = 2.
        result.append(dict(identity=str(i), family=family, answer=0, prediction=prediction, logits=logits))
    return result


def report(correct, epoch=0):
    kl, ce = .5-.01*epoch, 3.-.05*epoch
    return dict(epoch=epoch, records=rows(correct), metrics=dict(correct=correct, count=64,
        accuracy=correct/64, kl=kl, ce=ce, loss=2.8*kl+.3*ce))


def evidence(counts=(44,45,46,46,47,47,48,48,48,49,48,49,48)):
    epochs = [report(count, i) for i,count in enumerate(counts)]
    history = [dict(epoch=r["epoch"],metrics=r["metrics"],checkpoint_sha256=f"state-{r['epoch']}") for r in epochs]
    best = max(history,key=lambda r:(r["metrics"]["correct"],-r["metrics"]["ce"]))
    winner = best["epoch"]; gates = quality_gates(counts[winner],48,51)
    controls = {}
    for name, count in (("native",48),("parent",51),("seam",51)):
        controls[name] = dict(**report(count),passed=True,reference_records=rows(count))
    return dict(status="completed",confirmed=False,
        plan=dict(schema="jspace.incremental-suffix.v1",cut=29,parent_cut=30,layers=LAYERS.copy(),
                  config=deepcopy(CONFIG),fingerprint="frozen"),
        controls=controls,
        pilot=dict(systems_only=True,updates=1,master_changed_layers=LAYERS.copy(),norm_master_changed=True,
            stored_changed_layers=LAYERS.copy(),stored_norm_changed=False,
            initial_records=deepcopy(epochs[0]["records"]),
            gradient_layers=LAYERS.copy(),norm_gradient=True,fixed_state_unchanged=True,parent_inheritance_exact=True),
        training=dict(status="completed",systems_only=False,updates=384,strict_reload_exact=True,
            fixed_state_unchanged=True,history=history,selected_epoch=winner,selected=best["metrics"],
            initial=epochs[0]["metrics"],selected_checkpoint_sha256=best["checkpoint_sha256"],gates=gates),
        epochs=epochs,
        verification=dict(status="completed",fingerprint="frozen",selected_epoch=winner,
            selected_checkpoint_sha256=best["checkpoint_sha256"],passed=True,strict_state_load=True,
            fixed_state_unchanged=True,packed=True,records=deepcopy(epochs[winner]["records"]),
            metrics=deepcopy(epochs[winner]["metrics"]),quality_pass=gates["quality_pass"]),
        storage=dict(tied_embedding_readout_counted_once=True,native_prefix_layers=list(range(29)),
            compact_layers=LAYERS.copy(),packed_suffix_tensor_bytes=800,selected_suffix_tensor_bytes=900,
            native_prefix_and_embedding_bytes=1000,whole_model_tensor_bytes=1800))


def test_actual_seven_block_accuracy_and_both_controls_define_success():
    r = JSpaceIncrementalSuffixRecord.from_evidence(evidence())
    assert r.primary_metric == 49/64 and r.selected_epoch == 11
    assert r.native["correct"] == 48 and r.parent["correct"] == 51
    assert r.gates == quality_gates(49,48,51)
    assert r.classification == "valid_expansion_within_budget"
    assert r.selected["paired_to_initial"]["net_correct"] == 5
    assert r.selected["paired_to_parent"]["net_correct"] == -2
    assert r.packing_validated and not r.confirmed and not r.speed_measured
    assert r.independent_confirmations == 0 and r.compressed_layers == tuple(range(29,36))


@pytest.mark.parametrize("correct,passed", [(44,False),(45,False),(47,False),(48,True),(51,True)])
def test_five_percentage_point_budget_is_not_rounded_to_five_answers(correct,passed):
    gate = quality_gates(correct,48,51)
    assert gate["native_minimum_correct"] == 45 and gate["parent_minimum_correct"] == 48
    assert gate["quality_pass"] is passed
    assert gate["native_pass"] is (correct >= 45)


def test_passing_initializer_establishes_expansion_without_training_gain():
    r = JSpaceIncrementalSuffixRecord.from_evidence(evidence((49,)+(47,)*12))
    assert r.selected_epoch == 0 and r.initializer_selected
    assert r.classification == "valid_expansion_within_budget"
    assert r.selected["paired_to_initial"]["net_correct"] == 0


def test_accuracy_first_then_full_vocabulary_ce_and_earlier_exact_ties():
    value=evidence((48,)+(49,)*12)
    # Make epoch1 and12 exact objective/accuracy ties; earliest must remain selected.
    best=deepcopy(value["epochs"][1]["metrics"])
    for i in range(2,13):
        value["epochs"][i]["metrics"] = deepcopy(best)
        value["training"]["history"][i]["metrics"] = deepcopy(best)
    t=value["training"]; t.update(selected_epoch=1,selected=best,selected_checkpoint_sha256="state-1")
    value["verification"].update(selected_epoch=1,selected_checkpoint_sha256="state-1",metrics=deepcopy(best))
    assert JSpaceIncrementalSuffixRecord.from_evidence(value).selected_epoch == 1


@pytest.mark.parametrize("counts,classification",[
    ((44,)+(46,)*12,"partial_recovery_below_budget"),
    ((44,)+(43,)*12,"no_recovery_below_budget"),
])
def test_completed_quality_miss_is_an_observed_negative_not_invalid(counts,classification):
    r=JSpaceIncrementalSuffixRecord.from_evidence(evidence(counts))
    assert r.classification == classification and not r.gates["quality_pass"]


@pytest.mark.parametrize("mutation",[
    lambda v:v["plan"].update(cut=28),
    lambda v:v["plan"]["config"].update(kl_weight=.2),
    lambda v:v["plan"]["config"].update(dtype="float32"),
    lambda v:v["training"].update(updates=383),
    lambda v:v["training"]["history"].pop(5),
    lambda v:v["pilot"].update(parent_inheritance_exact=False),
    lambda v:v["pilot"].update(norm_master_changed=False),
    lambda v:v["pilot"]["gradient_layers"].remove(29),
    lambda v:v["training"].update(fixed_state_unchanged=False),
    lambda v:v["training"]["gates"].update(minimum_correct=47),
    lambda v:v["verification"].update(packed=False),
    lambda v:v["verification"].update(fingerprint="different"),
    lambda v:v["verification"].update(selected_checkpoint_sha256="different"),
    lambda v:v["verification"].update(strict_state_load=False),
    lambda v:v["verification"]["records"][0].update(identity="other"),
    lambda v:v["storage"].update(whole_model_tensor_bytes=800),
    lambda v:v["storage"].update(tied_embedding_readout_counted_once=False),
    lambda v:v.update(confirmed=True),
])
def test_incomplete_or_unmatched_evidence_is_rejected(mutation):
    value=evidence();mutation(value)
    with pytest.raises(ValueError):JSpaceIncrementalSuffixRecord.from_evidence(value)


def test_same_predictions_do_not_hide_bad_logits_or_missing_choices():
    good=rows(48); bad=deepcopy(good); bad[0]["logits"][0]+=10
    assert parity(bad,good)["predictions_exact"] and not parity(bad,good)["passed"]
    bad=deepcopy(good);bad[0]["logits"].pop()
    with pytest.raises(ValueError):parity(bad,good)
    with pytest.raises(ValueError):parity(good,good,atol=ATOL*2,rtol=RTOL)


def test_fresh_full_vocabulary_metrics_and_packed_logits_are_both_authenticated():
    v=evidence();v["verification"]["records"][0]["logits"][0]+=10
    with pytest.raises(ValueError):JSpaceIncrementalSuffixRecord.from_evidence(v)
    v=evidence();m=v["verification"]["metrics"];m["kl"]+=1;m["loss"]+=2.8
    with pytest.raises(ValueError):JSpaceIncrementalSuffixRecord.from_evidence(v)


def test_bf16_rounding_does_not_invalidate_real_master_updates():
    value=evidence();value["pilot"]["stored_changed_layers"]=[]
    assert not value["pilot"]["stored_norm_changed"]
    assert JSpaceIncrementalSuffixRecord.from_evidence(value).updates == 384
    value["pilot"]["master_changed_layers"].remove(35)
    with pytest.raises(ValueError):JSpaceIncrementalSuffixRecord.from_evidence(value)


def test_storage_includes_native_prefix_tied_readout_geometry_and_norm(tmp_path,monkeypatch):
    import torch
    from experiments.neural_architecture_lab import finalize_jspace_incremental_suffix as final
    state=dict(stack={"blocks.0.qkv_weight":torch.ones(2,3,dtype=torch.bfloat16),
                      "blocks.0.basis":torch.ones(3,2,dtype=torch.float32)},
               norm={"weight":torch.ones(3,dtype=torch.bfloat16)})
    path=tmp_path/"selected.pt";torch.save(state,path)
    native={"model.embed_tokens.weight":dict(dtype="BF16",elements=30,bytes=60),
            **{f"model.layers.{i}.weight":dict(dtype="BF16",elements=6,bytes=12) for i in range(36)}}
    monkeypatch.setattr(final,"native_inventory",lambda _:dict(tensors=native,source_tensor_bytes=498))
    inventory=final.selected_inventory(path,tmp_path,30)
    assert inventory["selected_suffix_tensor_bytes"] == 12+24+6
    assert inventory["native_prefix_and_embedding_bytes"] == 29*12+60
    assert inventory["whole_model_tensor_bytes"] == 29*12+60+30
    assert inventory["model_parameters"] == 29*6+30+6+3
    assert inventory["tied_embedding_readout_counted_once"]


def fixture_analysis(tmp_path,monkeypatch):
    from experiments.neural_architecture_lab import finalize_jspace_incremental_suffix as final
    root=tmp_path/"run";root.mkdir()
    source=tmp_path/"producer.py";source.write_text("frozen")
    archived=root/final.ARCHIVE/"producer.py";archived.parent.mkdir(parents=True);archived.write_text("frozen")
    artifact=tmp_path/"development.json";artifact.write_text("paired choices")
    large=tmp_path/"bank.npy";large.write_text("authenticated bank")
    value=evidence();value.update(record=JSpaceIncrementalSuffixRecord.from_evidence(value).to_dict(),
        artifact_sha256={str(artifact):final.sha(artifact)},analysis_source_sha256={str(source):final.sha(source)},
        authenticated_large_artifacts={str(large):dict(sha256=final.sha(large),identity=final.identity(large))})
    (root/final.ANALYSIS).write_text(json.dumps(value))
    monkeypatch.setattr(final,"ROOT",tmp_path)
    return final,root,source,archived,artifact,large


@pytest.mark.parametrize("changed",["source","archive","artifact","bank","record"])
def test_completed_closeout_rejects_changed_provenance(tmp_path,monkeypatch,changed):
    final,root,source,archive,artifact,bank=fixture_analysis(tmp_path,monkeypatch)
    assert final.expected(root)[1].primary_metric == 49/64
    if changed == "record":
        value=json.loads((root/final.ANALYSIS).read_text());value["record"]["confirmed"]=True
        (root/final.ANALYSIS).write_text(json.dumps(value))
    else:{"source":source,"archive":archive,"artifact":artifact,"bank":bank}[changed].write_text("changed")
    with pytest.raises(ValueError):final.expected(root)


def test_registry_metadata_does_not_confuse_parent_or_confirmation_with_primary(tmp_path,monkeypatch):
    final,root,*_=fixture_analysis(tmp_path,monkeypatch)
    _,record=final.expected(root);metadata=final.metadata(root,record)
    assert metadata["primary_metric"] == 49/64 and metadata["parent_correct"] == 51
    assert metadata["native_correct"] == 48 and metadata["minimum_correct"] == 48
    assert metadata["confirmed"] is False and metadata["speed_measured"] is False
    assert metadata["packed_prediction_parity"] is True and metadata["compressed_layer_count"] == 7


def producer_fixture(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_incremental_suffix as producer
    from experiments.neural_architecture_lab import finalize_jspace_incremental_suffix as final
    root=tmp_path/"run";root.mkdir();v=evidence();plan=v["plan"]
    parent=tmp_path/"parent.json";parent.write_text(json.dumps(rows(51)))
    plan.update(confirmed=False,logit_atol=ATOL,logit_rtol=RTOL,metric_atol=1e-7,metric_rtol=1e-6,
        input_sha256={str(parent):final.sha(parent)},source_sha256={},label_ids=[32,33,34,35],
        parent_records=str(parent),model={"path":str(tmp_path)})
    (root/"plan.json").write_text(json.dumps(plan));(root/"items.json").write_text("{}")
    controls={"passed":True}
    for name in ("native","parent","seam"):
        c=v["controls"][name];controls[name]=dict(**c["metrics"],records=c["records"],passed=True,
            predictions_exact=True,logits_pass=True,atol=ATOL,rtol=RTOL,max_abs_logit_delta=0.)
    controls["teacher_cache_diagnostic"]=dict(predictions_exact=True,correct=51,logit_gate_applied=False,
        max_abs_logit_delta=.25,logits_pass=False)
    pilot=dict(status="completed",confirmed=False,systems_only=True,updates=1,controls=controls,
        inheritance=dict(passed=True,unchanged_layers=list(range(30,36)),interior_bridges_preserved=5,
                         new_layer=29,new_bridge=[29,30],norm_exact=True),
        update=dict(gradients_present_by_block=[True]*7,optimizer={"unclipped_gradient_norm":.5}),
        changed_layers=LAYERS,norm_changed=True,bf16_changed_layers=LAYERS,bf16_norm_changed=False,
        parameter_change_gate="FP32 optimizer masters; BF16 stored changes are separate rounding diagnostics",
        initial=v["epochs"][0]["metrics"],initial_records=v["epochs"][0]["records"],
        fixed_state_unchanged=True,fixed_state={"sha256":"fixed"},checkpoint_strict_load=True)
    train=deepcopy(v["training"]);train.update(confirmed=False,completed_epochs=12,config=CONFIG,
        fixed_state={"sha256":"fixed"},quality=producer.gates(train["selected"]["correct"]),quality_pass=True)
    fresh=deepcopy(v["verification"]);fresh.update(confirmed=False,packing_validated=True,metrics_pass=True,
        quality=deepcopy(train["quality"]),comparison=dict(passed=True,predictions_exact=True,logits_pass=True,atol=ATOL,rtol=RTOL),
        storage=dict(packed_suffix_and_norm_bytes=800,rotary_bytes=16,frozen_native_prefix_bytes=600,
                     tied_embedding_readout_bytes=400,full_model_tensor_bytes=1816,native_full_tensor_bytes=3016))
    for stage in ("pilot","train","verification"):(root/stage).mkdir()
    checkpoint=root/"train/selected.pt";checkpoint.write_bytes(b"authenticated selected tensor fixture")
    digest=final.sha(checkpoint);train["selected_checkpoint_sha256"]=fresh["selected_checkpoint_sha256"]=digest
    train["history"][train["selected_epoch"]]["checkpoint_sha256"]=digest
    for history,epoch in zip(train["history"],v["epochs"]):
        path=root/"train"/f"epoch_{epoch['epoch']:02d}_development.json"
        path.write_text(json.dumps(dict(metrics=epoch["metrics"],records=epoch["records"])))
        history["development_sha256"]=final.sha(path)
    (root/"train/initializer_replay.json").write_text(json.dumps(dict(passed=True)))
    for stage,result in (("pilot",pilot),("train",train),("verification",fresh)):
        path=root/stage;(path/"result.json").write_text(json.dumps(result))
        manifest=dict(status="completed",fingerprint=plan["fingerprint"],passed=result.get("passed"),
            artifacts={p.name:final.sha(p) for p in path.iterdir()},allocation={"GPU-0370fdb0-test":8192},
            allocator=dict(physical_uuid="GPU-0370fdb0-test",lease_bytes=8192*1024**2,allocator_limit_bytes=7936*1024**2),
            peak_reserved_bytes=1024,elapsed_seconds=1.)
        (path/"manifest.json").write_text(json.dumps(manifest))
    events=[dict(kind="update",update=i,epoch=(i-1)//32+1,cases=16,gradients_present_by_block=[True]*7,
        optimizer=dict(updated_scalars=300,unclipped_gradient_norm=.2),loss=1.,kl=.25,ce=1.) for i in range(1,385)]
    (root/"progress.jsonl").write_text("\n".join(map(json.dumps,events)))
    class Data:
        def close(self):pass
    monkeypatch.setattr(producer,"verify",lambda _:plan)
    monkeypatch.setattr(producer,"dataset",lambda _:Data())
    def cached_reference(data,label_ids):
        assert isinstance(data,Data) and label_ids == plan["label_ids"]
        return rows(48)
    monkeypatch.setattr(producer,"rows_from_cache",cached_reference)
    inventory=dict(v["storage"],selected_suffix_parameters=300,model_parameters=1000,
        rotary_bytes=16,whole_model_tensor_bytes=1816,native_model_tensor_bytes=3016)
    monkeypatch.setattr(final,"selected_inventory",lambda *a,**kw:inventory)
    return final,root


def test_audit_maps_actual_producer_schema_and_retains_teacher_cache_diagnostic(tmp_path,monkeypatch):
    final,root=producer_fixture(tmp_path,monkeypatch)
    value,record=final.audit(root)
    assert record.primary_metric == 49/64 and record.packing_validated
    assert not value["pilot"]["stored_norm_changed"] and value["pilot"]["norm_master_changed"]
    assert value["teacher_cache_diagnostic"]["max_abs_logit_delta"] == .25
    assert value["update_audit"]["count"] == 384


@pytest.mark.parametrize("damage",["record_hash","reservation","missing_update"])
def test_audit_rejects_damaged_real_producer_receipts(tmp_path,monkeypatch,damage):
    final,root=producer_fixture(tmp_path,monkeypatch)
    if damage == "record_hash":(root/"train/epoch_03_development.json").write_text("{}")
    elif damage == "reservation":
        p=root/"verification/manifest.json";v=json.loads(p.read_text());v["peak_reserved_bytes"]=8192*1024**2
        p.write_text(json.dumps(v))
    else:
        p=root/"progress.jsonl";p.write_text("\n".join(p.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError):final.audit(root)
