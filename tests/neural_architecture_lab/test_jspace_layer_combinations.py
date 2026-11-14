"""The search record reports tested composition, not learned repair or confirmation."""
from copy import deepcopy
import json
import math

import pytest

from neural_architecture_lab.jspace_layer_combinations_record import (
    JSpaceLayerCombinationsRecord,STUDY_ID,PRIMARY_METRIC_NAME,CONTROLS,policy,classify)
from experiments.neural_architecture_lab import finalize_jspace_layer_combinations as finalizer


def observed(correct):
    records = []
    families = ("arc","boolq","hellaswag","winogrande")
    by_family = {f:dict(count=16,correct=0) for f in families}
    for i in range(64):
        right = i < correct; logits = [1.,0.] if right else [0.,1.]
        ce = math.log(math.exp(logits[0])+math.exp(logits[1]))-logits[0]
        records.append(dict(identity=str(i),family=families[i//16],answer=0,prediction=0 if right else 1,
            logits=logits,teacher_prediction=int(i>=50),choice_ce=ce,choice_margin=logits[0]-logits[1]))
        by_family[families[i//16]]["correct"] += right
    agreement = sum(r["prediction"] == r["teacher_prediction"] for r in records)
    metrics = dict(count=64,correct=correct,accuracy=correct/64,teacher_agreement_count=agreement,
        teacher_agreement=agreement/64,choice_ce=sum(r["choice_ce"] for r in records)/64,
        choice_margin=sum(r["choice_margin"] for r in records)/64,by_family=by_family)
    return dict(records=records,metrics=metrics)


def evidence(extended=True):
    templates = {n:observed(n) for n in range(65)}
    results = {}
    controls = {policy.mask_id(mask):count for mask,count in CONTROLS.values()}
    for key,mask in policy.screen_masks().items():
        restored = [i for i,x in enumerate(mask) if not x]
        count = (30+restored[0]%3 if len(restored) == 1 else
                 31+sum(restored)%4 if len(restored) == 2 else 36+sum(restored)%2)
        results[key] = templates[controls.get(key,count if extended else 46)]
    candidates = policy.select_candidates(results)
    for key,mask in policy.refinement_masks(candidates["layers"]).items():
        if key in results: continue
        compact = sum(mask)
        count = 47 if compact == 33 else 48 if compact <= 6 else 45
        results[key] = templates[count if extended else 46]
    summary = policy.summarize(results,candidates)
    references = {arm:dict(records=deepcopy(results[policy.mask_id(mask)]["records"])) for arm,(mask,_) in CONTROLS.items()}
    return dict(status="completed",confirmed=False,
        plan=dict(schema="jspace.layer-combinations.v1",study_id=STUDY_ID,fingerprint="search",confirmed=False,
            microbatch=4,candidate_count=6,repair_group_size=3,minimum_correct=47,native_correct=50,original_all_compact_correct=27),
        search=dict(status="completed",fingerprint="search",confirmed=False,passed=True,systems_only=False,
            fixed_state_unchanged=True,results=results,summary=summary,
            controls={arm:dict(passed=True,predictions_exact=True,logits_pass=True) for arm in CONTROLS}),
        pilot=dict(passed=True,systems_only=True,fixed_state_unchanged=True),references=references,
        selection_proof=dict(selected_layers=summary["selected"]["layers"],search_fingerprint="search"),
        parameter_inventory=dict(endpoints=20,native_by_layer=[100]*36,compact_by_layer=[60]*36))


@pytest.fixture(scope="module")
def complete(): return evidence()


def test_primary_is_tested_gate_passing_layer_count_not_accuracy_or_global_optimum(complete):
    record = JSpaceLayerCombinationsRecord.from_evidence(complete)
    assert record.primary_metric == 33 and record.most_compressed["correct"] == 47
    assert record.most_compressed["accuracy"] == 47/64
    assert record.classification == "additional_tested_compression_passes_development_gate"
    assert record.model_parameters == 20+33*60+3*100
    assert record.unique_masks == len(complete["search"]["results"]) >= 681
    assert not record.confirmed and record.independent_confirmations == 0
    assert not record.global_search_complete and not record.speed_measured and not record.learned_repair_evaluated


def test_only_accepted_two_passing_is_not_called_new_recovery():
    record = JSpaceLayerCombinationsRecord.from_evidence(evidence(False))
    assert record.primary_metric == 2 and record.most_compressed["correct"] == 49
    assert record.classification == "no_tested_extension_beyond_accepted_two_blocks"
    assert not record.learned_repair_evaluated


def test_all_same_count_comparisons_and_conditional_pair_interactions_retained(complete):
    record = JSpaceLayerCombinationsRecord.from_evidence(complete)
    assert len(record.same_count) == 14 and len(record.pair_interactions) == 630
    for row in record.same_count:
        assert row["best"]["compressed_count"] == row["worst"]["compressed_count"] == row["compressed_count"]
        assert row["best_correct"] >= row["worst_correct"]
    target = record.selected_repair_target
    assert target["group_size"] == len(target["layers"]) == 3
    assert target["score"]["compressed_count"] == 33 and target["rescue_correct"] == target["restored_correct"]-27
    pair = record.pair_interactions[0]; i,j = pair["layers"]
    results = complete["search"]["results"]
    counts = lambda layers:results[policy.mask_id(policy.restored(layers))]["metrics"]["correct"]
    assert pair["excess_rescue"] == counts((i,j))-counts((i,))-counts((j,))+27


@pytest.mark.parametrize("problem",["missing_mask","extra_mask","summary","selection","confirmed","fixed","pilot",
    "gate","counts","teacher","ce","margin","family","control","proof","inventory"])
def test_incomplete_or_unjustified_claims_rejected(complete,problem):
    value = deepcopy(complete); result = value["search"]
    key = policy.mask_id(policy.restored((0,)))
    if problem == "missing_mask": del result["results"][key]
    elif problem == "extra_mask": result["results"][policy.mask_id(policy.compressed(range(10,20)))] = observed(30)
    elif problem == "summary": result["summary"]["same_count_rankings"][0]["best_correct"] += 1
    elif problem == "selection": result["summary"]["selected"]["layers"] = [20,21,22]
    elif problem == "confirmed": value["confirmed"] = True
    elif problem == "fixed": result["fixed_state_unchanged"] = False
    elif problem == "pilot": value["pilot"]["passed"] = False
    elif problem == "gate": value["plan"]["minimum_correct"] = 46
    elif problem == "counts": result["results"][key]["metrics"]["correct"] += 1
    elif problem == "teacher": result["results"][key]["records"][0]["teacher_prediction"] = 1
    elif problem == "ce": result["results"][key]["records"][0]["choice_ce"] += .1
    elif problem == "margin": result["results"][key]["records"][0]["choice_margin"] += .1
    elif problem == "family": result["results"][key]["metrics"]["by_family"]["arc"]["correct"] -= 1
    elif problem == "control": value["references"]["native"]["records"][0]["logits"][0] += .1
    elif problem == "proof": value["selection_proof"]["selected_layers"] = [20,21,22]
    elif problem == "inventory": value["parameter_inventory"]["compact_by_layer"].pop()
    with pytest.raises(ValueError): JSpaceLayerCombinationsRecord.from_evidence(value)


def test_registry_labels_primary_unit_and_does_not_claim_learned_repair(complete,tmp_path):
    (tmp_path/finalizer.ANALYSIS).write_text("evidence")
    record = JSpaceLayerCombinationsRecord.from_evidence(complete)
    metadata = finalizer.metadata(tmp_path,record)
    assert metadata["primary_metric"] == 33 and metadata["primary_metric_name"] == PRIMARY_METRIC_NAME
    assert metadata["primary_metric_unit"] == "layers" and metadata["accuracy"] == 47/64
    assert not metadata["learned_repair_evaluated"] and not metadata["training_performed"] and not metadata["confirmed"]
    assert json.loads(metadata["selected_repair_layers_json"]) == record.selected_repair_target["layers"]


@pytest.mark.parametrize("which",["result","weights"])
def test_postanalysis_provenance_changes_rejected(complete,tmp_path,which):
    value = deepcopy(complete); value["record"] = JSpaceLayerCombinationsRecord.from_evidence(value).to_dict()
    receipt = tmp_path/"search.json"; receipt.write_text("completed")
    weights = tmp_path/"fixed.pt"; weights.write_bytes(b"original")
    value.update(artifact_sha256={str(receipt):finalizer.sha(receipt)},analysis_source_sha256={},
        authenticated_large_artifacts={str(weights):dict(sha256=finalizer.sha(weights),identity=finalizer.identity(weights))})
    (tmp_path/finalizer.ANALYSIS).write_text(json.dumps(value))
    finalizer.expected(tmp_path)
    if which == "result": receipt.write_text("changed")
    else: weights.write_bytes(b"modified")
    with pytest.raises(ValueError): finalizer.expected(tmp_path)


def test_impossible_qualifier_counts_rejected():
    for count in (-1,0,1,37,True,3.5):
        with pytest.raises(ValueError): classify(count)


@pytest.mark.parametrize("changed",["report","image","csv"])
def test_report_and_pair_artifacts_remain_authenticated(tmp_path,changed):
    paths = {name:tmp_path/name for name in ("report","image","csv")}
    value = {}
    for name,path in paths.items():
        path.write_text(name); value[name] = str(path); value[name+"_sha256"] = finalizer.sha(path)
    (tmp_path/"layer_combinations_report.json").write_text(json.dumps(value))
    assert finalizer.plot_receipt(tmp_path,paths["report"]) == value
    paths[changed].write_text("modified")
    with pytest.raises(ValueError): finalizer.plot_receipt(tmp_path,paths["report"])
