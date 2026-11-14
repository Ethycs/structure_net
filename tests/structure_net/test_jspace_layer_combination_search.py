"""Search entry points and saved-decision proof of the repair selection."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_layer_combination_search as search


def cases():
    return [dict(identity=str(i),family=("arc","boolq","hellaswag","winogrande")[i//16],
        split="development",input_ids=[1,2,3],choices=["A","B"],answer=0) for i in range(64)]


def score(correct):
    records = []
    for i,c in enumerate(cases()):
        prediction = int(i >= correct); logits = [1.,0.] if prediction == 0 else [0.,1.]
        records.append(dict(identity=c["identity"],family=c["family"],answer=0,prediction=prediction,
            logits=logits,teacher_prediction=int(i >= 50),choice_ce=math.log1p(math.exp(-1))+prediction,
            choice_margin=1. if prediction == 0 else -1.))
    agreement = sum(r["prediction"] == r["teacher_prediction"] for r in records)
    metrics = dict(count=64,correct=correct,accuracy=correct/64,teacher_agreement_count=agreement,
        teacher_agreement=agreement/64,choice_ce=sum(r["choice_ce"] for r in records)/64)
    return dict(records=records,metrics=metrics)


def plan_fixture(root):
    references = {}
    for arm,correct in dict(native=50,accepted_two=49,all_compact=27,hybrid=24).items():
        path = root/f"reference_{arm}.json";search.recipe.write(path,dict(records=score(correct)["records"]))
        references[arm] = str(path)
    return dict(fingerprint="search",assembly_fingerprint="assembly",assembly_root=str(root/"assembly"),
        cases=cases(),references=references,microbatch=4,label_ids=[0,1])


def test_normalization_uses_decisions_for_native_agreement_and_preserves_records():
    raw = score(31); value = dict(raw["metrics"],records=raw["records"])
    value.pop("teacher_agreement_count")
    converted = search.normalized(value)
    assert converted == raw
    assert converted["records"] is value["records"]


@pytest.mark.parametrize("damage", ["inflated_count","teacher_prediction","case_order","choice_ce"])
def test_saved_decisions_reject_inflated_metrics_or_reference_drift(tmp_path,damage):
    plan = plan_fixture(tmp_path); key = search.policy.mask_id(search.policy.restored(()))
    values = {key:score(27)};search.validate_results(values,plan)
    value = values[key]
    if damage == "inflated_count":value["metrics"]["correct"] += 1
    elif damage == "teacher_prediction":value["records"][0]["teacher_prediction"] = 1
    elif damage == "case_order":value["records"].reverse()
    else:value["records"][0]["choice_ce"] += .1
    with pytest.raises(ValueError):search.validate_results(values,plan)


def test_actual_pilot_entrypoint_emits_progress_without_keyword_collision(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    plan = plan_fixture(tmp_path);events = [];lease = dict(id="leased",allocation={"tiny":1})
    monkeypatch.setattr(search,"verify",lambda _:plan)
    monkeypatch.setattr(search.assembly,"verify",lambda _:dict(fingerprint="assembly"))
    monkeypatch.setattr(jspace_full_lease,"wait_ready",lambda devices:events.append(("lease",devices)) or lease)
    monkeypatch.setattr(search.runtime,"allocator_settings",lambda _:events.append(("allocator",)) or dict(settings=[],devices={}))
    monkeypatch.setattr(search.runtime,"device_peaks",lambda _: {})
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda *_:pytest.fail("Unexpected CUDA in CPU pilot harness"))
    bank = SimpleNamespace(storage_inventory=lambda:dict(storage_bytes=1))
    def load(*_a,guard,**_k):guard();events.append(("load",));return bank
    monkeypatch.setattr(search.runtime,"load_bank",load)
    monkeypatch.setattr(search.runtime,"state_digest",lambda *_a,**_k:dict(sha256="fixed"))
    seen = []
    def scoring(_bank,selected,masks,_labels,*,teacher_records,progress,guard,**kwargs):
        guard();seen.append((masks,kwargs["share_prefixes"]))
        rows = [dict(r) for r in teacher_records]
        progress(kind="batch",completed=len(selected),total=len(selected))
        return dict(results={key:dict(records=deepcopy(rows)) for key in masks},execution=dict(decoder_calls=1))
    monkeypatch.setattr(search.runtime,"score_masks",scoring)
    result = search.run(tmp_path,"pilot")
    assert result["status"] == "completed"
    assert events[0][0] == "lease" and events[1][0] == "allocator"
    assert seen[0][1] is True and seen[1][1] is False and seen[0][0] == seen[1][0]
    requested = seen[0][0]
    assert all(search.policy.mask_id(search.policy.restored((i,))) in requested for i in range(36))
    pilot = search.verify_pilot(tmp_path,plan)
    assert pilot["passed"] and pilot["fixed_state_unchanged"]
    assert len(pilot["longest_case_identities"]) == 4
    pilot["fixed_state_unchanged"] = False
    (tmp_path/"pilot/result.json").write_text(json.dumps(pilot))
    receipt = search.recipe.read(tmp_path/"pilot/manifest.json")
    receipt["result_sha256"] = search.recipe.sha(tmp_path/"pilot/result.json")
    (tmp_path/"pilot/manifest.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="systems pilot"):
        search.verify_pilot(tmp_path,plan)


def test_search_cannot_reserve_without_a_completed_matching_pilot(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(search,"verify",lambda _:dict(fingerprint="search"))
    monkeypatch.setattr(jspace_full_lease,"wait_ready",lambda *_:pytest.fail("Lease before authenticated pilot"))
    with pytest.raises(FileNotFoundError):search.run(tmp_path,"search")
    assert not (tmp_path/"search").exists()


def completed_fixture(root,monkeypatch):
    plan = plan_fixture(root)
    monkeypatch.setattr(search,"verify",lambda _:plan)
    monkeypatch.setattr(search,"verify_pilot",lambda *_:dict(passed=True))
    values = {k:score(27) for k in search.policy.screen_masks()}
    for arm,correct in dict(native=50,accepted_two=49,all_compact=27,hybrid=24).items():
        values[search.policy.mask_id(search.control_masks()[arm])] = score(correct)
    candidates = search.policy.select_candidates(values)
    for key in search.policy.refinement_masks(candidates["layers"]):values.setdefault(key,score(27))
    triple = candidates["layers"][:3]
    values[search.policy.mask_id(search.policy.restored(triple))] = score(43)
    summary = search.policy.summarize(values,candidates)
    result = dict(status="completed",passed=True,systems_only=False,fingerprint="search",fixed_state_unchanged=True,
        results=values,summary=summary)
    directory = root/"search";directory.mkdir()
    (directory/"result.json").write_text(json.dumps(result))
    (directory/"selection.json").write_text(json.dumps(summary["selected"]))
    manifest = dict(status="completed",passed=True,fingerprint="search",
        result_sha256=search.recipe.sha(directory/"result.json"),selection_sha256=search.recipe.sha(directory/"selection.json"))
    (directory/"manifest.json").write_text(json.dumps(manifest))
    return plan,result,manifest


@pytest.mark.parametrize("damage", ["missing_refinement_mask","rewritten_selected_group"])
def test_repair_group_is_recomputed_from_complete_decisions_not_just_matching_hashes(tmp_path,monkeypatch,damage):
    _,result,manifest = completed_fixture(tmp_path,monkeypatch)
    valid = search.verified_selection(tmp_path)
    assert valid["selected_layers"] == [0,1,2]
    directory = tmp_path/"search"
    if damage == "missing_refinement_mask":
        key = next(k for k in result["results"] if k not in search.policy.screen_masks())
        del result["results"][key]
    else:
        selected = deepcopy(result["summary"]["selected"]);selected["layers"] = [3,4,5]
        result["summary"]["selected"] = selected
        (directory/"selection.json").write_text(json.dumps(selected))
        manifest["selection_sha256"] = search.recipe.sha(directory/"selection.json")
    (directory/"result.json").write_text(json.dumps(result))
    manifest["result_sha256"] = search.recipe.sha(directory/"result.json")
    (directory/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):search.verified_selection(tmp_path)


def test_preparation_requires_CPU_visibility_before_reading_parent(tmp_path,monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    monkeypatch.setattr(search.assembly,"verify",lambda _:pytest.fail("Parent access before CPU guard"))
    with pytest.raises(ValueError,match="hidden CUDA"):search.prepare(tmp_path/"root",tmp_path/"parent")
