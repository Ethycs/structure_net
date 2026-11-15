"""CPU qualification, protected selection and reservation-chain contracts."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_3b_resident_campaign as study
from test_jspace_3b_staged_campaign import (
    stage as old_stage, training, write, write_phase, reseal,
)
from test_jspace_resident_adam_replay import training_receipt, phase_receipt, metrics


def qualified_replay(root, monkeypatch):
    plan = dict(fingerprint="replay", cut=24, resident_budget_mib=6144,
                historical_warm=metrics(), historical_selected=metrics(50))
    trained = training_receipt(root, plan, "resident")
    trained["controls"] = {"parent_correct": 52}
    trained["quality"] = study.training.inherited.gates(45, native_correct=48, parent_correct=52, count=64)
    write(root/"resident/train/result.json", trained)
    reseal(root/"resident/train")
    report = study.read(root/"resident/train/epoch_-1_development.json")
    for arm in study.replay.ARMS:
        probe = dict(passed=True, fixed_state_unchanged=True,
            fp32_master_changed_by_block=[True]*12, norm_fp32_master_changed=True,
            fp32_master_before=[{"sha": str(i)} for i in range(12)],
            norm_fp32_master_before={"sha": "norm"})
        phase_receipt(root, arm, "pilot", dict(status="completed", passed=True,
            systems_only=True, discarded=True, warm_replay_exact=True, probe=probe,
            warm=plan["historical_warm"], records=report["records"]), plan)
    phase_receipt(root, "resident", "verify", dict(status="completed", passed=True,
        exact_packed_replay=True, fixed_state_unchanged=True,
        metrics=trained["selected"], records=report["records"],
        selected_checkpoint_sha256=trained["selected_checkpoint_sha256"]), plan)
    monkeypatch.setattr(study.replay, "verify_plan", lambda path: (plan, {}, {}))
    return plan


def test_real_receipts_qualify_native_budget_without_cpu_weight_equality(tmp_path, monkeypatch):
    qualified_replay(tmp_path, monkeypatch)
    value = study.qualification(tmp_path)
    assert value["selected"]["correct"] == 45
    assert value["quality"]["passed"] and not value["quality"]["parent"]["passed"]
    assert not value["cross_backend_weight_equality_required"]
    assert not value["cpu_full_replay_required"]
    assert value["qualification_only_not_campaign_seed"]
    assert not (tmp_path/"cpu/train").exists()


@pytest.mark.parametrize("change", ["incomplete", "own_checkpoint", "initial_masters", "warm_rows", "budget"])
def test_qualification_refuses_invalid_replay_evidence(tmp_path, monkeypatch, change):
    qualified_replay(tmp_path, monkeypatch)
    if change == "incomplete":
        (tmp_path/"resident/verify/manifest.json").unlink()
    else:
        phase = "train" if change == "budget" else "verify" if change == "own_checkpoint" else "pilot"
        path = tmp_path/"resident"/phase
        value = study.read(path/"result.json")
        if change == "own_checkpoint": value["selected_checkpoint_sha256"] = "wrong"
        elif change == "initial_masters": value["probe"]["fp32_master_before"][0]["sha"] = "different"
        elif change == "warm_rows": value["records"][0]["logits"][0] += .1
        else: value["history"][-1]["training"].pop()
        write(path/"result.json", value); reseal(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.qualification(tmp_path)


def resident_receipt(path):
    value = dict(optimizer=study.OPTIMIZER, transfer_accounting=study.replay.TRANSFER_ACCOUNTING,
        memory=dict(resident_backend="cuda:1", resident_budget_bytes=6144*1024**2,
                    resident_state_bytes=1024, resident_planned_peak_bytes=4096))
    write(path/"resident_optimizer.json", value); reseal(path)
    manifest = study.read(path/"manifest.json")
    manifest.update(allocation={"GPU-0370fdb0-test":11264,"GPU-7b1db3cb-test":7168},
        allocator={role:dict(uuid=uuid,device=device,allocator_limit_bytes=(mib-256)*1024**2)
            for role,uuid,device,mib in (("front","GPU-0370fdb0-test",0,11264),("tail","GPU-7b1db3cb-test",1,7168))},
        peak_reserved_bytes={"front":1024,"tail":1024})
    write(path/"manifest.json", manifest)


def current_stage():
    value = old_stage(22)
    value.update(schema=study.SCHEMA, optimizer=study.OPTIMIZER)
    value["fingerprint"] = study.base.digest({k:v for k,v in value.items() if k != "fingerprint"})
    return value


def test_resident_checked_preserves_real_protected_minus_one_selection(tmp_path):
    current = current_stage()
    trained = training(tmp_path, current, best=-1)
    for phase in ("pilot", "train"): resident_receipt(tmp_path/phase)
    result = study.checked(tmp_path, "train", current)
    assert result["selected_epoch"] == -1 and result["updates"] == 384
    verify = dict(status="completed", passed=True, comparison={"passed":True},
        fixed_state_unchanged=True, metrics=trained["selected"],
        selected_checkpoint_sha256=trained["selected_checkpoint_sha256"],
        quality=trained["quality"], compressed_blocks=14)
    write_phase(tmp_path, "verify", verify, current)
    assert study.checked(tmp_path, "verify", current)["passed"]
    (tmp_path/"train/resident_optimizer.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        study.checked(tmp_path, "verify", current)


@pytest.mark.parametrize("change", ["missing", "over_budget", "cpu"])
def test_resident_receipt_required_even_with_valid_scientific_selection(tmp_path, change):
    current = current_stage(); training(tmp_path, current)
    for phase in ("pilot", "train"): resident_receipt(tmp_path/phase)
    path = tmp_path/"train/resident_optimizer.json"
    if change == "missing": path.unlink()
    else:
        value = study.read(path)
        if change == "over_budget": value["memory"]["resident_planned_peak_bytes"] = 6144*1024**2+1
        else: value["memory"]["resident_backend"] = "cpu"
        write(path, value)
    reseal(path.parent)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.checked(tmp_path, "train", current)


@pytest.mark.parametrize("change", ["wrong_card", "exceeded_peak"])
def test_resident_receipt_rejects_wrong_physical_card_or_memory_overrun(tmp_path, change):
    current = current_stage(); training(tmp_path, current)
    for phase in ("pilot", "train"): resident_receipt(tmp_path/phase)
    path=tmp_path/"train/manifest.json"; value=study.read(path)
    if change=="wrong_card": value["allocator"]["tail"]["uuid"]="GPU-8cf440df-test"
    else: value["peak_reserved_bytes"]["tail"]=value["allocator"]["tail"]["allocator_limit_bytes"]+1
    write(path,value)
    with pytest.raises(ValueError,match="reservation mapping or measured peak"):
        study.checked(tmp_path,"train",current)


def test_stage22_uses_original_seed_and_next_stage_keeps_minus_one_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    seed = tmp_path/"seed.pt"; seed.write_bytes(b"accepted thirteen block seed")
    reference = tmp_path/"epoch_00_development.json"; write(reference, {})
    fit = tmp_path/"fit"; fit.mkdir()
    for name in ("fit.pt", "fit.json", "relation.pt", "relations.json", "interface.json"):
        (fit/name).write_bytes(b"unchanged frozen fit")
    plan = dict(fingerprint="campaign", fits={str(i):str(fit) for i in range(23,36)},
        seed_checkpoint=str(seed), seed_records=str(reference), seed_correct=46, seed_fingerprint="seed")
    monkeypatch.setattr(study, "verify_root", lambda root: plan)
    first = study.prepare_stage(tmp_path, 22)
    assert first["parent_cut"] == 23 and first["parent_correct"] == 46
    assert first["parent_checkpoint_sha256"] == study.base.sha(seed)
    assert first["layers"] == list(range(22,36)) and first["calibration_reuse"] is None
    assert first["calibration_seed"] == 720861
    directory = study.stage_dir(tmp_path,22)
    for phase in ("train", "verify"):
        (directory/phase).mkdir(); write(directory/phase/"manifest.json", {})
    (directory/"train/selected.pt").write_bytes(b"selected protected raw")
    write(directory/"train/epoch_-1_development.json", {})
    # The new fit would have completed before this parent was accepted.
    new_fit = Path(first["fits"]["22"]); new_fit.mkdir(parents=True)
    for name in ("fit.pt", "fit.json", "relation.pt", "relations.json", "interface.json"):
        (new_fit/name).write_bytes(b"new fit")
    monkeypatch.setattr(study, "checked", lambda *a: dict(quality_pass=True, passed=True,
        selected_epoch=-1, selected={"correct":45}))
    second = study.prepare_stage(tmp_path,21)
    assert second["parent_cut"] == 22 and second["parent_correct"] == 45
    assert second["parent_records"].endswith("epoch_-1_development.json")
    assert second["parent_fingerprint"] == first["fingerprint"]
    with pytest.raises(ValueError): study.stage_dir(tmp_path,23)
    with pytest.raises(ValueError): study.stage_dir(tmp_path,True)


def test_campaign_six_phases_use_second_card_only_for_joint_optimizer(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ""); calls=[]
    monkeypatch.setattr(study,"verify_root",lambda root:{"fingerprint":"campaign"})
    def prepare(root,cut):
        directory=study.stage_dir(root,cut);directory.mkdir(parents=True)
        write(directory/"plan.json",dict(cut=cut,fingerprint=f"stage{cut}"))
    monkeypatch.setattr(study,"prepare_stage",prepare)
    monkeypatch.setattr(study,"verify_stage",lambda root,cut:({},study.read(study.stage_dir(root,cut)/"plan.json")))
    def checked(directory,phase,current):
        assert (directory/phase/"manifest.json").exists()
        return dict(quality_pass=True,passed=True,selected={"correct":45},quality={},storage={})
    monkeypatch.setattr(study,"checked",checked)
    def run(argv,**kwargs):
        calls.append(argv);phase=argv[argv.index("-m")+2];cut=int(argv[argv.index("--cut")+1])
        output=study.stage_dir(tmp_path,cut)/phase;output.mkdir();write(output/"manifest.json",{})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess,"run",run)
    outcome=study.campaign(tmp_path,stop_cut=22)
    assert outcome["compressed_blocks"]==14 and outcome["confirmed"] is False
    assert [c[c.index("-m")+2] for c in calls]==list(study.PHASES)
    for argv in calls:
        phase=argv[argv.index("-m")+2]
        # The deployed CLI accepts one string, not repeated allocation flags.
        assert argv.count("--allocation")==1
        allocation=argv[argv.index("--allocation")+1]
        assert allocation==("GPU-0370fdb0=11264,GPU-7b1db3cb=7168" if phase in ("pilot","train") else "GPU-0370fdb0=11264")
        assert argv[argv.index("--minutes")+1]=="120"
        assert argv[argv.index("-m")+1]==study.MODULE and "--locked" in argv
    prepare(tmp_path,21);(study.stage_dir(tmp_path,21)/"capture").mkdir();calls.clear()
    with pytest.raises(RuntimeError,match="Incomplete phase"):
        study.campaign(tmp_path,stop_cut=21)
    assert not calls


def test_reservation_denial_precedes_any_cuda_or_model_allocation(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    directory=study.stage_dir(tmp_path,22);directory.mkdir(parents=True)
    monkeypatch.setattr(study,"verify_stage",lambda *a:({},current_stage()))
    monkeypatch.setattr(study,"checked",lambda *a:{})
    def denied(*a):raise RuntimeError("lease denied")
    monkeypatch.setattr(jspace_full_lease,"wait_ready",denied)
    def forbidden(*a,**k):pytest.fail("CUDA or model allocation before lease")
    monkeypatch.setattr(study.torch.cuda,"set_device",forbidden)
    monkeypatch.setattr(study.replay,"allocator_settings",forbidden)
    monkeypatch.setattr(study,"make_stack",forbidden)
    with pytest.raises(RuntimeError,match="lease denied"):
        study.run_phase(tmp_path,22,"pilot")
    assert study.read(directory/"pilot/failure.json")["cuda_started"] is False
    assert not (directory/"pilot/initializer.pt").exists()
