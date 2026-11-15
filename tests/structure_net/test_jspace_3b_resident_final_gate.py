"""Explicit full36 admission before unchanged numerical final evaluation."""
from pathlib import Path

import pytest

from experiments.structure_net import jspace_3b_resident_final_gate as study


def complete_campaign(tmp_path, monkeypatch):
    root=tmp_path/"campaign";root.mkdir()
    plan=dict(schema=study.campaign.SCHEMA,fingerprint="campaign",seed_cut=23,seed_correct=46,
        seed_checkpoint_sha256="seed",input_sha256={},qualification={"qualified":True})
    study.base.write(root/"plan.json",plan)
    study.base.write(root/"outcome.json",dict(status="all_blocks_recovered",last_accepted_cut=0,compressed_blocks=36))
    previous="seed";results={}
    for cut in range(22,-1,-1):
        directory=study.campaign.stage_dir(root,cut);directory.mkdir(parents=True)
        stage=dict(schema=study.campaign.SCHEMA,root_fingerprint="campaign",cut=cut,
            parent_checkpoint_sha256=previous,parent_correct=46,fits={})
        stage["fingerprint"]=study.base.digest(stage)
        study.base.write(directory/"plan.json",stage)
        for phase in study.campaign.PHASES:
            path=directory/phase;path.mkdir()
            result=dict(status="completed",passed=True)
            if phase=="train":
                (path/"selected.pt").write_bytes(str(cut).encode())
                previous=study.base.sha(path/"selected.pt")
                result.update(quality_pass=True,selected={"correct":46},quality={"passed":True},
                    selected_epoch=-1,selected_checkpoint_sha256=previous)
                study.base.write(path/"epoch_-1_development.json",{"records":[]})
            study.base.write(path/"result.json",result)
            study.base.write(path/"manifest.json",dict(status="completed",fingerprint=stage["fingerprint"],
                artifacts={"result.json":study.base.sha(path/"result.json")}))
            results[(cut,phase)]=result
    monkeypatch.setattr(study.campaign,"verify_root",lambda path:plan)
    monkeypatch.setattr(study.campaign,"checked",lambda path,phase,stage:results[(stage["cut"],phase)])
    return root,plan,results


def test_full_chain_keeps_protected_minus_one_reference_and_six_phase_receipts(tmp_path,monkeypatch):
    root,_,_=complete_campaign(tmp_path,monkeypatch)
    accepted=study.admit_campaign(root)
    assert [row["cut"] for row in accepted["chain"]]==list(range(22,-1,-1))
    assert accepted["selected_reference"].endswith("block_00/train/epoch_-1_development.json")
    assert accepted["seed_gate_receipt"]=={"qualified":True}
    for cut in range(23):
        for phase in study.campaign.PHASES:
            assert str((root/"stages"/f"block_{cut:02d}"/phase/"manifest.json").resolve()) in accepted["input_sha256"]


@pytest.mark.parametrize("change",["partial","parent","receipt","quality"])
def test_incomplete_or_unaccepted_chain_cannot_be_admitted(tmp_path,monkeypatch,change):
    root,_,results=complete_campaign(tmp_path,monkeypatch)
    if change=="partial":
        (root/"outcome.json").write_text('{"status":"quality_stop","last_accepted_cut":1,"compressed_blocks":35}')
    elif change=="parent":
        path=root/"stages/block_10/plan.json";stage=study.read(path)
        stage["parent_checkpoint_sha256"]="wrong"
        stage["fingerprint"]=study.base.digest({k:v for k,v in stage.items() if k!="fingerprint"})
        path.write_text(__import__("json").dumps(stage))
    elif change=="receipt":
        path=root/"stages/block_10/capture/result.json";path.write_text('{"status":"different"}')
    else:
        results[(10,"train")]["quality_pass"]=False
    with pytest.raises(ValueError):study.admit_campaign(root)


def test_confirmation_is_not_opened_before_completed_admission(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    def deny(_):raise ValueError("Full36 absent")
    monkeypatch.setattr(study,"admit_campaign",deny)
    monkeypatch.setattr(study.numerical.confirmation,"verify",lambda *a:pytest.fail("Confirmation opened prematurely"))
    with pytest.raises(ValueError,match="Full36 absent"):
        study.prepare(tmp_path/"evaluation",tmp_path/"partial",tmp_path/"confirmation")
    assert not (tmp_path/"evaluation").exists()


def test_numerical_run_and_finalize_are_delegated_without_global_dispatch_changes(tmp_path,monkeypatch):
    plan={"admission_schema":study.ADMISSION_SCHEMA,"campaign_schema":study.campaign.SCHEMA}
    calls=[]
    monkeypatch.setattr(study.numerical,"verify",lambda root:plan)
    monkeypatch.setattr(study.numerical,"run",lambda root,phase:calls.append(("run",root,phase)) or {"unchanged":"run"})
    monkeypatch.setattr(study.numerical,"finalize",lambda root:calls.append(("finalize",root)) or {"unchanged":"finalize"})
    original_admission=study.numerical.admit_campaign
    assert study.run(tmp_path,"compact")=={"unchanged":"run"}
    assert study.finalize(tmp_path)=={"unchanged":"finalize"}
    assert calls==[("run",tmp_path,"compact"),("finalize",tmp_path)]
    assert study.numerical.admit_campaign is original_admission
    plan["campaign_schema"]="unqualified"
    with pytest.raises(ValueError,match="admission"):study.run(tmp_path,"compact")
    assert len(calls)==2
