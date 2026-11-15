"""Only accepted, freshly verified task artifacts enter full-model evaluation."""
from pathlib import Path

import pytest

from experiments.structure_net import jspace_3b_feature_resident_final_gate as study


def completed(tmp_path,monkeypatch):
    root=tmp_path/"campaign";root.mkdir()
    plan=dict(schema=study.campaign.SCHEMA,fingerprint="campaign",seed_cut=23,seed_correct=47,
        seed_checkpoint_sha256="seed",input_sha256={},qualification={"qualified":True})
    study.base.write(root/"plan.json",plan)
    study.base.write(root/"outcome.json",dict(status="all_blocks_recovered",last_accepted_cut=0,compressed_blocks=36))
    previous="seed";results={};calls=[]
    for cut in range(22,-1,-1):
        directory=study.campaign.stage_dir(root,cut);directory.mkdir(parents=True)
        stage=dict(schema=study.campaign.SCHEMA,root_fingerprint="campaign",cut=cut,
            parent_checkpoint_sha256=previous,parent_correct=47,fits={})
        stage["fingerprint"]=study.base.digest(stage);study.base.write(directory/"plan.json",stage)
        for phase in study.campaign.PHASES:
            path=directory/phase;path.mkdir();result=dict(status="completed",passed=True)
            if phase=="train":
                (path/"selected.pt").write_bytes(f"scientific{cut}".encode())
                result.update(quality_pass=False,selected={"correct":44},quality={"passed":False},
                    selected_epoch=-1,selected_checkpoint_sha256=study.base.sha(path/"selected.pt"))
            if phase=="verify":
                (path/"accepted.pt").write_bytes(f"accepted{cut}".encode());previous=study.base.sha(path/"accepted.pt")
                accepted=dict(accepted=True,candidate=dict(checkpoint_sha256=previous,metrics={"correct":47},kind="best_task"),quality={"passed":True})
                result["acceptance"]=accepted
                study.base.write(path/"acceptance.json",accepted)
                study.base.write(path/"accepted_development.json",dict(records=[],metrics={"correct":47}))
            study.base.write(path/"result.json",result)
            study.base.write(path/"manifest.json",dict(status="completed",fingerprint=stage["fingerprint"],
                artifacts={"result.json":study.base.sha(path/"result.json")}))
            results[(cut,phase)]=result
    monkeypatch.setattr(study.campaign,"verify_root",lambda path:plan)
    def checked(path,phase,stage):calls.append((stage["cut"],phase));return results[(stage["cut"],phase)]
    monkeypatch.setattr(study.campaign,"checked",checked)
    return root,results,calls


def test_full_chain_uses_accepted_hashes_despite_scientific_quality_miss(tmp_path,monkeypatch):
    root,_,calls=completed(tmp_path,monkeypatch)
    accepted=study.admit_campaign(root)
    assert len(accepted["chain"])==23
    assert accepted["checkpoint"].endswith("block_00/verify/accepted.pt")
    assert accepted["selected_reference"].endswith("block_00/verify/accepted_development.json")
    for row in accepted["chain"]:
        assert row["correct"]==47 and row["scientific_selected"]["correct"]==44
        assert row["quality"]["passed"] and not row["scientific_quality"]["passed"]
        assert row["checkpoint_sha256"]!=row["scientific_checkpoint_sha256"]
    assert set(calls)=={(cut,phase) for cut in range(23) for phase in study.campaign.PHASES}


@pytest.mark.parametrize("change",["partial","wrong_parent","missing_feature_pilot","rejected_task","fresh_failed"])
def test_final_admission_refuses_unfinished_or_wrong_accepted_chain(tmp_path,monkeypatch,change):
    root,results,_=completed(tmp_path,monkeypatch)
    if change=="partial":
        (root/"outcome.json").write_text('{"status":"quality_stop","last_accepted_cut":1,"compressed_blocks":35}')
    elif change=="wrong_parent":
        path=root/"stages/block_10/plan.json";value=study.read(path)
        value["parent_checkpoint_sha256"]=results[(11,"train")]["selected_checkpoint_sha256"]
        value["fingerprint"]=study.base.digest({k:v for k,v in value.items() if k!="fingerprint"})
        path.write_text(__import__("json").dumps(value))
    elif change=="missing_feature_pilot":(root/"stages/block_10/feature-pilot/manifest.json").unlink()
    elif change=="rejected_task":results[(10,"verify")]["acceptance"]["accepted"]=False
    else:results[(10,"verify")]["passed"]=False
    with pytest.raises((ValueError,FileNotFoundError)):study.admit_campaign(root)


def test_confirmation_is_closed_until_full_accepted_chain(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    def deny(*a):raise ValueError("Full36 absent")
    monkeypatch.setattr(study,"admit_campaign",deny)
    monkeypatch.setattr(study.numerical.confirmation,"verify",lambda *a:pytest.fail("Confirmation opened before admission"))
    with pytest.raises(ValueError,match="Full36 absent"):
        study.prepare(tmp_path/"new",tmp_path/"campaign",tmp_path/"confirmation")
    assert not (tmp_path/"new").exists()


def test_final_numerics_use_accepted_artifact_and_do_not_mutate_original_dispatch(tmp_path,monkeypatch):
    plan=dict(admission_schema=study.ADMISSION_SCHEMA,campaign_schema=study.campaign.SCHEMA,
        acceptance_policy=study.campaign.acceptance.POLICY,selected_checkpoint_sha256="accepted",
        accepted_task_candidate={"checkpoint_sha256":"accepted"},scientific_selected_checkpoint_sha256="scientific")
    monkeypatch.setattr(study.numerical,"verify",lambda root:plan)
    calls=[];original=study.numerical.admit_campaign
    monkeypatch.setattr(study.numerical,"run",lambda root,phase:calls.append((root,phase)) or {"delegated":True})
    assert study.run(tmp_path,"compact_validate")=={"delegated":True}
    assert study.numerical.admit_campaign is original
    plan["selected_checkpoint_sha256"]="scientific"
    with pytest.raises(ValueError,match="final-gate admission"):
        study.run(tmp_path,"compact_validate")
    assert calls==[(tmp_path,"compact_validate")]
