"""Coordinator contracts; GPU math has separate focused tests."""
import pytest
from experiments.structure_net import jspace_qwen_parallel as campaign


def test_groups_cover_every_block_once_in_reverse_stage_order():
    groups=[campaign.group_layers(i) for i in range(6)]
    assert groups[0]==list(range(30,36))
    assert groups[-1]==list(range(6))
    assert sorted(sum(groups,[]))==list(range(36))
    for invalid in (-1,6,True,1.5):
        with pytest.raises(ValueError):campaign.group_layers(invalid)


def test_worker_commands_use_separate_reservations_and_module_name(tmp_path):
    command=campaign.leased_command(tmp_path,"relation",index=2,layer=19,card=campaign.CARDS[1])
    assert command[:4]==["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run"]
    assert command[command.index("--allocation")+1]=="GPU-8cf440df=5632"
    assert command[command.index("-m")+1]=="experiments.structure_net.jspace_qwen_parallel"
    assert command[-4:]==["--group","2","--layer","19"]
    assert "--locked" in command and "--offline" in command


def test_fingerprint_changes_for_quality_or_resource_protocol():
    config=dict(campaign.CONFIG)
    for key in ("rank","microbatch","minimum_speedup","maximum_accuracy_drop"):
        changed={**config,key:config[key]+1}
        assert campaign.digest(changed)!=campaign.digest(config)
    with pytest.raises(ValueError):campaign.digest({"invalid":float("nan")})


def test_outputs_append_only(tmp_path):
    path=tmp_path/"result.json"
    campaign.write(path,{"status":"completed"})
    with pytest.raises(FileExistsError):campaign.write(path,{"status":"replaced"})
