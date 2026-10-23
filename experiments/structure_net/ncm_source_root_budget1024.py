"""Fresh fixed-budget follow-up; retain the failed256-update source-root trial."""
import argparse
from dataclasses import dataclass
from pathlib import Path

from experiments.structure_net import ncm_source_root_capacity as base
from experiments.structure_net.ncm_memory_data import file_hash,write_json

PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-source-root-budget1024-plan.md")
HYPOTHESIS="ncm-source-root-capacity-budget1024-v1"
PARENT=base.study.data.ROOT/"20261023_word_final_root_capacity_results_v1"
TRAIN=base.study.data.ROOT/"20261023_word_final_root_capacity_budget1024_train_v1"
PILOT=base.PILOT
ORIGINAL_PLAN=base.PLAN
_rate=base.rate
_producing=base.producing


@dataclass(frozen=True)
class Config(base.Config):
    updates:int=1024


def rate(step,config=Config()):
    return _rate(step,config)


def producing(root,pilot,rows,manifest):
    if pilot:raise ValueError("this follow-up reuses the exact unchanged two-update pilot")
    p=_producing(root,pilot,rows,manifest)
    files=[Path(__file__),PLAN]
    extra=[PARENT/n for n in ("campaign_results.json","ncm-word-final-source-root-capacity-v1_registry_readback.json")]
    for path in files+extra:p["input_hashes"][str(path)]=file_hash(path)
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (root/"producer_source"/f"budget1024_{i}_{path.name}").write_bytes(path.read_bytes())
    p["config"].update(actual_updates=1024,reused_exact_pilot=str(PILOT),
        parent256_preserved=True,initialization="fresh_from_word_final512_not_capacity256",
        scheduler_change="cosine_horizon1024_after_identical16warmup")
    p["fingerprint"]=base.study.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def configure():
    receipt=base.study.q.read(PARENT/"ncm-word-final-source-root-capacity-v1_registry_readback.json")
    if (receipt["status"]!="registered_and_read_back"
            or receipt["record"]["campaign_sha256"]!=file_hash(PARENT/"campaign_results.json")
            or base.study.q.read(PARENT/"campaign_results.json")["classification"]!="source_root_TRAIN_capacity_failed"):
        raise ValueError("archived registered256-update capacity failure required")
    if any(rate(i)!=_rate(i) for i in range(1,17)):
        raise ValueError("identical first16warmup rates required for pilot reuse")
    base.Config=Config
    base.rate=rate
    base.producing=producing
    base.TRAIN=TRAIN
    base.HYPOTHESIS=HYPOTHESIS
    return base


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["train"])
    parser.parse_args()
    configure().train(False)
