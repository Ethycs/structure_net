"""Prospectively recorded device-only recovery of the failed word-final pilot."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_word_final_train as base
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_memory_data import file_hash,write_json

PILOT=base.data.ROOT/"20261023_word_final_train_pilot_cuda_v2"
TRAIN=base.data.ROOT/"20261023_word_final_train_cuda_v2"
FAILED=base.PILOT
AMENDMENT=Path("docs/07 - Status Reports/2026-10-23_ncm-word-final-cuda-recovery.md")
_producing=base.producing


def producing(root,pilot,inputs):
    files=[Path(__file__),Path("experiments/structure_net/ncm_device_geometry.py"),AMENDMENT]
    p=_producing(root,pilot,list(inputs)+files+[FAILED/n for n in ("producer.json","worker_terminal_receipt.json")])
    p["config"]["systems_recovery"]="CPU_reference_RoPE_transferred_to_generator_device"
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (root/"producer_source"/f"cuda_recovery_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=base.data.prior.run.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def configure():
    if base.data.prior.run.q.read(FAILED/"worker_terminal_receipt.json")["exit_code"]!=1:
        raise ValueError("preserved observed failed pilot required")
    base.data.verify(base.data.prior.run.q.read(FAILED/"producer.json"))
    base.PILOT=PILOT
    base.TRAIN=TRAIN
    base.set_source_frame=set_source_frame
    base.producing=producing
    return base


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=["pilot","train"])
    configure().train(parser.parse_args().stage=="pilot")
