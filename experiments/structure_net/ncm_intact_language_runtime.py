"""Add runtime provenance without changing the verified decoder input producer."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform

import torch
from experiments.structure_net import ncm_intact_language_control as control
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json

BASE = control.producing


def producing(root,stage,inputs,config):
    p = BASE(root,stage,inputs,{**config,"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
        "transformers":importlib.metadata.version("transformers"),"python":platform.python_version(),
        "selected_physical_gpus":os.environ.get("CUDA_VISIBLE_DEVICES","no_local_model")})
    p["source_hashes"][str(Path(__file__))] = file_hash(__file__)
    (root/"producer_source/runtime_wrapper.py").write_bytes(Path(__file__).read_bytes())
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command",choices=["pilot","generate","judge-pilot","grade","primary"])
    command = parser.parse_args().command; control.producing = producing
    if command in ("pilot","generate"): control.generate(command=="pilot")
    else: control.grade(command)
