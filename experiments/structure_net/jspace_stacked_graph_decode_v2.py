"""Match Qwen's configured repetition penalty before launching graph decoding.

The prepared v1 was not executed: the checkpoint's generation configuration uses
repetition_penalty=1.05. Track prompt/generated-token membership on GPU and apply
that exact processor in both static modes. The HF control is unchanged.
"""
import argparse
import json
from pathlib import Path
import shutil
import torch
from torch.nn import functional as F

from . import jspace_stacked_graph_decode as original
from . import jspace_bench_tensor as base
from .jspace_bench_data import sha, write


DEFAULT_ROOT=base.ROOT/"data/experiments/jspace_bench_tensor/20261113_graph_v2"
original.CONFIG={**original.CONFIG,"repetition_penalty":1.05,
                 "prepared_v1":"not executed; required matching configured repetition penalty"}


def penalize(logits,seen,penalty):
    return torch.where(seen,torch.where(logits<0,logits*penalty,logits/penalty),logits)


class RepetitionGraphDecoder(original.GraphDecoder):
    def __init__(self,model):
        super().__init__(model)
        if model.generation_config.repetition_penalty!=original.CONFIG["repetition_penalty"]:
            raise ValueError("Unexpected repetition penalty")
        self.seen=torch.zeros(model.config.vocab_size,device="cuda:0",dtype=torch.bool)

    def choose(self,hidden):
        logits=penalize(F.linear(hidden,self.model.lm_head.weight).float(),self.seen,original.CONFIG["repetition_penalty"])
        logits.index_fill_(-1,self.eos,-torch.inf)
        token=logits.argmax(-1)
        self.seen.index_fill_(0,token,True)
        return token

    @torch.inference_mode()
    def prefill(self,ids):
        self.seen.zero_();self.seen.index_fill_(0,ids.reshape(-1),True)
        return super().prefill(ids)


original.GraphDecoder=RepetitionGraphDecoder


def prepare(root):
    base.cpu_only()
    if root.exists():raise FileExistsError(root)
    parent=base.verify(original.TIMING);base.checked(original.TIMING,"timing")
    extras=[Path(__file__),Path(original.__file__),base.ROOT/"tests/structure_net/test_jspace_graph_penalty.py"]
    value={**parent,"graph_config":original.CONFIG,
           "sources":{**parent["sources"],**{str(p):sha(p) for p in extras}},
           "input_sha256":{**parent["input_sha256"],str(original.TIMING/"result.json"):sha(original.TIMING/"result.json")}}
    value.pop("scientific_fingerprint");value["scientific_fingerprint"]=base.digest(value)
    root.mkdir(parents=True)
    for name in value["sources"]:
        path=Path(name);target=root/"producer"/path.relative_to(base.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(original.PACKED/"tasks.json",root/"tasks.json");write(root/"plan.json",value)
    return {"fingerprint":value["scientific_fingerprint"],"graph_config":original.CONFIG}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("prepare","run"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT);args=parser.parse_args()
    print(json.dumps({"prepare":prepare,"run":original.run}[args.stage](args.root),allow_nan=False),flush=True)
